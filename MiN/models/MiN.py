import math
import numpy as np
from tqdm import tqdm
import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
import gc 

from utils.inc_net import MiNbaseNet
from utils.toolkit import tensor2numpy, calculate_class_metrics, calculate_task_metrics
from utils.training_tool import get_optimizer, get_scheduler

# [FIX 1: Safe Import]
try:
    from torch.amp import autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast, GradScaler

class MinNet(object):
    def __init__(self, args, loger):
        super().__init__()
        self.args = args
        self.logger = loger
        self._network = MiNbaseNet(args)
        self.device = args['device']
        self.num_workers = args["num_workers"]

        self.init_epochs = args["init_epochs"]
        self.init_lr = args["init_lr"]
        self.init_weight_decay = args["init_weight_decay"]
        self.init_batch_size = args["init_batch_size"]
        self.lr = args["lr"]
        self.batch_size = args["batch_size"]
        self.weight_decay = args["weight_decay"]
        self.epochs = args["epochs"]
        self.init_class = args["init_class"]
        self.increment = args["increment"]
        self.buffer_batch = args["buffer_batch"]
        self.fit_epoch = args["fit_epochs"]

        self.known_class = 0
        self.cur_task = -1
        self.scaler = GradScaler()
        self.old_prototypes = [] 
        self.total_acc = []

    def _clear_gpu(self):
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    def get_task_prototype(self, model, train_loader):
        model = model.eval()
        model.to(self.device)
        features = []
        with torch.no_grad():
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(self.device)
                with autocast(enabled=True):
                    feature = model.extract_feature(inputs)
                features.append(feature.detach().cpu())
        
        all_features = torch.cat(features, dim=0)
        prototype = torch.mean(all_features, dim=0).to(self.device)
        self._clear_gpu()
        return prototype

    def compute_adaptive_scale(self, current_loader):
        curr_proto = self.get_task_prototype(self._network, current_loader)
        
        if not self.old_prototypes:
            self.old_prototypes.append(curr_proto)
            return 1.0 
            
        max_sim = 0.0
        curr_norm = F.normalize(curr_proto.unsqueeze(0), p=2, dim=1)
        for old_p in self.old_prototypes:
            old_norm = F.normalize(old_p.unsqueeze(0), p=2, dim=1)
            sim = torch.mm(curr_norm, old_norm.t()).item()
            if sim > max_sim: max_sim = sim
                
        self.old_prototypes.append(curr_proto)
        
        scale = 0.5 + 0.5 * (1.0 - max_sim)
        scale = max(0.5, min(scale, 1.0))
        
        self.logger.info(f"--> [ADAPTIVE SGP] Similarity: {max_sim:.4f} => Scale: {scale:.4f}")
        return scale

    def init_train(self, data_manger):
        self.cur_task += 1
        train_list, test_list, _ = data_manger.get_task_list(0)
        train_set = data_manger.get_task_data(source="train", class_list=train_list)
        
        # [FIX 2: IN-PLACE MAPPING] Khôi phục cách cũ để đảm bảo dataset nhận nhãn mới
        train_set.labels = self.cat2order(train_set.labels, data_manger)
        
        test_set = data_manger.get_task_data(source="test", class_list=test_list)
        test_set.labels = self.cat2order(test_set.labels, data_manger)

        train_loader = DataLoader(train_set, batch_size=self.init_batch_size, shuffle=True, num_workers=self.num_workers)
        self.test_loader = DataLoader(test_set, batch_size=self.init_batch_size, shuffle=False, num_workers=self.num_workers)

        if self.args['pretrained']:
            for param in self._network.backbone.parameters(): param.requires_grad = True

        self._network.update_fc(self.init_class)
        self._network.update_noise()
        
        self.run(train_loader)
        
        # [CHIẾN THUẬT]: Threshold 0.9, không limit
        self._network.collect_projections(mode='threshold', val=0.9)
        self._clear_gpu()
        
        train_loader_buf = DataLoader(train_set, batch_size=self.buffer_batch, shuffle=True, num_workers=self.num_workers)
        self.fit_fc(train_loader_buf)
        self.compute_adaptive_scale(train_loader) 

    def increment_train(self, data_manger):
        self.cur_task += 1
        train_list, test_list, _ = data_manger.get_task_list(self.cur_task)
        train_set = data_manger.get_task_data(source="train", class_list=train_list)
        train_set.labels = self.cat2order(train_set.labels, data_manger)
        test_set = data_manger.get_task_data(source="test", class_list=test_list)
        test_set.labels = self.cat2order(test_set.labels, data_manger)

        train_loader = DataLoader(train_set, batch_size=self.buffer_batch, shuffle=True, num_workers=self.num_workers)
        self.test_loader = DataLoader(test_set, batch_size=self.buffer_batch, shuffle=False, num_workers=self.num_workers)

        if self.args['pretrained']:
            for param in self._network.backbone.parameters(): param.requires_grad = False

        self.fit_fc(train_loader) 
        self._network.update_fc(self.increment)
        
        train_loader_run = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self._network.update_noise()
        self._clear_gpu()

        self.run(train_loader_run)
        
        self._network.collect_projections(mode='threshold', val=0.9)
        self._clear_gpu()

        train_set_no_aug = data_manger.get_task_data(source="train_no_aug", class_list=train_list)
        train_set_no_aug.labels = self.cat2order(train_set_no_aug.labels, data_manger)
        train_loader_no_aug = DataLoader(train_set_no_aug, batch_size=self.buffer_batch, shuffle=True, num_workers=self.num_workers)
        self.re_fit(train_loader_no_aug)

    def run(self, train_loader):
        epochs = self.init_epochs if self.cur_task == 0 else self.epochs
        lr = self.init_lr if self.cur_task == 0 else self.lr
        weight_decay = self.init_weight_decay if self.cur_task == 0 else self.weight_decay

        for param in self._network.parameters(): param.requires_grad = False
        for param in self._network.normal_fc.parameters(): param.requires_grad = True
        
        if self.cur_task == 0: self._network.init_unfreeze()
        else: self._network.unfreeze_noise()
            
        params = filter(lambda p: p.requires_grad, self._network.parameters())
        optimizer = get_optimizer(self.args['optimizer_type'], params, lr, weight_decay)
        scheduler = get_scheduler(self.args['scheduler_type'], optimizer, epochs)

        current_scale = 0.85
        if self.cur_task > 0:
            current_scale = self.compute_adaptive_scale(train_loader)

        self._network.train()
        self._network.to(self.device)
        prog_bar = tqdm(range(epochs))
        WARMUP_EPOCHS = 5

        for _, epoch in enumerate(prog_bar):
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad(set_to_none=True) 

                with autocast(enabled=True): 
                    if self.cur_task > 0:
                        with torch.no_grad():
                            out1 = self._network(inputs, new_forward=False)['logits']
                        out2 = self._network.forward_normal_fc(inputs, new_forward=False)['logits']
                        logits = out2 + out1
                    else:
                        logits = self._network.forward_normal_fc(inputs, new_forward=False)['logits']
                    loss = F.cross_entropy(logits, targets.long())

                self.scaler.scale(loss).backward()
                
                if self.cur_task > 0 and epoch >= WARMUP_EPOCHS:
                    self.scaler.unscale_(optimizer)
                    self._network.apply_gpm_to_grads(scale=current_scale)
                
                self.scaler.step(optimizer)
                self.scaler.update()
                losses += loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = 100. * correct / total
            info = f"Task {self.cur_task} | Ep {epoch+1}/{epochs} | Loss {losses/len(train_loader):.3f} | Acc {train_acc:.2f}"
            prog_bar.set_description(info)

    def fit_fc(self, train_loader):
        self._network.eval()
        self._network.to(self.device)
        for _, inputs, targets in train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            targets = F.one_hot(targets, num_classes=self._network.known_class)
            self._network.fit(inputs, targets)
        self._clear_gpu()

    def re_fit(self, train_loader):
        self._network.eval()
        self._network.to(self.device)
        for _, inputs, targets in train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            targets = F.one_hot(targets, num_classes=self._network.known_class)
            self._network.fit(inputs, targets)
        self._clear_gpu()

    def after_train(self, data_manger):
        if self.cur_task == 0: self.known_class = self.init_class
        else: self.known_class += self.increment
        _, test_list, _ = data_manger.get_task_list(self.cur_task)
        test_set = data_manger.get_task_data(source="test", class_list=test_list)
        test_set.labels = self.cat2order(test_set.labels, data_manger)
        test_loader = DataLoader(test_set, batch_size=self.init_batch_size, shuffle=False, num_workers=self.num_workers)
        eval_res = self.eval_task(test_loader)
        self.total_acc.append(round(float(eval_res['all_class_accy']*100.), 2))
        self.logger.info(f'total acc: {self.total_acc}')
        self.logger.info(f'avg_acc: {np.mean(self.total_acc):.2f}')
        print(f'total acc: {self.total_acc}')
        print(f'avg_acc: {np.mean(self.total_acc):.2f}')
        del test_set

    def save_check_point(self, path_name):
        torch.save(self._network.state_dict(), path_name)

    def compute_test_acc(self, test_loader):
        return self.eval_task(test_loader)['all_class_accy']

    def eval_task(self, test_loader):
        model = self._network.eval()
        pred, label = [], []
        with torch.no_grad():
            for i, (_, inputs, targets) in enumerate(test_loader):
                inputs = inputs.to(self.device)
                outputs = model(inputs)
                logits = outputs["logits"]
                predicts = torch.max(logits, dim=1)[1]
                pred.extend([int(predicts[i].cpu().numpy()) for i in range(predicts.shape[0])])
                label.extend([int(targets[i].cpu().numpy()) for i in range(targets.shape[0])])
        class_info = calculate_class_metrics(pred, label)
        task_info = calculate_task_metrics(pred, label, self.init_class, self.increment)
        return {
            "all_class_accy": class_info['all_accy'],
            "class_accy": class_info['class_accy'],
            "class_confusion": class_info['class_confusion_matrices'],
            "task_accy": task_info['all_accy'],
            "task_confusion": task_info['task_confusion_matrices'],
            "all_task_accy": task_info['task_accy'],
        }

    @staticmethod
    def cat2order(targets, datamanger):
        # [QUAN TRỌNG] Quay lại vòng lặp In-place để sửa đúng vào bộ nhớ của dataset
        for i in range(len(targets)):
            targets[i] = datamanger.map_cat2order(targets[i])
        return targets
