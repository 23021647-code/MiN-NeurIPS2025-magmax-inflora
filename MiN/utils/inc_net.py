import copy
import logging
import math
import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from backbones.pretrained_backbone import get_pretrained_backbone
from backbones.linears import SimpleLinear
# Import autocast để tắt nó trong quá trình tính toán ma trận chính xác cao
from torch.cuda.amp import autocast 

class BaseIncNet(nn.Module):
    def __init__(self, args: dict):
        super(BaseIncNet, self).__init__()
        self.args = args
        self.backbone = get_pretrained_backbone(args)
        self.feature_dim = self.backbone.out_dim
        self.fc = None

    def update_fc(self, nb_classes):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

    @staticmethod
    def generate_fc(in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc

    def forward(self, x):
        hyper_features = self.backbone(x)
        logits = self.fc(hyper_features)['logits']
        return {
            'features': hyper_features,
            'logits': logits
        }


class RandomBuffer(torch.nn.Linear):
    """
    Lớp mở rộng đặc trưng ngẫu nhiên (Random Projection).
    """
    def __init__(self, in_features: int, buffer_size: int, device):
        super(torch.nn.Linear, self).__init__()
        self.bias = None
        self.in_features = in_features
        self.out_features = buffer_size
        
        # [QUAN TRỌNG] Sử dụng float32 để đảm bảo độ chính xác khi tính RLS
        factory_kwargs = {"device": device, "dtype": torch.float32}
        
        self.W = torch.empty((self.in_features, self.out_features), **factory_kwargs)
        self.register_buffer("weight", self.W)

        self.reset_parameters()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # Ép kiểu input X về cùng kiểu với weight (float32)
        X = X.to(self.weight.dtype)
        return F.relu(X @ self.W)


class MiNbaseNet(nn.Module):
    def __init__(self, args: dict):
        super(MiNbaseNet, self).__init__()
        self.args = args
        self.backbone = get_pretrained_backbone(args)
        self.device = args['device']
        
        # Các tham số cho Analytic Learning (RLS)
        self.gamma = args['gamma']
        self.buffer_size = args['buffer_size']
        self.feature_dim = self.backbone.out_dim 

        # Random Buffer
        self.buffer = RandomBuffer(in_features=self.feature_dim, buffer_size=self.buffer_size, device=self.device)

        # Khởi tạo ma trận trọng số và ma trận hiệp phương sai cho RLS
        # Dùng float32 để tránh lỗi singular matrix khi tính nghịch đảo
        factory_kwargs = {"device": self.device, "dtype": torch.float32}

        weight = torch.zeros((self.buffer_size, 0), **factory_kwargs)
        self.register_buffer("weight", weight) # Trọng số của Analytic Classifier

        self.R: torch.Tensor
        R = torch.eye(self.weight.shape[0], **factory_kwargs) / self.gamma
        self.register_buffer("R", R) # Ma trận hiệp phương sai đảo (Inverse Covariance Matrix)

        # Normal FC: Dùng để train Gradient Descent cho Noise Generator
        self.normal_fc = None
        self.cur_task = -1
        self.known_class = 0

    def update_fc(self, nb_classes):
        """
        Cập nhật lớp Normal FC (cho việc training Noise).
        Lớp Analytic FC (self.weight) sẽ tự động mở rộng trong hàm fit().
        """
        self.cur_task += 1
        self.known_class += nb_classes
        
        # Tạo mới Normal FC cho task hiện tại
        if self.cur_task > 0:
            # Task sau: Không dùng Bias để tránh bias vào lớp mới quá nhiều
            new_fc = SimpleLinear(self.buffer_size, self.known_class, bias=False)
        else:
            # Task đầu: Có bias
            # [FIX LỖI TẠI ĐÂY]: Đổi fc thành new_fc
            new_fc = SimpleLinear(self.buffer_size, nb_classes, bias=True)
            
        if self.normal_fc is not None:
            # Sequential Init: Copy trọng số cũ
            old_nb_output = self.normal_fc.out_features
            with torch.no_grad():
                # Copy phần cũ
                new_fc.weight[:old_nb_output] = self.normal_fc.weight.data
                # Init phần mới về 0
                nn.init.constant_(new_fc.weight[old_nb_output:], 0.)
            
            del self.normal_fc
            self.normal_fc = new_fc
        else:
            # Task đầu tiên
            nn.init.constant_(new_fc.weight, 0.)
            if new_fc.bias is not None:
                nn.init.constant_(new_fc.bias, 0.)
            self.normal_fc = new_fc

    # =========================================================================
    # [MAGMAX & NOISE CONTROL SECTION]
    # =========================================================================
    
    def update_noise(self):
        """
        Gọi khi bắt đầu Task mới.
        Kích hoạt chế độ Sequential Initialization trong PiNoise.
        """
        for j in range(self.backbone.layer_num):
            self.backbone.noise_maker[j].update_noise()

    def after_task_magmax_merge(self):
        """
        Gọi sau khi kết thúc Task.
        Kích hoạt việc LƯU (Save) và TRỘN (Merge) tham số theo MagMax.
        """
        print(f"--> [IncNet] Task {self.cur_task}: Triggering Parameter-wise MagMax Merging...")
        for j in range(self.backbone.layer_num):
            # Hàm này nằm trong PiNoise
            self.backbone.noise_maker[j].after_task_training()

    def unfreeze_noise(self):
        """Gọi cho Task > 0: Chỉ unfreeze Noise thưa"""
        for j in range(len(self.backbone.noise_maker)):
            self.backbone.noise_maker[j].unfreeze_incremental()

    def init_unfreeze(self):
        for j in range(len(self.backbone.noise_maker)):
            self.backbone.noise_maker[j].unfreeze_task_0()
            
            # Giữ LayerNorm trainable ở Task 0 để ổn định base
            if hasattr(self.backbone.blocks[j], 'norm1'):
                for p in self.backbone.blocks[j].norm1.parameters(): p.requires_grad = True
            if hasattr(self.backbone.blocks[j], 'norm2'):
                for p in self.backbone.blocks[j].norm2.parameters(): p.requires_grad = True
                
        if hasattr(self.backbone, 'norm') and self.backbone.norm is not None:
            for p in self.backbone.norm.parameters(): p.requires_grad = True
    # =========================================================================
    # [ANALYTIC LEARNING (RLS) SECTION]
    # =========================================================================

    def forward_fc(self, features):
        """Forward qua Analytic Classifier"""
        # Đảm bảo features cùng kiểu với trọng số RLS (float32)
        features = features.to(self.weight.dtype) 
        return features @ self.weight
# Trong class MiNbaseNet
    @torch.no_grad()
    def fit(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        """
        Phiên bản RLS tối ưu bộ nhớ (Memory-Efficient RLS)
        """
        # Tắt Autocast để tính toán chính xác FP32 (tránh lỗi Singular Matrix)
        try:
            from torch.amp import autocast
        except ImportError:
            from torch.cuda.amp import autocast

        with autocast('cuda', enabled=False):
            # 1. Feature Extraction & Expansion
            X = self.backbone(X).float() 
            X = self.buffer(X) 
            
            # Đảm bảo cùng device
            X, Y = X.to(self.weight.device), Y.to(self.weight.device).float()

            # 2. Mở rộng chiều của classifier nếu có lớp mới
            num_targets = Y.shape[1]
            if num_targets > self.weight.shape[1]:
                increment_size = num_targets - self.weight.shape[1]
                tail = torch.zeros((self.weight.shape[0], increment_size), device=self.weight.device)
                self.weight = torch.cat((self.weight, tail), dim=1)
            elif num_targets < self.weight.shape[1]:
                # Trường hợp hiếm: Padding Y cho khớp weight cũ
                increment_size = self.weight.shape[1] - num_targets
                tail = torch.zeros((Y.shape[0], increment_size), device=Y.device)
                Y = torch.cat((Y, tail), dim=1)

            # 3. RLS Update (Tối ưu OOM)
            # Công thức: P = (I + X R X^T)^-1
            # term kích thước [Batch x Batch]. Nếu Batch lớn (Buffer), cái này rất nặng.
            
            term = torch.eye(X.shape[0], device=X.device) + X @ self.R @ X.T
            jitter = 1e-6 * torch.eye(term.shape[0], device=term.device)
            
            # Dùng linalg.solve nhanh và ổn định hơn torch.inverse
            try:
                # K = (X R X^T + I)^-1 @ (X R)
                # Kích thước [Batch x Buffer Dim]
                K = torch.linalg.solve(term + jitter, X @ self.R)
                K = K.T # Transpose về [Buffer Dim x Batch]
            except:
                # Fallback nếu lỗi
                K = self.R @ X.T @ torch.inverse(term + jitter)

            # Cập nhật R và Weight
            self.R -= K @ X @ self.R
            self.weight += K @ (Y - X @ self.weight)
            
            # [QUAN TRỌNG] Xóa ngay lập tức để giải phóng VRAM cho batch sau
            del term, jitter, K, X, Y    # =========================================================================
    # [FORWARD PASSES]
    # =========================================================================

    def forward(self, x, new_forward: bool = False):
        if new_forward:
            hyper_features = self.backbone(x, new_forward=True)
        else:
            hyper_features = self.backbone(x)
        
        # [SỬA]: Đảm bảo đặc trưng đồng nhất kiểu dữ liệu trước khi vào Buffer
        hyper_features = hyper_features.to(self.weight.dtype)
        
        # Buffer trả về ReLU(X @ W), forward_fc thực hiện X @ Weight
        logits = self.forward_fc(self.buffer(hyper_features))
        
        return {'logits': logits}
    def extract_feature(self, x):
        """Chỉ trích xuất đặc trưng từ Backbone"""
        return self.backbone(x)

    def forward_normal_fc(self, x, new_forward: bool = False):
        if new_forward:
            hyper_features = self.backbone(x, new_forward=True)
        else:
            hyper_features = self.backbone(x)
        
        # [SỬA]: Buffer thường chứa trọng số FP32, ép hyper_features lên FP32 
        # để phép nhân trong Buffer diễn ra chính xác trước khi đưa vào Classifier
        hyper_features = self.buffer(hyper_features.to(self.buffer.weight.dtype))
        
        # Sau đó ép về kiểu của normal_fc (thường là Half nếu dùng autocast)
        hyper_features = hyper_features.to(self.normal_fc.weight.dtype)
        
        logits = self.normal_fc(hyper_features)['logits']
        return {"logits": logits}
    def collect_projections(self):
        """
        Duyệt qua các lớp PiNoise và yêu cầu chúng tính toán ma trận chiếu 
        dựa trên các đặc trưng đã thu thập được trong Task vừa qua.
        """
        for j in range(self.backbone.layer_num):
            # Giả sử noise_maker[j] là đối tượng PiNoise đã được cập nhật hàm compute_projection_matrix
            self.backbone.noise_maker[j].compute_projection_matrix(threshold=0.95)

    def apply_gpm_to_grads(self):
        """
        Thực hiện chiếu trực giao gradient cho mu và sigma.
        """
        for j in range(self.backbone.layer_num):
            self.backbone.noise_maker[j].apply_gradient_projection()
