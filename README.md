# MIN: Mixture of Noise for Pre-Trained Model-Based Class-Incremental Learning --> [paper](https://arxiv.org/abs/2509.16738)

![Overall](source/overall_00.png)

# How to reproduce our method?

## Data
Six datasets are included in our experiments, i.e., CIFAR-100, CUB-200, ImageNet-A, ImageNet-R, FOOD-101 and Omnibenchmark.
1. **CIFAR-100**: it will be automatically downloaded by the code.
2. **ImageNet-A**: Onedrive: [link](https://entuedu-my.sharepoint.com/personal/n2207876b_e_ntu_edu_sg/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fn2207876b%5Fe%5Fntu%5Fedu%5Fsg%2FDocuments%2FRevisitingCIL%2Fina%2Ezip&parent=%2Fpersonal%2Fn2207876b%5Fe%5Fntu%5Fedu%5Fsg%2FDocuments%2FRevisitingCIL&ga=1)
3. **Omnibenchmark**: Onedrive: [link](https://entuedu-my.sharepoint.com/personal/n2207876b_e_ntu_edu_sg/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fn2207876b%5Fe%5Fntu%5Fedu%5Fsg%2FDocuments%2FRevisitingCIL%2Fomnibenchmark%2Ezip&parent=%2Fpersonal%2Fn2207876b%5Fe%5Fntu%5Fedu%5Fsg%2FDocuments%2FRevisitingCIL&ga=1)
4. **CUB-200**: AWS: [link](https://s3.amazonaws.com/fast-ai-imageclas/CUB_200_2011.tgz)
5. **FOOD-101**: AWS: [link](https://s3.amazonaws.com/fast-ai-imageclas/food-101.tgz)
6. **ImageNet-R**: github: [link](https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar)
Then, revise the data path in data_process/data.py.

## Install
The core dependencies are as follows:
1. **torch==2.0.0+cu118** or **torch==2.8.0+cu12.8**
2. **torchvision==0.15.1+cu118** or **torchvision==0.23.0+cu118**
3. **timm**
4. **scikit-learn**
5. **scipy**

## How to run?
Use the script "run.sh" to reproduce the restlts of 10 steps setting:

**"sh run.sh"**

## Log Files
We hasv tested our code before uploading. The log files can be found in MiN/log. Due to the equipment change, we have to use 5090 to reproduce our results with torch==2.8.0+cu12.8. The result is basically the same as that stated in the paper.

## Main results

**CIFAR ViT-B/16-IN21K:**

![CIFAR](source/cifar100_in21k_00.png)

**CUB ViT-B/16-IN21K:**

![CUB](source/cub_in21k_00.png)

**IN-A ViT-B/16-IN21K:**

![IN-A](source/ina_in21k_00.png)

**IN-R ViT-B/16-IN21K:**

![IN-R](source/inr_in21k_00.png)

**FOOD ViT-B/16-IN21K:**

![IN-R](source/food_in21k_00.png)

**Omni. ViT-B/16-IN21K:**

![IN-R](source/omni_in21k_00.png)

