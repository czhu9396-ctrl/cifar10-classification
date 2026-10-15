\# CIFAR-10 图像分类



基于 PyTorch 的 CIFAR-10 图像分类项目，对比了 CustomCNN 与 ResNet18 的性能。



\## 实验结果



| 模型 | 训练轮数 | 验证集准确率 | 测试集准确率 |

|------|---------|-------------|-------------|

| CustomCNN | 50 | 88.68% | 87.71% |

| ResNet18 | 30 | 87.88% | - |



\## 使用方法



```bash

\# 训练 CustomCNN

python scripts/train.py --config confits/default.yaml



\# 训练 ResNet18

python scripts/train.py --config confits/resnet18.yaml



\# 测试

python scripts/test.py --config confits/default.yaml

