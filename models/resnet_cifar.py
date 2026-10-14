import torch.nn as nn
from torchvision.models import resnet18


def get_resnet18_cifar(num_classes=10):
    """
    获取适配 CIFAR-10 (32x32) 的 ResNet18 模型。
    - 修改第一层卷积为 3x3, stride=1, padding=1
    - 去掉 MaxPool，防止特征图过早下采样
    - 修改全连接层输出为 num_classes
    """
    # 不加载 ImageNet 预训练权重，从头训练
    model = resnet18(weights=None)

    # 适配 CIFAR-10 的 32x32 输入
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()

    # 修改全连接层
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model