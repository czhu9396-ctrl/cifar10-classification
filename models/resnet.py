"""公开模型封装：ResNet18。

支持加载 ImageNet 预训练权重，替换最后一层全连接为 CIFAR10 的 10 分类。
"""

import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def build_resnet18(num_classes: int = 10, pretrained: bool = True):
    """构建 ResNet18 模型。

    Args:
        num_classes: 分类类别数（CIFAR10 为 10）。
        pretrained: 是否加载 ImageNet 预训练权重。
    """
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)

    # 替换分类头，适配 CIFAR10 的类别数
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model
