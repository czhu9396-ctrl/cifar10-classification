"""模型层：网络定义与模型工厂。

通过 get_model(cfg) 根据配置中的 model.name 分发到不同模型。
"""

import torch.nn as nn

from .custom_cnn import CustomCNN
from .resnet import build_resnet18

# 可用的模型名称注册表
MODEL_REGISTRY = {
    "custom_cnn": "自写轻量 CNN（特征提取 + 分类头）",
    "resnet18": "公开预训练模型 ResNet18（微调）",
}


def get_model(cfg):
    """根据配置构建模型。"""
    model_cfg = cfg["model"]
    name = model_cfg["name"]
    num_classes = model_cfg.get("num_classes", 10)

    if name == "custom_cnn":
        return CustomCNN(
            num_classes=num_classes,
            base_channels=model_cfg.get("base_channels", 32),
        )
    elif name == "resnet18":
        return build_resnet18(
            num_classes=num_classes,
            pretrained=model_cfg.get("pretrained", True),
        )
    else:
        raise ValueError(
            f"未知模型名称: {name}，可用模型: {list(MODEL_REGISTRY.keys())}"
        )


__all__ = ["get_model", "CustomCNN", "build_resnet18", "MODEL_REGISTRY"]
