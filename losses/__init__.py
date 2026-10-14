"""损失函数层：损失函数工厂。

支持:
    - cross_entropy: 标准交叉熵
    - label_smoothing: 带标签平滑的交叉熵（正则化，缓解过拟合）
"""

import torch.nn as nn


def get_loss(cfg):
    """根据配置构建损失函数。"""
    loss_cfg = cfg["loss"]
    name = loss_cfg.get("name", "cross_entropy")

    if name == "cross_entropy":
        return nn.CrossEntropyLoss()
    elif name == "label_smoothing":
        smoothing = loss_cfg.get("smoothing", 0.1)
        return nn.CrossEntropyLoss(label_smoothing=smoothing)
    else:
        raise ValueError(f"未知损失函数: {name}，可用: cross_entropy / label_smoothing")


__all__ = ["get_loss"]
