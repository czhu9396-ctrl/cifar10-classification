"""模型 checkpoint 保存与加载。"""

import os

import torch


def save_checkpoint(state, path):
    """保存 checkpoint 到指定路径（自动创建目录）。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path, model, device="cpu"):
    """加载 checkpoint，恢复模型权重。

    Returns:
        checkpoint dict（含 epoch、best_acc、config 等元信息）。
    """
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    return ckpt
