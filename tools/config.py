"""配置加载与设备选择工具。"""

import yaml
import torch


def load_config(path):
    """从 YAML 文件加载配置，返回 dict。"""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def select_device(device: str = "auto"):
    """解析设备名称。

    Args:
        device: "auto" / "cuda" / "cpu"
    """
    if device in ("cuda", "cpu"):
        return device
    # auto: 有 GPU 用 GPU，否则 CPU
    return "cuda" if torch.cuda.is_available() else "cpu"
