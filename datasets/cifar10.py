"""CIFAR10 数据集加载与 DataLoader 构建。

- 自动下载数据到 cfg["data"]["root"]
- 从训练集按比例划分验证集（可复现）
- 返回 (train_loader, val_loader, test_loader)
"""

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from .transforms import get_train_transform, get_eval_transform


def build_dataloaders(cfg):
    data_cfg = cfg["data"]
    root = data_cfg["root"]
    batch_size = data_cfg["batch_size"]
    num_workers = data_cfg.get("num_workers", 2)
    val_ratio = data_cfg.get("val_ratio", 0.1)
    seed = cfg["experiment"].get("seed", 42)

    # 训练集使用数据增强，验证/测试集使用一致的无增强预处理
    train_ds = datasets.CIFAR10(root=root, train=True, download=True, transform=get_train_transform())
    val_ds = datasets.CIFAR10(root=root, train=True, download=True, transform=get_eval_transform())
    test_ds = datasets.CIFAR10(root=root, train=False, download=True, transform=get_eval_transform())

    val_loader = None
    if val_ratio > 0:
        n_total = len(train_ds)
        n_val = int(n_total * val_ratio)
        n_train = n_total - n_val

        # 用固定随机种子生成划分索引，保证可复现
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(n_total, generator=generator).tolist()
        train_set = Subset(train_ds, indices[:n_train])
        val_set = Subset(val_ds, indices[n_train:])

        val_loader = DataLoader(
            val_set, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        )
    else:
        train_set = train_ds

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader
