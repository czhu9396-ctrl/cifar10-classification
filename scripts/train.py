"""CIFAR10 分类训练主脚本。

用法:
    python scripts/train.py --config confits/default.yaml
    python scripts/train.py --config confits/resnet18.yaml
"""

import argparse
import os
import sys
from datetime import datetime

import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools.config import load_config, select_device  # noqa: E402
from tools.seed import set_seed  # noqa: E402
from tools.logger import setup_logger  # noqa: E402
from tools.checkpoint import save_checkpoint  # noqa: E402
from tools.optim import get_optimizer, get_scheduler  # noqa: E402
from datasets.cifar10 import build_dataloaders  # noqa: E402
from models import get_model  # noqa: E402
from losses import get_loss  # noqa: E402
from evaluation.metrics import accuracy  # noqa: E402


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, logger, log_interval):
    """训练一个 epoch，返回 (平均 loss, 平均 top-1 准确率)。"""
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    n = 0

    for i, (images, targets) in enumerate(loader):
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        acc = accuracy(outputs, targets)[0]
        bs = targets.size(0)
        running_loss += loss.item() * bs
        running_acc += acc * bs
        n += bs

        if (i + 1) % log_interval == 0:
            logger.info(
                f"Epoch {epoch} [{i + 1}/{len(loader)}] "
                f"loss={loss.item():.4f} acc={acc:.2f}%"
            )

    return running_loss / n, running_acc / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """在验证/测试集上评估，返回 (平均 loss, 平均 top-1 准确率)。"""
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    n = 0

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        loss = criterion(outputs, targets)
        acc = accuracy(outputs, targets)[0]
        bs = targets.size(0)
        running_loss += loss.item() * bs
        running_acc += acc * bs
        n += bs

    return running_loss / n, running_acc / n


def main():
    parser = argparse.ArgumentParser(description="CIFAR10 分类训练")
    parser.add_argument("--config", type=str, default="confits/default.yaml", help="配置文件路径")
    parser.add_argument("--exp-dir", type=str, default=None, help="实验输出目录（默认自动生成）")
    args = parser.parse_args()

    # 1. 加载配置与设备
    cfg = load_config(args.config)
    device = select_device(cfg["experiment"].get("device", "auto"))
    set_seed(cfg["experiment"]["seed"])

    # 2. 实验输出目录
    exp_name = cfg["experiment"]["name"]
    if args.exp_dir:
        exp_dir = args.exp_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = os.path.join("experiments", f"{exp_name}_{timestamp}")

    log_dir = os.path.join(exp_dir, "logs")
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    logger, writer = setup_logger(log_dir)

    # 保存本次实验配置快照，便于复现
    with open(os.path.join(exp_dir, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    logger.info(f"实验目录: {exp_dir}")
    logger.info(f"设备: {device}")

    # 3. 构建数据 / 模型 / 损失 / 优化器 / 调度器
    train_loader, val_loader, test_loader = build_dataloaders(cfg)
    model = get_model(cfg).to(device)
    criterion = get_loss(cfg)
    optimizer = get_optimizer(cfg, model.parameters())
    scheduler = get_scheduler(cfg, optimizer)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型可训练参数量: {n_params:,}")

    epochs = cfg["train"]["epochs"]
    log_interval = cfg["train"]["log_interval"]
    save_best = cfg["train"].get("save_best", True)
    best_acc = 0.0

    # 4. 训练循环
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, logger, log_interval
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        if scheduler is not None:
            scheduler.step()

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Acc/train", train_acc, epoch)
        writer.add_scalar("Acc/val", val_acc, epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        logger.info(
            f"Epoch {epoch}/{epochs}  "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}%  "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.2f}%"
        )

        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc

        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_acc": best_acc,
            "config": cfg,
        }
        if save_best and is_best:
            save_checkpoint(state, os.path.join(ckpt_dir, "best.pth"))

    # 保存最后一个 epoch 的模型
    save_checkpoint(state, os.path.join(ckpt_dir, "last.pth"))

    logger.info(f"训练完成，最佳验证准确率: {best_acc:.2f}%")
    logger.info(f"日志目录: {log_dir}")
    logger.info(f"模型权重目录: {ckpt_dir}")
    writer.close()


if __name__ == "__main__":
    main()
