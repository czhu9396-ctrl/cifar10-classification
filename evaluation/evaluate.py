"""模型评估脚本：加载 checkpoint 在测试集上评估并输出指标与混淆矩阵。

用法:
    python evaluation/evaluate.py --config confits/default.yaml \
        --checkpoint experiments/<exp_dir>/checkpoints/best.pth
"""

import argparse
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools.config import load_config, select_device  # noqa: E402
from tools.seed import set_seed  # noqa: E402
from datasets.cifar10 import build_dataloaders  # noqa: E402
from models import get_model  # noqa: E402
from evaluation.metrics import (  # noqa: E402
    CIFAR10_CLASSES,
    collect_predictions,
    evaluate_metrics,
    plot_confusion_matrix,
)


def main():
    parser = argparse.ArgumentParser(description="CIFAR10 模型评估")
    parser.add_argument("--config", type=str, default="confits/default.yaml", help="配置文件路径")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型权重路径 (.pth)")
    parser.add_argument("--save-dir", type=str, default=None, help="评估结果输出目录")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = select_device(cfg["experiment"].get("device", "auto"))
    set_seed(cfg["experiment"]["seed"])

    # 输出目录：默认放到 checkpoint 所在目录的上一级
    if args.save_dir is None:
        save_dir = os.path.join(os.path.dirname(os.path.dirname(args.checkpoint)), "eval")
    else:
        save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    print(f"设备: {device}")
    print(f"配置文件: {args.config}")
    print(f"模型权重: {args.checkpoint}")

    # 构建数据与模型
    _, _, test_loader = build_dataloaders(cfg)
    model = get_model(cfg).to(device)

    # 加载权重
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 收集预测结果
    preds, targets = collect_predictions(model, test_loader, device)

    # 计算指标
    acc, cm, report = evaluate_metrics(preds, targets, CIFAR10_CLASSES)

    print("\n================ 测试集评估结果 ================")
    print(f"Top-1 准确率: {acc * 100:.2f}%")
    print("\n分类报告:\n")
    print(report)

    # 保存混淆矩阵图与报告
    cm_path = os.path.join(save_dir, "confusion_matrix.png")
    plot_confusion_matrix(cm, CIFAR10_CLASSES, cm_path, title="CIFAR10 Confusion Matrix")

    report_path = os.path.join(save_dir, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Top-1 Accuracy: {acc * 100:.2f}%\n\n")
        f.write(report)

    print(f"\n混淆矩阵已保存: {cm_path}")
    print(f"分类报告已保存: {report_path}")


if __name__ == "__main__":
    main()
