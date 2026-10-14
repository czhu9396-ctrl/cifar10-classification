"""评估指标：Top-k 准确率、混淆矩阵、分类报告。"""

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# CIFAR10 类别名称（按官方顺序）
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def accuracy(output, target, topk=(1,)):
    """计算 Top-k 准确率（用于训练循环，返回百分比）。

    Args:
        output: 模型 logits，shape (N, C)
        target: 标签，shape (N,)
        topk: 需要计算的 top-k 列表，如 (1, 5)
    Returns:
        list[float]: 每个 k 对应的准确率百分比
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res


@torch.no_grad()
def collect_predictions(model, loader, device):
    """在整个数据加载器上收集预测结果与真实标签。"""
    model.eval()
    all_preds, all_targets = [], []
    for images, targets in loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_targets.append(targets.numpy())
    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    return preds, targets


def evaluate_metrics(preds, targets, class_names=CIFAR10_CLASSES):
    """计算整体准确率、混淆矩阵与分类报告。"""
    acc = accuracy_score(targets, preds)
    cm = confusion_matrix(targets, preds)
    report = classification_report(
        targets, preds, target_names=class_names, digits=4
    )
    return acc, cm, report


def plot_confusion_matrix(cm, class_names, save_path, title="Confusion Matrix"):
    """绘制并保存混淆矩阵热力图。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # 在格子上标注数值
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
