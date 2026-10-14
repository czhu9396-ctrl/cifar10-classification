"""评估层：指标计算与模型评估。"""

from .metrics import (
    CIFAR10_CLASSES,
    accuracy,
    collect_predictions,
    evaluate_metrics,
    plot_confusion_matrix,
)

__all__ = [
    "CIFAR10_CLASSES",
    "accuracy",
    "collect_predictions",
    "evaluate_metrics",
    "plot_confusion_matrix",
]
