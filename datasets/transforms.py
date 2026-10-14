"""CIFAR10 数据增强策略。

训练集: 随机裁剪 + 随机水平翻转（标准数据增强，缓解过拟合）
验证/测试集: 仅归一化（保持与评估一致）
"""

import torchvision.transforms as T

# CIFAR10 训练集各通道均值与标准差（公开统计值）
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)


def get_train_transform():
    """训练集数据增强。"""
    return T.Compose(
        [
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(MEAN, STD),
        ]
    )


def get_eval_transform():
    """验证/测试集预处理（无随机增强）。"""
    return T.Compose(
        [
            T.ToTensor(),
            T.Normalize(MEAN, STD),
        ]
    )
