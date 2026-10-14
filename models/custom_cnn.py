"""自写轻量 CNN：特征提取网络 + 分类头。

结构（VGG 风格，逐步加深）:
    - 3 个卷积块，通道数 32 -> 64 -> 128，每块含两层 3x3 卷积 + BN + ReLU
    - 每块后接 2x2 最大池化，空间尺寸 32x32 -> 16x16 -> 8x8 -> 4x4
    - 自适应平均池化 + 全连接分类头

特点：结构清晰、可解释性强，适合科研入门理解特征提取流程。
"""

import torch.nn as nn


class CustomCNN(nn.Module):
    def __init__(self, num_classes: int = 10, base_channels: int = 32):
        super().__init__()
        c = base_channels

        # ---- 特征提取部分 ----
        self.features = nn.Sequential(
            # 块 1: 3 -> c 通道, 32x32 -> 16x16
            nn.Conv2d(3, c, kernel_size=3, padding=1),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, kernel_size=3, padding=1),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # 块 2: c -> 2c 通道, 16x16 -> 8x8
            nn.Conv2d(c, c * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(c * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c * 2, c * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(c * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # 块 3: 2c -> 4c 通道, 8x8 -> 4x4
            nn.Conv2d(c * 2, c * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(c * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(c * 4, c * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(c * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # ---- 分类头 ----
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c * 4, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x
