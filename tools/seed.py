"""随机种子设置，保证实验可复现。"""

import random

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 保证卷积等算子结果确定（会略微降低速度，科研复现时建议开启）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
