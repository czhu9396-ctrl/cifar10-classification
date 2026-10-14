"""日志工具：同时输出到终端、文件与 TensorBoard。"""

import logging
import os

from torch.utils.tensorboard import SummaryWriter


def setup_logger(log_dir, name="train"):
    """初始化日志记录器与 TensorBoard writer。

    Args:
        log_dir: 日志输出目录。
        name: 日志文件前缀。
    Returns:
        (logger, writer)
    """
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S")

    # 终端输出
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    # 文件输出
    fh = logging.FileHandler(os.path.join(log_dir, f"{name}.log"), encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # TensorBoard
    writer = SummaryWriter(log_dir)

    return logger, writer
