"""优化器与学习率调度器工厂。"""

import torch.optim as optim


def get_optimizer(cfg, params):
    """根据配置构建优化器。"""
    opt_cfg = cfg["optimizer"]
    name = opt_cfg.get("name", "sgd")
    lr = opt_cfg.get("lr", 0.01)
    weight_decay = opt_cfg.get("weight_decay", 0.0)

    if name == "sgd":
        return optim.SGD(
            params,
            lr=lr,
            momentum=opt_cfg.get("momentum", 0.9),
            weight_decay=weight_decay,
            nesterov=True,
        )
    elif name == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif name == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"未知优化器: {name}，可用: sgd / adam / adamw")


def get_scheduler(cfg, optimizer):
    """根据配置构建学习率调度器，返回 None 表示不调度。"""
    sch_cfg = cfg.get("scheduler", {})
    name = sch_cfg.get("name", "none")

    if name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=sch_cfg.get("T_max", 50))
    elif name == "step":
        return optim.lr_scheduler.StepLR(
            optimizer, step_size=sch_cfg.get("step_size", 30), gamma=sch_cfg.get("gamma", 0.1)
        )
    elif name in ("none", None):
        return None
    else:
        raise ValueError(f"未知调度器: {name}，可用: cosine / step / none")
