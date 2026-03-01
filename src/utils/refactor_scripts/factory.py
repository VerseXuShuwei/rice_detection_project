import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, ConstantLR

## Optimizer/Scheduler 工厂

def create_scheduler(optimizer, config):
    """
    专门负责构建复杂的 LR Scheduler，包括 Trapezoidal 策略。
    """
    sched_cfg = config.get('scheduler', {})
    sched_name = sched_cfg.get('name', 'cosine').lower()

    if sched_name == 'trapezoidal':
        # 把原脚本里 create_scheduler 的代码搬到这里
        # ...
        return scheduler_dict
    # ... 其他分支

