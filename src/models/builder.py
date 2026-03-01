from typing import Dict, Any
import torch.nn as nn

# 示例,需完善
# 1. 导入自定义模型
from .efficientnetv2_mil import MILEfficientNetV2S
# 未来如果有新模型（比如 ResNet版），就在这里导入: from .resnet_mil import MILResNet50

def get_model(model_name: str, config: Dict[str, Any]) -> nn.Module:
    """
    Model Factory: 根据名称实例化对应的模型类。
    
    Args:
        model_name: 模型名称 (e.g., 'mil_efficientnetv2-s')
        config: 完整的配置字典，将传递给模型的 __init__
    
    Returns:
        实例化后的 PyTorch 模型
    """
    # 统一转小写，防止配置写错大小写
    name = model_name.lower()

    if name == 'mil_efficientnetv2-s':
        return MILEfficientNetV2S(config)
    
    # === 未来扩展区域 ===
    # elif name == 'mil_resnet50':
    #     return MILResNet50(config)
    
    else:
        raise ValueError(f"Unknown model name: {model_name}. Available: ['mil_efficientnetv2-s']")