"""
Model architectures for MIL-based rice disease detection.

Exports:
    - get_model: Model factory function
    - MILEfficientNetV2S: EfficientNetV2-S backbone with MIL head
"""

from .builder import get_model
from .efficientnetv2_mil import MILEfficientNetV2S

__all__ = [
    'get_model',
    'MILEfficientNetV2S',
]
