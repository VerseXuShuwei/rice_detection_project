"""
Feature-Space Critics for MIL Training

Recent Updates:
    - [2026-01-14] Initial: Refactored from losses/critics/

Modules:
    - feature_critic: Background suppression via prototype similarity
"""

from .feature_critic import FeatureCritic, apply_feature_repulsion

__all__ = [
    'FeatureCritic',
    'apply_feature_repulsion'
]
