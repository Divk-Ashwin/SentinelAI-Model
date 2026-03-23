"""
Fusion package for late fusion of multi-modal predictions
Combines predictions from text, image, and metadata pipelines
"""
from fusion.decision_fusion import DecisionFusion

__all__ = [
    "DecisionFusion"
]