"""Stage-prefix feature engineering (protocol Sec. 8)."""

from .dictionary import FEATURE_DICTIONARY, availability_by_stage, feature_names
from .prefix_features import build_stage_features

__all__ = [
    "FEATURE_DICTIONARY",
    "availability_by_stage",
    "feature_names",
    "build_stage_features",
]
