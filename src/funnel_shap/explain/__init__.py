"""Stage-conditioned explanation layers (protocol Sec. 11)."""

from .stage_shap import (
    StageAttribution,
    attribution_migration,
    correlation_groups,
    grouped_attribution,
    stage_tree_shap,
)

__all__ = [
    "StageAttribution",
    "attribution_migration",
    "correlation_groups",
    "grouped_attribution",
    "stage_tree_shap",
]
