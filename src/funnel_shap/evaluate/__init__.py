"""Predictive evaluation and statistical testing (protocol Sec. 10, 12)."""

from .metrics import (
    METRIC_FUNCTIONS,
    ClassificationReport,
    evaluate_predictions,
    expected_calibration_error,
    reliability_curve,
    select_threshold,
)

__all__ = [
    "METRIC_FUNCTIONS",
    "ClassificationReport",
    "evaluate_predictions",
    "expected_calibration_error",
    "reliability_curve",
    "select_threshold",
]
