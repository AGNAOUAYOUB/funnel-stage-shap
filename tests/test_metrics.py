"""Tests for predictive metrics and calibration (protocol Sec. 10)."""

from __future__ import annotations

import numpy as np
import pytest

from funnel_shap.evaluate.metrics import (
    aggregate_over_seeds,
    evaluate_predictions,
    expected_calibration_error,
    reliability_curve,
    select_threshold,
)


@pytest.fixture
def imbalanced():
    rng = np.random.default_rng(42)
    n = 3000
    y = rng.binomial(1, 0.13, size=n)
    # A well-separated, roughly calibrated score.
    logit = -2.0 + 3.0 * y + rng.normal(scale=1.0, size=n)
    p = 1 / (1 + np.exp(-logit))
    return y, p


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def test_perfect_calibration_has_near_zero_ece() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(0.02, 0.98, size=200_000)
    y = rng.binomial(1, p)
    assert expected_calibration_error(y, p, n_bins=10) < 0.01


def test_systematically_overconfident_model_has_large_ece() -> None:
    rng = np.random.default_rng(0)
    n = 50_000
    true_p = np.full(n, 0.10)
    y = rng.binomial(1, true_p)
    claimed = np.full(n, 0.60)  # claims 60% when the truth is 10%
    assert expected_calibration_error(y, claimed, n_bins=10) == pytest.approx(0.5, abs=0.02)


def test_ece_handles_probabilities_at_the_boundaries() -> None:
    y = np.array([0, 0, 1, 1])
    p = np.array([0.0, 0.0, 1.0, 1.0])
    assert expected_calibration_error(y, p, n_bins=10) == pytest.approx(0.0)


def test_unknown_binning_strategy_is_rejected(imbalanced) -> None:
    y, p = imbalanced
    with pytest.raises(ValueError, match="binning strategy"):
        expected_calibration_error(y, p, strategy="sqrt")


def test_reliability_curve_returns_counts(imbalanced) -> None:
    """A bin's fraction-positive is unreadable without knowing its size."""
    y, p = imbalanced
    curve = reliability_curve(y, p, n_bins=10)

    assert set(curve) == {"mean_predicted", "fraction_positive", "count"}
    assert curve["count"].sum() == len(y)
    assert len(curve["mean_predicted"]) == len(curve["count"])
    # Quantile bins should be roughly equal-sized.
    assert curve["count"].std() / curve["count"].mean() < 0.5


def test_reliability_curve_is_monotone_for_a_good_model(imbalanced) -> None:
    y, p = imbalanced
    curve = reliability_curve(y, p, n_bins=5)
    frac = curve["fraction_positive"]
    assert frac[0] < frac[-1]


# ---------------------------------------------------------------------------
# Threshold selection
# ---------------------------------------------------------------------------


def test_selected_threshold_beats_the_default_on_f1(imbalanced) -> None:
    """With 13% prevalence, 0.5 is rarely the F1-optimal operating point."""
    from sklearn.metrics import f1_score

    y, p = imbalanced
    threshold = select_threshold(y, p, objective="f1")

    assert f1_score(y, (p >= threshold).astype(int)) >= f1_score(y, (p >= 0.5).astype(int))


def test_cost_objective_shifts_threshold_with_relative_costs(imbalanced) -> None:
    y, p = imbalanced
    cheap_fn = select_threshold(y, p, objective="cost", cost_fp=1.0, cost_fn=1.0)
    dear_fn = select_threshold(y, p, objective="cost", cost_fp=1.0, cost_fn=20.0)
    # Missing a converter costing 20x should make us predict positive more often.
    assert dear_fn < cheap_fn


def test_unknown_objective_is_rejected(imbalanced) -> None:
    y, p = imbalanced
    with pytest.raises(ValueError, match="unknown objective"):
        select_threshold(y, p, objective="youden")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def test_report_contains_every_protocol_metric(imbalanced) -> None:
    y, p = imbalanced
    report = evaluate_predictions(y, p, threshold=0.3, n_resamples=2000, seed=7)

    for name in ("pr_auc", "roc_auc", "f1", "precision", "recall", "accuracy", "brier", "ece"):
        assert name in report.point

    assert report.prevalence == pytest.approx(y.mean())
    assert report.n == len(y)
    assert report.n_positive == int(y.sum())


def test_intervals_bracket_their_point_estimates(imbalanced) -> None:
    y, p = imbalanced
    report = evaluate_predictions(y, p, threshold=0.3, n_resamples=2000, seed=7)

    for name, ci in report.intervals.items():
        assert ci.ci_low <= report.point[name] <= ci.ci_high, name


def test_accuracy_is_high_even_for_a_useless_model(imbalanced) -> None:
    """The reason Sec. 10 says report it but never headline it."""
    y, _ = imbalanced
    always_negative = np.zeros(len(y))
    report = evaluate_predictions(
        y, always_negative, threshold=0.5, with_intervals=False
    )

    assert report.point["accuracy"] > 0.85
    assert report.point["f1"] == 0.0
    assert report.point["recall"] == 0.0


def test_threshold_is_an_argument_not_a_choice(imbalanced) -> None:
    """Sec. 10: the threshold is fixed before test, so evaluation cannot pick it."""
    y, p = imbalanced
    low = evaluate_predictions(y, p, threshold=0.1, with_intervals=False)
    high = evaluate_predictions(y, p, threshold=0.9, with_intervals=False)

    assert low.point["recall"] > high.point["recall"]
    # Threshold-free metrics must be identical regardless.
    assert low.point["pr_auc"] == pytest.approx(high.point["pr_auc"])
    assert low.point["roc_auc"] == pytest.approx(high.point["roc_auc"])


def test_aggregate_over_seeds_gives_mean_and_std(imbalanced) -> None:
    y, p = imbalanced
    rng = np.random.default_rng(1)
    reports = [
        evaluate_predictions(
            y, np.clip(p + rng.normal(scale=0.01, size=len(p)), 0, 1),
            threshold=0.3, with_intervals=False,
        )
        for _ in range(5)
    ]

    agg = aggregate_over_seeds(reports)
    mean, std = agg["pr_auc"]
    assert 0 < mean < 1
    assert std > 0


def test_aggregate_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="no reports"):
        aggregate_over_seeds([])
