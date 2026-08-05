"""Tests for paired comparisons and multiple-comparison control (protocol Sec. 12)."""

from __future__ import annotations

import numpy as np
import pytest

from funnel_shap.evaluate.comparisons import (
    ComparisonPlan,
    benjamini_hochberg_adjust,
    holm_adjust,
    mcnemar_test,
    run_plan,
    wilcoxon_across_folds,
)


@pytest.fixture
def three_models():
    rng = np.random.default_rng(42)
    n = 4000
    y = rng.binomial(1, 0.13, size=n)
    strong = rng.normal(loc=y * 2.0, scale=1.0)
    medium = rng.normal(loc=y * 1.0, scale=1.0)
    weak = rng.normal(loc=y * 0.1, scale=1.0)
    return y, {"strong": strong, "medium": medium, "weak": weak}


# ---------------------------------------------------------------------------
# McNemar
# ---------------------------------------------------------------------------


def test_mcnemar_detects_asymmetric_disagreement() -> None:
    y = np.array([1] * 100 + [0] * 100)
    a = y.copy()
    b = y.copy()
    b[:30] = 1 - b[:30]  # B wrong on 30 cases A gets right

    result = mcnemar_test(y, a, b)
    assert result["n10"] == 30
    assert result["n01"] == 0
    assert result["p_value"] < 0.001


def test_mcnemar_is_null_for_identical_predictions() -> None:
    y = np.array([1, 0, 1, 0, 1, 0])
    result = mcnemar_test(y, y, y)
    assert result["p_value"] == 1.0
    assert result["n01"] == 0 and result["n10"] == 0


def test_mcnemar_uses_exact_test_when_discordant_count_is_small() -> None:
    """The chi-square approximation is unreliable on a handful of disagreements."""
    y = np.array([1] * 50 + [0] * 50)
    a = y.copy()
    b = y.copy()
    b[:4] = 1 - b[:4]

    result = mcnemar_test(y, a, b)
    assert result["exact"] == 1.0
    assert result["p_value"] == pytest.approx(2 * 0.5**4)


def test_mcnemar_symmetric_disagreement_is_not_significant() -> None:
    y = np.array([1] * 200 + [0] * 200)
    a = y.copy()
    b = y.copy()
    a[:40] = 1 - a[:40]
    b[200:240] = 1 - b[200:240]

    result = mcnemar_test(y, a, b)
    assert result["p_value"] > 0.05


# ---------------------------------------------------------------------------
# Wilcoxon
# ---------------------------------------------------------------------------


def test_wilcoxon_reports_min_attainable_p_for_five_seeds() -> None:
    """With 5 seeds the smallest two-sided p is 0.0625 -- it cannot reach 0.05."""
    a = np.array([0.71, 0.72, 0.73, 0.74, 0.75])
    b = np.array([0.60, 0.61, 0.62, 0.63, 0.64])

    result = wilcoxon_across_folds(a, b)
    assert result["min_attainable_p"] == pytest.approx(0.0625)
    assert result["p_value"] >= 0.0625
    assert result["median_diff"] > 0


def test_wilcoxon_handles_identical_scores() -> None:
    a = np.array([0.7, 0.7, 0.7, 0.7, 0.7])
    result = wilcoxon_across_folds(a, a)
    assert result["p_value"] == 1.0
    assert result["median_diff"] == 0.0


def test_wilcoxon_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="equal-length"):
        wilcoxon_across_folds(np.array([1.0, 2.0]), np.array([1.0]))


# ---------------------------------------------------------------------------
# Corrections
# ---------------------------------------------------------------------------


def test_holm_is_monotone_and_never_decreases_significance() -> None:
    raw = [0.001, 0.008, 0.039, 0.041, 0.9]
    adjusted = holm_adjust(raw)

    assert adjusted == sorted(adjusted)
    assert all(a >= r for a, r in zip(adjusted, raw, strict=True))
    assert all(a <= 1.0 for a in adjusted)


def test_holm_matches_hand_computation() -> None:
    raw = [0.01, 0.02, 0.03]
    # step-down: 3*0.01=0.03, 2*0.02=0.04, 1*0.03=0.03 -> monotone -> 0.04
    assert holm_adjust(raw) == pytest.approx([0.03, 0.04, 0.04])


def test_bh_is_less_conservative_than_holm() -> None:
    raw = [0.001, 0.008, 0.039, 0.041, 0.042, 0.9]
    holm = holm_adjust(raw)
    bh = benjamini_hochberg_adjust(raw)
    assert sum(p < 0.05 for p in bh) >= sum(p < 0.05 for p in holm)


def test_corrections_handle_empty_input() -> None:
    assert holm_adjust([]) == []
    assert benjamini_hochberg_adjust([]) == []


def test_correction_actually_suppresses_a_borderline_family() -> None:
    """The point of Sec. 12's mandate: 20 borderline tests should not all survive."""
    raw = [0.04] * 20
    adjusted = holm_adjust(raw)
    assert all(p > 0.05 for p in adjusted)


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------


def test_plan_requires_a_rationale() -> None:
    plan = ComparisonPlan()
    with pytest.raises(ValueError, match="rationale"):
        plan.add("a", "b", "   ")


def test_empty_plan_is_rejected(three_models) -> None:
    y, scores = three_models
    with pytest.raises(ValueError, match="pre-specified pairs"):
        run_plan(ComparisonPlan(), y, scores)


def test_run_plan_reports_effect_sizes_with_p_values(three_models) -> None:
    y, scores = three_models
    plan = ComparisonPlan().add("strong", "weak", "H1: does the stronger signal win?")

    frame = run_plan(plan, y, scores, n_resamples=2000)
    row = frame.to_dicts()[0]

    assert row["roc_auc_diff"] > 0
    assert row["roc_auc_diff_ci_low"] < row["roc_auc_diff"] < row["roc_auc_diff_ci_high"]
    assert row["delong_p"] < 1e-6
    assert row["pr_auc_diff"] > 0
    assert row["significant"] is True


def test_run_plan_applies_correction_across_the_family(three_models) -> None:
    y, scores = three_models
    plan = (
        ComparisonPlan()
        .add("strong", "medium", "confirmatory pair 1")
        .add("strong", "weak", "confirmatory pair 2")
        .add("medium", "weak", "confirmatory pair 3")
    )

    frame = run_plan(plan, y, scores, correction="holm", n_resamples=2000)
    assert frame.height == 3
    assert frame["family_size"][0] == 3
    # Adjusted p must never be below raw p.
    for row in frame.to_dicts():
        assert row["delong_p_holm"] >= row["delong_p"]


def test_run_plan_accepts_both_corrections(three_models) -> None:
    y, scores = three_models
    plan = ComparisonPlan().add("strong", "weak", "r")

    holm = run_plan(plan, y, scores, correction="holm", n_resamples=2000)
    bh = run_plan(plan, y, scores, correction="fdr_bh", n_resamples=2000)
    assert "delong_p_holm" in holm.columns
    assert "delong_p_fdr_bh" in bh.columns

    with pytest.raises(ValueError, match="unknown correction"):
        run_plan(plan, y, scores, correction="bonferroni")


def test_run_plan_includes_mcnemar_when_decisions_supplied(three_models) -> None:
    y, scores = three_models
    predictions = {name: (s > 0).astype(int) for name, s in scores.items()}
    plan = ComparisonPlan().add("strong", "weak", "r")

    frame = run_plan(plan, y, scores, predictions=predictions, n_resamples=2000)
    assert "mcnemar_p" in frame.columns
    assert "mcnemar_p_holm" in frame.columns


def test_run_plan_rejects_unknown_model_name(three_models) -> None:
    y, scores = three_models
    plan = ComparisonPlan().add("strong", "nonexistent", "r")
    with pytest.raises(KeyError, match="nonexistent"):
        run_plan(plan, y, scores, n_resamples=2000)
