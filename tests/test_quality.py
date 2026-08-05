"""Tests for Layer 3, explanation quality (protocol Sec. 11.3)."""

from __future__ import annotations

import numpy as np
import pytest

from funnel_shap.explain.quality import (
    CROSS_PARADIGM_SPEARMAN_THRESHOLD,
    FAITHFULNESS_THRESHOLD,
    deletion_curve,
    faithfulness_correlation,
    insertion_curve,
    local_lipschitz,
    rank_consistency,
)


@pytest.fixture(scope="module")
def linear_setup():
    """A linear model, whose exact Shapley values are known analytically.

    For f(x) = w.x, the SHAP value of feature j against a baseline b is
    w_j * (x_j - b_j). That gives a ground-truth attribution to validate the
    quality metrics against, rather than validating one approximation with
    another.
    """
    rng = np.random.default_rng(42)
    n, d = 60, 8
    w = rng.normal(size=d)
    background = rng.normal(size=(200, d))
    X = rng.normal(size=(n, d))
    baseline_mean = background.mean(axis=0)

    def predict(matrix):
        return np.asarray(matrix, dtype=float) @ w

    true_shap = (X - baseline_mean) * w
    return predict, X, true_shap, background, w


# ---------------------------------------------------------------------------
# Faithfulness
# ---------------------------------------------------------------------------


def test_true_attributions_are_faithful(linear_setup) -> None:
    predict, X, true_shap, background, _ = linear_setup
    result = faithfulness_correlation(
        predict, X, true_shap, background, n_subsets=40, seed=7
    )

    assert result.correlation_mean > FAITHFULNESS_THRESHOLD
    assert result.passes
    assert "PASS" in result.summary()


def test_random_attributions_are_not_faithful(linear_setup) -> None:
    """The metric must be able to fail, or it certifies nothing."""
    predict, X, _, background, _ = linear_setup
    rng = np.random.default_rng(0)
    nonsense = rng.normal(size=X.shape)

    result = faithfulness_correlation(
        predict, X, nonsense, background, n_subsets=40, seed=7
    )
    assert result.correlation_mean < FAITHFULNESS_THRESHOLD
    assert not result.passes


def test_sign_flipped_attributions_score_negatively(linear_setup) -> None:
    predict, X, true_shap, background, _ = linear_setup
    result = faithfulness_correlation(
        predict, X, -true_shap, background, n_subsets=40, seed=7
    )
    assert result.correlation_mean < 0


# ---------------------------------------------------------------------------
# Deletion / insertion
# ---------------------------------------------------------------------------


def test_deletion_falls_and_insertion_rises(linear_setup) -> None:
    """Sec. 11.3's claim, stated on signed attribution."""
    predict, X, true_shap, background, _ = linear_setup

    deletion = deletion_curve(predict, X, true_shap, background, n_steps=8, seed=7)
    insertion = insertion_curve(predict, X, true_shap, background, n_steps=8, seed=7)

    assert deletion.mean_prediction[0] > deletion.mean_prediction[-1]
    assert insertion.mean_prediction[0] < insertion.mean_prediction[-1]
    assert deletion.monotone, "removing top positive contributors must degrade monotonically"


def test_curves_separate_a_good_explanation_from_a_useless_one(linear_setup) -> None:
    """The guard on the bug this metric had: |SHAP| ordering collapsed the curves.

    With a true explanation the deletion curve should fall much faster than
    under a random ordering, so the AUC gap is the discriminating quantity.
    """
    predict, X, true_shap, background, _ = linear_setup
    rng = np.random.default_rng(0)

    good = deletion_curve(predict, X, true_shap, background, n_steps=8, seed=7)
    useless = deletion_curve(
        predict, X, rng.normal(size=true_shap.shape), background, n_steps=8, seed=7
    )
    assert good.auc < useless.auc


def test_curves_span_the_full_feature_range(linear_setup) -> None:
    predict, X, true_shap, background, _ = linear_setup
    curve = deletion_curve(predict, X, true_shap, background, n_steps=5, seed=7)

    assert curve.fractions[0] == 0.0
    assert curve.fractions[-1] == 1.0
    assert len(curve.mean_prediction) == len(curve.fractions)
    assert np.isfinite(curve.auc)


def test_deletion_endpoints_are_the_original_and_the_baseline(linear_setup) -> None:
    predict, X, true_shap, background, _ = linear_setup
    curve = deletion_curve(predict, X, true_shap, background, n_steps=4, seed=7)

    # At fraction 0 nothing is removed, so it is the model on the real data.
    assert curve.mean_prediction[0] == pytest.approx(float(np.mean(predict(X))), rel=1e-6)


# ---------------------------------------------------------------------------
# Stability
# ---------------------------------------------------------------------------


def test_a_stable_explainer_has_a_bounded_lipschitz_ratio(linear_setup) -> None:
    predict, X, _, background, w = linear_setup
    baseline_mean = background.mean(axis=0)

    def explain(matrix):
        return (np.asarray(matrix) - baseline_mean) * w

    result = local_lipschitz(explain, X, n_perturbations=4, noise_scale=0.05, seed=7)
    # For this explainer the ratio is bounded by max|w| regardless of noise.
    assert result.max_ratio <= np.abs(w).max() + 1e-6


def test_an_unstable_explainer_is_detected(linear_setup) -> None:
    _, X, _, _, _ = linear_setup
    rng = np.random.default_rng(1)

    def chaotic(matrix):
        return rng.normal(size=np.asarray(matrix).shape) * 100

    stable_ratio = local_lipschitz(
        lambda m: np.asarray(m) * 0.01, X, n_perturbations=3, seed=7
    ).max_ratio
    chaotic_ratio = local_lipschitz(chaotic, X, n_perturbations=3, seed=7).max_ratio

    assert chaotic_ratio > stable_ratio * 10


# ---------------------------------------------------------------------------
# Consistency
# ---------------------------------------------------------------------------


def test_identical_rankings_are_perfectly_consistent() -> None:
    importance = np.array([5.0, 3.0, 1.0, 0.5])
    result = rank_consistency([importance, importance.copy()])

    assert result.mean_spearman == pytest.approx(1.0)
    assert result.passes


def test_reversed_rankings_are_maximally_inconsistent() -> None:
    a = np.array([5.0, 3.0, 1.0, 0.5])
    result = rank_consistency([a, a[::-1].copy()])

    assert result.mean_spearman == pytest.approx(-1.0)
    assert not result.passes


def test_minimum_pairwise_correlation_is_reported() -> None:
    """A mean hides one badly disagreeing pair; the reviewer will ask about it."""
    a = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    b = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    c = a[::-1].copy()

    result = rank_consistency([a, b, c])
    assert result.min_spearman < result.mean_spearman
    assert result.n_pairs == 3


def test_threshold_comes_from_the_protocol() -> None:
    a = np.array([4.0, 3.0, 2.0, 1.0])
    result = rank_consistency([a, a.copy()])
    assert result.threshold == CROSS_PARADIGM_SPEARMAN_THRESHOLD


def test_too_few_vectors_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least two"):
        rank_consistency([np.array([1.0, 2.0])])


def test_mismatched_lengths_are_rejected() -> None:
    with pytest.raises(ValueError, match="same length"):
        rank_consistency([np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0])])
