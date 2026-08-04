"""Tests for the stratified bootstrap CI (protocol Sec. 4.2, Sec. 10)."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from funnel_shap.stats.bootstrap import bootstrap_ci, paired_bootstrap_diff


@pytest.fixture
def imbalanced_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """~15% prevalence, mirroring Dataset A (protocol Sec. 5.2)."""
    rng = np.random.default_rng(42)
    n = 1500
    y = rng.binomial(1, 0.15, size=n)
    strong = rng.normal(loc=y * 1.5, scale=1.0)
    weak = rng.normal(loc=y * 0.3, scale=1.0)
    return y, strong, weak


def test_ci_brackets_point_estimate(imbalanced_data) -> None:
    y, strong, _ = imbalanced_data
    res = bootstrap_ci(y, strong, roc_auc_score, n_resamples=2000, seed=7)

    assert res.point == pytest.approx(roc_auc_score(y, strong))
    assert res.ci_low < res.point < res.ci_high
    assert res.n_valid == 2000


def test_is_seeded_and_reproducible(imbalanced_data) -> None:
    y, strong, _ = imbalanced_data
    a = bootstrap_ci(y, strong, average_precision_score, n_resamples=2000, seed=17)
    b = bootstrap_ci(y, strong, average_precision_score, n_resamples=2000, seed=17)
    c = bootstrap_ci(y, strong, average_precision_score, n_resamples=2000, seed=23)

    assert (a.ci_low, a.ci_high) == (b.ci_low, b.ci_high)
    assert (a.ci_low, a.ci_high) != (c.ci_low, c.ci_high)


def test_stratification_preserves_prevalence(imbalanced_data) -> None:
    """The whole point of stratifying: every resample has the same class counts."""
    from funnel_shap.stats.bootstrap import _stratified_indices

    y, _, _ = imbalanced_data
    rng = np.random.default_rng(1)
    for _ in range(50):
        idx = _stratified_indices(y, rng)
        assert idx.size == y.size
        assert y[idx].sum() == y.sum()


def test_resample_floor_is_enforced(imbalanced_data) -> None:
    y, strong, _ = imbalanced_data
    with pytest.raises(ValueError, match=">=2000 resamples"):
        bootstrap_ci(y, strong, roc_auc_score, n_resamples=500)


def test_paired_diff_excludes_zero_for_separated_models(imbalanced_data) -> None:
    y, strong, weak = imbalanced_data
    res = paired_bootstrap_diff(
        y, strong, weak, average_precision_score, n_resamples=2000, seed=42
    )

    assert res.point > 0
    assert res.ci_low > 0
    assert res.ci_low < res.point < res.ci_high


def test_paired_diff_of_identical_models_is_zero(imbalanced_data) -> None:
    y, strong, _ = imbalanced_data
    res = paired_bootstrap_diff(
        y, strong, strong, average_precision_score, n_resamples=2000, seed=42
    )

    assert res.point == pytest.approx(0.0)
    assert res.ci_low == pytest.approx(0.0)
    assert res.ci_high == pytest.approx(0.0)


def test_paired_diff_is_tighter_than_differencing_independent_cis(imbalanced_data) -> None:
    """Sharing resample indices is what buys the power; verify it actually does."""
    y, strong, weak = imbalanced_data
    paired = paired_bootstrap_diff(
        y, strong, weak, average_precision_score, n_resamples=2000, seed=42
    )
    a = bootstrap_ci(y, strong, average_precision_score, n_resamples=2000, seed=42)
    b = bootstrap_ci(y, weak, average_precision_score, n_resamples=2000, seed=101)

    paired_width = paired.ci_high - paired.ci_low
    naive_width = (a.ci_high - b.ci_low) - (a.ci_low - b.ci_high)
    assert paired_width < naive_width
