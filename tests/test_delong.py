"""Verification of the vendored DeLong implementation (protocol Sec. 4.2).

The protocol requires this file to be unit-tested against a known example "so a
reviewer can reproduce it". Two independent references are used:

1. The AUC point estimate must equal ``sklearn.metrics.roc_auc_score``.
2. The AUC variance must equal a naive O(N^2) transcription of the DeLong (1988)
   structural-component definition, written below without any of Sun & Xu's
   sorting tricks. If the fast path had a ranking or tie-handling bug, these two
   would disagree.

A worked numerical example with hand-checkable values is pinned in
``test_reference_example`` so the file also serves as documentation.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from funnel_shap.stats.delong import delong_roc_test, delong_roc_variance


def _kernel(x: float, y: float) -> float:
    """DeLong's psi kernel: 1 if positive scores above negative, 0.5 on a tie."""
    if x > y:
        return 1.0
    if x == y:
        return 0.5
    return 0.0


def naive_delong_cov(y_true: np.ndarray, scores: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Reference implementation straight from DeLong et al. (1988).

    Deliberately O(m*n) per model with explicit loops. Slow, but it is a direct
    transcription of the published definition and shares no code with the fast
    path.
    """
    pos_idx = np.flatnonzero(y_true == 1)
    neg_idx = np.flatnonzero(y_true == 0)
    m, n = len(pos_idx), len(neg_idx)
    k = len(scores)

    v10 = np.zeros((k, m))  # component for each positive case
    v01 = np.zeros((k, n))  # component for each negative case
    aucs = np.zeros(k)

    for r, s in enumerate(scores):
        pos = s[pos_idx]
        neg = s[neg_idx]
        psi = np.array([[_kernel(p, q) for q in neg] for p in pos])
        aucs[r] = psi.mean()
        v10[r] = psi.mean(axis=1)
        v01[r] = psi.mean(axis=0)

    s10 = np.cov(v10, ddof=1).reshape(k, k)
    s01 = np.cov(v01, ddof=1).reshape(k, k)
    return aucs, s10 / m + s01 / n


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_auc_matches_sklearn(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = 300
    y = rng.binomial(1, 0.15, size=n)
    if y.sum() < 2 or (1 - y).sum() < 2:
        pytest.skip("degenerate draw")
    score = rng.normal(loc=y * 0.8, scale=1.0)

    auc, _ = delong_roc_variance(y, score)
    assert auc == pytest.approx(roc_auc_score(y, score), abs=1e-12)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_variance_matches_naive_reference(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = 200
    y = rng.binomial(1, 0.3, size=n)
    a = rng.normal(loc=y * 1.0, scale=1.0)
    b = rng.normal(loc=y * 0.4, scale=1.0)

    naive_aucs, naive_cov = naive_delong_cov(y, [a, b])
    fast = delong_roc_test(y, a, b)

    assert fast.auc_a == pytest.approx(naive_aucs[0], abs=1e-12)
    assert fast.auc_b == pytest.approx(naive_aucs[1], abs=1e-12)

    naive_var_diff = naive_cov[0, 0] + naive_cov[1, 1] - 2 * naive_cov[0, 1]
    assert fast.se_diff == pytest.approx(np.sqrt(naive_var_diff), rel=1e-10)


def test_ties_are_handled_by_midranks() -> None:
    """Heavily tied scores are where a naive rank implementation breaks."""
    rng = np.random.default_rng(7)
    y = rng.binomial(1, 0.4, size=240)
    # Coarse quantisation forces many exact ties.
    a = np.round(rng.normal(loc=y * 0.9, scale=1.0), 1)
    b = np.round(rng.normal(loc=y * 0.3, scale=1.0), 1)

    naive_aucs, naive_cov = naive_delong_cov(y, [a, b])
    fast = delong_roc_test(y, a, b)

    assert fast.auc_a == pytest.approx(naive_aucs[0], abs=1e-12)
    naive_var_diff = naive_cov[0, 0] + naive_cov[1, 1] - 2 * naive_cov[0, 1]
    assert fast.se_diff == pytest.approx(np.sqrt(naive_var_diff), rel=1e-10)


def test_identical_models_give_null_result() -> None:
    rng = np.random.default_rng(3)
    y = rng.binomial(1, 0.2, size=150)
    s = rng.normal(size=150)

    res = delong_roc_test(y, s, s)
    assert res.diff == pytest.approx(0.0, abs=1e-12)
    assert res.se_diff == pytest.approx(0.0, abs=1e-12)
    assert res.p_value == pytest.approx(1.0)


def test_clearly_better_model_is_detected() -> None:
    rng = np.random.default_rng(11)
    n = 2000
    y = rng.binomial(1, 0.15, size=n)
    strong = rng.normal(loc=y * 2.0, scale=1.0)
    weak = rng.normal(loc=y * 0.1, scale=1.0)

    res = delong_roc_test(y, strong, weak)
    assert res.diff > 0.2
    assert res.p_value < 1e-10
    # The effect-size CI must exclude zero and bracket the point estimate.
    assert res.ci_low > 0
    assert res.ci_low < res.diff < res.ci_high


def test_reference_example() -> None:
    """Small hand-checkable case: 4 positives, 4 negatives, perfect separation.

    Every positive scores above every negative, so AUC = 1.0 and both sets of
    structural components are constant, giving zero variance.
    """
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    s = np.array([0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1])

    auc, var = delong_roc_variance(y, s)
    assert auc == pytest.approx(1.0)
    assert var == pytest.approx(0.0, abs=1e-15)

    # Reversing the scores flips the AUC to 0.0.
    auc_rev, _ = delong_roc_variance(y, -s)
    assert auc_rev == pytest.approx(0.0)


def test_rejects_non_binary_labels() -> None:
    y = np.array([0, 1, 2, 1, 0, 1])
    s = np.arange(6, dtype=float)
    with pytest.raises(ValueError, match="binary"):
        delong_roc_variance(y, s)


def test_rejects_too_few_per_class() -> None:
    y = np.array([1, 0, 0, 0, 0])
    s = np.arange(5, dtype=float)
    with pytest.raises(ValueError, match=">=2 samples per class"):
        delong_roc_variance(y, s)
