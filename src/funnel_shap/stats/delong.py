"""Fast DeLong test for paired ROC-AUC comparison (protocol Sec. 4.2, Sec. 12).

Neither scikit-learn nor statsmodels ships this test, so the algorithm is
vendored here rather than pulled from an unpinned third-party gist. The
implementation follows:

    X. Sun and W. Xu (2014), "Fast Implementation of DeLong's Algorithm for
    Comparing the Areas Under Correlated Receiver Operating Characteristic
    Curves", IEEE Signal Processing Letters 21(11), 1389-1393.

which computes the same structural components as

    E. R. DeLong, D. M. DeLong and D. L. Clarke-Pearson (1988), "Comparing the
    Areas under Two or More Correlated Receiver Operating Characteristic
    Curves: A Nonparametric Approach", Biometrics 44(3), 837-845

in O(N log N) instead of O(N^2), using midranks so that tied scores are handled
correctly.

`tests/test_delong.py` pins this module against (a) scikit-learn's
`roc_auc_score` for the point estimate and (b) a deliberately naive O(N^2)
transcription of the 1988 definition for the covariance, so a reviewer can
verify the fast path without trusting it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


def _midrank(x: np.ndarray) -> np.ndarray:
    """Midranks of ``x`` (1-based), averaging ranks within tied groups."""
    order = np.argsort(x)
    sorted_x = x[order]
    n = len(x)
    ranks_sorted = np.zeros(n, dtype=float)

    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks_sorted[i:j] = 0.5 * (i + j - 1)
        i = j

    ranks = np.empty(n, dtype=float)
    ranks[order] = ranks_sorted + 1
    return ranks


def _fast_delong(scores_sorted: np.ndarray, n_pos: int) -> tuple[np.ndarray, np.ndarray]:
    """Core Sun & Xu computation.

    Parameters
    ----------
    scores_sorted:
        Array of shape (k, N): k models' scores for the same N samples, with the
        columns ordered so that all ``n_pos`` positive-label samples come first.
    n_pos:
        Number of positive-label samples.

    Returns
    -------
    (aucs, cov):
        ``aucs`` of shape (k,) and the estimated AUC covariance matrix of shape
        (k, k).
    """
    m = n_pos
    n = scores_sorted.shape[1] - m
    if m < 2 or n < 2:
        raise ValueError(f"DeLong needs >=2 samples per class, got {m} positive / {n} negative")

    k = scores_sorted.shape[0]
    pos = scores_sorted[:, :m]
    neg = scores_sorted[:, m:]

    tx = np.empty((k, m), dtype=float)
    ty = np.empty((k, n), dtype=float)
    tz = np.empty((k, m + n), dtype=float)
    for r in range(k):
        tx[r] = _midrank(pos[r])
        ty[r] = _midrank(neg[r])
        tz[r] = _midrank(scores_sorted[r])

    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / (2.0 * n)

    # Structural components (DeLong 1988, eq. for V10/V01).
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m

    sx = np.atleast_2d(np.cov(v01))
    sy = np.atleast_2d(np.cov(v10))
    cov = sx / m + sy / n
    return aucs, cov


def _prepare(y_true: np.ndarray, *score_arrays: np.ndarray) -> tuple[np.ndarray, int]:
    y_true = np.asarray(y_true).ravel()
    uniq = np.unique(y_true)
    if not np.all(np.isin(uniq, [0, 1])):
        raise ValueError(f"y_true must be binary 0/1, found labels {uniq}")

    scores = np.vstack([np.asarray(s, dtype=float).ravel() for s in score_arrays])
    if scores.shape[1] != y_true.shape[0]:
        raise ValueError("y_true and score arrays must have the same length")

    # Positives first, as the fast algorithm assumes.
    order = np.argsort(-y_true, kind="mergesort")
    return scores[:, order], int(y_true.sum())


def delong_roc_variance(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    """Return ``(auc, variance)`` for a single model's ROC-AUC."""
    scores_sorted, n_pos = _prepare(y_true, y_score)
    aucs, cov = _fast_delong(scores_sorted, n_pos)
    return float(aucs[0]), float(cov[0, 0])


@dataclass(frozen=True)
class DelongResult:
    """Outcome of a paired DeLong comparison.

    The protocol (Sec. 12) requires the effect size and its CI alongside the
    p-value, so all three are carried together.
    """

    auc_a: float
    auc_b: float
    diff: float
    se_diff: float
    z: float
    p_value: float
    ci_low: float
    ci_high: float
    alpha: float

    def as_row(self) -> dict[str, float]:
        return {
            "auc_a": self.auc_a,
            "auc_b": self.auc_b,
            "auc_diff": self.diff,
            "se_diff": self.se_diff,
            "z": self.z,
            "p_value": self.p_value,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
        }


def delong_roc_test(
    y_true: np.ndarray,
    y_score_a: np.ndarray,
    y_score_b: np.ndarray,
    *,
    alpha: float = 0.05,
) -> DelongResult:
    """Two-sided DeLong test for AUC(a) - AUC(b) on the same samples.

    Both score vectors must be predictions for the *same* test rows in the same
    order; that pairing is what makes the test more powerful than comparing two
    independent AUC confidence intervals.
    """
    scores_sorted, n_pos = _prepare(y_true, y_score_a, y_score_b)
    aucs, cov = _fast_delong(scores_sorted, n_pos)

    diff = float(aucs[0] - aucs[1])
    var_diff = float(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    se = float(np.sqrt(max(var_diff, 0.0)))

    if se == 0.0:
        z, p = 0.0, 1.0
        half = 0.0
    else:
        z = diff / se
        p = float(2 * stats.norm.sf(abs(z)))
        half = float(stats.norm.ppf(1 - alpha / 2) * se)

    return DelongResult(
        auc_a=float(aucs[0]),
        auc_b=float(aucs[1]),
        diff=diff,
        se_diff=se,
        z=float(z),
        p_value=p,
        ci_low=diff - half,
        ci_high=diff + half,
        alpha=alpha,
    )
