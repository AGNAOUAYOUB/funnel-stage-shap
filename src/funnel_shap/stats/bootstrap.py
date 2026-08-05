"""Stratified bootstrap confidence intervals (protocol Sec. 4.2, Sec. 10).

Every reported metric carries a bootstrapped 95% CI. Resampling is *stratified*
by the label: with ~15% conversion prevalence (Dataset A) an unstratified
resample can produce folds whose positive count swings enough to dominate the
interval width, and PR-AUC is especially sensitive to prevalence drift.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

Metric = Callable[[np.ndarray, np.ndarray], float]


@dataclass(frozen=True)
class BootstrapResult:
    point: float
    ci_low: float
    ci_high: float
    n_resamples: int
    n_valid: int
    alpha: float

    def as_row(self) -> dict[str, float]:
        return {
            "point": self.point,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "n_resamples": float(self.n_valid),
        }

    def __str__(self) -> str:
        return f"{self.point:.4f} [{self.ci_low:.4f}, {self.ci_high:.4f}]"


def _stratified_indices(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    return np.concatenate(
        [
            rng.choice(pos, size=pos.size, replace=True),
            rng.choice(neg, size=neg.size, replace=True),
        ]
    )


def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric: Metric,
    *,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> BootstrapResult:
    """Percentile bootstrap CI for ``metric(y_true, y_score)``.

    Resamples that a metric cannot be computed on (e.g. a degenerate draw) are
    dropped and counted in ``n_valid`` rather than silently treated as zero.
    """
    if n_resamples < 2000:
        raise ValueError("protocol Sec. 4.2 requires >=2000 resamples")

    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()
    rng = np.random.default_rng(seed)

    point = float(metric(y_true, y_score))
    draws: list[float] = []
    for _ in range(n_resamples):
        idx = _stratified_indices(y_true, rng)
        try:
            draws.append(float(metric(y_true[idx], y_score[idx])))
        except ValueError:
            continue

    if not draws:
        raise RuntimeError("no bootstrap resample produced a valid metric value")

    arr = np.asarray(draws)
    low, high = np.percentile(arr, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return BootstrapResult(
        point=point,
        ci_low=float(low),
        ci_high=float(high),
        n_resamples=n_resamples,
        n_valid=len(draws),
        alpha=alpha,
    )


def bootstrap_ci_multi(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metrics: dict[str, Metric],
    *,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, BootstrapResult]:
    """CIs for several metrics, drawing each resample **once**.

    Calling :func:`bootstrap_ci` per metric redraws the indices every time, so
    reporting three metrics costs three times the resampling for no extra
    information. Sharing the draw is both faster and slightly more correct: all
    three intervals then describe the same resampled populations, so they can be
    read against one another rather than as three independent experiments.
    """
    if n_resamples < 2000:
        raise ValueError("protocol Sec. 4.2 requires >=2000 resamples")
    if not metrics:
        raise ValueError("no metrics given")

    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()
    rng = np.random.default_rng(seed)

    points = {name: float(fn(y_true, y_score)) for name, fn in metrics.items()}
    draws: dict[str, list[float]] = {name: [] for name in metrics}

    for _ in range(n_resamples):
        idx = _stratified_indices(y_true, rng)
        resampled_true = y_true[idx]
        resampled_score = y_score[idx]
        for name, fn in metrics.items():
            try:
                draws[name].append(float(fn(resampled_true, resampled_score)))
            except ValueError:
                continue

    results: dict[str, BootstrapResult] = {}
    for name in metrics:
        values = draws[name]
        if not values:
            raise RuntimeError(f"no bootstrap resample produced a valid value for {name!r}")
        arr = np.asarray(values)
        low, high = np.percentile(arr, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        results[name] = BootstrapResult(
            point=points[name],
            ci_low=float(low),
            ci_high=float(high),
            n_resamples=n_resamples,
            n_valid=len(values),
            alpha=alpha,
        )
    return results


def paired_bootstrap_diff(
    y_true: np.ndarray,
    y_score_a: np.ndarray,
    y_score_b: np.ndarray,
    metric: Metric,
    *,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> BootstrapResult:
    """CI for ``metric(a) - metric(b)`` using the *same* resample for both models.

    Used for PR-AUC differences, where no analytic analogue of DeLong exists.
    Sharing the resample indices preserves the pairing and yields a much tighter
    interval than differencing two independent CIs.
    """
    if n_resamples < 2000:
        raise ValueError("protocol Sec. 4.2 requires >=2000 resamples")

    y_true = np.asarray(y_true).ravel()
    y_score_a = np.asarray(y_score_a).ravel()
    y_score_b = np.asarray(y_score_b).ravel()
    rng = np.random.default_rng(seed)

    point = float(metric(y_true, y_score_a)) - float(metric(y_true, y_score_b))
    draws: list[float] = []
    for _ in range(n_resamples):
        idx = _stratified_indices(y_true, rng)
        try:
            draws.append(
                float(metric(y_true[idx], y_score_a[idx]))
                - float(metric(y_true[idx], y_score_b[idx]))
            )
        except ValueError:
            continue

    if not draws:
        raise RuntimeError("no bootstrap resample produced a valid metric value")

    arr = np.asarray(draws)
    low, high = np.percentile(arr, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return BootstrapResult(
        point=point,
        ci_low=float(low),
        ci_high=float(high),
        n_resamples=n_resamples,
        n_valid=len(draws),
        alpha=alpha,
    )
