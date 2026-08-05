"""Predictive metrics with bootstrapped CIs (protocol Sec. 10).

PR-AUC is the headline because conversion is rare: with ~13% prevalence, ROC-AUC
rewards a model for ranking the vast negative class correctly, which is not the
task. Accuracy is computed because Sec. 10 asks for it, and is explicitly
marked near-meaningless: predicting "no purchase" for everyone scores 87%.

Calibration is not an afterthought here. The intervention story in RQ4 depends
on predicted probabilities meaning what they say — "this visitor has a 40%
chance" has to be true 40% of the time, or a cost-based threshold is arbitrary.
Hence reliability curves, Brier score, and Expected Calibration Error, none of
which scikit-learn provides in the form Sec. 10 asks for.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ..stats.bootstrap import BootstrapResult, bootstrap_ci

#: Threshold-free metrics, safe to bootstrap directly on scores.
METRIC_FUNCTIONS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "pr_auc": average_precision_score,
    "roc_auc": roc_auc_score,
    "brier": brier_score_loss,
}

#: Sec. 10: reported, never headlined.
NEAR_MEANINGLESS = ("accuracy",)


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> float:
    """Weighted mean gap between confidence and accuracy across probability bins.

    ECE = sum_b (n_b / N) * |accuracy(b) - mean_confidence(b)|.

    ``strategy='quantile'` puts equal counts in each bin, which is the more
    honest choice under heavy class imbalance: with uniform bins most of the
    mass lands in the lowest bin and the statistic is dominated by it.
    """
    y_true = np.asarray(y_true).ravel().astype(float)
    y_prob = np.asarray(y_prob).ravel().astype(float)

    if strategy == "quantile":
        edges = np.unique(np.quantile(y_prob, np.linspace(0, 1, n_bins + 1)))
        if edges.size < 2:
            return 0.0
    elif strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        raise ValueError(f"unknown binning strategy {strategy!r}")

    # np.digitize is right-open; clip so the maximum lands in the last bin.
    idx = np.clip(np.digitize(y_prob, edges[1:-1], right=False), 0, len(edges) - 2)

    total = len(y_true)
    error = 0.0
    for b in range(len(edges) - 1):
        mask = idx == b
        n_b = int(mask.sum())
        if n_b == 0:
            continue
        error += (n_b / total) * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(error)


def reliability_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bins: int = 10,
    strategy: str = "quantile",
) -> dict[str, np.ndarray]:
    """Points for the Sec. 10 reliability diagram, plus per-bin counts.

    Counts are returned alongside because a reliability curve without them
    invites reading a wild swing in a bin holding nine samples as a calibration
    failure.
    """
    y_true = np.asarray(y_true).ravel().astype(float)
    y_prob = np.asarray(y_prob).ravel().astype(float)

    if strategy == "quantile":
        edges = np.unique(np.quantile(y_prob, np.linspace(0, 1, n_bins + 1)))
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)

    idx = np.clip(np.digitize(y_prob, edges[1:-1], right=False), 0, len(edges) - 2)

    mean_pred, frac_pos, counts = [], [], []
    for b in range(len(edges) - 1):
        mask = idx == b
        if not mask.any():
            continue
        mean_pred.append(float(y_prob[mask].mean()))
        frac_pos.append(float(y_true[mask].mean()))
        counts.append(int(mask.sum()))

    return {
        "mean_predicted": np.asarray(mean_pred),
        "fraction_positive": np.asarray(frac_pos),
        "count": np.asarray(counts),
    }


def select_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    objective: str = "f1",
    cost_fp: float = 1.0,
    cost_fn: float = 1.0,
) -> float:
    """Choose an operating threshold on the *validation* split (Sec. 10).

    Sec. 10 requires the threshold to be justified and fixed before test. This
    must never be called on test scores: doing so tunes the operating point to
    the evaluation set and inflates every threshold-dependent metric.
    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()

    candidates = np.unique(y_prob)
    if candidates.size > 512:
        candidates = np.quantile(y_prob, np.linspace(0.001, 0.999, 512))

    best_threshold, best_score = 0.5, -np.inf
    for threshold in candidates:
        pred = (y_prob >= threshold).astype(int)
        if objective == "f1":
            score = f1_score(y_true, pred, zero_division=0)
        elif objective == "cost":
            fp = int(((pred == 1) & (y_true == 0)).sum())
            fn = int(((pred == 0) & (y_true == 1)).sum())
            score = -(cost_fp * fp + cost_fn * fn)
        else:
            raise ValueError(f"unknown objective {objective!r}")

        if score > best_score:
            best_threshold, best_score = float(threshold), float(score)

    return best_threshold


@dataclass
class ClassificationReport:
    """Every Sec. 10 metric for one model on one partition."""

    n: int
    n_positive: int
    prevalence: float
    threshold: float
    point: dict[str, float] = field(default_factory=dict)
    intervals: dict[str, BootstrapResult] = field(default_factory=dict)

    def as_row(self) -> dict[str, float]:
        row: dict[str, float] = {
            "n": float(self.n),
            "n_positive": float(self.n_positive),
            "prevalence": self.prevalence,
            "threshold": self.threshold,
        }
        row.update(self.point)
        for name, ci in self.intervals.items():
            row[f"{name}_ci_low"] = ci.ci_low
            row[f"{name}_ci_high"] = ci.ci_high
        return row

    def summary(self) -> str:
        lines = [
            f"n={self.n:,}  positives={self.n_positive:,}  "
            f"prevalence={self.prevalence:.4f}  threshold={self.threshold:.4f}"
        ]
        for name, value in self.point.items():
            ci = self.intervals.get(name)
            note = "  (near-meaningless alone -- Sec. 10)" if name in NEAR_MEANINGLESS else ""
            if ci is None:
                lines.append(f"  {name:<12} {value:.4f}{note}")
            else:
                lines.append(
                    f"  {name:<12} {value:.4f}  [{ci.ci_low:.4f}, {ci.ci_high:.4f}]{note}"
                )
        return "\n".join(lines)


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    threshold: float,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
    with_intervals: bool = True,
) -> ClassificationReport:
    """Compute the full Sec. 10 metric set at a *pre-selected* threshold.

    ``threshold`` is an argument rather than something chosen here, so that the
    Sec. 10 requirement -- fixed before test -- is structurally enforced: this
    function cannot tune it.
    """
    y_true = np.asarray(y_true).ravel().astype(int)
    y_prob = np.asarray(y_prob).ravel().astype(float)
    pred = (y_prob >= threshold).astype(int)

    point = {
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece": expected_calibration_error(y_true, y_prob, strategy="quantile"),
    }

    intervals: dict[str, BootstrapResult] = {}
    if with_intervals:
        for name, fn in METRIC_FUNCTIONS.items():
            intervals[name] = bootstrap_ci(
                y_true, y_prob, fn, n_resamples=n_resamples, alpha=alpha, seed=seed
            )

    return ClassificationReport(
        n=len(y_true),
        n_positive=int(y_true.sum()),
        prevalence=float(y_true.mean()),
        threshold=float(threshold),
        point=point,
        intervals=intervals,
    )


def aggregate_over_seeds(reports: list[ClassificationReport]) -> dict[str, tuple[float, float]]:
    """Mean and std of each metric across seeds (Sec. 6.2, Sec. 10).

    Sec. 6.2 forbids reporting single runs. This is the function that turns five
    seeded runs into the mean +/- std the paper reports.
    """
    if not reports:
        raise ValueError("no reports to aggregate")

    names = reports[0].point.keys()
    out: dict[str, tuple[float, float]] = {}
    for name in names:
        values = np.asarray([r.point[name] for r in reports], dtype=float)
        out[name] = (float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else 0.0)
    return out
