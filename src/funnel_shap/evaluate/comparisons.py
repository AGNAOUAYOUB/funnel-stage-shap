"""Paired model comparisons with multiple-comparison control (protocol Sec. 12).

Sec. 12 is unusually specific, and for good reason: with five model families
across three stages there are dozens of possible pairwise tests, and an
uncorrected sweep of them will manufacture significance. Three rules are
enforced structurally here rather than left to discipline:

1. **Confirmatory comparisons are declared before running.** `ComparisonPlan`
   holds the pairs; `run_plan` tests exactly those and nothing else. Anything
   discovered afterwards is exploratory and must be labelled so.
2. **Correction is mandatory, not optional.** `run_plan` always applies Holm or
   Benjamini-Hochberg across the whole family and returns both raw and adjusted
   p-values.
3. **Effect sizes travel with p-values.** Every result carries the AUC or PR-AUC
   difference and its CI, because "p < 0.05" on 5 million sessions can mean an
   AUC difference of 0.0004.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl
from scipy import stats

from ..stats.bootstrap import paired_bootstrap_diff
from ..stats.delong import delong_roc_test


@dataclass(frozen=True)
class Comparison:
    """One pre-declared confirmatory comparison."""

    name_a: str
    name_b: str
    rationale: str


@dataclass
class ComparisonPlan:
    """The confirmatory comparisons, fixed before the test set is opened."""

    comparisons: list[Comparison] = field(default_factory=list)

    def add(self, name_a: str, name_b: str, rationale: str) -> ComparisonPlan:
        if not rationale.strip():
            raise ValueError(
                "every confirmatory comparison needs a rationale (Sec. 12: pre-specify "
                "which pairs, to avoid fishing)"
            )
        self.comparisons.append(Comparison(name_a, name_b, rationale))
        return self

    def __len__(self) -> int:
        return len(self.comparisons)


def mcnemar_test(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    *,
    exact: bool | None = None,
) -> dict[str, float]:
    """Paired disagreement between two classifiers' *decisions* (Sec. 12).

    Uses the exact binomial test when the discordant count is small, where the
    chi-square approximation is unreliable, and the continuity-corrected
    chi-square otherwise. `mlxtend.mcnemar` is the protocol's named tool; this
    reproduces it without adding a dependency on its table helper, and returns
    the discordant counts so the effect is visible rather than just its p-value.
    """
    y_true = np.asarray(y_true).ravel().astype(int)
    a = np.asarray(pred_a).ravel().astype(int)
    b = np.asarray(pred_b).ravel().astype(int)

    a_right = a == y_true
    b_right = b == y_true

    n01 = int((~a_right & b_right).sum())  # only B correct
    n10 = int((a_right & ~b_right).sum())  # only A correct
    discordant = n01 + n10

    if discordant == 0:
        return {"n01": 0.0, "n10": 0.0, "statistic": 0.0, "p_value": 1.0, "exact": 1.0}

    use_exact = discordant < 25 if exact is None else exact
    if use_exact:
        p = float(stats.binomtest(n10, discordant, 0.5).pvalue)
        statistic = float(min(n01, n10))
    else:
        statistic = (abs(n10 - n01) - 1.0) ** 2 / discordant
        p = float(stats.chi2.sf(statistic, df=1))

    return {
        "n01": float(n01),
        "n10": float(n10),
        "statistic": float(statistic),
        "p_value": p,
        "exact": float(use_exact),
    }


def wilcoxon_across_folds(scores_a: np.ndarray, scores_b: np.ndarray) -> dict[str, float]:
    """Paired Wilcoxon signed-rank across CV folds or seeds (Sec. 12).

    With the protocol's five seeds the smallest attainable two-sided p is 0.0625,
    so this test *cannot* reach 0.05 on seeds alone. That is a property of the
    design, not a bug, and the returned ``min_attainable_p`` makes it explicit
    so a null result is not misread as evidence of no difference.
    """
    a = np.asarray(scores_a, dtype=float).ravel()
    b = np.asarray(scores_b, dtype=float).ravel()
    if a.shape != b.shape:
        raise ValueError("paired test needs equal-length score vectors")

    n = len(a)
    min_attainable = 2.0 / (2**n) if n < 20 else 0.0

    if np.allclose(a, b):
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "n_pairs": float(n),
            "median_diff": 0.0,
            "min_attainable_p": min_attainable,
        }

    result = stats.wilcoxon(a, b)
    return {
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "n_pairs": float(n),
        "median_diff": float(np.median(a - b)),
        "min_attainable_p": min_attainable,
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values (Sec. 12).

    Controls the family-wise error rate. Implemented directly rather than via
    statsmodels so the enforced monotonicity is visible: an adjusted p can never
    be smaller than that of a more significant hypothesis.
    """
    m = len(p_values)
    if m == 0:
        return []

    order = np.argsort(p_values)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        value = (m - rank) * p_values[idx]
        running = max(running, value)
        adjusted[idx] = min(running, 1.0)
    return adjusted.tolist()


def benjamini_hochberg_adjust(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg adjusted p-values, controlling the false discovery rate."""
    m = len(p_values)
    if m == 0:
        return []

    order = np.argsort(p_values)
    adjusted = np.empty(m, dtype=float)
    running = 1.0
    for rank in range(m - 1, -1, -1):
        idx = order[rank]
        value = m / (rank + 1) * p_values[idx]
        running = min(running, value)
        adjusted[idx] = min(running, 1.0)
    return adjusted.tolist()


ADJUSTERS = {"holm": holm_adjust, "fdr_bh": benjamini_hochberg_adjust}


def run_plan(
    plan: ComparisonPlan,
    y_true: np.ndarray,
    scores: dict[str, np.ndarray],
    *,
    predictions: dict[str, np.ndarray] | None = None,
    correction: str = "holm",
    alpha: float = 0.05,
    n_resamples: int = 2000,
    seed: int = 42,
) -> pl.DataFrame:
    """Run every declared comparison and correct across the whole family.

    Returns one row per comparison with the DeLong ROC-AUC test, a paired
    bootstrap PR-AUC difference (no analytic analogue of DeLong exists for
    PR-AUC), McNemar on decisions where predictions are supplied, and both raw
    and adjusted p-values.
    """
    if correction not in ADJUSTERS:
        raise ValueError(f"unknown correction {correction!r}, expected one of {list(ADJUSTERS)}")
    if len(plan) == 0:
        raise ValueError("the comparison plan is empty; Sec. 12 requires pre-specified pairs")

    from sklearn.metrics import average_precision_score

    y_true = np.asarray(y_true).ravel().astype(int)
    rows: list[dict] = []

    for comparison in plan.comparisons:
        for name in (comparison.name_a, comparison.name_b):
            if name not in scores:
                raise KeyError(f"no scores supplied for model {name!r}")

        a = np.asarray(scores[comparison.name_a]).ravel()
        b = np.asarray(scores[comparison.name_b]).ravel()

        delong = delong_roc_test(y_true, a, b, alpha=alpha)
        pr_diff = paired_bootstrap_diff(
            y_true, a, b, average_precision_score, n_resamples=n_resamples, alpha=alpha, seed=seed
        )

        row = {
            "model_a": comparison.name_a,
            "model_b": comparison.name_b,
            "rationale": comparison.rationale,
            "roc_auc_a": delong.auc_a,
            "roc_auc_b": delong.auc_b,
            "roc_auc_diff": delong.diff,
            "roc_auc_diff_ci_low": delong.ci_low,
            "roc_auc_diff_ci_high": delong.ci_high,
            "delong_p": delong.p_value,
            "pr_auc_diff": pr_diff.point,
            "pr_auc_diff_ci_low": pr_diff.ci_low,
            "pr_auc_diff_ci_high": pr_diff.ci_high,
        }

        if predictions is not None:
            mc = mcnemar_test(
                y_true, predictions[comparison.name_a], predictions[comparison.name_b]
            )
            row |= {
                "mcnemar_p": mc["p_value"],
                "mcnemar_n01": mc["n01"],
                "mcnemar_n10": mc["n10"],
            }

        rows.append(row)

    frame = pl.DataFrame(rows)
    adjust = ADJUSTERS[correction]

    frame = frame.with_columns(
        pl.Series(f"delong_p_{correction}", adjust(frame["delong_p"].to_list()))
    )
    if "mcnemar_p" in frame.columns:
        frame = frame.with_columns(
            pl.Series(f"mcnemar_p_{correction}", adjust(frame["mcnemar_p"].to_list()))
        )

    return frame.with_columns(
        (pl.col(f"delong_p_{correction}") < alpha).alias("significant"),
        pl.lit(correction).alias("correction"),
        pl.lit(alpha).alias("alpha"),
        pl.lit(len(plan)).alias("family_size"),
    )
