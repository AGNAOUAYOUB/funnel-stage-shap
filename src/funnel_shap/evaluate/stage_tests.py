"""Confirmatory statistical tests on the stage runs (protocol Sec. 12).

Sec. 12 requires the confirmatory comparisons to be pre-specified, corrected for
multiplicity, and reported with effect sizes rather than bare p-values. This
module declares the family once, up front, and runs exactly it.

**Only within-stage contrasts are paired.** Two feature sets at the same stage
score the *same* test sessions, so DeLong and the paired bootstrap apply. Two
different stages score different populations — S3's test set is the 2,973
sessions that reached a cart, S1's is 38,059 — so no paired test is defined
between them, and an unpaired comparison of their PR-AUCs would be comparing
different questions. The improvement curve is therefore descriptive, reported
with reach and prevalence (amendment A2), and is not in this family.

Two complementary views are reported for each contrast:

* **Paired bootstrap on the test set**, using seed-averaged scores. Gives the
  PR-AUC difference and its CI on the actual evaluation data.
* **Wilcoxon across the five seeds**, as Sec. 12 names. Note that with five
  pairs the smallest attainable two-sided p is 0.0625, so this test *cannot*
  reach 0.05 — `wilcoxon_across_folds` returns `min_attainable_p` to make that
  explicit rather than letting a null read as evidence of no effect.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score

from ..stats.bootstrap import paired_bootstrap_diff
from .comparisons import ADJUSTERS, wilcoxon_across_folds

#: The pre-specified confirmatory family (Sec. 12), declared before the runs.
#: Each entry is (feature_set_a, feature_set_b, rationale); a is the richer set.
CONFIRMATORY_CONTRASTS: tuple[tuple[str, str, str], ...] = (
    (
        "+temporal",
        "baseline",
        "Does the temporal family contribute beyond aggregate counts?",
    ),
    (
        "baseline+entropy_velocity",
        "baseline",
        "H3 as written: do navigation entropy and click velocity carry non-trivial "
        "attribution beyond baseline aggregate features?",
    ),
    (
        "+entropy_velocity",
        "+temporal",
        "Is that entropy/velocity signal additional to the temporal family, or shared "
        "with it?",
    ),
)


def _seed_average(runs: list) -> tuple[np.ndarray, np.ndarray]:
    """Mean predicted probability across seeds, plus the shared labels."""
    scores = np.mean([r.test_scores for r in runs], axis=0)
    return scores, runs[0].y_test


def run_stage_comparisons(
    runs: list,
    *,
    correction: str = "holm",
    alpha: float = 0.05,
    n_resamples: int = 2000,
    seed: int = 42,
) -> pl.DataFrame:
    """Run the pre-declared family across every stage and correct over all of it.

    Correction spans the *whole* family — every contrast at every stage — not
    each stage separately. Correcting within stages would treat three families
    of three as if multiplicity only applied inside each, which is precisely the
    inflation Sec. 12 makes correction mandatory to prevent.
    """
    if correction not in ADJUSTERS:
        raise ValueError(f"unknown correction {correction!r}")

    grouped: dict[tuple[str, str], list] = defaultdict(list)
    for run in runs:
        grouped[(run.stage, run.feature_set)].append(run)

    rows: list[dict] = []
    for stage in sorted({s for s, _ in grouped}):
        for set_a, set_b, rationale in CONFIRMATORY_CONTRASTS:
            runs_a = grouped.get((stage, set_a))
            runs_b = grouped.get((stage, set_b))
            if not runs_a or not runs_b:
                continue

            scores_a, y = _seed_average(runs_a)
            scores_b, _ = _seed_average(runs_b)

            diff = paired_bootstrap_diff(
                y,
                scores_a,
                scores_b,
                average_precision_score,
                n_resamples=n_resamples,
                alpha=alpha,
                seed=seed,
            )

            per_seed_a = np.array([r.test.point["pr_auc"] for r in sorted(runs_a, key=lambda r: r.seed)])
            per_seed_b = np.array([r.test.point["pr_auc"] for r in sorted(runs_b, key=lambda r: r.seed)])
            wilcoxon = wilcoxon_across_folds(per_seed_a, per_seed_b)

            rows.append(
                {
                    "stage": stage,
                    "feature_set_a": set_a,
                    "feature_set_b": set_b,
                    "rationale": rationale,
                    "pr_auc_a": float(per_seed_a.mean()),
                    "pr_auc_b": float(per_seed_b.mean()),
                    "pr_auc_diff": diff.point,
                    "ci_low": diff.ci_low,
                    "ci_high": diff.ci_high,
                    "excludes_zero": bool(diff.ci_low > 0 or diff.ci_high < 0),
                    "wilcoxon_p": wilcoxon["p_value"],
                    "wilcoxon_min_attainable_p": wilcoxon["min_attainable_p"],
                    "n_seeds": int(len(per_seed_a)),
                }
            )

    if not rows:
        raise ValueError("no contrasts could be formed; were the ablation rungs run?")

    frame = pl.DataFrame(rows)
    adjusted = ADJUSTERS[correction](frame["wilcoxon_p"].to_list())

    return frame.with_columns(
        pl.Series(f"wilcoxon_p_{correction}", adjusted),
        pl.lit(correction).alias("correction"),
        pl.lit(len(rows)).alias("family_size"),
    )


def summarise(frame: pl.DataFrame) -> str:
    """Human-readable verdicts, effect size first."""
    lines = [
        f"Confirmatory family: {frame['family_size'][0]} contrasts, "
        f"{frame['correction'][0]} corrected",
        "-" * 88,
    ]
    for row in frame.to_dicts():
        verdict = "CI excludes 0" if row["excludes_zero"] else "CI includes 0"
        lines.append(
            f"{row['stage']}  {row['feature_set_a']:<26} vs {row['feature_set_b']:<12} "
            f"{row['pr_auc_diff']:+.4f} [{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]  {verdict}"
        )
    lines.append("")
    lines.append(
        "Wilcoxon across 5 seeds cannot reach p<0.05 (minimum attainable 0.0625); "
        "read the bootstrap CIs as the effect-size evidence."
    )
    return "\n".join(lines)
