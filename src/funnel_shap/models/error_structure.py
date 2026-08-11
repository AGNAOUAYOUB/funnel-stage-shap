"""Confusion structure at the operating threshold, over the frozen seed list.

The decision-support argument opens with the error structure each stage model
produces at the threshold a standard rule selects. That table was originally
computed from one fitted model at seed 42 while every other headline number in
the paper is a five-seed mean --- the same mismatch amendment A40 corrected in
the decision-economics table, left uncorrected here.

Two things vary with the seed and both matter. The fit varies, so the scores
vary; and the threshold is itself selected on validation, so it varies too. A
single-seed confusion matrix therefore conditions on one draw of both. This
module recomputes the structure per seed, each at that seed's own
validation-selected threshold, and reports the mean with its dispersion.

The per-seed arithmetic is deliberately identical to the single-seed version in
``report.appendix_tables``: margins are derived from the *rounded* components so
that the published columns subtract correctly, rather than from an independently
rounded difference, which can leave the printed table off by one in the last
place and read as an arithmetic error.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl

from ..data.journey import MODELLING_STAGES


def confusion_at_threshold(
    y_true: np.ndarray, scores: np.ndarray, threshold: float
) -> dict[str, float]:
    """Confusion counts and derived rates for one seed at one threshold."""
    y_true = np.asarray(y_true).astype(int).ravel()
    scores = np.asarray(scores, dtype=float).ravel()
    if y_true.shape != scores.shape:
        raise ValueError("labels and scores must have the same shape")

    predicted = (scores >= threshold).astype(int)
    tp = int(((predicted == 1) & (y_true == 1)).sum())
    fp = int(((predicted == 1) & (y_true == 0)).sum())
    fn = int(((predicted == 0) & (y_true == 1)).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    prevalence = float(y_true.mean())
    # The always-positive rule: the only comparator that costs nothing to build.
    trivial_f1 = 2 * prevalence / (1 + prevalence) if prevalence else 0.0

    return {
        "prevalence": prevalence,
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "trivial_positive_f1": trivial_f1,
        "f1_over_trivial": f1 - trivial_f1,
        "flag_rate": float((predicted == 1).mean()),
        "fp_per_tp": fp / tp if tp else float("nan"),
        "n_test": int(y_true.size),
    }


def error_structure_table(
    runs: Sequence,
    *,
    stage_order: Sequence[str] = MODELLING_STAGES,
) -> pl.DataFrame:
    """Mean and dispersion of the confusion structure across seeds.

    ``runs`` are ``StageRun`` records, each carrying its own
    validation-selected threshold alongside the test scores it produced, so the
    seed-to-seed variation reported here includes both sources.
    """
    by_stage: dict[str, list] = {}
    for run in runs:
        by_stage.setdefault(run.stage, []).append(run)
    if not by_stage:
        raise ValueError("no runs supplied; the table would be empty")

    rows = []
    for stage in stage_order:
        stage_runs = by_stage.get(stage)
        if not stage_runs:
            continue
        per_seed = [
            confusion_at_threshold(r.y_test, r.test_scores, r.threshold)
            for r in stage_runs
        ]
        row: dict[str, object] = {"stage": stage, "n_seeds": len(per_seed)}
        row["n_test"] = per_seed[0]["n_test"]
        for key in ("prevalence", "threshold", "flag_rate", "precision", "recall",
                    "f1", "trivial_positive_f1", "f1_over_trivial", "fp_per_tp"):
            values = np.array([p[key] for p in per_seed], dtype=float)
            row[key] = round(float(np.nanmean(values)), 4)
            row[key + "_sd"] = (
                round(float(np.nanstd(values, ddof=1)), 4) if values.size > 1 else None
            )
        # Derive the published margin from the rounded components so the columns
        # subtract correctly in print.
        row["f1_over_trivial"] = round(
            float(row["f1"]) - float(row["trivial_positive_f1"]), 4
        )
        rows.append(row)

    return pl.DataFrame(rows)
