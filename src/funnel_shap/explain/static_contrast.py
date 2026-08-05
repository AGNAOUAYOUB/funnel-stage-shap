"""RQ4 — what stage-conditioned attribution shows that a static one cannot (Sec. 11.1).

RQ4 asks whether stage conditioning "identifies intervention points that a
static whole-session SHAP analysis misses". The honest way to answer it is to
build the static counterfactual on the *same* data and compare, rather than to
assert that a static analysis would have missed something.

The static baseline here is the attribution a researcher would obtain by
pooling all stages into one model — the standard practice this paper argues
against. Comparing against Dataset A instead would confound the question with a
change of dataset, schema and grain.

The quantity that matters is **range**: how much a driver's attribution share
moves across stages. A static analysis reports one number per feature, so it can
only recover something close to the stage-average. Any driver whose share varies
widely across stages is one the static view flattens, and the flattening is
worst — and most misleading — for non-monotone drivers, whose average sits in a
region they never actually occupy.
"""

from __future__ import annotations

import numpy as np
import polars as pl


def static_contrast_table(migration: pl.DataFrame) -> pl.DataFrame:
    """Per-driver comparison of the stage view against its static average.

    ``static_share`` is the attribution share a pooled analysis would report,
    approximated by the mean across stages where the driver exists.
    ``range_share`` is how far it actually travels, and ``non_monotone`` marks
    drivers whose peak or trough is at an interior stage — the cases where the
    static average is not merely imprecise but unrepresentative of any stage.
    """
    available = migration.filter(pl.col("available") & pl.col("share").is_not_null())
    if available.is_empty():
        raise ValueError("migration table has no available rows")

    stage_order = {"S1": 0, "S2": 1, "S3": 2}
    rows = []

    for group in available["group"].unique().to_list():
        sub = (
            available.filter(pl.col("group") == group)
            .with_columns(pl.col("stage").replace_strict(stage_order).alias("_o"))
            .sort("_o")
        )
        shares = sub["share"].to_numpy()
        stages = sub["stage"].to_list()
        if len(shares) < 2:
            continue

        peak = int(np.argmax(shares))
        trough = int(np.argmin(shares))
        interior = range(1, len(shares) - 1)
        non_monotone = peak in interior or trough in interior

        static_share = float(shares.mean())
        rows.append(
            {
                "group": group,
                "static_share": static_share,
                "range_share": float(shares.max() - shares.min()),
                "peak_stage": stages[peak],
                "peak_share": float(shares.max()),
                "trough_stage": stages[trough],
                "non_monotone": non_monotone,
                # How far the peak sits above the number a static view reports.
                "static_understates_peak_by": float(shares.max() - static_share),
                "n_stages": len(shares),
            }
        )

    return pl.DataFrame(rows).sort("range_share", descending=True)


def summarise_contrast(table: pl.DataFrame, *, top: int = 3) -> str:
    """Prose-ready summary of the RQ4 evidence."""
    lines = ["Drivers a static whole-session attribution would flatten:", "-" * 78]
    for row in table.head(top).to_dicts():
        shape = "non-monotone (peaks mid-funnel)" if row["non_monotone"] else "monotone"
        lines.append(
            f"{row['group'][:44]:<46} static {row['static_share']:.3f}  "
            f"range {row['range_share']:.3f}  peak {row['peak_share']:.3f} "
            f"at {row['peak_stage']}  [{shape}]"
        )

    worst = table.filter(pl.col("non_monotone"))
    if worst.height:
        top_row = worst.sort("range_share", descending=True).to_dicts()[0]
        lines.append("")
        lines.append(
            f"Strongest case: {top_row['group'][:50]} peaks at {top_row['peak_stage']} "
            f"({top_row['peak_share']:.3f}) but a static analysis would report "
            f"{top_row['static_share']:.3f} -- understating it by "
            f"{top_row['static_understates_peak_by']:.3f} and giving no indication that "
            "the driver is stage-specific."
        )
    return "\n".join(lines)
