"""Dataset A: UCI Online Shoppers Purchasing Intention (protocol Sec. 5.2, 7.5).

The static benchmark. Its role is a reproducible sanity check and the
static-vs-stage SHAP contrast, not the headline result.

Sec. 7.4 is blunt about the caveat that governs this whole module: Dataset A
carries whole-session aggregates with no event order, so *every* feature here is
measured over the entire session including whatever happened at the moment of
purchase. `PageValues` in particular is a post-hoc quantity — the average value
of the pages a visitor completed a transaction on — and it is the single
strongest predictor in the published literature on this dataset largely for that
reason. Models fitted here are therefore flagged non-causal, and any comparison
to Dataset B's prefix models is a comparison of what leakage buys, not of two
honest predictors.
"""

from __future__ import annotations

import polars as pl

from .schema import (
    DATASET_A_CATEGORICAL,
    DATASET_A_NUMERIC,
    DATASET_A_STAGE_PROXY,
    DATASET_A_TARGET,
)

#: Sec. 7.4: these are whole-session aggregates and are used for the static
#: benchmark only. The name is deliberately loud at every call site.
NON_CAUSAL_WARNING = (
    "Dataset A features are whole-session aggregates (Sec. 7.4). Results are a static "
    "benchmark and must be labelled non-causal; PageValues in particular is measured "
    "post-transaction."
)


def prepare_dataset_a(frame: pl.DataFrame) -> tuple[pl.DataFrame, pl.Series]:
    """Split Dataset A into a typed feature frame and a 0/1 label series.

    Boolean columns arrive as the strings ``TRUE``/``FALSE`` in the UCI CSV;
    they are cast explicitly rather than left to inference, which silently
    produces an all-null column on some mirrors.
    """
    features = frame.select([*DATASET_A_NUMERIC, *DATASET_A_CATEGORICAL])

    features = features.with_columns(
        [pl.col(c).cast(pl.Float64) for c in DATASET_A_NUMERIC]
        + [_to_bool_int(pl.col("Weekend")).alias("Weekend")]
    )

    label = _to_bool_int(frame[DATASET_A_TARGET]).rename("label")
    return features, label


def _to_bool_int(col):
    """Coerce TRUE/FALSE strings or booleans to 0/1 integers."""
    if isinstance(col, pl.Series):
        if col.dtype == pl.Boolean:
            return col.cast(pl.Int8)
        return (col.cast(pl.Utf8).str.to_uppercase() == "TRUE").cast(pl.Int8)
    return (
        pl.when(col.cast(pl.Utf8).str.to_uppercase() == "TRUE")
        .then(1)
        .otherwise(0)
        .cast(pl.Int8)
    )


def stage_proxy_columns(stage: str) -> tuple[str, ...]:
    """Aggregate columns standing in for a funnel stage (Sec. 5.2).

    A coarse but reproducible decomposition. It supports the static-vs-stage
    SHAP contrast; it is *not* a prefix and yields no anti-leakage guarantee.
    """
    if stage not in DATASET_A_STAGE_PROXY:
        raise ValueError(
            f"unknown Dataset A stage proxy {stage!r}; "
            f"expected one of {sorted(DATASET_A_STAGE_PROXY)}"
        )
    return DATASET_A_STAGE_PROXY[stage]


def describe(frame: pl.DataFrame) -> dict[str, float]:
    """Headline shape statistics for the methods section."""
    label = _to_bool_int(frame[DATASET_A_TARGET])
    return {
        "n_sessions": float(frame.height),
        "n_features": float(len(DATASET_A_NUMERIC) + len(DATASET_A_CATEGORICAL)),
        "n_positive": float(label.sum()),
        "prevalence": float(label.mean()),
    }
