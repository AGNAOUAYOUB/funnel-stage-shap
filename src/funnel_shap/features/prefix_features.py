"""Stage-prefix feature construction (protocol Sec. 7.4, Sec. 8).

Every feature is computed over the events of a single stage prefix and nothing
else. Two leakage traps are handled explicitly, because both look harmless:

**Dwell time.** Dwell on an event is normally the gap until the *next* event.
For the final event of a prefix, that next event lies beyond the cut-point, so
using it would import information from the future. Dwell is therefore defined
only for prefix events that have a successor *inside the same prefix*; the last
event contributes no dwell. This makes `dwell_total_s` slightly conservative and
that is the correct direction.

**Category transitions and revisits.** Computed strictly within the prefix, so a
session's transition count at S1 is a prefix of its count at S3 rather than a
whole-session quantity sliced afterwards.

The intent-composite weights are declared here rather than tuned, because Sec. 8
requires them documented and Sec. 10 tests the composite's contribution by
ablation. Tuning them against the label would make the ablation circular.
"""

from __future__ import annotations

import polars as pl

from ..data.journey import MODELLING_STAGES, JourneyConfig, StageError, StageName
from .dictionary import feature_names

#: Documented, not fitted (Sec. 8). Ordering encodes funnel proximity: a cart add
#: is a stronger intent signal than a view, a removal partially undoes one.
INTENT_WEIGHTS: dict[str, float] = {
    "view": 1.0,
    "cart": 4.0,
    "remove_from_cart": -2.0,
}

_EPS = 1e-9


def build_stage_features(
    prefix: pl.LazyFrame,
    stage: StageName,
    config: JourneyConfig | None = None,
) -> pl.DataFrame:
    """Aggregate a stage prefix into one feature row per session.

    ``prefix`` must come from :func:`funnel_shap.data.journey.prefix_events`,
    which is the only place the cut-point is applied.
    """
    if stage not in MODELLING_STAGES:
        raise StageError(
            f"stage {stage!r} has no feature matrix; modelling stages are {MODELLING_STAGES} "
            "(Sec. 9.2). S4's cut-point is the label event itself."
        )

    config = config or JourneyConfig()
    s = config.session_column
    etype = pl.col(config.type_column).cast(pl.Utf8)
    has_category = config.category_column in prefix.collect_schema().names()
    has_price = "price" in prefix.collect_schema().names()

    # Per-event derived quantities. All window functions are ordered by
    # event_idx within the session, which journey.order_events guarantees.
    gap_s = (
        pl.col(config.time_column)
        .diff()
        .over(s)
        .dt.total_microseconds()
        .truediv(1_000_000)
    )

    # Dwell = gap to the NEXT event, defined only when that next event is still
    # inside the prefix. shift(-1) is null on the last row of each session, so
    # the last event contributes nothing -- exactly the intended behaviour.
    dwell_s = (
        pl.col(config.time_column)
        .shift(-1)
        .over(s)
        .sub(pl.col(config.time_column))
        .dt.total_microseconds()
        .truediv(1_000_000)
    )

    if has_category:
        prev_category = pl.col(config.category_column).shift().over(s)
        is_transition = (
            prev_category.is_not_null()
            & pl.col(config.category_column).is_not_null()
            & (pl.col(config.category_column) != prev_category)
        )
        category_seen_before = (
            pl.col(config.category_column).cum_count().over([s, config.category_column]) > 1
        )
        transition_pair = (
            prev_category.cast(pl.Utf8) + pl.lit("->") + pl.col(config.category_column).cast(pl.Utf8)
        )
    else:
        is_transition = pl.lit(False)
        category_seen_before = pl.lit(False)
        transition_pair = pl.lit(None, dtype=pl.Utf8)

    product_seen_before = (
        pl.col(config.product_column).cum_count().over([s, config.product_column]) > 1
    )

    enriched = prefix.with_columns(
        gap_s.alias("_gap_s"),
        dwell_s.alias("_dwell_s"),
        is_transition.alias("_is_transition"),
        category_seen_before.alias("_cat_seen"),
        product_seen_before.alias("_prod_seen"),
        pl.when(is_transition).then(transition_pair).otherwise(None).alias("_transition_pair"),
    )

    intent_expr = pl.lit(0.0)
    for event_type, weight in INTENT_WEIGHTS.items():
        intent_expr = intent_expr + (etype == event_type).sum() * weight

    aggregations = [
        pl.first("label").alias("label"),
        # counts
        pl.len().alias("n_events"),
        (etype == "view").sum().alias("n_views"),
        pl.col(config.product_column).n_unique().alias("n_unique_products"),
        # temporal
        (pl.col(config.time_column).max() - pl.col(config.time_column).min())
        .dt.total_microseconds()
        .truediv(1_000_000)
        .alias("prefix_duration_s"),
        pl.col("_gap_s").mean().alias("inter_event_mean_s"),
        pl.col("_gap_s").std().alias("inter_event_std_s"),
        pl.col("_gap_s").last().alias("last_gap_s"),
        pl.col(config.time_column).min().dt.hour().alias("hour_of_day"),
        pl.col(config.time_column).min().dt.weekday().alias("_weekday"),
        # engagement
        pl.col("_dwell_s").sum().alias("dwell_total_s"),
        pl.when(etype == "view").then(pl.col("_dwell_s")).otherwise(None)
        .mean()
        .alias("dwell_mean_product_s"),
        # transitions / revisits
        pl.col("_is_transition").sum().alias("n_category_transitions"),
        pl.col("_transition_pair").n_unique().alias("_n_distinct_transitions"),
        pl.col("_cat_seen").mean().alias("category_revisit_rate"),
        pl.col("_prod_seen").mean().alias("product_revisit_rate"),
        # composite
        intent_expr.alias("purchase_intent_score"),
    ]

    if has_category:
        aggregations.append(
            pl.col(config.category_column).n_unique().alias("n_unique_categories")
        )
        aggregations.append(
            pl.col(config.category_column).drop_nulls().alias("_categories")
        )
    if has_price:
        aggregations += [
            pl.col("price").mean().alias("price_mean"),
            pl.col("price").max().alias("price_max"),
        ]
    # No cart-specific aggregations: see the amendment A8 note in dictionary.py.
    # At S3's cut-point the prefix holds exactly one cart event, so every such
    # feature is a constant.

    grouped = enriched.group_by(s).agg(aggregations).collect()

    if has_category:
        grouped = grouped.with_columns(
            _shannon_entropy_from_list(pl.col("_categories")).alias("category_entropy")
        ).drop("_categories")
    else:
        grouped = grouped.with_columns(
            pl.lit(0.0).alias("category_entropy"),
            pl.lit(1).alias("n_unique_categories"),
        )

    grouped = grouped.with_columns(
        # Normalised entropy: 0 = single category, 1 = uniform over all seen.
        pl.when(pl.col("n_unique_categories") > 1)
        .then(pl.col("category_entropy") / pl.col("n_unique_categories").log())
        .otherwise(0.0)
        .alias("category_entropy_normalised"),
        (pl.col("n_events") / (pl.col("prefix_duration_s") + _EPS)).alias("click_velocity"),
        (pl.col("n_events") / (pl.col("prefix_duration_s") / 60.0 + _EPS)).alias(
            "events_per_active_minute"
        ),
        (pl.col("_weekday") >= 6).cast(pl.Int8).alias("is_weekend"),
        pl.when(pl.col("n_category_transitions") > 0)
        .then(pl.col("_n_distinct_transitions") / pl.col("n_category_transitions"))
        .otherwise(0.0)
        .alias("distinct_transition_rate"),
    ).drop("_weekday", "_n_distinct_transitions")

    # A one-event prefix has no gaps and no dwell; null there means "not
    # applicable", and zero is the right encoding for a tree model.
    zero_fill = [
        "inter_event_mean_s",
        "inter_event_std_s",
        "last_gap_s",
        "dwell_total_s",
        "dwell_mean_product_s",
    ]
    grouped = grouped.with_columns([pl.col(c).fill_null(0.0) for c in zero_fill])

    expected = feature_names(stage)
    missing = [c for c in expected if c not in grouped.columns]
    if missing:
        raise ValueError(
            f"stage {stage} feature matrix is missing {missing}; the feature dictionary and "
            "the builder have drifted apart"
        )

    return grouped.select([s, "label", *expected])


def _shannon_entropy_from_list(col: pl.Expr) -> pl.Expr:
    """H = -sum p_i log p_i over the value distribution of a list column.

    Natural log, matching the Sec. 8 definition. Computed on the aggregated list
    so it stays a single pass over the grouped frame.
    """
    counts = col.list.eval(pl.element().value_counts(sort=False).struct.field("count"))
    total = counts.list.sum()
    return (
        counts.list.eval(
            pl.element() / pl.element().sum() * (pl.element() / pl.element().sum()).log()
        )
        .list.sum()
        .mul(-1.0)
        .fill_nan(0.0)
        * pl.when(total > 0).then(1.0).otherwise(0.0)
    )
