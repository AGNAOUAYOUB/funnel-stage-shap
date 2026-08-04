"""Journey reconstruction and funnel-stage cut-points (protocol Sec. 7.3-7.4).

This module is the scientific core of the "sequential" claim. It converts an
ordered event stream into, per session, the index of each stage's cut-point —
from which the feature layer builds the *nested prefixes* S1 subset S2 subset S3
that make the design leakage-free.

Three invariants are enforced here and unit-tested in `tests/test_journey.py`:

1. **No prefix ever contains a purchase event.** The label is `purchase`, so any
   feature computed at or after it leaks the outcome (Sec. 7.4). Every cut-point
   is hard-clipped to strictly before the first purchase.
2. **Cut-points are monotone**: `s1 <= s2 <= s3`, so the prefixes really do nest.
3. **A stage a session never reached yields no row for that stage's model.**

On (3): the Sk model is therefore trained on sessions *conditional on having
reached Sk*. This is the honest reading of "features use only the prefix up to
Sk's cut-point", but it means the H1 prediction-improvement curve is partly a
composition effect — sessions that reach S3 convert far more often than the
population. The paper must report per-stage prevalence and N alongside PR-AUC
(the `stage_prevalence_table` below produces exactly that) and interpret the
curve as conditional, not marginal. Comparing raw PR-AUC across stages without
that caveat would overstate H1.

Timestamps in REES46-class logs have one-second granularity and many events
collide, so ordering carries an explicit file-order tie-break rather than
relying on sort stability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import polars as pl

StageName = Literal["S1", "S2", "S3", "S4"]

STAGES: tuple[StageName, ...] = ("S1", "S2", "S3", "S4")

#: Stages that admit a predictive model (Sec. 9.2). S4 is the conversion window
#: itself: its cut-point *is* the label event, so no S4 feature matrix can exist
#: without leaking. S4 remains defined for descriptive journey statistics.
MODELLING_STAGES: tuple[StageName, ...] = ("S1", "S2", "S3")

STAGE_DEFINITIONS: dict[StageName, str] = {
    "S1": "session entry -> initial catalogue contact; cut-point = second product "
          "interaction (amendment A6)",
    "S2": "product/category browsing; cut-point = first repeat view or category switch",
    "S3": "cart activity begins; cut-point = first cart event",
    "S4": "checkout/purchase window; cut-point = purchase or session end",
}


class StageError(ValueError):
    """Raised when a stage is used in a way the protocol forbids."""


@dataclass(frozen=True)
class JourneyConfig:
    session_column: str = "session_id"
    time_column: str = "event_time"
    type_column: str = "event_type"
    product_column: str = "product_id"
    category_column: str = "category_id"


def order_events(lazy: pl.LazyFrame, config: JourneyConfig | None = None) -> pl.LazyFrame:
    """Add a deterministic within-session event index.

    ``_file_order`` preserves the source ordering so that events sharing a
    one-second timestamp resolve identically on every rerun.
    """
    config = config or JourneyConfig()
    return (
        lazy.with_row_index("_file_order")
        .sort([config.session_column, config.time_column, "_file_order"])
        .with_columns(
            pl.int_range(pl.len()).over(config.session_column).alias("event_idx"),
            pl.len().over(config.session_column).alias("session_n_events"),
        )
    )


def _first_index_where(condition: pl.Expr, session: str) -> pl.Expr:
    """Index of the first event in the session satisfying ``condition``, else null."""
    return (
        pl.when(condition).then(pl.col("event_idx")).otherwise(None).min().over(session)
    )


def stage_cutpoints(lazy: pl.LazyFrame, config: JourneyConfig | None = None) -> pl.DataFrame:
    """Per-session cut-point index for each funnel stage (Sec. 7.3).

    Returns one row per session with columns ``session_id``, ``label``,
    ``cut_S1``..``cut_S4``, ``reached_S1``..``reached_S4`` and the session-level
    counts needed for the stage-prevalence table.

    Indices are *inclusive*: the prefix for stage Sk is ``event_idx <= cut_Sk``.
    The cut-point event itself is part of the stage that it opens (the first
    cart event belongs to S3), which is why the purchase clip below is strict.
    """
    config = config or JourneyConfig()
    s = config.session_column

    ordered = order_events(lazy, config)
    etype = pl.col(config.type_column).cast(pl.Utf8)

    # Product interaction: any event carrying a product. In a REES46-class log
    # every event does, but the predicate is written explicitly so a source with
    # non-product events (site search, static pages) degrades correctly.
    product_interaction = pl.col(config.product_column).is_not_null()
    interaction_ordinal = (
        pl.when(product_interaction)
        .then(product_interaction.cum_sum().over(s))
        .otherwise(None)
    )

    # Repeat view: this product has been seen earlier in the same session.
    repeat_view = (etype == "view") & (
        pl.col(config.product_column).cum_count().over([s, config.product_column]) > 1
    )

    # Category switch: category differs from the previous event's category.
    has_category = config.category_column in ordered.collect_schema().names()
    if has_category:
        prev_category = pl.col(config.category_column).shift().over(s)
        category_switch = (
            prev_category.is_not_null()
            & pl.col(config.category_column).is_not_null()
            & (pl.col(config.category_column) != prev_category)
        )
    else:
        category_switch = pl.lit(False)

    per_session = (
        ordered.with_columns(
            _first_index_where(etype == "view", s).alias("_first_view"),
            _first_index_where(interaction_ordinal == 2, s).alias("_second_interaction"),
            _first_index_where(repeat_view, s).alias("_first_repeat_view"),
            _first_index_where(category_switch, s).alias("_first_cat_switch"),
            _first_index_where(etype == "cart", s).alias("_first_cart"),
            _first_index_where(etype == "purchase", s).alias("_first_purchase"),
        )
        .group_by(s)
        .agg(
            pl.first("_first_view").alias("first_view"),
            pl.first("_second_interaction").alias("second_interaction"),
            pl.first("_first_repeat_view").alias("first_repeat_view"),
            pl.first("_first_cat_switch").alias("first_cat_switch"),
            pl.first("_first_cart").alias("first_cart"),
            pl.first("_first_purchase").alias("first_purchase"),
            pl.first("session_n_events").alias("n_events"),
            (etype == "purchase").any().alias("label"),
        )
        .collect()
    )

    # The last index any prefix may include: strictly before the first purchase.
    # Sessions whose very first event is a purchase have no admissible prefix.
    max_idx = (
        pl.when(pl.col("first_purchase").is_not_null())
        .then(pl.col("first_purchase") - 1)
        .otherwise(pl.col("n_events") - 1)
    )

    # S1 closes at the *second* product interaction, not the first (amendment
    # A6). Cutting at the first view produced a one-event prefix in which 19 of
    # 23 features were constant, which would have made the H1 curve an artefact
    # of S1 having no features and H2 circular at S1. Two interactions is the
    # minimum that gives awareness a measurable dwell, inter-event gap and
    # category distribution.
    cut_s1 = pl.col("second_interaction")
    # min_horizontal ignores nulls, which is what we want here: the earlier of
    # whichever S2 trigger actually fired.
    cut_s2 = pl.min_horizontal(pl.col("first_repeat_view"), pl.col("first_cat_switch"))
    cut_s3 = pl.col("first_cart")
    # S4 opens at the purchase (or session end) and is descriptive only.
    cut_s4 = pl.col("first_purchase").fill_null(pl.col("n_events") - 1)

    frame = per_session.with_columns(
        max_admissible_idx=max_idx,
    ).with_columns(
        cut_S1=_clip(cut_s1, pl.col("max_admissible_idx")),
        cut_S2=_clip(_monotone(cut_s2, cut_s1), pl.col("max_admissible_idx")),
        cut_S3=_clip(_monotone(cut_s3, cut_s2, cut_s1), pl.col("max_admissible_idx")),
        cut_S4=cut_s4,
    )

    frame = frame.with_columns(
        [pl.col(f"cut_{st}").is_not_null().alias(f"reached_{st}") for st in STAGES]
    )

    return frame.select(
        [s, "label", "n_events", "max_admissible_idx"]
        + [f"cut_{st}" for st in STAGES]
        + [f"reached_{st}" for st in STAGES]
    )


def stage_distinctness_table(cutpoints: pl.DataFrame) -> pl.DataFrame:
    """How often consecutive stages share a cut-point, i.e. carry identical prefixes.

    Moving S1's cut-point forward to the second interaction (amendment A6) buys
    S1 real features, but it moves S1 closer to S2, whose earliest possible
    trigger is also the second interaction. If a large share of sessions have
    ``cut_S1 == cut_S2`` then the two stages are the same model on the same rows
    and the attribution "migration" between them is trivially empty.

    This table must be reported alongside the migration figure: a migration
    trajectory is only interpretable between stages that are actually distinct.
    """
    rows = []
    for early, late in (("S1", "S2"), ("S2", "S3")):
        both = cutpoints.filter(
            pl.col(f"cut_{early}").is_not_null() & pl.col(f"cut_{late}").is_not_null()
        )
        n = both.height
        identical = int((both[f"cut_{early}"] == both[f"cut_{late}"]).sum()) if n else 0
        rows.append(
            {
                "pair": f"{early}->{late}",
                "n_sessions_reaching_both": n,
                "n_identical_cutpoint": identical,
                "share_identical": identical / n if n else float("nan"),
                "mean_extra_events": (
                    float((both[f"cut_{late}"] - both[f"cut_{early}"]).mean()) if n else float("nan")
                ),
            }
        )
    return pl.DataFrame(rows)


def _clip(expr: pl.Expr, upper: pl.Expr) -> pl.Expr:
    """Null out a cut-point that would fall at or after the purchase event."""
    return pl.when(expr.is_not_null() & (expr <= upper)).then(expr).otherwise(None)


def _monotone(own_trigger: pl.Expr, *earlier: pl.Expr) -> pl.Expr:
    """Push a cut-point forward past earlier stages, but only if its own trigger fired.

    ``max_horizontal`` skips nulls, so a bare ``max_horizontal(first_cart,
    first_view)`` would quietly hand a cart-less session its S1 index as an S3
    cut-point -- inventing an intent stage the visitor never reached and
    inflating S3's N to the whole population. The ``when`` guard is what keeps
    "did not reach this stage" distinct from "reached it at the same moment as
    the previous stage".
    """
    return (
        pl.when(own_trigger.is_not_null())
        .then(pl.max_horizontal(own_trigger, *earlier))
        .otherwise(None)
    )


def prefix_events(
    lazy: pl.LazyFrame,
    cutpoints: pl.DataFrame,
    stage: StageName,
    config: JourneyConfig | None = None,
) -> pl.LazyFrame:
    """Events forming the stage-``stage`` prefix, for sessions that reached it.

    This is the single gate through which the feature layer sees data. Nothing
    downstream re-reads the raw stream, so the anti-leakage guarantee holds by
    construction rather than by convention.
    """
    if stage not in STAGES:
        raise StageError(f"unknown stage {stage!r}, expected one of {STAGES}")
    if stage == "S4":
        raise StageError(
            "S4's cut-point is the purchase event itself, so an S4 prefix would contain "
            "the label (Sec. 7.4). S4 is descriptive only; model stages are "
            f"{MODELLING_STAGES} (Sec. 9.2)."
        )

    config = config or JourneyConfig()
    s = config.session_column
    cut_col = f"cut_{stage}"

    reached = cutpoints.filter(pl.col(cut_col).is_not_null()).select([s, cut_col, "label"])

    return (
        order_events(lazy, config)
        .join(reached.lazy(), on=s, how="inner")
        .filter(pl.col("event_idx") <= pl.col(cut_col))
    )


def stage_prevalence_table(cutpoints: pl.DataFrame) -> pl.DataFrame:
    """N, reach rate and conversion prevalence per stage.

    Required reading alongside the H1 improvement curve: PR-AUC gains across
    stages are only interpretable next to the shifting prevalence and the
    shrinking, increasingly selected population they are computed on.
    """
    total = cutpoints.height
    rows = []
    for stage in STAGES:
        sub = cutpoints.filter(pl.col(f"cut_{stage}").is_not_null())
        n = sub.height
        rows.append(
            {
                "stage": stage,
                "definition": STAGE_DEFINITIONS[stage],
                "modelled": stage in MODELLING_STAGES,
                "n_sessions": n,
                "reach_rate": n / total if total else float("nan"),
                "n_positive": int(sub["label"].sum()) if n else 0,
                "prevalence": float(sub["label"].mean()) if n else float("nan"),
                "mean_prefix_events": (
                    float((sub[f"cut_{stage}"] + 1).mean()) if n else float("nan")
                ),
            }
        )
    return pl.DataFrame(rows)
