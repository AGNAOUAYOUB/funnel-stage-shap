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
    "S2": "product/category browsing; cut-point = second repeat view or category switch "
          "(amendment A8)",
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
    s = config.session_column
    return (
        lazy.with_row_index("_file_order")
        .sort([s, config.time_column, "_file_order"])
        # On a frame already sorted by session, the within-session index is the
        # global row number minus the session's first global row number. That is
        # one window pass instead of the two that int_range().over() plus
        # len().over() would cost, which matters at 37M rows / 5.4M sessions.
        # Cast to a *signed* type before subtracting: cum_count returns UInt32,
        # and the anti-leakage clip computes `first_purchase - 1`, which for a
        # session whose first event is a purchase would wrap to 4294967295 and
        # silently admit every cut-point including the label event itself.
        .with_columns(
            (pl.col("_file_order").cum_count().over(s).cast(pl.Int64) - 1).alias("event_idx"),
        )
    )


def _first_index_where(condition: pl.Expr) -> pl.Expr:
    """Index of the first event satisfying ``condition``, else null.

    Written as an *aggregation* rather than a window expression. The window form
    (``.min().over(session)``) needs a full group-and-broadcast pass per call,
    and there are five of them; folding them into a single ``group_by().agg()``
    turned a run that had burned 2.6 CPU-hours without finishing into one that
    completes, with identical semantics.
    """
    return pl.when(condition).then(pl.col("event_idx")).otherwise(None).min()


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

    has_category = config.category_column in ordered.collect_schema().names()

    # Each window expression is materialised as a column before anything
    # references it. Composing them inline instead -- `browsing_signal` embeds
    # `repeat_view`, and `browsing_ordinal` then names `browsing_signal` twice --
    # makes Polars re-evaluate the innermost windows several times per pass,
    # which is what turned a few million events into minutes of CPU.
    #
    # Product interaction: any event carrying a product. In a REES46-class log
    # every event does, but the predicate is written explicitly so a source with
    # non-product events (site search, static pages) degrades correctly.
    flags = ordered.with_columns(
        pl.col(config.product_column).is_not_null().alias("_is_interaction"),
        # Repeat view: this product has been seen earlier in the same session.
        # `is_first_distinct().over(session)` is one single-key window pass;
        # `cum_count().over([session, product])` was a two-key window over
        # millions of distinct (session, product) groups.
        (
            (etype == "view")
            & ~pl.col(config.product_column).is_first_distinct().over(s)
        ).alias("_repeat_view"),
    )

    # Category switch: a *view* whose category differs from the previous event's.
    #
    # The `etype == "view"` guard is load-bearing. Without it a cart event whose
    # category differs from the preceding view counts as a browsing signal, so it
    # can open S2 and S3 on the very same event: on real REES46 data that made
    # 45% of S2/S3 cut-points identical, collapsing the S2->S3 leg of the
    # migration figure exactly as the first-signal rule had collapsed S1->S2
    # (amendment A8). The synthetic fixture never showed it, because there carts
    # always inherit the preceding view's category. S2 is defined as
    # "product/category browsing", so only browsing events may trigger it.
    if has_category:
        prev_category = pl.col(config.category_column).shift().over(s)
        flags = flags.with_columns(
            (
                (etype == "view")
                & prev_category.is_not_null()
                & pl.col(config.category_column).is_not_null()
                & (pl.col(config.category_column) != prev_category)
            ).alias("_cat_switch")
        )
    else:
        flags = flags.with_columns(pl.lit(False).alias("_cat_switch"))

    # S2 opens on the *second* browsing signal, not the first (amendment A8).
    # A single repeat view or category switch is incidental and, for most
    # sessions, lands on the very same event as S1's cut-point -- which made 83%
    # of S1 and S2 prefixes identical and the S1->S2 leg of the migration figure
    # empty. Requiring two signals is the "consideration is established, not
    # incidental" reading, and it separates the stages by construction: the
    # earliest possible second signal is the third event.
    flags = flags.with_columns(
        (pl.col("_repeat_view") | pl.col("_cat_switch")).alias("_browsing_signal")
    ).with_columns(
        pl.col("_is_interaction").cum_sum().over(s).alias("_interaction_ordinal"),
        pl.col("_browsing_signal").cum_sum().over(s).alias("_browsing_ordinal"),
    )

    per_session = (
        flags.with_columns(
            (pl.col("_is_interaction") & (pl.col("_interaction_ordinal") == 2)).alias(
                "_is_second_interaction"
            ),
            (pl.col("_browsing_signal") & (pl.col("_browsing_ordinal") == 2)).alias(
                "_is_second_browsing_signal"
            ),
        )
        .group_by(s)
        .agg(
            _first_index_where(etype == "view").alias("first_view"),
            _first_index_where(pl.col("_is_second_interaction")).alias("second_interaction"),
            _first_index_where(pl.col("_is_second_browsing_signal")).alias(
                "second_browsing_signal"
            ),
            _first_index_where(etype == "cart").alias("first_cart"),
            _first_index_where(etype == "purchase").alias("first_purchase"),
            pl.len().alias("n_events"),
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
    #
    # Stages are *ordered*, and a stage whose trigger fires out of order is not
    # reached rather than shunted into place. The earlier rule pushed a later
    # stage's cut-point forward past an earlier stage's, which meant a visitor
    # who carted before browsing much had S3 dragged onto S2's cut-point: on real
    # data that made 45% of S2/S3 prefixes identical and emptied the S2->S3 leg
    # of the migration figure. A visitor who carts before establishing
    # consideration did not pass through consideration -- they skipped it, and
    # S2 is simply unreached for them.
    cut_s1 = pl.col("second_interaction")
    cut_s2 = pl.when(
        pl.col("second_browsing_signal").is_not_null()
        & (pl.col("second_browsing_signal") > cut_s1)
        & (
            pl.col("first_cart").is_null()
            | (pl.col("second_browsing_signal") < pl.col("first_cart"))
        )
    ).then(pl.col("second_browsing_signal"))
    # S3 is anchored on the cart itself and never moved.
    cut_s3 = pl.when(pl.col("first_cart") >= cut_s1).then(pl.col("first_cart"))
    # S4 opens at the purchase (or session end) and is descriptive only.
    cut_s4 = pl.col("first_purchase").fill_null(pl.col("n_events") - 1)

    frame = per_session.with_columns(
        max_admissible_idx=max_idx,
    ).with_columns(
        cut_S1=_clip(cut_s1, pl.col("max_admissible_idx")),
        cut_S2=_clip(cut_s2, pl.col("max_admissible_idx")),
        cut_S3=_clip(cut_s3, pl.col("max_admissible_idx")),
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


#
# `_monotone` used to live here: it pushed a stage's cut-point forward past the
# earlier stages' cut-points. It was removed in favour of the ordering rule in
# `stage_cutpoints`, which marks an out-of-order stage unreached instead of
# shunting it into place -- see the comment there. Shunting made 45% of S2/S3
# prefixes identical on real data.


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
