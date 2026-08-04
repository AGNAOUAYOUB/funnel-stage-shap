"""The feature dictionary and availability-by-stage table (protocol Sec. 8).

Sec. 8 requires a "feature-availability-by-stage table so it's auditable which
features exist at which cut-point". That table is generated from this single
declaration, so the manuscript table, the ablation groups and the actual matrix
columns cannot drift apart.

Availability is not a matter of taste. A feature is unavailable at a stage when
its defining event cannot have occurred inside that stage's prefix — cart
recency is undefined before the first cart event, which by construction is the
S3 cut-point. Emitting it at S1 as a sentinel value would hand the model a
constant, and emitting it as "time since a cart that hasn't happened yet" would
leak.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from ..data.journey import MODELLING_STAGES, StageName


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    family: str
    description: str
    stages: tuple[StageName, ...]
    #: Ablation group for the Sec. 10 / H3 study.
    ablation_group: str


BASELINE = "baseline_aggregate"
TEMPORAL = "temporal"
ENTROPY_VELOCITY = "entropy_velocity"
COMPOSITE = "composite"


FEATURE_DICTIONARY: tuple[FeatureSpec, ...] = (
    # ---- Behavioural counts (Sec. 8) -----------------------------------
    FeatureSpec("n_events", "counts", "events in the prefix", MODELLING_STAGES, BASELINE),
    FeatureSpec("n_views", "counts", "view events in the prefix", MODELLING_STAGES, BASELINE),
    FeatureSpec(
        "n_unique_products", "counts", "distinct product_id in the prefix",
        MODELLING_STAGES, BASELINE,
    ),
    FeatureSpec(
        "n_unique_categories", "counts", "distinct category_id in the prefix",
        MODELLING_STAGES, BASELINE,
    ),
    FeatureSpec(
        "n_cart_adds", "counts", "cart events in the prefix",
        ("S3",), BASELINE,
    ),
    FeatureSpec(
        "n_cart_removes", "counts", "remove_from_cart events in the prefix",
        ("S3",), BASELINE,
    ),
    # ---- Temporal (Sec. 8) ---------------------------------------------
    FeatureSpec(
        "prefix_duration_s", TEMPORAL, "last minus first event time in the prefix",
        MODELLING_STAGES, TEMPORAL,
    ),
    FeatureSpec(
        "inter_event_mean_s", TEMPORAL, "mean gap between consecutive prefix events",
        MODELLING_STAGES, TEMPORAL,
    ),
    FeatureSpec(
        "inter_event_std_s", TEMPORAL, "std of gaps between consecutive prefix events",
        MODELLING_STAGES, TEMPORAL,
    ),
    FeatureSpec(
        "last_gap_s", TEMPORAL, "gap between the final two prefix events",
        MODELLING_STAGES, TEMPORAL,
    ),
    FeatureSpec(
        "time_since_last_cart_s", TEMPORAL,
        "seconds from the most recent cart event to the cut-point; undefined before "
        "the first cart event, hence S3 only",
        ("S3",), TEMPORAL,
    ),
    FeatureSpec("hour_of_day", TEMPORAL, "hour of the session's first event",
                MODELLING_STAGES, TEMPORAL),
    FeatureSpec("is_weekend", TEMPORAL, "session start falls on Sat/Sun",
                MODELLING_STAGES, TEMPORAL),
    # ---- Engagement (Sec. 8) -------------------------------------------
    FeatureSpec(
        "dwell_total_s", "engagement",
        "summed dwell over prefix events that have a successor inside the prefix",
        MODELLING_STAGES, TEMPORAL,
    ),
    FeatureSpec(
        "dwell_mean_product_s", "engagement",
        "mean dwell on product-view events; documented scroll-depth proxy (Sec. 5.1)",
        MODELLING_STAGES, TEMPORAL,
    ),
    FeatureSpec(
        "events_per_active_minute", "engagement", "prefix events per minute of prefix duration",
        MODELLING_STAGES, TEMPORAL,
    ),
    # ---- Navigation entropy and click velocity (Sec. 8, H3) ------------
    FeatureSpec(
        "category_entropy", ENTROPY_VELOCITY,
        "Shannon entropy over the prefix category distribution: focus vs. wandering",
        MODELLING_STAGES, ENTROPY_VELOCITY,
    ),
    FeatureSpec(
        "category_entropy_normalised", ENTROPY_VELOCITY,
        "category entropy divided by log(n distinct categories), comparable across prefixes",
        MODELLING_STAGES, ENTROPY_VELOCITY,
    ),
    FeatureSpec(
        "click_velocity", ENTROPY_VELOCITY, "prefix events divided by elapsed prefix seconds",
        MODELLING_STAGES, ENTROPY_VELOCITY,
    ),
    # ---- Category transitions (Sec. 8) ---------------------------------
    FeatureSpec(
        "n_category_transitions", ENTROPY_VELOCITY,
        "consecutive prefix events with differing category", MODELLING_STAGES, ENTROPY_VELOCITY,
    ),
    FeatureSpec(
        "distinct_transition_rate", ENTROPY_VELOCITY,
        "distinct (from, to) category pairs per transition", MODELLING_STAGES, ENTROPY_VELOCITY,
    ),
    FeatureSpec(
        "category_revisit_rate", ENTROPY_VELOCITY,
        "share of prefix events on an already-seen category", MODELLING_STAGES,
        ENTROPY_VELOCITY,
    ),
    FeatureSpec(
        "product_revisit_rate", ENTROPY_VELOCITY,
        "share of prefix events on an already-seen product", MODELLING_STAGES,
        ENTROPY_VELOCITY,
    ),
    # ---- Purchase-intent composite (Sec. 8) ----------------------------
    FeatureSpec(
        "purchase_intent_score", COMPOSITE,
        "weighted sum of prefix view/cart/remove signals; weights in "
        "prefix_features.INTENT_WEIGHTS, tested via ablation (H3)",
        MODELLING_STAGES, COMPOSITE,
    ),
    # ---- Price context (available in REES46, not in the Sec. 5.1 ideal set)
    FeatureSpec(
        "price_mean", "context", "mean price of products in the prefix",
        MODELLING_STAGES, BASELINE,
    ),
    FeatureSpec(
        "price_max", "context", "max price of products in the prefix",
        MODELLING_STAGES, BASELINE,
    ),
)


def feature_names(stage: StageName) -> list[str]:
    """Columns the feature matrix carries at ``stage``."""
    return [f.name for f in FEATURE_DICTIONARY if stage in f.stages]


def ablation_groups(stage: StageName) -> dict[str, list[str]]:
    """Feature families for the Sec. 10 ablation ladder.

    baseline -> +temporal -> +entropy/velocity -> full, matching H3.
    """
    groups: dict[str, list[str]] = {}
    for spec in FEATURE_DICTIONARY:
        if stage in spec.stages:
            groups.setdefault(spec.ablation_group, []).append(spec.name)
    return groups


ABLATION_LADDER: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("baseline", (BASELINE,)),
    ("+temporal", (BASELINE, TEMPORAL)),
    ("+entropy_velocity", (BASELINE, TEMPORAL, ENTROPY_VELOCITY)),
    ("full", (BASELINE, TEMPORAL, ENTROPY_VELOCITY, COMPOSITE)),
)


def availability_by_stage() -> pl.DataFrame:
    """The Sec. 8 auditable feature-availability-by-stage table."""
    return pl.DataFrame(
        {
            "feature": [f.name for f in FEATURE_DICTIONARY],
            "family": [f.family for f in FEATURE_DICTIONARY],
            "ablation_group": [f.ablation_group for f in FEATURE_DICTIONARY],
            **{
                stage: [stage in f.stages for f in FEATURE_DICTIONARY]
                for stage in MODELLING_STAGES
            },
            "description": [f.description for f in FEATURE_DICTIONARY],
        }
    )
