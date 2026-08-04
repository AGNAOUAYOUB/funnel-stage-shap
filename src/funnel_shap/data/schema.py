"""Dataset schemas and the honest variable-availability table (protocol Sec. 5.1).

Sec. 5.1 is explicit: no single public dataset carries all ten idealised
variables, and the paper must report true availability rather than claim
variables it does not have. That table is encoded here as data so the manuscript
table and the feature builder read from one source instead of drifting apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import polars as pl


class Availability(StrEnum):
    PRESENT = "present"
    PROXY = "proxy"
    ABSENT = "absent"


@dataclass(frozen=True)
class VariableAvailability:
    variable: str
    dataset_a: Availability
    dataset_b: Availability
    note: str


#: The ten idealised journey variables from Sec. 5.1, with the true situation in
#: each dataset. ``PROXY`` entries must cite their proxy in the note; ``ABSENT``
#: entries become limitations-section text, never silently dropped features.
IDEAL_VARIABLES: tuple[VariableAvailability, ...] = (
    VariableAvailability(
        "page_views",
        Availability.PRESENT,
        Availability.PRESENT,
        "A: Administrative/Informational/ProductRelated counts. B: view events.",
    ),
    VariableAvailability(
        "session_duration",
        Availability.PRESENT,
        Availability.PRESENT,
        "A: *_Duration sums. B: last minus first event_time within the session.",
    ),
    VariableAvailability(
        "device",
        Availability.PRESENT,
        Availability.ABSENT,
        "A: Browser/OperatingSystems. B: REES46 event log carries no device field.",
    ),
    VariableAvailability(
        "referrer_traffic_source",
        Availability.PRESENT,
        Availability.ABSENT,
        "A: TrafficType. B: absent -> limitation, H2's context arm rests on A.",
    ),
    VariableAvailability(
        "time_of_day",
        Availability.PROXY,
        Availability.PRESENT,
        "A: Month/SpecialDay/Weekend only, no clock time. B: exact event_time.",
    ),
    VariableAvailability(
        "bounce",
        Availability.PRESENT,
        Availability.PROXY,
        "A: BounceRates. B: proxied by single-event sessions before the <2-event filter.",
    ),
    VariableAvailability(
        "cart_events",
        Availability.ABSENT,
        Availability.PRESENT,
        "A: no cart field at all. B: cart / remove_from_cart events. Core to stage S3.",
    ),
    VariableAvailability(
        "search_queries",
        Availability.ABSENT,
        Availability.ABSENT,
        "Absent in both. Sec. 5.1 flags this as one of the two never-jointly-present "
        "variables; reported as a limitation, no proxy claimed.",
    ),
    VariableAvailability(
        "scroll_depth",
        Availability.PROXY,
        Availability.PROXY,
        "A: ProductRelated_Duration per page. B: mean product-page dwell time. "
        "Documented proxy per Sec. 5.1(b), never labelled 'scroll depth' in results.",
    ),
    VariableAvailability(
        "purchase",
        Availability.PRESENT,
        Availability.PRESENT,
        "A: Revenue. B: purchase event. The label.",
    ),
)


def availability_table() -> pl.DataFrame:
    """The Sec. 5.1 availability table, ready to write to reports/tables/."""
    return pl.DataFrame(
        {
            "variable": [v.variable for v in IDEAL_VARIABLES],
            "dataset_A": [v.dataset_a.value for v in IDEAL_VARIABLES],
            "dataset_B": [v.dataset_b.value for v in IDEAL_VARIABLES],
            "note": [v.note for v in IDEAL_VARIABLES],
        }
    )


# --------------------------------------------------------------------------
# Dataset A — UCI Online Shoppers Purchasing Intention (Sec. 5.2)
# --------------------------------------------------------------------------

DATASET_A_NUMERIC: tuple[str, ...] = (
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
)

DATASET_A_CATEGORICAL: tuple[str, ...] = (
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
)

DATASET_A_TARGET = "Revenue"
DATASET_A_N_ROWS = 12_330


#: Coarse funnel proxy for the aggregate benchmark (Sec. 5.2). These are *not*
#: prefixes -- Dataset A has no event order -- so any model built on them is
#: flagged non-causal in the paper (Sec. 7.4).
DATASET_A_STAGE_PROXY: dict[str, tuple[str, ...]] = {
    "awareness": ("Administrative", "Administrative_Duration", "Informational",
                  "Informational_Duration"),
    "consideration": ("ProductRelated", "ProductRelated_Duration"),
    "intent": ("PageValues", "BounceRates", "ExitRates"),
}


# --------------------------------------------------------------------------
# Dataset B — REES46-class event log (Sec. 5.3)
# --------------------------------------------------------------------------

#: Fields the pipeline requires. Events missing any of the critical ones are
#: dropped during cleaning (Sec. 7.1 step 2).
DATASET_B_CRITICAL: tuple[str, ...] = ("user_session", "event_time", "event_type")
DATASET_B_REQUIRED: tuple[str, ...] = (
    "event_time",
    "event_type",
    "product_id",
    "user_id",
    "user_session",
)
DATASET_B_OPTIONAL: tuple[str, ...] = ("category_id", "category_code", "brand", "price")

VALID_EVENT_TYPES: frozenset[str] = frozenset(
    {"view", "cart", "remove_from_cart", "purchase"}
)

DATASET_B_SCHEMA: dict[str, pl.DataType] = {
    "event_time": pl.Utf8,
    "event_type": pl.Categorical,
    "product_id": pl.Int64,
    "category_id": pl.Int64,
    "category_code": pl.Utf8,
    "brand": pl.Utf8,
    "price": pl.Float64,
    "user_id": pl.Int64,
    "user_session": pl.Utf8,
}


class SchemaError(ValueError):
    """Raised when an input file does not match the expected schema."""


def validate_event_columns(columns: list[str]) -> None:
    """Fail fast if a Dataset-B file is missing required columns (Sec. 7.1 step 1)."""
    missing = [c for c in DATASET_B_REQUIRED if c not in columns]
    if missing:
        raise SchemaError(
            f"event log is missing required columns {missing}; "
            f"got {sorted(columns)}. Expected a REES46-class schema (Sec. 5.3)."
        )
