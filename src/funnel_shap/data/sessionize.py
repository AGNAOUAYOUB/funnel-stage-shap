"""Cleaning and sessionisation for the event stream (protocol Sec. 7.1-7.2).

Implemented on the Polars lazy API (Sec. 4.1) so the multi-GB event log is
streamed rather than materialised.

Two decisions here are protocol-visible and deliberately explicit:

* **The 30-minute inactivity rule is a choice, not a law.** Sec. 7.2 states it
  as such and requires a 15/30/60 sensitivity appendix, so the gap is a
  first-class parameter and `sessionisation_agreement` quantifies how far the
  derived sessions drift from the provider's own `user_session` labels.
* **Session IDs are re-derived rather than trusted.** The vendor's
  `user_session` column encodes *its* (undocumented) timeout. Re-cutting from
  `user_id` + `event_time` is what makes the sensitivity check meaningful; the
  original column is retained only for the agreement statistic.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from .flow import DataFlow
from .schema import DATASET_B_CRITICAL, VALID_EVENT_TYPES, validate_event_columns

#: Sec. 7.2 headline rule; 15 and 60 are the sensitivity arms.
DEFAULT_GAP_MINUTES = 30
SENSITIVITY_GAPS = (15, 30, 60)

#: Sec. 7.1 step 3.
MIN_EVENTS_PER_SESSION = 2
LONG_TAIL_PERCENTILE = 0.995


@dataclass(frozen=True)
class SessionizeConfig:
    gap_minutes: int = DEFAULT_GAP_MINUTES
    min_events: int = MIN_EVENTS_PER_SESSION
    long_tail_percentile: float = LONG_TAIL_PERCENTILE

    def __post_init__(self) -> None:
        if self.gap_minutes <= 0:
            raise ValueError("gap_minutes must be positive")
        if not 0.5 < self.long_tail_percentile < 1.0:
            raise ValueError("long_tail_percentile must lie in (0.5, 1.0)")


def parse_event_time(lazy: pl.LazyFrame, column: str = "event_time") -> pl.LazyFrame:
    """Parse REES46 timestamps (``2019-10-01 00:00:00 UTC``) to Datetime.

    The trailing ``UTC`` literal is stripped rather than parsed as a timezone
    because Polars' strptime rejects the bare abbreviation.
    """
    dtype = lazy.collect_schema()[column]
    if dtype == pl.Datetime:
        return lazy

    return lazy.with_columns(
        pl.col(column)
        .cast(pl.Utf8)
        .str.replace(r"\s*UTC$", "")
        .str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False)
        .alias(column)
    )


def clean_events(lazy: pl.LazyFrame, flow: DataFlow | None = None) -> pl.LazyFrame:
    """Sec. 7.1 steps 1-2: schema validation, malformed rows, critical nulls.

    Bot/outlier removal (step 3) happens after sessionisation, since it is
    defined on session length.
    """
    validate_event_columns(lazy.collect_schema().names())

    if flow is not None:
        flow.record(
            "raw events",
            "as delivered by the source",
            _count(lazy),
        )

    lazy = parse_event_time(lazy)

    # Malformed: unparseable timestamp, null critical field, unknown event type.
    lazy = lazy.filter(
        pl.all_horizontal([pl.col(c).is_not_null() for c in DATASET_B_CRITICAL])
        & pl.col("event_type").cast(pl.Utf8).is_in(list(VALID_EVENT_TYPES))
    )
    if flow is not None:
        flow.record(
            "schema-valid events",
            f"drop null {list(DATASET_B_CRITICAL)}, unparseable event_time, "
            f"event_type not in {sorted(VALID_EVENT_TYPES)}",
            _count(lazy),
        )

    # Exact duplicate events (same user, product, type, timestamp) are logging
    # artefacts; keeping them inflates count features.
    lazy = lazy.unique(
        subset=["user_id", "event_time", "event_type", "product_id"], keep="first"
    )
    if flow is not None:
        flow.record(
            "deduplicated events",
            "drop exact duplicates on (user_id, event_time, event_type, product_id)",
            _count(lazy),
        )

    return lazy


def assign_sessions(
    lazy: pl.LazyFrame,
    *,
    gap_minutes: int = DEFAULT_GAP_MINUTES,
    session_column: str = "session_id",
) -> pl.LazyFrame:
    """Cut sessions on an inactivity gap (Sec. 7.2).

    Events are sorted by ``(user_id, event_time)``; a new session starts
    whenever the gap since the previous event of the same user exceeds
    ``gap_minutes``. The resulting id is ``{user_id}_{session_index}`` so it is
    stable and human-readable across gap settings.
    """
    gap_us = gap_minutes * 60 * 1_000_000

    return (
        lazy.sort(["user_id", "event_time"])
        .with_columns(
            (
                pl.col("event_time")
                .diff()
                .dt.total_microseconds()
                .over("user_id")
                .fill_null(gap_us + 1)
                > gap_us
            ).alias("_new_session")
        )
        .with_columns(pl.col("_new_session").cum_sum().over("user_id").alias("_session_index"))
        .with_columns(
            (
                pl.col("user_id").cast(pl.Utf8)
                + pl.lit("_")
                + pl.col("_session_index").cast(pl.Utf8)
            ).alias(session_column)
        )
        .drop("_new_session", "_session_index")
    )


def filter_sessions(
    lazy: pl.LazyFrame,
    config: SessionizeConfig,
    flow: DataFlow | None = None,
    *,
    session_column: str = "session_id",
) -> pl.LazyFrame:
    """Sec. 7.1 step 3: drop implausibly short and long-tail sessions.

    The long-tail cap is computed on the *post-short-filter* distribution, so
    the percentile is not dragged down by the 1-event sessions that are being
    removed anyway.
    """
    per_session = lazy.group_by(session_column).agg(
        pl.len().alias("_n_events"),
        (pl.col("event_time").max() - pl.col("event_time").min())
        .dt.total_seconds()
        .alias("_duration_s"),
    )

    if flow is not None:
        flow.record(
            "sessionised",
            f"inactivity gap = {config.gap_minutes} min",
            _count(lazy),
            _count(per_session),
        )

    long_enough = per_session.filter(pl.col("_n_events") >= config.min_events)

    # Percentile thresholds must be materialised before they can be used as a
    # filter bound; this is the one unavoidable collect in the pipeline.
    caps = long_enough.select(
        pl.col("_n_events").quantile(config.long_tail_percentile).alias("cap_events"),
        pl.col("_duration_s").quantile(config.long_tail_percentile).alias("cap_duration_s"),
    ).collect()
    cap_events = float(caps["cap_events"][0])
    cap_duration = float(caps["cap_duration_s"][0])

    keep = long_enough.filter(
        (pl.col("_n_events") <= cap_events) & (pl.col("_duration_s") <= cap_duration)
    ).select(session_column)

    filtered = lazy.join(keep, on=session_column, how="inner")

    if flow is not None:
        flow.record(
            "length-filtered sessions",
            f">= {config.min_events} events and <= p{config.long_tail_percentile * 100:g} "
            f"on length ({cap_events:.0f} events) and duration ({cap_duration:.0f} s)",
            _count(filtered),
            _count(keep),
        )

    return filtered


def sessionize(
    lazy: pl.LazyFrame,
    config: SessionizeConfig | None = None,
    flow: DataFlow | None = None,
) -> pl.LazyFrame:
    """Full Sec. 7.1-7.2 path: clean -> assign sessions -> filter."""
    config = config or SessionizeConfig()
    cleaned = clean_events(lazy, flow)
    assigned = assign_sessions(cleaned, gap_minutes=config.gap_minutes)
    return filter_sessions(assigned, config, flow)


def sessionisation_agreement(
    lazy: pl.LazyFrame,
    *,
    derived: str = "session_id",
    provided: str = "user_session",
) -> dict[str, float]:
    """How far derived sessions drift from the provider's own labels.

    Reported in the Sec. 7.2 sensitivity appendix. Agreement is the share of
    events whose derived session maps one-to-one onto a provided session: high
    agreement means the vendor's undocumented timeout is close to the gap you
    chose, low agreement means the choice materially reshapes the unit of
    analysis and must be discussed.
    """
    columns = lazy.collect_schema().names()
    if provided not in columns:
        return {"agreement": float("nan"), "n_derived": float("nan"), "n_provided": float("nan")}

    pairs = (
        lazy.select([derived, provided])
        .group_by([derived, provided])
        .agg(pl.len().alias("n"))
        .collect()
    )

    total = int(pairs["n"].sum())
    # For each derived session, the largest overlap with a single provided one.
    dominant = pairs.group_by(derived).agg(pl.col("n").max().alias("n"))["n"].sum()

    return {
        "agreement": float(dominant) / total if total else float("nan"),
        "n_derived": float(pairs[derived].n_unique()),
        "n_provided": float(pairs[provided].n_unique()),
    }


def _count(lazy: pl.LazyFrame) -> int:
    return int(lazy.select(pl.len()).collect().item())
