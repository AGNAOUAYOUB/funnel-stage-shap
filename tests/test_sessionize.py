"""Tests for cleaning and sessionisation (protocol Sec. 7.1-7.2)."""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from funnel_shap.data.flow import DataFlow
from funnel_shap.data.schema import SchemaError
from funnel_shap.data.sessionize import (
    SENSITIVITY_GAPS,
    SessionizeConfig,
    assign_sessions,
    clean_events,
    parse_event_time,
    sessionisation_agreement,
    sessionize,
)


def _events(rows: list[tuple[int, str, str]]) -> pl.LazyFrame:
    """Build a minimal event log from (user_id, 'YYYY-MM-DD HH:MM:SS', event_type)."""
    return pl.DataFrame(
        {
            "user_id": [r[0] for r in rows],
            "event_time": [r[1] + " UTC" for r in rows],
            "event_type": [r[2] for r in rows],
            "product_id": list(range(len(rows))),
            "user_session": ["s"] * len(rows),
        }
    ).lazy()


def test_parses_rees46_timestamp_format() -> None:
    lazy = _events([(1, "2019-10-01 00:00:00", "view")])
    parsed = parse_event_time(lazy).collect()
    assert parsed["event_time"][0] == dt.datetime(2019, 10, 1, 0, 0, 0)


def test_missing_required_column_is_rejected() -> None:
    lazy = pl.DataFrame({"event_time": ["2019-10-01 00:00:00 UTC"]}).lazy()
    with pytest.raises(SchemaError, match="missing required columns"):
        clean_events(lazy)


def test_unknown_event_types_are_dropped() -> None:
    lazy = _events(
        [
            (1, "2019-10-01 00:00:00", "view"),
            (1, "2019-10-01 00:00:10", "teleport"),
            (1, "2019-10-01 00:00:20", "cart"),
        ]
    )
    cleaned = clean_events(lazy).collect()
    assert set(cleaned["event_type"].cast(pl.Utf8)) == {"view", "cart"}


def test_sessions_cut_on_the_inactivity_gap() -> None:
    # Two events 10 min apart, then one 45 min later => two sessions at gap=30.
    lazy = _events(
        [
            (1, "2019-10-01 00:00:00", "view"),
            (1, "2019-10-01 00:10:00", "view"),
            (1, "2019-10-01 00:55:00", "view"),
        ]
    )
    out = assign_sessions(clean_events(lazy), gap_minutes=30).collect()
    assert out["session_id"].n_unique() == 2

    # The same stream is a single session at gap=60.
    out60 = assign_sessions(clean_events(lazy), gap_minutes=60).collect()
    assert out60["session_id"].n_unique() == 1

    # ...and three sessions at gap=5.
    out5 = assign_sessions(clean_events(lazy), gap_minutes=5).collect()
    assert out5["session_id"].n_unique() == 3


def test_gap_exactly_at_threshold_does_not_cut() -> None:
    """A gap of exactly 30 min is within the 30-min rule, not beyond it."""
    lazy = _events(
        [
            (1, "2019-10-01 00:00:00", "view"),
            (1, "2019-10-01 00:30:00", "view"),
        ]
    )
    out = assign_sessions(clean_events(lazy), gap_minutes=30).collect()
    assert out["session_id"].n_unique() == 1


def test_sessions_never_span_users() -> None:
    lazy = _events(
        [
            (1, "2019-10-01 00:00:00", "view"),
            (2, "2019-10-01 00:00:01", "view"),
            (1, "2019-10-01 00:00:02", "view"),
        ]
    )
    out = assign_sessions(clean_events(lazy), gap_minutes=30).collect()
    per_session_users = out.group_by("session_id").agg(pl.col("user_id").n_unique().alias("n"))
    assert per_session_users["n"].max() == 1


def test_single_event_sessions_are_dropped(raw_events) -> None:
    out = sessionize(raw_events.lazy(), SessionizeConfig()).collect()
    counts = out.group_by("session_id").agg(pl.len().alias("n"))
    assert counts["n"].min() >= 2


def test_long_tail_sessions_are_capped(raw_events) -> None:
    config = SessionizeConfig(long_tail_percentile=0.90)
    strict = sessionize(raw_events.lazy(), config).collect()
    loose = sessionize(raw_events.lazy(), SessionizeConfig()).collect()
    assert strict["session_id"].n_unique() < loose["session_id"].n_unique()


def test_data_flow_is_monotone_and_complete(raw_events) -> None:
    flow = DataFlow("synthetic")
    sessionize(raw_events.lazy(), SessionizeConfig(), flow).collect()

    counts = [s.n_events for s in flow.steps]
    assert counts == sorted(counts, reverse=True), "a filter increased the event count"
    assert [s.step for s in flow.steps][0] == "raw events"
    assert flow.steps[-1].step == "length-filtered sessions"
    assert flow.to_frame().height == len(flow.steps)


@pytest.mark.parametrize("gap", SENSITIVITY_GAPS)
def test_sensitivity_gaps_all_run(raw_events, gap: int) -> None:
    """Sec. 7.2 requires 15/30/60 to be reported; all three must be runnable."""
    out = sessionize(raw_events.lazy(), SessionizeConfig(gap_minutes=gap)).collect()
    assert out.height > 0
    assert out["session_id"].n_unique() > 0


def test_shorter_gap_yields_more_sessions(raw_events) -> None:
    counts = {
        gap: sessionize(raw_events.lazy(), SessionizeConfig(gap_minutes=gap))
        .collect()["session_id"]
        .n_unique()
        for gap in SENSITIVITY_GAPS
    }
    assert counts[15] >= counts[30] >= counts[60]


def test_agreement_statistic_is_reported(sessionised) -> None:
    stats = sessionisation_agreement(sessionised.lazy())
    assert 0.0 <= stats["agreement"] <= 1.0
    assert stats["n_derived"] > 0


def test_invalid_config_is_rejected() -> None:
    with pytest.raises(ValueError, match="gap_minutes"):
        SessionizeConfig(gap_minutes=0)
    with pytest.raises(ValueError, match="long_tail_percentile"):
        SessionizeConfig(long_tail_percentile=1.5)
