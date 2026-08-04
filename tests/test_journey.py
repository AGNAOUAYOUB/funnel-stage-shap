"""Tests for journey reconstruction and the anti-leakage protocol (Sec. 7.3-7.4).

The leakage tests here are the ones that matter most in the whole suite. A
silent leak would not crash anything; it would just produce an implausibly good
PR-AUC that a reviewer would (correctly) reject the paper for.
"""

from __future__ import annotations

import polars as pl
import pytest

from funnel_shap.data.journey import (
    MODELLING_STAGES,
    STAGES,
    StageError,
    prefix_events,
    stage_cutpoints,
    stage_prevalence_table,
)
from funnel_shap.data.sessionize import SessionizeConfig, sessionize


def _log(rows: list[tuple[str, str, int, int]]) -> pl.LazyFrame:
    """Build one user's single session from (time, event_type, product, category)."""
    return pl.DataFrame(
        {
            "user_id": [1] * len(rows),
            "session_id": ["1_0"] * len(rows),
            "event_time": [r[0] for r in rows],
            "event_type": [r[1] for r in rows],
            "product_id": [r[2] for r in rows],
            "category_id": [r[3] for r in rows],
            "price": [10.0] * len(rows),
        }
    ).lazy().with_columns(
        pl.col("event_time").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S")
    )


# ---------------------------------------------------------------------------
# Leakage invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stage", MODELLING_STAGES)
def test_no_prefix_ever_contains_a_purchase(sessionised, cutpoints, stage) -> None:
    """The invariant the whole design rests on (Sec. 7.4)."""
    prefix = prefix_events(sessionised.lazy(), cutpoints, stage).collect()
    assert (prefix["event_type"].cast(pl.Utf8) == "purchase").sum() == 0


@pytest.mark.parametrize("stage", MODELLING_STAGES)
def test_cutpoints_precede_the_purchase(cutpoints, stage) -> None:
    reached = cutpoints.filter(pl.col(f"cut_{stage}").is_not_null())
    assert (reached[f"cut_{stage}"] <= reached["max_admissible_idx"]).all()


def test_prefixes_nest(cutpoints) -> None:
    """S1 subset S2 subset S3 -- otherwise the migration series is incoherent."""
    both = cutpoints.filter(
        pl.col("cut_S1").is_not_null() & pl.col("cut_S2").is_not_null()
    )
    assert (both["cut_S1"] <= both["cut_S2"]).all()

    both = cutpoints.filter(
        pl.col("cut_S2").is_not_null() & pl.col("cut_S3").is_not_null()
    )
    assert (both["cut_S2"] <= both["cut_S3"]).all()


def test_prefix_event_counts_are_monotone_across_stages(sessionised, cutpoints) -> None:
    sizes = {}
    for stage in MODELLING_STAGES:
        prefix = prefix_events(sessionised.lazy(), cutpoints, stage).collect()
        sizes[stage] = prefix.group_by("session_id").agg(pl.len().alias("n"))

    merged = (
        sizes["S1"].rename({"n": "n1"})
        .join(sizes["S3"].rename({"n": "n3"}), on="session_id", how="inner")
    )
    assert (merged["n1"] <= merged["n3"]).all()


def test_purchase_only_session_yields_no_admissible_prefix() -> None:
    """A session whose first event is a purchase has no leakage-free prefix."""
    lazy = _log(
        [
            ("2019-10-01 00:00:00", "purchase", 1, 1),
            ("2019-10-01 00:00:30", "view", 2, 1),
        ]
    )
    cuts = stage_cutpoints(lazy)
    assert cuts["max_admissible_idx"][0] == -1
    for stage in MODELLING_STAGES:
        assert cuts[f"cut_{stage}"][0] is None


def test_s4_prefix_is_refused() -> None:
    lazy = _log([("2019-10-01 00:00:00", "view", 1, 1)])
    cuts = stage_cutpoints(lazy)
    with pytest.raises(StageError, match="label"):
        prefix_events(lazy, cuts, "S4")


def test_unknown_stage_is_refused() -> None:
    lazy = _log([("2019-10-01 00:00:00", "view", 1, 1)])
    cuts = stage_cutpoints(lazy)
    with pytest.raises(StageError, match="unknown stage"):
        prefix_events(lazy, cuts, "S9")


# ---------------------------------------------------------------------------
# Cut-point semantics
# ---------------------------------------------------------------------------


def test_s1_cuts_at_the_first_view() -> None:
    lazy = _log(
        [
            ("2019-10-01 00:00:00", "view", 1, 1),
            ("2019-10-01 00:00:30", "view", 2, 2),
        ]
    )
    cuts = stage_cutpoints(lazy)
    assert cuts["cut_S1"][0] == 0


def test_s2_cuts_at_the_first_category_switch() -> None:
    lazy = _log(
        [
            ("2019-10-01 00:00:00", "view", 1, 1),
            ("2019-10-01 00:00:30", "view", 2, 1),
            ("2019-10-01 00:01:00", "view", 3, 2),  # category switch
        ]
    )
    cuts = stage_cutpoints(lazy)
    assert cuts["cut_S2"][0] == 2


def test_s2_cuts_at_the_first_repeat_view() -> None:
    lazy = _log(
        [
            ("2019-10-01 00:00:00", "view", 1, 1),
            ("2019-10-01 00:00:30", "view", 2, 1),
            ("2019-10-01 00:01:00", "view", 1, 1),  # repeat of product 1
        ]
    )
    cuts = stage_cutpoints(lazy)
    assert cuts["cut_S2"][0] == 2


def test_s2_is_null_when_the_session_only_browses_once() -> None:
    lazy = _log(
        [
            ("2019-10-01 00:00:00", "view", 1, 1),
            ("2019-10-01 00:00:30", "view", 2, 1),
        ]
    )
    cuts = stage_cutpoints(lazy)
    assert cuts["cut_S2"][0] is None
    assert cuts["reached_S2"][0] is False


def test_s3_cuts_at_the_first_cart_event() -> None:
    lazy = _log(
        [
            ("2019-10-01 00:00:00", "view", 1, 1),
            ("2019-10-01 00:00:30", "cart", 1, 1),
            ("2019-10-01 00:01:00", "purchase", 1, 1),
        ]
    )
    cuts = stage_cutpoints(lazy)
    assert cuts["cut_S3"][0] == 1
    assert cuts["label"][0] is True


def test_cart_after_purchase_does_not_open_s3() -> None:
    """A cart event that only occurs post-purchase must not create an S3 prefix."""
    lazy = _log(
        [
            ("2019-10-01 00:00:00", "view", 1, 1),
            ("2019-10-01 00:00:30", "purchase", 1, 1),
            ("2019-10-01 00:01:00", "cart", 2, 1),
        ]
    )
    cuts = stage_cutpoints(lazy)
    assert cuts["cut_S3"][0] is None
    assert cuts["cut_S1"][0] == 0


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def test_prevalence_table_covers_all_stages(cutpoints) -> None:
    table = stage_prevalence_table(cutpoints)
    assert table.height == len(STAGES)
    assert set(table["stage"]) == set(STAGES)
    assert table.filter(pl.col("stage") == "S4")["modelled"][0] is False


def test_reach_rate_decreases_down_the_funnel(cutpoints) -> None:
    table = stage_prevalence_table(cutpoints)
    rates = {r["stage"]: r["reach_rate"] for r in table.to_dicts()}
    assert rates["S1"] >= rates["S3"]


def test_deeper_stages_have_higher_prevalence(cutpoints) -> None:
    """The selection effect the paper must disclose alongside the H1 curve."""
    table = stage_prevalence_table(cutpoints)
    prevalence = {r["stage"]: r["prevalence"] for r in table.to_dicts()}
    assert prevalence["S3"] > prevalence["S1"]


def test_ordering_is_deterministic_under_timestamp_collisions() -> None:
    """One-second granularity means ties are the norm, not the exception."""
    rows = [("2019-10-01 00:00:00", "view", p, 1) for p in range(1, 6)]
    lazy = _log(rows)
    first = stage_cutpoints(lazy)
    second = stage_cutpoints(lazy)
    assert first.equals(second)


def test_pipeline_runs_end_to_end(raw_events) -> None:
    sessions = sessionize(raw_events.lazy(), SessionizeConfig()).collect()
    cuts = stage_cutpoints(sessions.lazy())
    assert cuts.height == sessions["session_id"].n_unique()
    assert cuts["reached_S1"].sum() > 0
    assert cuts["reached_S3"].sum() > 0
