"""Tests for stage-prefix feature construction (protocol Sec. 8)."""

from __future__ import annotations

import math

import polars as pl
import pytest

from funnel_shap.data.journey import MODELLING_STAGES, StageError, prefix_events, stage_cutpoints
from funnel_shap.features.dictionary import (
    ABLATION_LADDER,
    availability_by_stage,
    feature_names,
)
from funnel_shap.features.prefix_features import build_stage_features


def _log(rows: list[tuple[str, str, int, int]]) -> pl.LazyFrame:
    return pl.DataFrame(
        {
            "user_id": [1] * len(rows),
            "session_id": ["1_0"] * len(rows),
            "event_time": [r[0] for r in rows],
            "event_type": [r[1] for r in rows],
            "product_id": [r[2] for r in rows],
            "category_id": [r[3] for r in rows],
            "price": [10.0 * (i + 1) for i in range(len(rows))],
        }
    ).lazy().with_columns(
        pl.col("event_time").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S")
    )


def _features_for(rows, stage):
    lazy = _log(rows)
    cuts = stage_cutpoints(lazy)
    prefix = prefix_events(lazy, cuts, stage)
    return build_stage_features(prefix, stage)


# ---------------------------------------------------------------------------
# Contract with the feature dictionary
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stage", MODELLING_STAGES)
def test_columns_match_the_dictionary(sessionised, cutpoints, stage) -> None:
    prefix = prefix_events(sessionised.lazy(), cutpoints, stage)
    frame = build_stage_features(prefix, stage)
    assert frame.columns == ["session_id", "label", *feature_names(stage)]


def test_cart_features_exist_only_at_s3() -> None:
    assert "n_cart_adds" not in feature_names("S1")
    assert "time_since_last_cart_s" not in feature_names("S2")
    assert "n_cart_adds" in feature_names("S3")


def test_s4_has_no_feature_matrix() -> None:
    with pytest.raises(StageError, match="modelling stages"):
        build_stage_features(_log([("2019-10-01 00:00:00", "view", 1, 1)]), "S4")


def test_availability_table_is_auditable() -> None:
    table = availability_by_stage()
    assert set(MODELLING_STAGES).issubset(table.columns)
    # Every feature must be available at at least one modelling stage.
    reachable = [
        any(row[s] for s in MODELLING_STAGES) for row in table.to_dicts()
    ]
    assert all(reachable)


def test_ablation_ladder_is_nested() -> None:
    """baseline -> +temporal -> +entropy/velocity -> full (Sec. 10, H3)."""
    seen: set[str] = set()
    for _, groups in ABLATION_LADDER:
        current = set(groups)
        assert seen.issubset(current), "ablation rungs must be nested"
        seen = current


# ---------------------------------------------------------------------------
# Feature semantics
# ---------------------------------------------------------------------------


def test_counts_are_computed_on_the_prefix_only() -> None:
    rows = [
        ("2019-10-01 00:00:00", "view", 1, 1),
        ("2019-10-01 00:00:30", "view", 2, 1),
        ("2019-10-01 00:01:00", "view", 3, 2),  # S2 cut here
        ("2019-10-01 00:02:00", "view", 4, 3),
        ("2019-10-01 00:03:00", "cart", 4, 3),
    ]
    s1 = _features_for(rows, "S1")
    s2 = _features_for(rows, "S2")

    # S1 closes at the second interaction (amendment A6), S2 at the category
    # switch on event 3.
    assert s1["n_events"][0] == 2
    assert s2["n_events"][0] == 3
    assert s2["n_views"][0] == 3


def test_dwell_excludes_the_final_event() -> None:
    """The last prefix event's dwell would need an event past the cut-point."""
    rows = [
        ("2019-10-01 00:00:00", "view", 1, 1),
        ("2019-10-01 00:00:30", "view", 2, 1),
        ("2019-10-01 00:01:30", "view", 1, 1),  # repeat -> S2 cut at idx 2
    ]
    s2 = _features_for(rows, "S2")
    # Dwells inside the prefix: 30s (event 0 -> 1) and 60s (event 1 -> 2).
    # Event 2 is last, contributing nothing.
    assert s2["dwell_total_s"][0] == pytest.approx(90.0)


def test_minimal_prefix_has_measurable_features_not_constants() -> None:
    """Amendment A6's purpose: the smallest S1 prefix still carries real signal."""
    rows = [
        ("2019-10-01 00:00:00", "view", 1, 1),
        ("2019-10-01 00:00:30", "view", 2, 2),
    ]
    s1 = _features_for(rows, "S1")
    assert s1["n_events"][0] == 2
    assert s1["prefix_duration_s"][0] == pytest.approx(30.0)
    assert s1["inter_event_mean_s"][0] == pytest.approx(30.0)
    assert s1["dwell_total_s"][0] == pytest.approx(30.0)
    assert s1["category_entropy"][0] == pytest.approx(math.log(2))
    assert math.isfinite(s1["click_velocity"][0])

    # A two-event prefix has exactly one gap, so its std is undefined; the
    # zero-fill must turn that into 0.0 rather than leaking a null into the
    # model matrix.
    assert s1["inter_event_std_s"][0] == 0.0


def test_category_entropy_is_zero_for_a_single_category() -> None:
    rows = [
        ("2019-10-01 00:00:00", "view", 1, 1),
        ("2019-10-01 00:00:30", "view", 2, 1),
        ("2019-10-01 00:01:00", "view", 1, 1),  # repeat -> S2 cut at idx 2
    ]
    s2 = _features_for(rows, "S2")
    assert s2["n_unique_categories"][0] == 1
    assert s2["category_entropy"][0] == pytest.approx(0.0)
    assert s2["category_entropy_normalised"][0] == pytest.approx(0.0)


def test_category_entropy_matches_the_closed_form() -> None:
    """Two categories, 2 events each => H = log(2)."""
    rows = [
        ("2019-10-01 00:00:00", "view", 1, 1),
        ("2019-10-01 00:00:30", "view", 2, 2),  # switch -> S2 cut at idx 1
    ]
    lazy = _log(rows)
    cuts = stage_cutpoints(lazy)
    frame = build_stage_features(prefix_events(lazy, cuts, "S2"), "S2")
    assert frame["category_entropy"][0] == pytest.approx(math.log(2))
    assert frame["category_entropy_normalised"][0] == pytest.approx(1.0)


def test_transition_count_counts_category_changes() -> None:
    rows = [
        ("2019-10-01 00:00:00", "view", 1, 1),
        ("2019-10-01 00:00:30", "view", 2, 2),  # transition, S2 cut
        ("2019-10-01 00:01:00", "view", 3, 3),
    ]
    s2 = _features_for(rows, "S2")
    assert s2["n_category_transitions"][0] == 1


def test_product_revisit_rate() -> None:
    rows = [
        ("2019-10-01 00:00:00", "view", 1, 1),
        ("2019-10-01 00:00:30", "view", 2, 1),
        ("2019-10-01 00:01:00", "view", 1, 1),  # repeat -> S2 cut at idx 2
    ]
    s2 = _features_for(rows, "S2")
    # 1 of 3 prefix events lands on an already-seen product.
    assert s2["product_revisit_rate"][0] == pytest.approx(1 / 3)


def test_intent_composite_uses_documented_weights() -> None:
    rows = [
        ("2019-10-01 00:00:00", "view", 1, 1),
        ("2019-10-01 00:00:30", "view", 2, 2),
        ("2019-10-01 00:01:00", "cart", 2, 2),  # S3 cut
    ]
    s3 = _features_for(rows, "S3")
    # 2 views * 1.0 + 1 cart * 4.0
    assert s3["purchase_intent_score"][0] == pytest.approx(6.0)


def test_time_since_last_cart_is_zero_at_the_cut_point() -> None:
    """S3's cut-point IS the first cart, so recency is zero unless a later cart exists."""
    rows = [
        ("2019-10-01 00:00:00", "view", 1, 1),
        ("2019-10-01 00:01:00", "cart", 1, 1),
    ]
    s3 = _features_for(rows, "S3")
    assert s3["time_since_last_cart_s"][0] == pytest.approx(0.0)


def test_click_velocity_is_finite_and_positive() -> None:
    rows = [
        ("2019-10-01 00:00:00", "view", 1, 1),
        ("2019-10-01 00:00:30", "view", 2, 2),
    ]
    s2 = _features_for(rows, "S2")
    assert s2["click_velocity"][0] == pytest.approx(2 / 30, rel=1e-3)


# ---------------------------------------------------------------------------
# Leakage
# ---------------------------------------------------------------------------


def test_features_ignore_events_after_the_cut_point() -> None:
    """Appending post-cut-point events must not change earlier stages' features."""
    base = [
        ("2019-10-01 00:00:00", "view", 1, 1),
        ("2019-10-01 00:00:30", "view", 2, 2),  # S2 cut at idx 1
    ]
    extended = base + [
        ("2019-10-01 00:05:00", "view", 3, 3),
        ("2019-10-01 00:06:00", "cart", 3, 3),
        ("2019-10-01 00:07:00", "purchase", 3, 3),
    ]

    for stage in ("S1", "S2"):
        a = _features_for(base, stage).drop("label")
        b = _features_for(extended, stage).drop("label")
        assert a.equals(b), f"{stage} features changed when future events were appended"


@pytest.mark.parametrize("stage", MODELLING_STAGES)
def test_no_feature_is_all_null(sessionised, cutpoints, stage) -> None:
    prefix = prefix_events(sessionised.lazy(), cutpoints, stage)
    frame = build_stage_features(prefix, stage)
    for column in feature_names(stage):
        assert frame[column].null_count() < frame.height, f"{column} is entirely null at {stage}"
