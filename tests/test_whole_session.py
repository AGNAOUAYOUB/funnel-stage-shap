"""Tests for the empirical whole-session comparator (RQ4)."""

from __future__ import annotations

import polars as pl
import pytest

from funnel_shap.explain.whole_session import (
    WholeSessionResult,
    contrast_table,
    member_to_group,
    whole_session_events,
)


def test_member_to_group_inverts_cluster_names() -> None:
    names = ["a+b+c", "solo", "x+y"]
    mapping = member_to_group(names)

    assert mapping["a"] == "a+b+c"
    assert mapping["c"] == "a+b+c"
    assert mapping["solo"] == "solo"
    assert mapping["y"] == "x+y"


def test_whole_session_keeps_every_event(sessionised, cutpoints) -> None:
    """The comparator's whole point is that no cut-point is applied."""
    from funnel_shap.data.journey import prefix_events

    whole = whole_session_events(sessionised, cutpoints).collect()
    s1 = prefix_events(sessionised.lazy(), cutpoints, "S1").collect()

    assert whole.height > s1.height, "whole-session must see more events than an S1 prefix"
    # Every session with a label appears, not only those reaching a given stage.
    assert whole["session_id"].n_unique() == cutpoints.height


def test_whole_session_features_carry_the_label(sessionised, cutpoints) -> None:
    from funnel_shap.explain.whole_session import build_whole_session_features

    frame = build_whole_session_features(sessionised, cutpoints)
    assert "label" in frame.columns
    assert "session_id" in frame.columns
    assert frame.height == cutpoints.height


def test_contrast_table_reports_peak_and_flattening() -> None:
    importance = pl.DataFrame(
        {
            "stage": ["S1", "S2", "S3"] * 2,
            "group": ["nav"] * 3 + ["price"] * 3,
            "share": [0.035, 0.224, 0.074, 0.437, 0.239, 0.379],
        }
    )
    whole = WholeSessionResult(
        n_train=10, n_test=5, prevalence=0.2, pr_auc=0.5, roc_auc=0.6,
        shares={"nav": 0.10, "price": 0.40},
    )

    table = contrast_table(importance, whole)
    rows = {r["group"]: r for r in table.to_dicts()}

    assert rows["nav"]["stage_peak"] == pytest.approx(0.224)
    assert rows["nav"]["stage_peak_at"] == "S2"
    assert rows["nav"]["flattening"] == pytest.approx(0.124)
    # Price peaks early and the static model already carries it: little flattening.
    assert rows["price"]["flattening"] == pytest.approx(0.037)


def test_contrast_table_can_report_negative_flattening() -> None:
    """A family the static model over-weights must not be silently clipped."""
    importance = pl.DataFrame(
        {"stage": ["S1", "S2"], "group": ["cart", "cart"], "share": [0.05, 0.08]}
    )
    whole = WholeSessionResult(
        n_train=10, n_test=5, prevalence=0.2, pr_auc=0.5, roc_auc=0.6,
        shares={"cart": 0.40},
    )

    table = contrast_table(importance, whole)
    assert table["flattening"][0] == pytest.approx(-0.32)


def test_disjoint_grouping_is_rejected_not_silently_zeroed() -> None:
    """An earlier draft grouped by the wrong taxonomy and reported 0 for everything."""
    import numpy as np

    from funnel_shap.explain.whole_session import aggregate_shares

    with pytest.raises(ValueError, match="disjoint|meaningless"):
        aggregate_shares(
            ["click_velocity", "price_mean"],
            np.array([0.6, 0.4]),
            grouping={"totally_unrelated": "some_group"},
        )


def test_aggregate_shares_sums_within_clusters() -> None:
    import numpy as np

    from funnel_shap.explain.whole_session import aggregate_shares

    shares = aggregate_shares(
        ["a", "b", "solo"],
        np.array([0.5, 0.2, 0.3]),
        grouping={"a": "a+b", "b": "a+b", "solo": "solo"},
    )
    assert shares["a+b"] == pytest.approx(0.7)
    assert shares["solo"] == pytest.approx(0.3)
    assert sum(shares.values()) == pytest.approx(1.0)


def test_exclude_purchase_drops_only_purchase_events() -> None:
    """The fair comparator must lose the label without losing the session."""
    import polars as pl

    from funnel_shap.explain.whole_session import whole_session_events

    sessions = pl.DataFrame({
        "session_id": ["a", "a", "a", "b", "b"],
        "user_id": ["u1", "u1", "u1", "u2", "u2"],
        "event_time": pl.Series(
            ["2019-10-01T00:00:00", "2019-10-01T00:01:00", "2019-10-01T00:02:00",
             "2019-10-01T00:00:00", "2019-10-01T00:01:00"]
        ).str.to_datetime(),
        "event_type": ["view", "cart", "purchase", "view", "view"],
        "product_id": ["p1", "p1", "p1", "p2", "p3"],
        "category_id": ["c1", "c1", "c1", "c2", "c2"],
        "price": [10.0, 10.0, 10.0, 5.0, 6.0],
    })
    cutpoints = pl.DataFrame({"session_id": ["a", "b"], "label": [1, 0]})

    keep_all = whole_session_events(sessions, cutpoints).collect()
    no_buy = whole_session_events(sessions, cutpoints, exclude_purchase=True).collect()

    assert (keep_all["event_type"] == "purchase").sum() == 1
    assert (no_buy["event_type"] == "purchase").sum() == 0
    # The purchasing session survives, minus its purchase event.
    assert no_buy.filter(pl.col("session_id") == "a").height == 2
    # And it keeps its positive label: the outcome comes from the cut-point table.
    assert set(no_buy.filter(pl.col("session_id") == "a")["label"].unique()) == {1}
