"""Tests for stage-conditioned TreeSHAP and migration (protocol Sec. 11.1)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from funnel_shap.explain.stage_shap import (
    StageAttribution,
    attribution_migration,
    correlation_groups,
    grouped_attribution,
    stage_tree_shap,
)


@pytest.fixture(scope="module")
def fitted_model():
    """A tree model on data where feature 0 is the only real signal."""
    from lightgbm import LGBMClassifier

    rng = np.random.default_rng(42)
    n = 2000
    y = rng.binomial(1, 0.2, size=n)
    signal = rng.normal(loc=y * 2.0, scale=1.0, size=n)
    frame = pl.DataFrame(
        {
            "signal": signal,
            # Two near-duplicates of signal, to exercise the grouping path.
            "signal_copy": signal + rng.normal(scale=0.005, size=n),
            "signal_scaled": signal * 3.0 + rng.normal(scale=0.005, size=n),
            "noise": rng.normal(size=n),
        }
    )
    model = LGBMClassifier(n_estimators=60, verbose=-1, random_state=7)
    model.fit(frame.to_pandas(), y)
    return model, frame, y


def _attribution(fitted_model, stage="S1", n_explain=300):
    model, frame, _ = fitted_model
    return stage_tree_shap(
        model,
        background=frame.head(500),
        explain=frame.tail(n_explain),
        stage=stage,
        feature_names=frame.columns,
        background_size=200,
        seed=7,
    )


# ---------------------------------------------------------------------------
# TreeSHAP
# ---------------------------------------------------------------------------


def test_shap_shape_matches_explained_rows_and_features(fitted_model) -> None:
    attribution = _attribution(fitted_model)
    assert attribution.shap_values.shape == (300, 4)
    assert attribution.n_explained == 300
    assert attribution.background_size == 200


def test_background_is_subsampled_to_the_requested_size(fitted_model) -> None:
    """Sec. 11.1 requires the background set to be documented; it must be honoured."""
    model, frame, _ = fitted_model
    attribution = stage_tree_shap(
        model, background=frame, explain=frame.head(50), stage="S1",
        feature_names=frame.columns, background_size=100, seed=7,
    )
    assert attribution.background_size == 100


def test_correlated_copies_split_credit_below_noise(fitted_model) -> None:
    """Why Sec. 11.1 mandates grouped attribution, demonstrated.

    ``signal``, ``signal_copy`` and ``signal_scaled`` are the same variable up
    to noise of sd 0.005. The trees split arbitrarily among them, so their
    Shapley credit is divided three ways -- far enough that at least one genuine
    signal copy ranks *below* the pure-noise feature. Ranking individual
    features would put that artefact straight onto the migration figure.
    """
    importance = _attribution(fitted_model).global_importance()
    ranks = {r["feature"]: r["rank"] for r in importance.to_dicts()}
    copies = ["signal", "signal_copy", "signal_scaled"]

    assert max(ranks[c] for c in copies) > ranks["noise"], (
        "expected credit-splitting to push a redundant copy below noise; "
        "if this stops holding the grouping demonstration needs revisiting"
    )


def test_grouping_restores_signal_dominance_over_noise(fitted_model) -> None:
    """And the fix: grouped, the signal cluster beats noise decisively."""
    attribution = _attribution(fitted_model)
    _, frame, _ = fitted_model
    groups = correlation_groups(frame, threshold=0.95)

    grouped = grouped_attribution(attribution, groups)
    ranks = {r["group"]: r["rank"] for r in grouped.to_dicts()}
    signal_group = next(g for g, ms in groups.items() if "signal" in ms)

    assert ranks[signal_group] < ranks["noise"]
    shares = {r["group"]: r["share"] for r in grouped.to_dicts()}
    assert shares[signal_group] > 3 * shares["noise"]


def test_global_importance_shares_sum_to_one(fitted_model) -> None:
    importance = _attribution(fitted_model).global_importance()
    assert importance["share"].sum() == pytest.approx(1.0)
    assert importance["rank"].to_list() == sorted(importance["rank"].to_list())


def test_signed_mean_is_reported_beside_magnitude(fitted_model) -> None:
    """RQ2 asks about sign reversal, which |SHAP| alone cannot show."""
    importance = _attribution(fitted_model).global_importance()
    assert "mean_signed_shap" in importance.columns
    assert (importance["mean_abs_shap"] >= importance["mean_signed_shap"].abs() - 1e-9).all()


def test_seeded_background_is_reproducible(fitted_model) -> None:
    a = _attribution(fitted_model)
    b = _attribution(fitted_model)
    np.testing.assert_allclose(a.shap_values, b.shap_values)


# ---------------------------------------------------------------------------
# Correlation grouping
# ---------------------------------------------------------------------------


def test_near_duplicate_features_are_grouped(fitted_model) -> None:
    _, frame, _ = fitted_model
    groups = correlation_groups(frame, threshold=0.95)

    member_of = {m: g for g, ms in groups.items() for m in ms}
    assert member_of["signal"] == member_of["signal_copy"] == member_of["signal_scaled"]
    assert member_of["noise"] != member_of["signal"]


def test_grouping_is_transitive(fitted_model) -> None:
    """Single linkage: a chain of pairwise-redundant features is one group."""
    rng = np.random.default_rng(0)
    base = rng.normal(size=800)
    frame = pl.DataFrame(
        {
            "a": base,
            "b": base + rng.normal(scale=0.001, size=800),
            "c": base + rng.normal(scale=0.002, size=800),
            "z": rng.normal(size=800),
        }
    )
    groups = correlation_groups(frame, threshold=0.95)
    member_of = {m: g for g, ms in groups.items() for m in ms}
    assert member_of["a"] == member_of["b"] == member_of["c"]


def test_constant_columns_stand_alone() -> None:
    frame = pl.DataFrame(
        {"x": np.random.default_rng(0).normal(size=100), "const": np.ones(100)}
    )
    groups = correlation_groups(frame)
    assert ["const"] in groups.values()


def test_every_feature_appears_in_exactly_one_group(fitted_model) -> None:
    _, frame, _ = fitted_model
    groups = correlation_groups(frame)
    members = [m for ms in groups.values() for m in ms]
    assert sorted(members) == sorted(frame.columns)


# ---------------------------------------------------------------------------
# Grouped attribution
# ---------------------------------------------------------------------------


def test_grouped_attribution_sums_rather_than_averages(fitted_model) -> None:
    """A group of duplicates must not look less important than one of them."""
    attribution = _attribution(fitted_model)
    _, frame, _ = fitted_model
    groups = correlation_groups(frame, threshold=0.95)

    grouped = grouped_attribution(attribution, groups)
    individual = attribution.global_importance()

    signal_group = [g for g, ms in groups.items() if "signal" in ms][0]
    row = grouped.filter(pl.col("group") == signal_group).to_dicts()[0]
    members = groups[signal_group]
    expected = sum(
        r["mean_abs_shap"] for r in individual.to_dicts() if r["feature"] in members
    )
    assert row["mean_abs_shap"] == pytest.approx(expected)


def test_grouped_shares_sum_to_one(fitted_model) -> None:
    attribution = _attribution(fitted_model)
    _, frame, _ = fitted_model
    grouped = grouped_attribution(attribution, correlation_groups(frame))
    assert grouped["share"].sum() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------


def test_migration_covers_every_group_at_every_stage(fitted_model) -> None:
    attributions = {
        "S1": _attribution(fitted_model, "S1"),
        "S2": _attribution(fitted_model, "S2"),
    }
    table = attribution_migration(attributions)

    n_groups = table["group"].n_unique()
    assert table.height == n_groups * 2


def test_migration_marks_absent_features_rather_than_zeroing_them() -> None:
    """A missing feature must break the line, not be drawn at zero."""
    rng = np.random.default_rng(0)
    s1 = StageAttribution(
        stage="S1", feature_names=["a", "b"],
        shap_values=rng.normal(size=(50, 2)), background_size=10, n_explained=50, seed=7,
    )
    s3 = StageAttribution(
        stage="S3", feature_names=["a", "b", "cart_only"],
        shap_values=rng.normal(size=(50, 3)), background_size=10, n_explained=50, seed=7,
    )
    table = attribution_migration({"S1": s1, "S3": s3})

    cart = table.filter(pl.col("group") == "cart_only").sort("stage")
    assert cart.filter(pl.col("stage") == "S1")["available"][0] is False
    assert cart.filter(pl.col("stage") == "S1")["share"][0] is None
    assert cart.filter(pl.col("stage") == "S3")["available"][0] is True


def test_migration_detects_sign_reversal() -> None:
    """RQ2 asks explicitly whether any feature reverses sign across stages."""
    n = 200
    flip_s1 = np.column_stack([np.full(n, 0.5), np.full(n, 0.1)])
    flip_s3 = np.column_stack([np.full(n, -0.5), np.full(n, 0.1)])

    s1 = StageAttribution("S1", ["flipper", "steady"], flip_s1, 10, n, 7)
    s3 = StageAttribution("S3", ["flipper", "steady"], flip_s3, 10, n, 7)
    table = attribution_migration({"S1": s1, "S3": s3})

    flipper = table.filter((pl.col("group") == "flipper") & (pl.col("stage") == "S3"))
    steady = table.filter((pl.col("group") == "steady") & (pl.col("stage") == "S3"))
    assert flipper["reversed_sign"][0] is True
    assert steady["reversed_sign"][0] is False


def test_migration_reports_rank_and_share_change() -> None:
    n = 100
    s1 = StageAttribution(
        "S1", ["a", "b"], np.column_stack([np.full(n, 1.0), np.full(n, 0.1)]), 10, n, 7
    )
    s3 = StageAttribution(
        "S3", ["a", "b"], np.column_stack([np.full(n, 0.1), np.full(n, 1.0)]), 10, n, 7
    )
    table = attribution_migration({"S1": s1, "S3": s3})

    a_s3 = table.filter((pl.col("group") == "a") & (pl.col("stage") == "S3")).to_dicts()[0]
    assert a_s3["rank_change"] > 0, "feature a should lose rank"
    assert a_s3["share_change"] < 0


def test_empty_attributions_are_rejected() -> None:
    with pytest.raises(ValueError, match="no attributions"):
        attribution_migration({})
