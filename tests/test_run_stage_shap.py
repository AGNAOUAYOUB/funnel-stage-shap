"""Tests for the stage-SHAP runner (protocol Sec. 11.1, 6.2)."""

from __future__ import annotations

import polars as pl
import pytest

from funnel_shap.data.journey import MODELLING_STAGES, prefix_events
from funnel_shap.data.splits import freeze_splits
from funnel_shap.features.prefix_features import build_stage_features


@pytest.fixture(scope="module")
def explained(tmp_path_factory, sessionised, cutpoints):
    import funnel_shap.explain.run_stage_shap as module

    directory = tmp_path_factory.mktemp("shapsplits")
    freeze_splits(sessionised, cutpoints, suffix="sh", directory=directory)

    features = {}
    for stage in MODELLING_STAGES:
        prefix = prefix_events(sessionised.lazy(), cutpoints, stage)
        features[stage] = build_stage_features(prefix, stage)

    original = module.load_split
    module.load_split = lambda suffix, protocol: original(suffix, protocol, directory=directory)
    try:
        return module.explain_stages(
            features, suffix="sh", seed=42, background_size=100, max_explain=200
        )
    finally:
        module.load_split = original


def test_every_reachable_stage_is_explained(explained) -> None:
    assert set(explained).issubset(set(MODELLING_STAGES))
    assert len(explained) >= 2


def test_background_and_explained_sizes_are_recorded(explained) -> None:
    """Sec. 11.1 requires the background set documented."""
    for explanation in explained.values():
        assert explanation.attribution.background_size <= 100
        assert explanation.attribution.n_explained <= 200
        assert explanation.n_train > 0


def test_shap_matrix_matches_feature_count(explained) -> None:
    for explanation in explained.values():
        n_features = len(explanation.attribution.feature_names)
        assert explanation.attribution.shap_values.shape[1] == n_features


def test_migration_uses_one_grouping_across_all_stages(explained) -> None:
    """Re-clustering per stage would compare different bundles at each point."""
    from funnel_shap.explain.run_stage_shap import migration_table

    table = migration_table(explained)
    per_stage_groups = (
        table.group_by("stage").agg(pl.col("group").n_unique().alias("n"))["n"].to_list()
    )
    assert len(set(per_stage_groups)) == 1, "group set differs between stages"


def test_migration_covers_the_full_grid(explained) -> None:
    from funnel_shap.explain.run_stage_shap import migration_table

    table = migration_table(explained)
    assert table.height == table["group"].n_unique() * table["stage"].n_unique()


def test_migration_reports_rq2_quantities(explained) -> None:
    from funnel_shap.explain.run_stage_shap import migration_table

    table = migration_table(explained)
    for column in ("share", "rank", "rank_change", "reversed_sign", "available"):
        assert column in table.columns


def test_importance_shares_sum_to_one_per_stage(explained) -> None:
    from funnel_shap.explain.run_stage_shap import stage_importance_table

    table = stage_importance_table(explained)
    for stage in table["stage"].unique():
        share = table.filter(pl.col("stage") == stage)["share"].sum()
        assert share == pytest.approx(1.0)


def test_empty_input_is_rejected() -> None:
    from funnel_shap.explain.run_stage_shap import migration_table

    with pytest.raises(ValueError, match="no explanations"):
        migration_table({})
