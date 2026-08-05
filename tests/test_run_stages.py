"""Tests for the stage-model runner (protocol Sec. 9.2, 10)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from funnel_shap.data.journey import MODELLING_STAGES, prefix_events
from funnel_shap.data.splits import freeze_splits
from funnel_shap.features.prefix_features import build_stage_features
from funnel_shap.models.run_stages import (
    ablation_table,
    improvement_curve,
    run_stage_models,
    stage_results_table,
)


@pytest.fixture(scope="module")
def stage_setup(tmp_path_factory, sessionised, cutpoints):
    directory = tmp_path_factory.mktemp("splits")
    freeze_splits(sessionised, cutpoints, suffix="st", directory=directory)

    features = {}
    for stage in MODELLING_STAGES:
        prefix = prefix_events(sessionised.lazy(), cutpoints, stage)
        features[stage] = build_stage_features(prefix, stage)
    return features, directory


def _run(features, directory, **kwargs):
    import funnel_shap.models.run_stages as module

    original = module.load_split
    module.load_split = lambda suffix, protocol: original(
        suffix, protocol, directory=directory
    )
    try:
        return run_stage_models(features, suffix="st", n_resamples=2000, **kwargs)
    finally:
        module.load_split = original


def test_a_model_is_fitted_per_stage_and_seed(stage_setup) -> None:
    features, directory = stage_setup
    runs = _run(features, directory, seeds=(7, 17))

    assert {r.stage for r in runs}.issubset(set(MODELLING_STAGES))
    assert {r.seed for r in runs} == {7, 17}
    assert all(r.feature_set == "full" for r in runs)


def test_results_carry_prevalence_and_lift(stage_setup) -> None:
    """PR-AUC alone is not comparable across stages; the lift is."""
    features, directory = stage_setup
    table = stage_results_table(_run(features, directory, seeds=(7,)))

    for column in ("prevalence", "pr_auc", "pr_auc_lift", "n_test", "n_train"):
        assert column in table.columns

    for row in table.to_dicts():
        assert row["pr_auc_lift"] == pytest.approx(row["pr_auc"] / row["prevalence"])


def test_lift_is_at_least_chance_for_a_fitted_model(stage_setup) -> None:
    features, directory = stage_setup
    table = stage_results_table(_run(features, directory, seeds=(7,)))
    # Chance-level PR-AUC equals the prevalence, so lift ~1.0 is chance.
    assert (table["pr_auc_lift"] > 0.5).all()


def test_improvement_curve_is_ordered_by_stage(stage_setup) -> None:
    features, directory = stage_setup
    curve = improvement_curve(_run(features, directory, seeds=(7, 17)))

    stages = curve["stage"].to_list()
    assert stages == [s for s in MODELLING_STAGES if s in stages]


def test_improvement_curve_reports_context_not_just_pr_auc(stage_setup) -> None:
    """Amendment A2: the curve is conditional and unreadable without N and prevalence."""
    features, directory = stage_setup
    curve = improvement_curve(_run(features, directory, seeds=(7,)))

    for column in ("n_test", "prevalence", "pr_auc_lift_mean", "n_seeds"):
        assert column in curve.columns


def test_ablation_ladder_runs_every_rung(stage_setup) -> None:
    features, directory = stage_setup
    runs = _run(
        features,
        directory,
        seeds=(7,),
        feature_sets=("baseline", "+temporal", "+entropy_velocity", "full"),
    )
    table = ablation_table(runs)

    per_stage = table.group_by("stage").agg(pl.col("feature_set").n_unique().alias("n"))
    assert per_stage["n"].min() >= 2


def test_full_feature_set_uses_more_columns_than_baseline(stage_setup) -> None:
    features, directory = stage_setup
    runs = _run(features, directory, seeds=(7,), feature_sets=("baseline", "full"))

    by_set = {}
    for run in runs:
        by_set.setdefault(run.feature_set, []).append(run)
    assert set(by_set) == {"baseline", "full"}


def test_seeds_produce_variation_but_not_chaos(stage_setup) -> None:
    features, directory = stage_setup
    table = stage_results_table(_run(features, directory, seeds=(7, 17, 23)))

    spread = table.group_by("stage").agg(pl.col("pr_auc").std().alias("s"))
    assert spread["s"].max() < 0.25, "seed-to-seed variation is implausibly large"


def test_scores_are_probabilities(stage_setup) -> None:
    features, directory = stage_setup
    for run in _run(features, directory, seeds=(7,)):
        assert np.all((run.test_scores >= 0) & (run.test_scores <= 1))
        assert len(run.test_scores) == len(run.y_test)
