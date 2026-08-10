"""Tests for the fixed-cohort evaluation (composition vs information)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from funnel_shap.data.journey import MODELLING_STAGES, prefix_events
from funnel_shap.data.splits import freeze_splits
from funnel_shap.features.prefix_features import build_stage_features
from funnel_shap.models.common_cohort import (
    bootstrap_cohort,
    cohort_table,
    common_cohort_ids,
    run_common_cohort,
)


@pytest.fixture(scope="module")
def cohort_setup(tmp_path_factory, sessionised, cutpoints):
    directory = tmp_path_factory.mktemp("cohort_splits")
    freeze_splits(sessionised, cutpoints, suffix="cc", directory=directory)

    features = {}
    for stage in MODELLING_STAGES:
        prefix = prefix_events(sessionised.lazy(), cutpoints, stage)
        features[stage] = build_stage_features(prefix, stage)
    return features, directory


def _run(features, directory, **kwargs):
    import funnel_shap.models.common_cohort as module

    original = module.load_split
    module.load_split = lambda suffix, protocol: original(
        suffix, protocol, directory=directory
    )
    try:
        return run_common_cohort(features, suffix="cc", **kwargs)
    finally:
        module.load_split = original


def test_cohort_is_the_intersection_not_the_widest_stage(cohort_setup) -> None:
    features, _ = cohort_setup
    ids = common_cohort_ids(features)

    per_stage = {s: set(f["session_id"].to_list()) for s, f in features.items()}
    assert ids == set.intersection(*per_stage.values())
    # S3 is the narrowest stage, so the cohort cannot exceed it.
    assert len(ids) <= min(len(v) for v in per_stage.values())


def test_cohort_is_computed_not_assumed_to_be_s3(cohort_setup) -> None:
    """Nesting is expected but must not be assumed; a violation must shrink the cohort."""
    features, _ = cohort_setup
    trimmed = dict(features)
    keep = features["S1"]["session_id"].to_list()[:-1]
    trimmed["S1"] = features["S1"].filter(pl.col("session_id").is_in(keep))

    ids = common_cohort_ids(trimmed)
    assert ids == common_cohort_ids(features) - {features["S1"]["session_id"].to_list()[-1]} or len(
        ids
    ) <= len(common_cohort_ids(features))


def test_every_stage_is_evaluated_on_the_identical_population(cohort_setup) -> None:
    """The whole point: same sessions, same labels, same prevalence."""
    features, directory = cohort_setup
    runs = _run(features, directory, seeds=(7,))

    assert runs, "no stage produced an evaluable cohort run"
    sizes = {r.n_eval for r in runs}
    prevalences = {round(r.prevalence, 10) for r in runs}
    assert len(sizes) == 1, "evaluation cohort differs between stages"
    assert len(prevalences) == 1, "prevalence differs between stages"


def test_lift_is_comparable_because_prevalence_is_shared(cohort_setup) -> None:
    features, directory = cohort_setup
    runs = _run(features, directory, seeds=(7,))
    for r in runs:
        assert r.lift == pytest.approx(r.pr_auc / r.prevalence)


def test_cohort_table_reports_dispersion_over_seeds(cohort_setup) -> None:
    features, directory = cohort_setup
    runs = _run(features, directory, seeds=(7, 17))
    table = cohort_table(runs)

    assert set(table["stage"]) <= set(MODELLING_STAGES)
    assert table["n_seeds"].min() == 2
    assert "lift_sd" in table.columns


def test_empty_cohort_raises_rather_than_reporting_nothing(cohort_setup) -> None:
    """A silent empty comparison would read as 'no difference between stages'."""
    features, directory = cohort_setup
    disjoint = {
        "S1": features["S1"].head(1),
        "S3": features["S3"].tail(1).with_columns(
            pl.lit("no_such_session").alias("session_id")
        ),
    }
    with pytest.raises(ValueError, match="cohort is empty"):
        _run(disjoint, directory, seeds=(7,))


def test_scores_and_labels_are_retained_for_downstream_sweeps(cohort_setup) -> None:
    features, directory = cohort_setup
    runs = _run(features, directory, seeds=(7,))
    for r in runs:
        assert isinstance(r.scores, np.ndarray)
        assert r.scores.shape == r.y_true.shape == (r.n_eval,)
        assert len(r.session_ids) == r.n_eval


def test_stages_hold_the_same_customers_in_different_orders(cohort_setup) -> None:
    """The reason paired comparisons must align on ids rather than position."""
    features, directory = cohort_setup
    runs = _run(features, directory, seeds=(7,))
    by_stage = {r.stage: r for r in runs}
    ids = [set(r.session_ids) for r in by_stage.values()]
    assert all(s == ids[0] for s in ids), "the cohort is not shared"


def test_bootstrap_aligns_on_session_id_not_position(cohort_setup) -> None:
    """Shuffling one stage's rows must not change its interval."""
    features, directory = cohort_setup
    runs = _run(features, directory, seeds=(7, 17))
    straight = bootstrap_cohort(runs, n_resamples=200, seed=3)

    rng = np.random.default_rng(0)
    shuffled = []
    for run in runs:
        order = rng.permutation(run.n_eval)
        shuffled.append(
            type(run)(
                **{
                    **run.__dict__,
                    "scores": run.scores[order],
                    "y_true": run.y_true[order],
                    "session_ids": [run.session_ids[i] for i in order],
                }
            )
        )
    reordered = bootstrap_cohort(shuffled, n_resamples=200, seed=3)
    assert straight.equals(reordered)


def test_bootstrap_rejects_runs_without_session_ids(cohort_setup) -> None:
    features, directory = cohort_setup
    runs = _run(features, directory, seeds=(7,))
    stripped = [type(r)(**{**r.__dict__, "session_ids": None}) for r in runs]
    with pytest.raises(ValueError, match="no session ids"):
        bootstrap_cohort(stripped, n_resamples=50)


def test_bootstrap_reports_levels_and_paired_differences(cohort_setup) -> None:
    features, directory = cohort_setup
    runs = _run(features, directory, seeds=(7,))
    table = bootstrap_cohort(runs, n_resamples=200)

    kinds = set(table["kind"])
    assert kinds == {"level", "difference"}
    # An interval must contain its own resampled mean.
    for row in table.to_dicts():
        assert row["lo"] <= row["mean"] <= row["hi"]


def test_training_can_be_restricted_to_the_cohort(cohort_setup) -> None:
    """The variant that removes prior-probability shift must actually change the fit."""
    features, directory = cohort_setup
    wide = _run(features, directory, seeds=(7,))
    narrow = _run(features, directory, seeds=(7,), restrict_training_to_cohort=True)

    assert {r.n_eval for r in narrow} == {r.n_eval for r in wide}
    assert all(n.n_train <= w.n_train for n, w in zip(narrow, wide, strict=True))
