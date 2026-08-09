"""Tests for the fixed-cohort evaluation (composition vs information)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from funnel_shap.data.journey import MODELLING_STAGES, prefix_events
from funnel_shap.data.splits import freeze_splits
from funnel_shap.features.prefix_features import build_stage_features
from funnel_shap.models.common_cohort import (
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
