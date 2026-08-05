"""Tests for the confirmatory comparison family (protocol Sec. 12)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from funnel_shap.evaluate.stage_tests import (
    CONFIRMATORY_CONTRASTS,
    run_stage_comparisons,
    summarise,
)


@dataclass
class _Report:
    point: dict


@dataclass
class _FakeRun:
    stage: str
    feature_set: str
    seed: int
    test_scores: np.ndarray
    y_test: np.ndarray
    test: _Report


def _make_runs(effect: float = 0.8, n: int = 3000, stages=("S1", "S3")):
    """Runs where the richer feature set is genuinely better by ``effect``."""
    from sklearn.metrics import average_precision_score

    rng = np.random.default_rng(42)
    runs = []
    for stage in stages:
        y = rng.binomial(1, 0.2, size=n)
        strengths = {
            "baseline": 0.4,
            "+temporal": 0.4 + effect,
            "baseline+entropy_velocity": 0.4 + effect / 2,
            "+entropy_velocity": 0.4 + effect,
        }
        for feature_set, strength in strengths.items():
            for seed in (7, 17, 23, 42, 101):
                r = np.random.default_rng(seed)
                logit = r.normal(loc=y * strength, scale=1.0)
                scores = 1 / (1 + np.exp(-logit))
                runs.append(
                    _FakeRun(
                        stage=stage,
                        feature_set=feature_set,
                        seed=seed,
                        test_scores=scores,
                        y_test=y,
                        test=_Report({"pr_auc": float(average_precision_score(y, scores))}),
                    )
                )
    return runs


def test_family_covers_every_declared_contrast_at_every_stage() -> None:
    frame = run_stage_comparisons(_make_runs())

    assert frame["family_size"][0] == len(CONFIRMATORY_CONTRASTS) * 2
    assert set(frame["stage"]) == {"S1", "S3"}


def test_effect_sizes_and_cis_are_reported(frame_cache={}) -> None:
    """Sec. 12: effect sizes and CIs, not bare p-values."""
    frame = frame_cache.setdefault("f", run_stage_comparisons(_make_runs()))

    for column in ("pr_auc_diff", "ci_low", "ci_high", "excludes_zero"):
        assert column in frame.columns
    for row in frame.to_dicts():
        assert row["ci_low"] <= row["pr_auc_diff"] <= row["ci_high"]


def test_a_real_improvement_has_a_ci_excluding_zero() -> None:
    frame = run_stage_comparisons(_make_runs(effect=0.8))
    temporal = frame.filter(
        (frame["feature_set_a"] == "+temporal") & (frame["feature_set_b"] == "baseline")
    )
    assert temporal["pr_auc_diff"].min() > 0
    assert bool(temporal["excludes_zero"].all())


def test_no_difference_gives_a_ci_including_zero() -> None:
    """Identical models must not be reported as different."""
    runs = _make_runs(effect=0.0)
    frame = run_stage_comparisons(runs)
    redundancy = frame.filter(
        (frame["feature_set_a"] == "+entropy_velocity")
        & (frame["feature_set_b"] == "+temporal")
    )
    assert bool((~redundancy["excludes_zero"]).all())


def test_correction_spans_the_whole_family_not_each_stage() -> None:
    frame = run_stage_comparisons(_make_runs())
    assert frame["family_size"][0] == frame.height
    for row in frame.to_dicts():
        assert row["wilcoxon_p_holm"] >= row["wilcoxon_p"]


def test_wilcoxon_floor_is_surfaced() -> None:
    """Five seeds cannot reach p<0.05; the table must say so."""
    frame = run_stage_comparisons(_make_runs())
    assert frame["wilcoxon_min_attainable_p"].to_list() == pytest.approx([0.0625] * frame.height)
    assert (frame["wilcoxon_p"] >= 0.0625).all()


def test_summary_leads_with_effect_size() -> None:
    text = summarise(run_stage_comparisons(_make_runs()))
    assert "minimum attainable 0.0625" in text
    assert "+temporal" in text


def test_unknown_correction_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown correction"):
        run_stage_comparisons(_make_runs(), correction="bonferroni")


def test_missing_rungs_raise_rather_than_report_nothing() -> None:
    runs = [r for r in _make_runs() if r.feature_set == "baseline"]
    with pytest.raises(ValueError, match="no contrasts"):
        run_stage_comparisons(runs)
