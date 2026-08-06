"""Tests for MLflow run logging (protocol Sec. 6.2)."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import pytest

from funnel_shap import tracking
from funnel_shap.tracking import RunRecorder, stage_run_metrics, track_run


@pytest.fixture(autouse=True)
def _reset_warning_latch():
    """The module warns once per process; tests must not inherit that state."""
    tracking._WARNED = False
    yield
    tracking._WARNED = False


def _read_runs(directory):
    import mlflow

    mlflow.set_tracking_uri(directory.resolve().as_uri())
    return mlflow


def test_a_run_is_written_with_params_metrics_and_artifact(tmp_path) -> None:
    artifact = tmp_path / "results.csv"
    artifact.write_text("stage,pr_auc\nS1,0.13\n", encoding="utf-8")

    with track_run(
        "unit", experiment="test-exp", params={"seed": 7}, tracking_dir=tmp_path / "runs",
        strict=True,
    ) as run:
        assert run.active
        run.log_metrics({"pr_auc": 0.1312})
        run.log_artifact(artifact)

    mlflow = _read_runs(tmp_path / "runs")
    exp = mlflow.get_experiment_by_name("test-exp")
    assert exp is not None
    found = mlflow.search_runs(experiment_ids=[exp.experiment_id])
    assert len(found) == 1
    assert found.iloc[0]["params.seed"] == "7"
    assert found.iloc[0]["metrics.pr_auc"] == pytest.approx(0.1312)


def test_git_commit_is_tagged(tmp_path) -> None:
    """A run that cannot be traced to a code state is not reproducible (Sec. 6.2)."""
    with track_run("unit", experiment="tags", tracking_dir=tmp_path, strict=True) as run:
        assert run.active

    mlflow = _read_runs(tmp_path)
    exp = mlflow.get_experiment_by_name("tags")
    found = mlflow.search_runs(experiment_ids=[exp.experiment_id])
    assert "tags.git_commit" in found.columns


def test_disabled_tracking_yields_an_inert_recorder(tmp_path) -> None:
    with track_run("unit", experiment="off", tracking_dir=tmp_path, enabled=False) as run:
        assert not run.active
        run.log_metrics({"pr_auc": 1.0})  # must not raise
    assert not (tmp_path / "0").exists()


def test_a_backend_failure_does_not_propagate(tmp_path, monkeypatch) -> None:
    """Tracking must never fail the analysis it is only observing."""
    import mlflow

    def boom(*_args, **_kwargs):
        raise RuntimeError("tracking store unreachable")

    monkeypatch.setattr(mlflow, "set_experiment", boom)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with track_run("unit", experiment="dead", tracking_dir=tmp_path) as run:
            assert not run.active
            run.log_metrics({"pr_auc": 0.5})

    assert any("MLflow logging disabled" in str(w.message) for w in caught)


def test_strict_mode_surfaces_failures(tmp_path, monkeypatch) -> None:
    import mlflow

    def boom(*_args, **_kwargs):
        raise RuntimeError("tracking store unreachable")

    monkeypatch.setattr(mlflow, "set_experiment", boom)

    with pytest.raises(RuntimeError, match="unreachable"):
        with track_run("unit", experiment="dead", tracking_dir=tmp_path, strict=True):
            pass


def test_non_numeric_metrics_are_skipped_not_fatal(tmp_path) -> None:
    with track_run("unit", experiment="mixed", tracking_dir=tmp_path, strict=True) as run:
        run.log_metrics({"pr_auc": 0.5, "model": "lightgbm", "passes": True})

    mlflow = _read_runs(tmp_path)
    exp = mlflow.get_experiment_by_name("mixed")
    found = mlflow.search_runs(experiment_ids=[exp.experiment_id])
    assert found.iloc[0]["metrics.pr_auc"] == pytest.approx(0.5)
    assert found.iloc[0]["metrics.passes"] == pytest.approx(1.0)
    assert "metrics.model" not in found.columns


def test_nan_metrics_are_dropped(tmp_path) -> None:
    """A NaN in the results should not silently become a logged 'value'."""
    with track_run("unit", experiment="nan", tracking_dir=tmp_path, strict=True) as run:
        run.log_metrics({"good": 1.0, "bad": float("nan"), "worse": float("inf")})

    mlflow = _read_runs(tmp_path)
    exp = mlflow.get_experiment_by_name("nan")
    found = mlflow.search_runs(experiment_ids=[exp.experiment_id])
    assert "metrics.good" in found.columns
    assert "metrics.bad" not in found.columns
    assert "metrics.worse" not in found.columns


def test_missing_artifact_is_reported_not_raised(tmp_path) -> None:
    with track_run("unit", experiment="art", tracking_dir=tmp_path, strict=True) as run:
        run.log_artifact(tmp_path / "does_not_exist.csv")
        assert run.active


@dataclass
class _Report:
    n: int = 100
    n_positive: int = 10
    prevalence: float = 0.10
    threshold: float = 0.5
    point: dict = field(default_factory=lambda: {"pr_auc": 0.2, "roc_auc": 0.7})
    intervals: dict = field(default_factory=dict)

    def as_row(self):
        row = {
            "n": float(self.n),
            "n_positive": float(self.n_positive),
            "prevalence": self.prevalence,
            "threshold": self.threshold,
        }
        row.update(self.point)
        return row


@dataclass
class _Run:
    stage: str
    feature_set: str
    seed: int
    test: _Report


def test_stage_metrics_are_summarised_over_seeds() -> None:
    runs = [
        _Run("S1", "full", 7, _Report(point={"pr_auc": 0.10, "roc_auc": 0.6})),
        _Run("S1", "full", 17, _Report(point={"pr_auc": 0.20, "roc_auc": 0.7})),
    ]
    metrics = stage_run_metrics(runs)

    assert metrics["pr_auc.S1.full.mean"] == pytest.approx(0.15)
    assert metrics["pr_auc.S1.full.sd"] == pytest.approx(0.0707, abs=1e-3)
    # Lift is what compares across stages, so it must be logged (Sec. 9.2).
    assert metrics["pr_auc_lift.S1.full.mean"] == pytest.approx(1.5)


def test_stage_metrics_separate_stages_and_feature_sets() -> None:
    runs = [
        _Run("S1", "full", 7, _Report()),
        _Run("S2", "full", 7, _Report()),
        _Run("S1", "baseline", 7, _Report()),
    ]
    metrics = stage_run_metrics(runs)

    assert "pr_auc.S1.full.mean" in metrics
    assert "pr_auc.S2.full.mean" in metrics
    assert "pr_auc.S1.baseline.mean" in metrics
    # A single seed has no standard deviation to report.
    assert "pr_auc.S1.full.sd" not in metrics


def test_inert_recorder_absorbs_every_call() -> None:
    run = RunRecorder(None)
    run.log_params({"a": 1})
    run.log_metrics({"b": 2.0})
    run.set_tags({"c": "d"})
    assert not run.active


def test_grouped_feature_names_are_sanitised(tmp_path) -> None:
    """Correlated groups are named by joining members with '+', which MLflow rejects."""
    group = "click_velocity+dwell_total_s+events_per_active_minute"

    with track_run("unit", experiment="names", tracking_dir=tmp_path, strict=True) as run:
        run.log_metrics({f"share.S1.{group}": 0.169})
        assert run.active, "an unsanitised name would have disabled the recorder"

    mlflow = _read_runs(tmp_path)
    exp = mlflow.get_experiment_by_name("names")
    found = mlflow.search_runs(experiment_ids=[exp.experiment_id])
    logged = [c for c in found.columns if c.startswith("metrics.share.S1.")]
    assert len(logged) == 1
    assert "+" not in logged[0]
    assert found.iloc[0][logged[0]] == pytest.approx(0.169)


def test_over_long_metric_names_are_truncated(tmp_path) -> None:
    with track_run("unit", experiment="long", tracking_dir=tmp_path, strict=True) as run:
        run.log_metrics({"x" * 400: 1.0})
        assert run.active
