"""MLflow run logging (protocol Sec. 6.2).

Sec. 6.2 requires that every run producing a reported number be logged with
its parameters, metrics and output artefacts, so a reviewer can reconstruct
which configuration produced which table. This module provides that logging
as a thin wrapper over MLflow's file store, writing to `experiments/mlruns/`
with no server to stand up.

**Tracking never fails a run.** A logging backend that can crash the analysis
is worse than no logging: the numbers are the deliverable, the audit trail is
support for them. Every MLflow call is therefore wrapped, and a failure
degrades to a single warning and a no-op recorder rather than an exception.
`strict=True` inverts this for tests, which must be able to see breakage.
"""

from __future__ import annotations

import logging
import re
import warnings
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from . import paths
from .config import current_git_commit

logger = logging.getLogger(__name__)

#: Set once a degradation warning has been emitted, so a long run does not
#: print the same warning per stage/seed.
_WARNED = False


def _warn_once(message: str) -> None:
    global _WARNED
    if not _WARNED:
        warnings.warn(f"MLflow logging disabled: {message}", RuntimeWarning, stacklevel=3)
        _WARNED = True


#: MLflow accepts only alphanumerics, underscore, dash, period, space and slash
#: in metric names. Correlated-feature groups are named by joining their members
#: with "+", so unsanitised group names are rejected by the backend.
_ILLEGAL_IN_NAME = re.compile(r"[^A-Za-z0-9_\-./ ]")


def _metric_name(key: str) -> str:
    """Coerce a metric key into a name the MLflow backend will accept."""
    cleaned = _ILLEGAL_IN_NAME.sub("_", str(key).replace(" ", "_"))
    # MLflow caps names at 250 characters; grouped feature names can exceed it.
    if len(cleaned) > 250:
        cleaned = cleaned[:240] + "_trunc"
    return cleaned


def _scalar(value: Any) -> float | None:
    """Coerce to a float MLflow will accept, or None if it is not a metric."""
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        f = float(value)
        # MLflow stores NaN/inf badly and they are never meaningful metrics.
        return f if f == f and abs(f) != float("inf") else None
    return None


class RunRecorder:
    """Handle for one tracked run. A no-op instance is returned on failure."""

    def __init__(self, client: Any = None, *, strict: bool = False) -> None:
        self._mlflow = client
        self._strict = strict

    @property
    def active(self) -> bool:
        return self._mlflow is not None

    def _guard(self, what: str, fn) -> None:
        if self._mlflow is None:
            return
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - tracking must not fail a run
            if self._strict:
                raise
            logger.warning("MLflow %s failed: %s", what, exc)
            self._mlflow = None
            _warn_once(str(exc))

    def log_params(self, params: Mapping[str, Any]) -> None:
        """Log parameters, stringified and truncated to MLflow's 500-char limit."""
        cleaned = {
            str(k): (str(v)[:500] if v is not None else "")
            for k, v in params.items()
        }
        self._guard("log_params", lambda: self._mlflow.log_params(cleaned))

    def log_metrics(self, metrics: Mapping[str, Any]) -> None:
        """Log the numeric entries of `metrics`; non-numeric entries are skipped."""
        numeric = {}
        for key, value in metrics.items():
            scalar = _scalar(value)
            if scalar is not None:
                numeric[_metric_name(key)] = scalar
        if numeric:
            self._guard("log_metrics", lambda: self._mlflow.log_metrics(numeric))

    def log_artifact(self, path: Path) -> None:
        """Attach an output file (a results table, a figure) to the run."""
        path = Path(path)
        if not path.exists():
            logger.warning("MLflow artifact missing, not logged: %s", path)
            return
        self._guard("log_artifact", lambda: self._mlflow.log_artifact(str(path)))

    def set_tags(self, tags: Mapping[str, Any]) -> None:
        cleaned = {str(k): str(v)[:500] for k, v in tags.items() if v is not None}
        self._guard("set_tags", lambda: self._mlflow.set_tags(cleaned))


@contextmanager
def track_run(
    run_name: str,
    *,
    experiment: str,
    params: Mapping[str, Any] | None = None,
    tracking_dir: Path | None = None,
    enabled: bool = True,
    strict: bool = False,
) -> Iterator[RunRecorder]:
    """Open an MLflow run writing to the local file store.

    Yields a :class:`RunRecorder`; if MLflow is unavailable or misconfigured
    the recorder is inert and the caller proceeds unaffected. The git commit
    is tagged automatically, since a run that cannot be traced to a code
    state is not reproducible in the sense Sec. 6.2 requires.
    """
    if not enabled:
        yield RunRecorder(None, strict=strict)
        return

    directory = Path(tracking_dir) if tracking_dir is not None else paths.MLRUNS
    try:
        import mlflow
    except Exception as exc:  # noqa: BLE001
        if strict:
            raise
        _warn_once(f"import failed ({exc})")
        yield RunRecorder(None, strict=strict)
        return

    try:
        directory.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(directory.resolve().as_uri())
        mlflow.set_experiment(experiment)
        active = mlflow.start_run(run_name=run_name)
    except Exception as exc:  # noqa: BLE001
        if strict:
            raise
        _warn_once(str(exc))
        yield RunRecorder(None, strict=strict)
        return

    recorder = RunRecorder(mlflow, strict=strict)
    try:
        recorder.set_tags({"git_commit": current_git_commit(), "run_name": run_name})
        if params:
            recorder.log_params(params)
        yield recorder
    finally:
        try:
            mlflow.end_run()
        except Exception as exc:  # noqa: BLE001
            if strict:
                raise
            logger.warning("MLflow end_run failed: %s", exc)
        del active


def stage_run_metrics(runs: list) -> dict[str, float]:
    """Flatten StageRun results into `metric.stage.feature_set` MLflow keys.

    Seeds are averaged rather than logged individually: the protocol reports
    mean +/- sd over the frozen seed list, so the per-seed values belong in
    the results CSV (logged as an artefact) and the summary belongs here.
    """
    import statistics

    grouped: dict[tuple[str, str, str], list[float]] = {}
    for run in runs:
        row = run.test.as_row()
        row["pr_auc_lift"] = row["pr_auc"] / row["prevalence"] if row["prevalence"] else 0.0
        for metric, value in row.items():
            scalar = _scalar(value)
            if scalar is None:
                continue
            grouped.setdefault((metric, run.stage, run.feature_set), []).append(scalar)

    metrics: dict[str, float] = {}
    for (metric, stage, feature_set), values in grouped.items():
        key = f"{metric}.{stage}.{feature_set}"
        metrics[f"{key}.mean"] = statistics.fmean(values)
        if len(values) > 1:
            metrics[f"{key}.sd"] = statistics.stdev(values)
    return metrics
