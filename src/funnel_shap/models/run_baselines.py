"""Dataset A baseline runner (protocol run-sheet step 7, Sec. 9, 10).

The static benchmark: five model families (Sec. 9.1) x five seeds (Appendix B),
fitted on the month-ordered split (amendment A12), calibrated on validation
(Sec. 9.6), with the operating threshold selected on validation and *fixed*
before test (Sec. 10).

The test partition is read exactly once per model, at the end. Nothing in this
module inspects test scores to make a decision -- calibration, threshold, and
model selection all happen on validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.calibration import CalibratedClassifierCV

from ..data.dataset_a import prepare_dataset_a
from ..data.schema import DATASET_A_CATEGORICAL, DATASET_A_NUMERIC
from ..data.splits import dataset_a_split
from ..evaluate.metrics import ClassificationReport, evaluate_predictions, select_threshold
from ..seeds import SEEDS, set_global_seed
from .baselines import BASELINE_MODELS, build_pipeline


@dataclass
class BaselineRun:
    model: str
    seed: int
    threshold: float
    val: ClassificationReport
    test: ClassificationReport
    test_scores: np.ndarray = field(repr=False)
    y_test: np.ndarray = field(repr=False)


def _to_pandas(frame: pl.DataFrame):
    return frame.to_pandas()


def run_dataset_a_baselines(
    frame: pl.DataFrame,
    *,
    models: tuple[str, ...] = BASELINE_MODELS,
    seeds: tuple[int, ...] = SEEDS,
    imbalance: str = "class_weight",
    calibrate: str = "isotonic",
    n_resamples: int = 2000,
) -> list[BaselineRun]:
    """Fit, calibrate, threshold and evaluate every (model, seed) pair.

    Calibration uses ``cv='prefit'``: the base pipeline is fitted on train, then
    the calibrator is fitted on validation. Refitting the calibrator on training
    predictions would calibrate against scores the model has already memorised
    and report a flatteringly low Brier score.
    """
    assigned, split_report = dataset_a_split(frame)

    features, label = prepare_dataset_a(frame)
    features = features.with_row_index("session_id").with_columns(
        pl.col("session_id").cast(pl.Utf8)
    )
    features = features.join(
        assigned.select(["session_id", "partition"]), on="session_id", how="inner"
    )

    y = label.to_numpy()
    order = features["session_id"].cast(pl.Int64).to_numpy()
    y = y[order]

    partition = features["partition"].to_numpy()
    X = _to_pandas(features.drop(["session_id", "partition"]))

    numeric = list(DATASET_A_NUMERIC)
    categorical = [c for c in DATASET_A_CATEGORICAL if c in X.columns]

    masks = {name: partition == name for name in ("train", "val", "test")}
    runs: list[BaselineRun] = []

    for model_type in models:
        for seed in seeds:
            set_global_seed(seed)

            pipeline = build_pipeline(
                model_type, numeric, categorical, seed=seed, imbalance=imbalance
            )
            pipeline.fit(X[masks["train"]], y[masks["train"]])

            if calibrate != "none":
                calibrated = CalibratedClassifierCV(pipeline, method=calibrate, cv="prefit")
                calibrated.fit(X[masks["val"]], y[masks["val"]])
                scorer = calibrated
            else:
                scorer = pipeline

            val_scores = scorer.predict_proba(X[masks["val"]])[:, 1]
            # Fixed here, on validation, and never revisited.
            threshold = select_threshold(y[masks["val"]], val_scores, objective="f1")

            test_scores = scorer.predict_proba(X[masks["test"]])[:, 1]

            runs.append(
                BaselineRun(
                    model=model_type,
                    seed=seed,
                    threshold=threshold,
                    val=evaluate_predictions(
                        y[masks["val"]], val_scores, threshold=threshold,
                        n_resamples=n_resamples, seed=seed, with_intervals=False,
                    ),
                    test=evaluate_predictions(
                        y[masks["test"]], test_scores, threshold=threshold,
                        n_resamples=n_resamples, seed=seed,
                    ),
                    test_scores=test_scores,
                    y_test=y[masks["test"]],
                )
            )

    _ = split_report
    return runs


def results_table(runs: list[BaselineRun]) -> pl.DataFrame:
    """Per-(model, seed) test metrics, long form."""
    rows = []
    for run in runs:
        row = {"model": run.model, "seed": run.seed, "threshold": run.threshold}
        row |= {k: v for k, v in run.test.point.items()}
        for name, ci in run.test.intervals.items():
            row[f"{name}_ci_low"] = ci.ci_low
            row[f"{name}_ci_high"] = ci.ci_high
        rows.append(row)
    return pl.DataFrame(rows)


def summary_table(runs: list[BaselineRun]) -> pl.DataFrame:
    """Mean +/- std across seeds per model (Sec. 6.2: never single runs)."""
    table = results_table(runs)
    metrics = [
        c
        for c in table.columns
        if c not in ("model", "seed", "threshold") and not c.endswith(("_ci_low", "_ci_high"))
    ]
    return (
        table.group_by("model")
        .agg(
            [pl.len().alias("n_seeds")]
            + [pl.col(m).mean().alias(f"{m}_mean") for m in metrics]
            + [pl.col(m).std().alias(f"{m}_std") for m in metrics]
        )
        .sort("pr_auc_mean", descending=True)
    )


def write_results(runs: list[BaselineRun], directory: Path, *, prefix: str = "dataset_a") -> None:
    directory.mkdir(parents=True, exist_ok=True)
    results_table(runs).write_csv(directory / f"{prefix}_baselines_per_seed.csv")
    summary_table(runs).write_csv(directory / f"{prefix}_baselines_summary.csv")
