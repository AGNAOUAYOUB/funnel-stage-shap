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
from ..evaluate.calibration import correct_prior_shift
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
    #: Sec. 9.6 requires pre/post calibration reporting.
    test_uncalibrated: ClassificationReport | None = None
    test_prior_corrected: ClassificationReport | None = None
    prior_estimated: float | None = None
    prior_calibration: float | None = None


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
    calibration_fraction: float = 0.20,
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

    # Sec. 9.6 asks for a dedicated *calibration split*. Carving it from the
    # training period rather than reusing validation matters here: validation is
    # November, whose 25.4% prevalence is double December's, and calibrating on
    # it left every model over-predicting by that same ratio (amendment A12).
    # The calibration slice is stratified across the whole training period, so
    # its base rate tracks the training months rather than one extreme one.
    rng = np.random.default_rng(0)
    train_idx = np.flatnonzero(masks["train"])
    calib_idx = _stratified_sample(y[train_idx], rng, fraction=calibration_fraction)
    calibration_rows = train_idx[calib_idx]
    fit_rows = np.setdiff1d(train_idx, calibration_rows, assume_unique=False)

    runs: list[BaselineRun] = []

    for model_type in models:
        for seed in seeds:
            set_global_seed(seed)

            pipeline = build_pipeline(
                model_type, numeric, categorical, seed=seed, imbalance=imbalance
            )
            pipeline.fit(X.iloc[fit_rows], y[fit_rows])

            raw_test = pipeline.predict_proba(X[masks["test"]])[:, 1]

            if calibrate != "none":
                calibrated = CalibratedClassifierCV(pipeline, method=calibrate, cv="prefit")
                calibrated.fit(X.iloc[calibration_rows], y[calibration_rows])
                scorer = calibrated
            else:
                scorer = pipeline

            val_scores = scorer.predict_proba(X[masks["val"]])[:, 1]
            # Fixed here, on validation, and never revisited.
            threshold = select_threshold(y[masks["val"]], val_scores, objective="f1")

            test_scores = scorer.predict_proba(X[masks["test"]])[:, 1]

            # Residual shift, estimated from the test *scores* only -- never the
            # test labels, which would defeat the single-look rule.
            prior_calibration = float(y[calibration_rows].mean())
            corrected, shift = correct_prior_shift(
                test_scores, prior_source=prior_calibration
            )

            def report(scores, *, intervals=True, _seed=seed, _threshold=threshold):
                return evaluate_predictions(
                    y[masks["test"]], scores, threshold=_threshold,
                    n_resamples=n_resamples, seed=_seed, with_intervals=intervals,
                )

            runs.append(
                BaselineRun(
                    model=model_type,
                    seed=seed,
                    threshold=threshold,
                    val=evaluate_predictions(
                        y[masks["val"]], val_scores, threshold=threshold,
                        n_resamples=n_resamples, seed=seed, with_intervals=False,
                    ),
                    test=report(test_scores),
                    test_scores=test_scores,
                    y_test=y[masks["test"]],
                    test_uncalibrated=report(raw_test, intervals=False),
                    test_prior_corrected=report(corrected, intervals=False),
                    prior_estimated=shift.prior_estimated,
                    prior_calibration=prior_calibration,
                )
            )

    _ = split_report
    return runs


def _stratified_sample(
    y: np.ndarray, rng: np.random.Generator, *, fraction: float
) -> np.ndarray:
    """Positional indices of a label-stratified sample, so the slice keeps the base rate."""
    picked = []
    for label in (0, 1):
        idx = np.flatnonzero(y == label)
        n = max(1, int(round(len(idx) * fraction)))
        picked.append(rng.choice(idx, size=n, replace=False))
    return np.sort(np.concatenate(picked))


def results_table(runs: list[BaselineRun]) -> pl.DataFrame:
    """Per-(model, seed) test metrics, long form."""
    rows = []
    for run in runs:
        row = {"model": run.model, "seed": run.seed, "threshold": run.threshold}
        row |= dict(run.test.point)
        for name, ci in run.test.intervals.items():
            row[f"{name}_ci_low"] = ci.ci_low
            row[f"{name}_ci_high"] = ci.ci_high
        # Sec. 9.6: pre/post calibration, plus the prior-shift diagnostics.
        if run.test_uncalibrated is not None:
            row["ece_uncalibrated"] = run.test_uncalibrated.point["ece"]
            row["brier_uncalibrated"] = run.test_uncalibrated.point["brier"]
        if run.test_prior_corrected is not None:
            row["ece_prior_corrected"] = run.test_prior_corrected.point["ece"]
            row["brier_prior_corrected"] = run.test_prior_corrected.point["brier"]
            row["pr_auc_prior_corrected"] = run.test_prior_corrected.point["pr_auc"]
        row["prior_calibration"] = run.prior_calibration
        row["prior_estimated"] = run.prior_estimated
        row["prevalence_test"] = run.test.prevalence
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
