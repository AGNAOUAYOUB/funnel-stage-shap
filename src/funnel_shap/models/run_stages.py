"""Dataset B stage models (protocol run-sheet step 8, Sec. 9.2, 10).

Trains one model per funnel stage on that stage's prefix features, yielding the
prediction-improvement curve (RQ1/H1) and the fitted models that Layer 1 of the
explanation protocol will attribute over (Sec. 11.1).

Two properties of this design have to travel with every number it produces:

**The curve is conditional, not marginal.** The Sk model is fitted on sessions
that reached Sk, because a session with no cart has no S3 prefix. Reach and
prevalence differ sharply by stage (S1 97.9%/8.8%, S2 48.0%/6.8%, S3 9.3%/52.1%),
so a PR-AUC difference between stages is partly a difference in population.
`improvement_curve` therefore reports N, reach and prevalence in the same table
as PR-AUC; they are not optional context.

**PR-AUC is not comparable across stages without its baseline.** Chance-level
PR-AUC equals the prevalence, so S3's 52.1% prevalence gives it a floor of 0.52
where S1's is 0.088. Comparing raw PR-AUC across stages would say S3 is
dramatically better when it may be no better than its own base rate. The table
carries `pr_auc_lift` -- PR-AUC divided by prevalence -- which is the comparable
quantity, and H1 should be evaluated on it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl
from sklearn.calibration import CalibratedClassifierCV

from ..data.journey import MODELLING_STAGES, StageName
from ..data.splits import load_split
from ..evaluate.calibration import correct_prior_shift
from ..evaluate.metrics import ClassificationReport, evaluate_predictions, select_threshold
from ..features.dictionary import ALL_FEATURE_SETS, FEATURE_DICTIONARY, feature_names
from ..seeds import SEEDS, set_global_seed
from .baselines import build_pipeline
from .run_baselines import _stratified_sample


@dataclass
class StageRun:
    stage: StageName
    model: str
    seed: int
    feature_set: str
    threshold: float
    n_train: int
    test: ClassificationReport
    test_scores: np.ndarray = field(repr=False)
    y_test: np.ndarray = field(repr=False)
    prior_estimated: float | None = None
    fitted: object | None = field(default=None, repr=False)


def _split_features(
    features: pl.DataFrame, split: pl.DataFrame
) -> tuple[pl.DataFrame, np.ndarray, np.ndarray]:
    joined = features.join(
        split.select(["session_id", "partition"]), on="session_id", how="inner"
    )
    y = joined["label"].cast(pl.Int8).to_numpy()
    partition = joined["partition"].to_numpy()
    return joined, y, partition


def run_stage_models(
    features_by_stage: dict[StageName, pl.DataFrame],
    *,
    suffix: str,
    protocol: str = "temporal",
    models: tuple[str, ...] = ("lightgbm",),
    seeds: tuple[int, ...] = SEEDS,
    feature_sets: tuple[str, ...] = ("full",),
    imbalance: str = "class_weight",
    n_resamples: int = 2000,
    keep_fitted: bool = False,
    calibrate: str = "isotonic",
    calibration_fraction: float = 0.20,
    params_by_stage: dict[StageName, dict] | None = None,
) -> list[StageRun]:
    """Fit and evaluate a model per (stage, model, seed, feature set).

    `params_by_stage` carries tuned hyperparameters (Sec. 9.5) into the
    estimator; it only makes sense with a single model type, since the params
    were searched for that model.
    """
    if params_by_stage and len(models) != 1:
        raise ValueError("params_by_stage requires exactly one model type")
    split = load_split(suffix, protocol)
    ladder = ALL_FEATURE_SETS
    unknown = set(feature_sets) - set(ladder)
    if unknown:
        raise ValueError(f"unknown feature sets {sorted(unknown)}; expected {sorted(ladder)}")
    runs: list[StageRun] = []

    for stage in MODELLING_STAGES:
        if stage not in features_by_stage:
            continue

        joined, y, partition = _split_features(features_by_stage[stage], split)
        masks = {name: partition == name for name in ("train", "val", "test")}
        if not masks["test"].any() or y[masks["test"]].sum() == 0:
            continue

        # Calibration slice, drawn once per stage so every feature set and seed
        # is fitted and calibrated on identical rows -- otherwise an ablation
        # difference would partly reflect a different training sample.
        rng = np.random.default_rng(0)
        train_idx = np.flatnonzero(masks["train"])
        calibration_rows = train_idx[
            _stratified_sample(y[train_idx], rng, fraction=calibration_fraction)
        ]
        fit_rows = np.setdiff1d(train_idx, calibration_rows)

        available = feature_names(stage)

        for feature_set in feature_sets:
            groups = set(ladder[feature_set])
            columns = [
                spec.name
                for spec in FEATURE_DICTIONARY
                if spec.name in available and spec.ablation_group in groups
            ]
            if not columns:
                continue

            X = joined.select(columns).to_pandas()

            for model_type in models:
                for seed in seeds:
                    set_global_seed(seed)

                    pipeline = build_pipeline(
                        model_type, columns, [], seed=seed, imbalance=imbalance,
                        params=(params_by_stage or {}).get(stage),
                    )
                    pipeline.fit(X.iloc[fit_rows], y[fit_rows])

                    # Same correction as A18 on Dataset A: calibrate on a slice
                    # of the training period, not on validation. Uncalibrated
                    # stage models ran to ECE 0.33, which makes their
                    # probabilities unusable for the RQ4 intervention argument
                    # even though their rankings are fine.
                    if calibrate != "none":
                        scorer = CalibratedClassifierCV(pipeline, method=calibrate, cv="prefit")
                        scorer.fit(X.iloc[calibration_rows], y[calibration_rows])
                    else:
                        scorer = pipeline

                    val_scores = scorer.predict_proba(X[masks["val"]])[:, 1]
                    threshold = select_threshold(y[masks["val"]], val_scores, objective="f1")

                    test_scores = scorer.predict_proba(X[masks["test"]])[:, 1]
                    _, shift = correct_prior_shift(
                        test_scores, prior_source=float(y[calibration_rows].mean())
                    )

                    # Bootstrap CIs only on the headline feature set. The
                    # ablation ladder is reported as mean +/- std across seeds
                    # (Sec. 10), so per-rung intervals add cost, not evidence.
                    runs.append(
                        StageRun(
                            stage=stage,
                            model=model_type,
                            seed=seed,
                            feature_set=feature_set,
                            threshold=threshold,
                            n_train=int(masks["train"].sum()),
                            test=evaluate_predictions(
                                y[masks["test"]], test_scores, threshold=threshold,
                                n_resamples=n_resamples, seed=seed,
                                with_intervals=feature_set == "full",
                            ),
                            test_scores=test_scores,
                            y_test=y[masks["test"]],
                            prior_estimated=shift.prior_estimated,
                            fitted=pipeline if keep_fitted else None,
                        )
                    )

    return runs


def stage_results_table(runs: list[StageRun]) -> pl.DataFrame:
    rows = []
    for run in runs:
        row = {
            "stage": run.stage,
            "model": run.model,
            "feature_set": run.feature_set,
            "seed": run.seed,
            "n_train": run.n_train,
            "n_test": run.test.n,
            "prevalence": run.test.prevalence,
            "threshold": run.threshold,
        }
        row |= dict(run.test.point)
        # Chance-level PR-AUC is the prevalence, so the lift is what compares
        # across stages with very different base rates.
        row["pr_auc_lift"] = run.test.point["pr_auc"] / run.test.prevalence
        for name, ci in run.test.intervals.items():
            row[f"{name}_ci_low"] = ci.ci_low
            row[f"{name}_ci_high"] = ci.ci_high
        rows.append(row)
    return pl.DataFrame(rows)


def improvement_curve(runs: list[StageRun], *, feature_set: str = "full") -> pl.DataFrame:
    """The RQ1/H1 curve, with the context needed to read it honestly."""
    table = stage_results_table(runs).filter(pl.col("feature_set") == feature_set)
    order = {stage: i for i, stage in enumerate(MODELLING_STAGES)}
    return (
        table.group_by("stage")
        .agg(
            pl.first("n_test").alias("n_test"),
            pl.first("prevalence").alias("prevalence"),
            pl.col("pr_auc").mean().alias("pr_auc_mean"),
            pl.col("pr_auc").std().alias("pr_auc_std"),
            pl.col("pr_gain").mean().alias("pr_gain_mean"),
            pl.col("pr_gain").std().alias("pr_gain_std"),
            pl.col("pr_auc_lift").mean().alias("pr_auc_lift_mean"),
            pl.col("roc_auc").mean().alias("roc_auc_mean"),
            pl.col("roc_auc").std().alias("roc_auc_std"),
            pl.col("ece").mean().alias("ece_mean"),
            pl.len().alias("n_seeds"),
        )
        .with_columns(pl.col("stage").replace_strict(order).alias("_o"))
        .sort("_o")
        .drop("_o")
    )


def ablation_table(runs: list[StageRun]) -> pl.DataFrame:
    """Marginal PR-AUC contribution of each feature family (Sec. 10, H3)."""
    table = stage_results_table(runs)
    return (
        table.group_by(["stage", "feature_set"])
        .agg(
            pl.col("pr_auc").mean().alias("pr_auc_mean"),
            pl.col("pr_auc").std().alias("pr_auc_std"),
            pl.len().alias("n_seeds"),
        )
        .sort(["stage", "pr_auc_mean"])
    )
