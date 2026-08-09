"""Fixed-cohort evaluation: information accumulation versus composition.

The stage models are fitted and evaluated on the sessions that reach each
stage, and those populations are not nested: a session that goes straight from
browsing to a cart bypasses the consideration milestone, so
$R_2 \\not\\supseteq R_3$. Every cross-stage comparison in the paper therefore
confounds two things:

1. **Information accumulation.** Later stages observe longer prefixes, so their
   models have strictly more to work with.
2. **Composition.** Later stages describe a population filtered by the
   customer's own prior behaviour, and filtering removes the variance that
   discriminates outcomes.

Naming the confound is not testing it. This module tests it. Every stage model
is scored on one fixed population --- the sessions that reach S3, which by
construction have a prefix at all three stages --- so the evaluated cohort,
its labels and its prevalence are identical across stages. Whatever difference
remains is attributable to the information in the prefix rather than to who is
in the sample.

Reading the result:

* If lift still falls across stages within the fixed cohort, the decline is
  about information and H1's rejection is a clean finding.
* If it flattens, the funnel effect was composition, and the honest conclusion
  is that the apparent decline was a property of who survives to each stage.

Both outcomes are publishable; only one of them is what the paper currently
claims, which is why the test is worth running rather than describing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, roc_auc_score

from ..data.journey import MODELLING_STAGES, StageName
from ..data.splits import load_split
from ..features.dictionary import ALL_FEATURE_SETS, FEATURE_DICTIONARY, feature_names
from ..seeds import SEEDS, set_global_seed
from .baselines import build_pipeline
from .run_baselines import _stratified_sample


@dataclass
class CohortRun:
    stage: StageName
    seed: int
    n_train: int
    n_eval: int
    prevalence: float
    pr_auc: float
    roc_auc: float
    lift: float
    scores: np.ndarray = field(repr=False, default=None)
    y_true: np.ndarray = field(repr=False, default=None)


def common_cohort_ids(features_by_stage: dict[StageName, pl.DataFrame]) -> set[str]:
    """Session ids present at every modelled stage.

    In principle this is the S3 population, since prefixes nest; it is computed
    as an intersection rather than assumed so that any violation of nesting
    shows up as a smaller cohort instead of a silent mismatch.
    """
    ids: set[str] | None = None
    for stage in MODELLING_STAGES:
        if stage not in features_by_stage:
            continue
        stage_ids = set(features_by_stage[stage]["session_id"].to_list())
        ids = stage_ids if ids is None else (ids & stage_ids)
    return ids or set()


def run_common_cohort(
    features_by_stage: dict[StageName, pl.DataFrame],
    *,
    suffix: str,
    protocol: str = "temporal",
    model: str = "lightgbm",
    seeds: tuple[int, ...] = SEEDS,
    feature_set: str = "full",
    imbalance: str = "class_weight",
    calibration_fraction: float = 0.20,
) -> list[CohortRun]:
    """Fit each stage as usual, then evaluate all stages on the shared cohort.

    Training is deliberately unchanged: a stage model should be built the way
    the paper builds it, or the comparison would test a different artefact.
    Only the evaluation population is held fixed.
    """
    split = load_split(suffix, protocol)
    cohort = common_cohort_ids(features_by_stage)
    if not cohort:
        raise ValueError("no session reaches every stage; the cohort is empty")

    groups = set(ALL_FEATURE_SETS[feature_set])
    runs: list[CohortRun] = []

    for stage in MODELLING_STAGES:
        if stage not in features_by_stage:
            continue

        joined = features_by_stage[stage].join(
            split.select(["session_id", "partition"]), on="session_id", how="inner"
        )
        y = joined["label"].cast(pl.Int8).to_numpy()
        partition = joined["partition"].to_numpy()
        in_cohort = np.array(
            [s in cohort for s in joined["session_id"].to_list()], dtype=bool
        )

        train_mask = partition == "train"
        # The evaluation set is the shared cohort inside the test partition, so
        # the split discipline is preserved: the test partition still opens once.
        eval_mask = (partition == "test") & in_cohort
        if not eval_mask.any() or y[eval_mask].sum() == 0:
            continue

        available = feature_names(stage)
        columns = [
            spec.name
            for spec in FEATURE_DICTIONARY
            if spec.name in available and spec.ablation_group in groups
        ]
        X = joined.select(columns).to_pandas()

        rng = np.random.default_rng(0)
        train_idx = np.flatnonzero(train_mask)
        calibration_rows = train_idx[
            _stratified_sample(y[train_idx], rng, fraction=calibration_fraction)
        ]
        fit_rows = np.setdiff1d(train_idx, calibration_rows)

        for seed in seeds:
            set_global_seed(seed)
            pipeline = build_pipeline(model, columns, [], seed=seed, imbalance=imbalance)
            pipeline.fit(X.iloc[fit_rows], y[fit_rows])
            scorer = CalibratedClassifierCV(pipeline, method="isotonic", cv="prefit")
            scorer.fit(X.iloc[calibration_rows], y[calibration_rows])

            scores = scorer.predict_proba(X[eval_mask])[:, 1]
            y_eval = y[eval_mask]
            prevalence = float(y_eval.mean())
            pr_auc = float(average_precision_score(y_eval, scores))

            runs.append(
                CohortRun(
                    stage=stage,
                    seed=seed,
                    n_train=int(train_mask.sum()),
                    n_eval=int(eval_mask.sum()),
                    prevalence=prevalence,
                    pr_auc=pr_auc,
                    roc_auc=float(roc_auc_score(y_eval, scores)),
                    lift=pr_auc / prevalence if prevalence else float("nan"),
                    scores=scores,
                    y_true=y_eval,
                )
            )

    return runs


def cohort_table(runs: list[CohortRun]) -> pl.DataFrame:
    """Mean and dispersion per stage on the shared cohort."""
    frame = pl.DataFrame(
        [
            {
                "stage": r.stage, "seed": r.seed, "n_eval": r.n_eval,
                "prevalence": r.prevalence, "pr_auc": r.pr_auc,
                "roc_auc": r.roc_auc, "lift": r.lift,
            }
            for r in runs
        ]
    )
    order = {stage: i for i, stage in enumerate(MODELLING_STAGES)}
    return (
        frame.group_by("stage")
        .agg(
            pl.first("n_eval"),
            pl.first("prevalence"),
            pl.col("pr_auc").mean().alias("pr_auc_mean"),
            pl.col("pr_auc").std().alias("pr_auc_sd"),
            pl.col("roc_auc").mean().alias("roc_auc_mean"),
            pl.col("roc_auc").std().alias("roc_auc_sd"),
            pl.col("lift").mean().alias("lift_mean"),
            pl.col("lift").std().alias("lift_sd"),
            pl.len().alias("n_seeds"),
        )
        .with_columns(pl.col("stage").replace_strict(order).alias("_o"))
        .sort("_o")
        .drop("_o")
    )
