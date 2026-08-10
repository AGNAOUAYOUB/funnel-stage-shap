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
    #: Session ids of the evaluated rows, in the same order as ``scores``. Each
    #: stage's feature frame carries its own row order, so the cohort is the
    #: same set of customers in a different sequence at every stage. Any paired
    #: comparison must align on these ids; comparing the arrays positionally
    #: would pair one customer's score with another's label.
    session_ids: list[str] = field(repr=False, default=None)


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
    restrict_training_to_cohort: bool = False,
) -> list[CohortRun]:
    """Fit each stage as usual, then evaluate all stages on the shared cohort.

    Training is deliberately unchanged by default: a stage model should be built
    the way the paper builds it, or the comparison would test a different
    artefact. Only the evaluation population is held fixed.

    That default carries a cost worth naming. A model trained where prevalence
    is 0.089 and scored where it is 0.463 is being applied under
    prior-probability shift, so its calibration on the cohort is meaningless and
    its ranking is arguably out of domain. Ranking metrics are invariant to a
    monotone recalibration, so the comparison survives --- but only as a
    comparison of *rankings*. Setting ``restrict_training_to_cohort`` refits each
    stage on cohort members alone, which removes the shift at the cost of a much
    smaller training set; reporting both is what separates the two objections.
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
        session_ids = joined["session_id"].to_list()
        in_cohort = np.array([s in cohort for s in session_ids], dtype=bool)

        train_mask = partition == "train"
        if restrict_training_to_cohort:
            train_mask = train_mask & in_cohort
        # The evaluation set is the shared cohort inside the test partition, so
        # the split discipline is preserved: the test partition still opens once.
        eval_mask = (partition == "test") & in_cohort
        if not eval_mask.any() or y[eval_mask].sum() == 0:
            continue
        if train_mask.sum() < 2 or y[train_mask].sum() == 0:
            raise ValueError(
                f"stage {stage} has no usable training rows inside the cohort; "
                "a silently skipped stage would read as a missing comparison"
            )

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
                    session_ids=[
                        s for s, keep in zip(session_ids, eval_mask, strict=True) if keep
                    ],
                )
            )

    return runs


def bootstrap_cohort(
    runs: list[CohortRun],
    *,
    n_resamples: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> pl.DataFrame:
    """Stratified bootstrap intervals over the cohort, per stage and pairwise.

    The seed standard deviations reported by :func:`cohort_table` measure how
    much the *fit* moves when the random seed changes. They say nothing about
    how much the *estimate* would move on another sample of customers, and on a
    cohort of a few hundred sessions with an AUC near one half the second is far
    larger than the first. Judging a between-stage difference against seed
    dispersion is the same mistake the amendment log records under A18, so the
    cohort comparison is made against resampling intervals instead.

    Resampling is stratified by label and shared across stages: every stage sees
    the same resampled customers in the same draw, which is what makes the
    paired difference interval meaningful. Scores are held fixed --- this is
    uncertainty about the evaluation sample, not about the fit, and the seed
    dispersion already covers the latter.
    """
    if not runs:
        raise ValueError("no cohort runs to resample")

    by_stage: dict[StageName, list[CohortRun]] = {}
    for run in runs:
        by_stage.setdefault(run.stage, []).append(run)

    # Align every stage on session id before anything is compared. The stages
    # hold the same customers in different row orders, so positional pairing
    # would compare one customer's score against another's label.
    if any(run.session_ids is None for run in runs):
        raise ValueError(
            "cohort runs carry no session ids; they predate the alignment fix "
            "and cannot be paired"
        )
    reference = sorted(by_stage[next(iter(by_stage))][0].session_ids)
    order = {sid: i for i, sid in enumerate(reference)}

    def aligned(run: CohortRun) -> tuple[np.ndarray, np.ndarray]:
        if sorted(run.session_ids) != reference:
            raise ValueError(
                f"stage {run.stage} was evaluated on a different set of sessions; "
                "the cohort is not shared and a paired interval would be meaningless"
            )
        index = np.array([order[s] for s in run.session_ids])
        scores = np.empty_like(run.scores)
        labels = np.empty_like(run.y_true)
        scores[index] = run.scores
        labels[index] = run.y_true
        return scores, labels

    y = None
    # Seed-averaged scores per stage: the question here is whether the stages
    # differ, not whether the seeds do.
    mean_scores: dict[StageName, np.ndarray] = {}
    for stage, stage_runs in by_stage.items():
        per_seed = []
        for run in stage_runs:
            scores, labels = aligned(run)
            if y is None:
                y = labels
            elif not np.array_equal(labels, y):
                raise ValueError(
                    f"stage {stage} carries different labels for the same sessions"
                )
            per_seed.append(scores)
        mean_scores[stage] = np.mean(per_seed, axis=0)

    rng = np.random.default_rng(seed)
    positives = np.flatnonzero(y == 1)
    negatives = np.flatnonzero(y == 0)

    stages = [s for s in MODELLING_STAGES if s in by_stage]
    draws: dict[str, list[float]] = {f"lift.{s}": [] for s in stages}
    draws.update({f"roc_auc.{s}": [] for s in stages})
    for i, a in enumerate(stages):
        for b in stages[i + 1 :]:
            draws[f"lift.{b}-{a}"] = []
            draws[f"roc_auc.{b}-{a}"] = []

    for _ in range(n_resamples):
        idx = np.concatenate(
            [
                rng.choice(positives, size=positives.size, replace=True),
                rng.choice(negatives, size=negatives.size, replace=True),
            ]
        )
        y_b = y[idx]
        prevalence = float(y_b.mean())
        lift, roc = {}, {}
        for stage in stages:
            s_b = mean_scores[stage][idx]
            lift[stage] = float(average_precision_score(y_b, s_b)) / prevalence
            roc[stage] = float(roc_auc_score(y_b, s_b))
            draws[f"lift.{stage}"].append(lift[stage])
            draws[f"roc_auc.{stage}"].append(roc[stage])
        for i, a in enumerate(stages):
            for b in stages[i + 1 :]:
                draws[f"lift.{b}-{a}"].append(lift[b] - lift[a])
                draws[f"roc_auc.{b}-{a}"].append(roc[b] - roc[a])

    rows = []
    for name, values in draws.items():
        metric, target = name.split(".", 1)
        arr = np.asarray(values)
        is_difference = "-" in target
        rows.append(
            {
                "metric": metric,
                "target": target,
                "kind": "difference" if is_difference else "level",
                "mean": round(float(arr.mean()), 4),
                "lo": round(float(np.quantile(arr, alpha / 2)), 4),
                "hi": round(float(np.quantile(arr, 1 - alpha / 2)), 4),
                # For a difference, whether the interval clears zero; for a
                # level, whether it clears the no-skill value.
                "excludes_null": bool(
                    (np.quantile(arr, alpha / 2) > 0 or np.quantile(arr, 1 - alpha / 2) < 0)
                    if is_difference
                    else (
                        np.quantile(arr, alpha / 2) > (1.0 if metric == "lift" else 0.5)
                        or np.quantile(arr, 1 - alpha / 2)
                        < (1.0 if metric == "lift" else 0.5)
                    )
                ),
                "n_resamples": n_resamples,
            }
        )
    return pl.DataFrame(rows)


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
