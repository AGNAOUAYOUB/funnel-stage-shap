"""Wire the stage models to Layer 1 and produce the migration table (Sec. 11.1).

Fits one model per stage, explains it with interventional TreeSHAP against a
training-drawn background, and assembles the cross-stage attribution
trajectories that answer RQ2.

**Explanations are computed on the validation split, not the test split.** The
test partition opens exactly once, for the final predictive evaluation
(Sec. 6.2). Nothing about the explanation layer requires test data — the
attributions describe the *model*, and validation rows are drawn from the same
distribution — so spending the single look on SHAP plots would be a poor trade.
If a reviewer asks for test-set attributions, they can be produced in the same
pass that opens the partition, not before.

**The background set comes from training rows only.** Interventional TreeSHAP
marginalises over a background sample; drawing it from the rows being explained
would make the explanation conditional on the evaluation data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from ..data.journey import MODELLING_STAGES, StageName
from ..data.splits import load_split
from ..features.dictionary import feature_names
from ..models.baselines import build_pipeline
from ..seeds import set_global_seed
from .stage_shap import (
    DEFAULT_BACKGROUND_SIZE,
    DEFAULT_CORRELATION_THRESHOLD,
    StageAttribution,
    attribution_migration,
    correlation_groups,
    grouped_attribution,
)


@dataclass
class StageExplanation:
    attribution: StageAttribution
    groups: dict[str, list[str]]
    n_train: int
    #: The exact rows the attributions were computed on, in the same order, and
    #: the background they were computed against. Carried here rather than
    #: re-derived downstream: the explained set is a *random subsample* of
    #: validation when it exceeds max_explain, so any attempt to reconstruct it
    #: by re-slicing pairs each attribution with the wrong instance. That
    #: silently drove the Layer 3 faithfulness correlation to zero before this
    #: field existed.
    explained_matrix: np.ndarray = None
    background_matrix: np.ndarray = None
    estimator: object = None


def explain_stages(
    features_by_stage: dict[StageName, pl.DataFrame],
    *,
    suffix: str,
    protocol: str = "temporal",
    model_type: str = "lightgbm",
    seed: int = 42,
    imbalance: str = "class_weight",
    background_size: int = DEFAULT_BACKGROUND_SIZE,
    max_explain: int = 5000,
    correlation_threshold: float = DEFAULT_CORRELATION_THRESHOLD,
    params: dict | None = None,
) -> dict[StageName, StageExplanation]:
    """Fit and explain one model per stage."""
    split = load_split(suffix, protocol)
    rng = np.random.default_rng(seed)
    out: dict[StageName, StageExplanation] = {}

    for stage in MODELLING_STAGES:
        if stage not in features_by_stage:
            continue

        joined = features_by_stage[stage].join(
            split.select(["session_id", "partition"]), on="session_id", how="inner"
        )
        y = joined["label"].cast(pl.Int8).to_numpy()
        partition = joined["partition"].to_numpy()
        columns = feature_names(stage)
        X = joined.select(columns)

        train_mask = partition == "train"
        explain_mask = partition == "val"
        if not train_mask.any() or not explain_mask.any():
            continue

        set_global_seed(seed)
        pipeline = build_pipeline(
            model_type, columns, [], seed=seed, params=params, imbalance=imbalance
        )
        pipeline.fit(X.filter(train_mask).to_pandas(), y[train_mask])

        # SHAP needs the bare estimator and a matrix already in model space.
        preprocess = pipeline.named_steps["preprocess"]
        estimator = pipeline.named_steps["model"]
        model_space_names = list(preprocess.get_feature_names_out())

        background = preprocess.transform(X.filter(train_mask).to_pandas())
        explain_rows = X.filter(explain_mask).to_pandas()
        if len(explain_rows) > max_explain:
            keep = rng.choice(len(explain_rows), size=max_explain, replace=False)
            explain_rows = explain_rows.iloc[np.sort(keep)]
        explain_matrix = preprocess.transform(explain_rows)

        from .stage_shap import stage_tree_shap

        attribution = stage_tree_shap(
            estimator,
            background=background,
            explain=explain_matrix,
            stage=stage,
            feature_names=model_space_names,
            background_size=background_size,
            seed=seed,
        )

        groups = correlation_groups(
            pl.DataFrame(explain_matrix, schema=model_space_names),
            threshold=correlation_threshold,
        )
        out[stage] = StageExplanation(
            attribution=attribution,
            groups=groups,
            n_train=int(train_mask.sum()),
            explained_matrix=explain_matrix,
            background_matrix=background,
            estimator=estimator,
        )

    return out


def reference_groups(explanations: dict[StageName, StageExplanation]) -> dict[str, list[str]]:
    """The single grouping used at every stage, taken from the earliest one.

    A trajectory is only interpretable if the thing being tracked is the same at
    each point, so the grouping must be fixed across stages. Which stage should
    supply it is not arbitrary: the earliest one is the **coarsest**, because
    early prefixes are short and collapse many features onto each other. At S1 a
    two-event prefix makes seven engagement and temporal features numerically
    identical (amendment A8's residual note), so their individual attributions
    there are arbitrary splits of one quantity.

    Taking the finest grouping instead — from a later stage, where those
    features separate — would restore that arbitrary split at S1 and put it
    straight onto the figure. Taking the coarsest keeps every group meaningful
    everywhere, at the cost of merging features at later stages that are
    distinguishable there. That cost is real and belongs in the caption: within
    a merged group, later-stage migration is invisible.
    """
    if not explanations:
        raise ValueError("no explanations supplied")
    return explanations[next(iter(explanations))].groups


def migration_table(
    explanations: dict[StageName, StageExplanation], *, grouped: bool = True
) -> pl.DataFrame:
    """The RQ2 trajectory table across stages."""
    if not explanations:
        raise ValueError("no explanations supplied")

    attributions = {stage: e.attribution for stage, e in explanations.items()}
    if not grouped:
        return attribution_migration(attributions)
    return attribution_migration(attributions, groups=reference_groups(explanations))


def stage_importance_table(explanations: dict[StageName, StageExplanation]) -> pl.DataFrame:
    """Per-stage grouped importance ranking (Sec. 11.1).

    Uses the same reference grouping as :func:`migration_table`. Grouping each
    stage by its own correlation structure would make the per-stage rankings
    incomparable with the trajectory they are supposed to summarise -- and with
    each other, which is the error this whole layer exists to avoid.
    """
    groups = reference_groups(explanations)
    frames = [grouped_attribution(e.attribution, groups) for e in explanations.values()]
    return pl.concat(frames, how="diagonal")
