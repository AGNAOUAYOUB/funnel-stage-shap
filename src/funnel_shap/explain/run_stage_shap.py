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
            attribution=attribution, groups=groups, n_train=int(train_mask.sum())
        )

    return out


def migration_table(
    explanations: dict[StageName, StageExplanation], *, grouped: bool = True
) -> pl.DataFrame:
    """The RQ2 trajectory table across stages.

    Correlation groups are taken from the **earliest** stage present and applied
    to all of them. Re-clustering per stage would let a group's membership
    change between stages, so a trajectory would silently compare different
    bundles of features at each point and any movement would be uninterpretable.
    """
    if not explanations:
        raise ValueError("no explanations supplied")

    attributions = {stage: e.attribution for stage, e in explanations.items()}
    if not grouped:
        return attribution_migration(attributions)

    first_stage = next(iter(explanations))
    groups = explanations[first_stage].groups
    return attribution_migration(attributions, groups=groups)


def stage_importance_table(explanations: dict[StageName, StageExplanation]) -> pl.DataFrame:
    """Per-stage grouped importance ranking (Sec. 11.1)."""
    frames = [
        grouped_attribution(e.attribution, e.groups) for e in explanations.values()
    ]
    return pl.concat(frames, how="diagonal")
