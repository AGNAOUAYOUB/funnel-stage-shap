"""Layer 1 — stage-conditioned TreeSHAP and attribution migration (Sec. 11.1).

This is the module the paper's centrepiece figure comes from: per stage, the
global mean-|SHAP| ranking, and across stages, each feature's attribution
trajectory.

Three protocol requirements are enforced rather than assumed.

**Interventional, not path-dependent.** Sec. 11.1 fixes this. Path-dependent
TreeSHAP attributes through the tree's own splits, so a feature that is merely
correlated with a used feature inherits credit from it; the interventional
estimator breaks that dependence against an explicit background sample. The
background set is drawn from *training* rows only, seeded, and its size is
recorded — an explanation computed against a background the model was evaluated
on is not a held-out explanation.

**Correlated features are grouped before they are ranked.** Sec. 11.1 requires
it, and at S1 it binds hard: a two-event prefix makes `prefix_duration_s`,
`inter_event_mean_s`, `last_gap_s` and `dwell_total_s` numerically identical
(amendment A8), so Shapley credit splits arbitrarily among six redundant
columns. Ranking them individually would produce a migration figure whose S1
column is an artefact of that split.

**Trajectories are only comparable where the stages are.** A feature absent at a
stage has no attribution there — not zero. `attribution_migration` records
absence explicitly so the figure can break the line rather than draw it to the
floor, which would read as "this driver stopped mattering".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from ..data.journey import StageName

#: Sec. 11.1 / Appendix A default.
DEFAULT_BACKGROUND_SIZE = 2000

#: Absolute Spearman correlation at or above which two features are treated as
#: one group. Deliberately strict: grouping too eagerly hides real distinctions,
#: grouping too little leaves Shapley credit split across near-duplicates.
DEFAULT_CORRELATION_THRESHOLD = 0.95


@dataclass
class StageAttribution:
    """Per-stage SHAP output (Sec. 11.1)."""

    stage: StageName
    feature_names: list[str]
    #: (n_explained, n_features)
    shap_values: np.ndarray = field(repr=False)
    background_size: int
    n_explained: int
    seed: int

    def global_importance(self) -> pl.DataFrame:
        """Mean |SHAP| ranking, with the mean signed value beside it.

        The signed mean is what makes a *reversal* visible: a feature can hold
        its rank across stages while flipping direction, and Sec. 11.1's RQ2
        asks specifically whether any feature reverses sign.
        """
        mean_abs = np.abs(self.shap_values).mean(axis=0)
        mean_signed = self.shap_values.mean(axis=0)
        total = mean_abs.sum()

        return (
            pl.DataFrame(
                {
                    "stage": [self.stage] * len(self.feature_names),
                    "feature": self.feature_names,
                    "mean_abs_shap": mean_abs,
                    "mean_signed_shap": mean_signed,
                    "share": mean_abs / total if total > 0 else mean_abs,
                }
            )
            .sort("mean_abs_shap", descending=True)
            .with_columns(pl.int_range(1, pl.len() + 1).alias("rank"))
        )


def stage_tree_shap(
    model,
    background: pl.DataFrame | np.ndarray,
    explain: pl.DataFrame | np.ndarray,
    *,
    stage: StageName,
    feature_names: list[str],
    background_size: int = DEFAULT_BACKGROUND_SIZE,
    seed: int = 42,
) -> StageAttribution:
    """Interventional TreeSHAP for one stage's model (Sec. 11.1).

    ``model`` must be the bare tree estimator, not a pipeline: SHAP needs the
    booster, and the feature matrix must already be in model space.
    """
    import shap

    rng = np.random.default_rng(seed)

    bg = background.to_numpy() if isinstance(background, pl.DataFrame) else np.asarray(background)
    xs = explain.to_numpy() if isinstance(explain, pl.DataFrame) else np.asarray(explain)

    if bg.shape[0] > background_size:
        bg = bg[rng.choice(bg.shape[0], size=background_size, replace=False)]

    explainer = shap.TreeExplainer(
        model,
        data=bg,
        feature_perturbation="interventional",
        model_output="raw",
    )
    values = explainer.shap_values(xs, check_additivity=False)

    # Binary classifiers return either (n, f) or a length-2 list / (n, f, 2)
    # array depending on the library; normalise to the positive class.
    if isinstance(values, list):
        values = values[-1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, -1]

    return StageAttribution(
        stage=stage,
        feature_names=list(feature_names),
        shap_values=values,
        background_size=int(bg.shape[0]),
        n_explained=int(xs.shape[0]),
        seed=seed,
    )


def correlation_groups(
    frame: pl.DataFrame,
    *,
    threshold: float = DEFAULT_CORRELATION_THRESHOLD,
) -> dict[str, list[str]]:
    """Cluster near-duplicate features by absolute Spearman correlation (Sec. 11.1).

    Single-linkage over the thresholded correlation graph, so a chain of
    pairwise-redundant features collapses into one group. Group names are the
    member names joined by ``+``, which keeps the figure legend readable and
    makes the grouping visible rather than hidden behind an index.
    """
    from scipy.stats import spearmanr

    columns = frame.columns
    values = frame.to_numpy()

    # Constant columns have undefined correlation; they cannot be redundant
    # with anything in an informative sense, so they stand alone.
    varying = [i for i in range(values.shape[1]) if np.std(values[:, i]) > 0]
    if len(varying) < 2:
        return {c: [c] for c in columns}

    rho, _ = spearmanr(values[:, varying])
    rho = np.atleast_2d(rho)
    adjacency = np.abs(np.nan_to_num(rho)) >= threshold

    parent = list(range(len(varying)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a in range(len(varying)):
        for b in range(a + 1, len(varying)):
            if adjacency[a, b]:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra

    clusters: dict[int, list[str]] = {}
    for idx, col_idx in enumerate(varying):
        clusters.setdefault(find(idx), []).append(columns[col_idx])

    groups = {"+".join(sorted(members)): sorted(members) for members in clusters.values()}
    for i, column in enumerate(columns):
        if i not in varying:
            groups[column] = [column]
    return groups


def grouped_attribution(
    attribution: StageAttribution, groups: dict[str, list[str]]
) -> pl.DataFrame:
    """Sum mean-|SHAP| within correlation groups (Sec. 11.1).

    Summing rather than averaging is the right aggregation: Shapley values are
    additive contributions to a prediction, so a group's contribution is the sum
    of its members'. Averaging would make a group of six redundant features look
    six times less important than the single feature they collectively duplicate.
    """
    index = {name: i for i, name in enumerate(attribution.feature_names)}
    mean_abs = np.abs(attribution.shap_values).mean(axis=0)
    mean_signed = attribution.shap_values.mean(axis=0)

    rows = []
    for group, members in groups.items():
        present = [index[m] for m in members if m in index]
        if not present:
            continue
        rows.append(
            {
                "stage": attribution.stage,
                "group": group,
                "n_members": len(present),
                "mean_abs_shap": float(mean_abs[present].sum()),
                "mean_signed_shap": float(mean_signed[present].sum()),
            }
        )

    frame = pl.DataFrame(rows)
    total = frame["mean_abs_shap"].sum()
    return (
        frame.with_columns(
            (pl.col("mean_abs_shap") / total if total else pl.col("mean_abs_shap")).alias("share")
        )
        .sort("mean_abs_shap", descending=True)
        .with_columns(pl.int_range(1, pl.len() + 1).alias("rank"))
    )


def attribution_migration(
    attributions: dict[StageName, StageAttribution],
    *,
    groups: dict[str, list[str]] | None = None,
) -> pl.DataFrame:
    """The RQ2 trajectory table: how each feature's attribution moves across stages.

    One row per (feature, stage) with the share of total attribution mass, its
    rank, its signed direction, and an explicit ``available`` flag. Downstream,
    ``reversed_sign`` and ``rank_change`` answer RQ2 directly: which drivers
    dominate early versus late, and does anything flip.
    """
    if not attributions:
        raise ValueError("no attributions supplied")

    frames = []
    for stage, attribution in attributions.items():
        if groups is None:
            table = attribution.global_importance().rename({"feature": "group"})
            table = table.with_columns(pl.lit(1).alias("n_members"))
        else:
            table = grouped_attribution(attribution, groups)
        frames.append(table.with_columns(pl.lit(stage).alias("stage")))

    long = pl.concat(frames, how="diagonal")

    stage_order = {stage: i for i, stage in enumerate(attributions)}
    long = long.with_columns(
        pl.col("stage").replace_strict(stage_order).alias("_stage_idx"),
        pl.lit(True).alias("available"),
    ).sort(["group", "_stage_idx"])

    # A feature absent at a stage must not be drawn as zero: the figure should
    # break the line, not run it to the floor, which reads as "stopped
    # mattering" rather than "not measurable here".
    all_groups = long["group"].unique().to_list()
    complete = pl.DataFrame(
        {
            "group": [g for g in all_groups for _ in stage_order],
            "stage": [s for _ in all_groups for s in stage_order],
        }
    ).with_columns(pl.col("stage").replace_strict(stage_order).alias("_stage_idx"))

    merged = complete.join(long, on=["group", "stage", "_stage_idx"], how="left").with_columns(
        pl.col("available").fill_null(False)
    )

    return (
        merged.sort(["group", "_stage_idx"])
        .with_columns(
            (pl.col("share") - pl.col("share").first().over("group")).alias("share_change"),
            (pl.col("rank") - pl.col("rank").first().over("group")).alias("rank_change"),
            (
                pl.col("mean_signed_shap").sign()
                != pl.col("mean_signed_shap").first().over("group").sign()
            ).alias("reversed_sign"),
        )
        .drop("_stage_idx")
    )
