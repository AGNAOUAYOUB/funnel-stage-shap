"""Run Layer 3 against the real stage models (protocol Sec. 11.3).

Produces the RQ3 evidence: are the stage-conditioned explanations faithful, are
they stable under perturbation, and do they agree across seeds?

The seed-consistency arm is the one that gates Figure 3. A migration trajectory
computed from a single seed is one draw; if the importance ordering does not
survive reseeding, the figure is noise dressed as a finding.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from ..data.journey import StageName
from ..seeds import SEEDS
from .quality import (
    ConsistencyResult,
    FaithfulnessResult,
    StabilityResult,
    deletion_curve,
    faithfulness_correlation,
    insertion_curve,
    local_lipschitz,
    rank_consistency,
)


@dataclass
class StageQuality:
    stage: StageName
    faithfulness: FaithfulnessResult
    deletion_auc: float
    insertion_auc: float
    deletion_monotone: bool
    stability: StabilityResult | None


def evaluate_stage_quality(
    explanations: dict,
    *,
    seed: int = 42,
    n_faithfulness: int = 120,
    n_subsets: int = 40,
    n_stability: int = 25,
    with_stability: bool = True,
) -> dict[StageName, StageQuality]:
    """Faithfulness, curves and stability for each explained stage."""
    from .stage_shap import stage_tree_shap

    rng = np.random.default_rng(seed)
    out: dict[StageName, StageQuality] = {}

    for stage, explanation in explanations.items():
        if explanation.explained_matrix is None:
            raise ValueError(
                f"stage {stage} carries no explained_matrix; Layer 3 must use the exact "
                "rows the attributions were computed on, not a re-derived slice"
            )

        estimator = explanation.estimator
        background = explanation.background_matrix
        shap_values = explanation.attribution.shap_values
        matrix = explanation.explained_matrix

        # Row i of shap_values corresponds to row i of matrix, guaranteed by
        # construction, so a shared index keeps the pairing exact.
        explained = min(n_faithfulness, shap_values.shape[0])
        pick = rng.choice(shap_values.shape[0], size=explained, replace=False)
        val_matrix = matrix[pick]
        subset_shap = shap_values[pick]

        def predict(m, _est=estimator):
            return _est.predict_proba(m)[:, 1]

        faith = faithfulness_correlation(
            predict, val_matrix, subset_shap, background,
            n_subsets=n_subsets, seed=seed,
        )
        deletion = deletion_curve(
            predict, val_matrix, subset_shap, background, n_steps=10, seed=seed
        )
        insertion = insertion_curve(
            predict, val_matrix, subset_shap, background, n_steps=10, seed=seed
        )

        stability = None
        if with_stability:
            small = val_matrix[: min(n_stability, len(val_matrix))]

            def explain_fn(matrix, _est=estimator, _bg=background, _stage=stage):
                return stage_tree_shap(
                    _est, background=_bg, explain=matrix, stage=_stage,
                    feature_names=list(range(matrix.shape[1])),
                    background_size=100, seed=seed,
                ).shap_values

            stability = local_lipschitz(
                explain_fn, small, n_perturbations=3, noise_scale=0.05, seed=seed
            )

        out[stage] = StageQuality(
            stage=stage,
            faithfulness=faith,
            deletion_auc=deletion.auc,
            insertion_auc=insertion.auc,
            deletion_monotone=deletion.monotone,
            stability=stability,
        )

    return out


def seed_consistency(
    features_by_stage: dict[StageName, pl.DataFrame],
    *,
    suffix: str,
    protocol: str = "temporal",
    model_type: str = "lightgbm",
    seeds: tuple[int, ...] = SEEDS,
    background_size: int = 300,
    max_explain: int = 1000,
) -> dict[StageName, ConsistencyResult]:
    """Spearman agreement of the grouped importance ordering across seeds.

    This is what licenses the migration figure. The comparison is made on the
    *grouped* importances, and on a grouping fixed from the first seed, so that
    a change in the correlation clustering between seeds cannot masquerade as a
    change in importance.
    """
    from .run_stage_shap import explain_stages, reference_groups
    from .stage_shap import grouped_attribution

    per_seed: dict[StageName, list[np.ndarray]] = {}
    reference: dict | None = None

    for seed in seeds:
        explanations = explain_stages(
            features_by_stage, suffix=suffix, protocol=protocol,
            model_type=model_type, seed=seed,
            background_size=background_size, max_explain=max_explain,
        )
        if reference is None:
            reference = reference_groups(explanations)

        for stage, explanation in explanations.items():
            table = grouped_attribution(explanation.attribution, reference).sort("group")
            per_seed.setdefault(stage, []).append(table["mean_abs_shap"].to_numpy())

    return {
        stage: rank_consistency(vectors)
        for stage, vectors in per_seed.items()
        if len(vectors) >= 2
    }
