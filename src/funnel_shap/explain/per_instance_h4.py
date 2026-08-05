"""Per-instance cross-paradigm agreement (H4, better powered than the aggregate test).

The pre-registered H4 test correlates two *global* importance rankings over the
four or five concepts the paradigms share. With four items Spearman can only take
a handful of discrete values, so the pre-registered 0.6 threshold is applied to
something that is barely a statistic (amendment A22).

This module asks a different and better-powered question on the same data:
**for a given driver, do the two paradigms agree about which journeys it
mattered for?** For each mapped concept, the tree attribution and the sequence
attribution are correlated *across sessions*, giving one correlation per concept
with hundreds or thousands of observations behind it rather than five.

It is also the stronger reading of convergent validity. Two methods can produce
the same average ranking while disagreeing completely about individual cases;
an explanation that is only right on average is not much use to someone
deciding what to do about a particular visitor.

Both arms explain the **same sessions**, matched by id. Comparing different
sessions would measure sampling noise rather than paradigm agreement.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy.stats import spearmanr

from ..data.journey import StageName
from .sequence_shap import SEQUENCE_TO_TABULAR


@dataclass
class PerInstanceH4:
    stage: StageName
    concept: str
    spearman: float
    pearson: float
    n_sessions: int
    tree_features: tuple[str, ...]


def _concept_matrix(
    shap_values: np.ndarray, feature_names: list[str], members: tuple[str, ...]
) -> np.ndarray | None:
    """Sum the SHAP values of a concept's member features, per instance.

    Summed, not averaged: Shapley values are additive contributions, so a
    concept's contribution to a prediction is the sum of its members'.
    """
    index = {name: i for i, name in enumerate(feature_names)}
    columns = [index[m] for m in members if m in index]
    if not columns:
        return None
    return shap_values[:, columns].sum(axis=1)


def compare_per_instance(
    tree_shap: np.ndarray,
    tree_features: list[str],
    tree_session_ids: list[str],
    sequence_shap: np.ndarray,
    sequence_features: list[str],
    sequence_session_ids: list[str],
    *,
    stage: StageName,
) -> list[PerInstanceH4]:
    """Correlate the two paradigms' per-session attributions, concept by concept."""
    tree_pos = {sid: i for i, sid in enumerate(tree_session_ids)}
    shared = [sid for sid in sequence_session_ids if sid in tree_pos]
    if len(shared) < 30:
        return []

    seq_pos = {sid: i for i, sid in enumerate(sequence_session_ids)}
    tree_rows = np.array([tree_pos[s] for s in shared])
    seq_rows = np.array([seq_pos[s] for s in shared])

    seq_index = {name: i for i, name in enumerate(sequence_features)}
    results: list[PerInstanceH4] = []

    for seq_name, tabular_names in SEQUENCE_TO_TABULAR.items():
        if seq_name not in seq_index or not tabular_names:
            continue
        tree_vector = _concept_matrix(tree_shap, tree_features, tabular_names)
        if tree_vector is None:
            continue

        a = tree_vector[tree_rows]
        b = sequence_shap[seq_rows, seq_index[seq_name]]
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            continue

        rho, _ = spearmanr(a, b)
        results.append(
            PerInstanceH4(
                stage=stage,
                concept=seq_name,
                spearman=float(np.nan_to_num(rho)),
                pearson=float(np.nan_to_num(np.corrcoef(a, b)[0, 1])),
                n_sessions=len(shared),
                tree_features=tabular_names,
            )
        )

    return results


def per_instance_table(results: list[PerInstanceH4], *, threshold: float = 0.6) -> pl.DataFrame:
    rows = [
        {
            "stage": r.stage,
            "concept": r.concept,
            "tree_features": ", ".join(r.tree_features),
            "spearman": r.spearman,
            "pearson": r.pearson,
            "n_sessions": r.n_sessions,
            "passes": bool(r.spearman > threshold),
        }
        for r in results
    ]
    if not rows:
        raise ValueError("no concept could be compared per instance")
    return pl.DataFrame(rows).sort(["stage", "spearman"], descending=[False, True])


def summarise(table: pl.DataFrame, *, threshold: float = 0.6) -> str:
    lines = [
        "Per-instance cross-paradigm agreement (H4, better powered than the aggregate test)",
        "-" * 86,
        f"{'stage':<6}{'concept':<26}{'rho':>8}{'r':>8}{'n':>9}  verdict",
    ]
    for row in table.to_dicts():
        verdict = "agree" if row["spearman"] > threshold else "below threshold"
        lines.append(
            f"{row['stage']:<6}{row['concept']:<26}{row['spearman']:>8.3f}"
            f"{row['pearson']:>8.3f}{row['n_sessions']:>9,}  {verdict}"
        )
    mean = float(table["spearman"].mean())
    lines.append("")
    lines.append(
        f"mean rho {mean:.3f} over {table.height} (stage, concept) pairs, "
        f"each backed by {int(table['n_sessions'].min()):,}+ sessions"
    )
    return "\n".join(lines)
