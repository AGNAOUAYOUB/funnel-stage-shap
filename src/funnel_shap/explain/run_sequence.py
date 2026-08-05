"""Run the sequence arm and the H4 cross-paradigm comparison (Sec. 9.3, 11.2).

Trains one GRU per funnel stage on that stage's prefix — the same information
the tree models see — then attributes it and correlates the resulting feature
ranking against the TreeSHAP ranking. Sec. 11.2's question is whether two
different paradigms tell the same story about the same data; training the GRU on
different data would answer a different question.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy.stats import spearmanr

from ..data.journey import MODELLING_STAGES, StageName, prefix_events
from ..data.splits import load_split
from ..models.sequence import build_sequences, predict_sequences, train_gru
from .sequence_shap import (
    SequenceAttribution,
    align_for_h4,
    permutation_feature_attribution,
    timeshap_available,
    timeshap_feature_attribution,
)


@dataclass
class SequenceStageResult:
    stage: StageName
    val_pr_auc: float
    test_pr_auc: float
    n_train: int
    n_test: int
    attribution: SequenceAttribution
    h4_spearman: float | None = None
    h4_n_concepts: int = 0
    #: Carried so the per-instance H4 comparison can align the two arms by
    #: session and recompute attributions without retraining.
    explained_session_ids: list[str] | None = None
    explained_X: np.ndarray | None = None
    predict_fn: object = None
    feature_names: list[str] | None = None
    train_X: np.ndarray | None = None


def run_sequence_arm(
    sessions: pl.DataFrame,
    cutpoints: pl.DataFrame,
    *,
    suffix: str,
    protocol: str = "temporal",
    seed: int = 42,
    max_len: int = 32,
    epochs: int = 6,
    max_sessions: int = 60_000,
    n_explain: int = 150,
    use_timeshap: bool | None = None,
    tabular_importance: dict[StageName, dict[str, float]] | None = None,
    restrict_to_sessions: dict[StageName, list[str]] | None = None,
) -> dict[StageName, SequenceStageResult]:
    """Train, evaluate and attribute one GRU per stage."""
    from sklearn.metrics import average_precision_score

    split = load_split(suffix, protocol)
    rng = np.random.default_rng(seed)
    if use_timeshap is None:
        use_timeshap = timeshap_available()

    out: dict[StageName, SequenceStageResult] = {}

    for stage in MODELLING_STAGES:
        prefix = prefix_events(sessions.lazy(), cutpoints, stage).collect()
        prefix = prefix.join(
            split.select(["session_id", "partition"]), on="session_id", how="inner"
        )
        if prefix.is_empty():
            continue

        # Cap sessions before padding: the padded tensor is
        # n_sessions x max_len x n_features and dominates memory.
        ids = prefix["session_id"].unique()
        if ids.len() > max_sessions:
            keep = pl.Series("session_id", rng.choice(ids.to_numpy(), max_sessions, replace=False))
            prefix = prefix.filter(pl.col("session_id").is_in(keep))

        batch = build_sequences(prefix, max_len=max_len)
        partition = (
            prefix.group_by("session_id", maintain_order=True)
            .agg(pl.first("partition"))
            .join(pl.DataFrame({"session_id": batch.session_ids}), on="session_id", how="right")
        )["partition"].to_numpy()

        idx = {name: np.flatnonzero(partition == name) for name in ("train", "val", "test")}
        if any(len(v) == 0 for v in idx.values()) or batch.y[idx["test"]].sum() == 0:
            continue

        model, val_score = train_gru(
            batch, idx["train"], idx["val"], seed=seed, epochs=epochs
        )
        test_scores = predict_sequences(model, batch.X[idx["test"]])
        test_score = float(average_precision_score(batch.y[idx["test"]], test_scores))

        # Prefer sessions the tree arm has already explained. Both arms
        # subsampling independently leaves only an incidental overlap -- 49
        # sessions at S1 on the first run -- and a per-instance comparison is
        # only as powerful as the intersection.
        candidates = idx["val"]
        if restrict_to_sessions:
            wanted = set(restrict_to_sessions.get(stage, ()))
            if wanted:
                preferred = [i for i in candidates if batch.session_ids[i] in wanted]
                if len(preferred) >= 30:
                    candidates = np.asarray(preferred)
        explain_idx = candidates[: min(n_explain, len(candidates))]

        def predict(matrix, _m=model):
            return predict_sequences(_m, np.asarray(matrix, dtype=np.float32))

        if use_timeshap:
            attribution = timeshap_feature_attribution(
                predict, batch.X[explain_idx], batch.X[idx["train"]],
                batch.feature_names, n_explain=len(explain_idx), seed=seed,
            )
        else:
            attribution = permutation_feature_attribution(
                predict, batch.X[explain_idx], batch.feature_names, seed=seed
            )

        result = SequenceStageResult(
            stage=stage,
            val_pr_auc=float(val_score),
            test_pr_auc=test_score,
            n_train=len(idx["train"]),
            n_test=len(idx["test"]),
            attribution=attribution,
            explained_session_ids=[batch.session_ids[i] for i in explain_idx],
            explained_X=batch.X[explain_idx],
            predict_fn=predict,
            feature_names=list(batch.feature_names),
            train_X=batch.X[idx["train"]],
        )

        if tabular_importance and stage in tabular_importance:
            seq_vec, tab_vec, labels = align_for_h4(attribution, tabular_importance[stage])
            if len(labels) >= 3:
                rho, _ = spearmanr(seq_vec, tab_vec)
                result.h4_spearman = float(np.nan_to_num(rho))
                result.h4_n_concepts = len(labels)

        out[stage] = result

    return out


def h4_table(results: dict[StageName, SequenceStageResult], *, threshold: float = 0.6) -> pl.DataFrame:
    """The H4 convergent-validity table (Sec. 11.3, threshold from Appendix A)."""
    rows = []
    for stage, r in results.items():
        rows.append(
            {
                "stage": stage,
                "method": r.attribution.method,
                "seq_test_pr_auc": r.test_pr_auc,
                "n_train": r.n_train,
                "n_test": r.n_test,
                "h4_spearman": r.h4_spearman,
                "n_concepts": r.h4_n_concepts,
                "passes": (
                    None if r.h4_spearman is None else bool(r.h4_spearman > threshold)
                ),
                "threshold": threshold,
            }
        )
    return pl.DataFrame(rows)
