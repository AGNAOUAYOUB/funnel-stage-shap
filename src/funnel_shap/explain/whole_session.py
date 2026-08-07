"""Empirical whole-session comparator for RQ4 (protocol Sec. 2, RQ4).

RQ4 asks whether stage-conditioned attributions identify intervention points a
static whole-session SHAP analysis would miss. Answering it requires an actual
whole-session model to compare against.

**Why the previous comparator did not answer the question.** Earlier revisions
contrasted the stage trajectory against the arithmetic mean of its own stage
shares. That quantity is true by construction --- averaging 3.5/22.4/7.4 must
give 11.1 --- so it demonstrates the arithmetic of averaging rather than
anything about how a whole-session model behaves. A reviewer is entitled to
reject it, and this module replaces it.

**What is computed here.** A LightGBM model is fitted on features aggregated
over the *entire* session, exactly as conventional practice does, and explained
with the same interventional TreeSHAP and the same correlated-feature grouping
used for the stage models. The comparison is therefore between two real models
differing in one respect: whether the prefix constraint was applied.

**This model is deliberately leaky.** Whole-session features see cart and
purchase-adjacent behaviour, which is precisely the practice the paper
criticises. Its predictive score is reported to quantify what that leakage buys
-- it is a measure of the practice, not a baseline we endorse.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from ..data.journey import JourneyConfig, order_events
from ..data.splits import load_split
from ..features.dictionary import FEATURE_DICTIONARY
from ..features.prefix_features import build_stage_features
from ..seeds import set_global_seed


@dataclass
class WholeSessionResult:
    n_train: int
    n_test: int
    prevalence: float
    pr_auc: float
    roc_auc: float
    shares: dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"whole-session model: n_train={self.n_train:,} n_test={self.n_test:,} "
            f"prevalence={self.prevalence:.4f} PR-AUC={self.pr_auc:.4f} "
            f"ROC-AUC={self.roc_auc:.4f}"
        )


def whole_session_events(sessions: pl.DataFrame, cutpoints: pl.DataFrame) -> pl.LazyFrame:
    """Every event of every session, with the label attached.

    This is the deliberate complement of :func:`prefix_events`: no cut-point is
    applied, so the frame contains exactly the information a conventional
    whole-session pipeline would use.
    """
    config = JourneyConfig()
    s = config.session_column
    labels = cutpoints.select([s, "label"])
    return order_events(sessions.lazy(), config).join(labels.lazy(), on=s, how="inner")


def build_whole_session_features(
    sessions: pl.DataFrame, cutpoints: pl.DataFrame
) -> pl.DataFrame:
    """Aggregate whole sessions with the S3 feature recipe.

    S3 carries the widest feature vocabulary of the three stages, so using its
    recipe keeps the whole-session and stage-conditioned feature spaces as close
    as possible; the only difference that remains is the window they see, which
    is the comparison RQ4 is about.
    """
    events = whole_session_events(sessions, cutpoints)
    return build_stage_features(events, "S3")


def member_to_group(group_names: list[str]) -> dict[str, str]:
    """Invert the stage grouping: feature -> correlated-cluster name.

    Cluster names are ``"+".join(sorted(members))`` (see
    ``explain.stage_shap.correlation_groups``), so the membership is recoverable
    from the name alone. The whole-session model *must* be aggregated under the
    stage models' reference grouping; grouping it by any other taxonomy would
    compare two different partitions of attribution mass and silently report
    zero for every family, which is what an earlier draft of this module did.
    """
    mapping: dict[str, str] = {}
    for name in group_names:
        for member in name.split("+"):
            mapping[member] = name
    return mapping


def _fallback_group(feature: str) -> str | None:
    for spec in FEATURE_DICTIONARY:
        if spec.name == feature:
            return spec.ablation_group
    return None


def aggregate_shares(
    columns: list[str],
    normalised: "np.ndarray",
    grouping: dict[str, str] | None = None,
) -> dict[str, float]:
    """Sum normalised attribution mass into the supplied grouping.

    Raises rather than returning zeros when no feature maps, because a silent
    all-zero result reads as "the static model ignores everything", which is a
    fabricated finding rather than a measurement.
    """
    shares: dict[str, float] = {}
    unmapped = 0
    for name, value in zip(columns, normalised, strict=True):
        group = (grouping or {}).get(name)
        if group is None:
            unmapped += 1
            group = _fallback_group(name) or name
        shares[group] = shares.get(group, 0.0) + float(value)

    if grouping and columns and unmapped == len(columns):
        raise ValueError(
            "no whole-session feature matched the stage grouping; the two "
            "taxonomies are disjoint and the contrast would be meaningless"
        )
    return shares


def run_whole_session_contrast(
    sessions: pl.DataFrame,
    cutpoints: pl.DataFrame,
    *,
    suffix: str,
    protocol: str = "temporal",
    seed: int = 42,
    background_size: int = 300,
    max_explain: int = 4000,
    grouping: dict[str, str] | None = None,
) -> WholeSessionResult:
    """Fit and explain a whole-session model on the frozen split."""
    import shap
    from sklearn.metrics import average_precision_score, roc_auc_score

    from ..models.baselines import build_pipeline

    set_global_seed(seed)
    features = build_whole_session_features(sessions, cutpoints)
    split = load_split(suffix, protocol)

    joined = features.join(split.select(["session_id", "partition"]), on="session_id")
    y = joined["label"].cast(pl.Int8).to_numpy()
    partition = joined["partition"].to_numpy()

    columns = [c for c in joined.columns if c not in {"session_id", "label", "partition"}]
    X = joined.select(columns).to_pandas()

    train, test = partition == "train", partition == "test"
    if not train.any() or not test.any():
        raise ValueError("whole-session split produced an empty train or test partition")

    pipeline = build_pipeline("lightgbm", columns, [], seed=seed)
    pipeline.fit(X[train], y[train])
    scores = pipeline.predict_proba(X[test])[:, 1]

    # Attribute on validation, matching the stage protocol: the test partition
    # is for predictive evaluation only.
    val = partition == "val"
    explain_rows = np.flatnonzero(val if val.any() else train)[:max_explain]
    background = X.iloc[np.flatnonzero(train)[:background_size]]

    model = pipeline.named_steps["model"]
    transformed = pipeline.named_steps["preprocess"].transform(X.iloc[explain_rows])
    background_t = pipeline.named_steps["preprocess"].transform(background)
    explainer = shap.TreeExplainer(
        model, data=background_t, feature_perturbation="interventional"
    )
    values = explainer.shap_values(transformed, check_additivity=False)
    if isinstance(values, list):
        values = values[-1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, -1]

    mean_abs = np.abs(values).mean(axis=0)
    total = float(mean_abs.sum()) or 1.0

    shares = aggregate_shares(columns, mean_abs / total, grouping)

    return WholeSessionResult(
        n_train=int(train.sum()),
        n_test=int(test.sum()),
        prevalence=float(y[test].mean()),
        pr_auc=float(average_precision_score(y[test], scores)),
        roc_auc=float(roc_auc_score(y[test], scores)),
        shares=shares,
    )


def contrast_table(
    stage_importance: pl.DataFrame, whole: WholeSessionResult
) -> pl.DataFrame:
    """Stage-conditioned shares against the empirical whole-session shares.

    ``flattening`` is the stage-conditioned peak minus the whole-session share:
    how much of a family's stage-specific prominence a static analysis loses.
    """
    pivot = stage_importance.pivot(values="share", index="group", on="stage")
    stages = [c for c in pivot.columns if c != "group"]

    rows = []
    for row in pivot.to_dicts():
        group = row["group"]
        values = [row[s] for s in stages if row[s] is not None]
        if not values:
            continue
        peak = max(values)
        static_share = whole.shares.get(group, 0.0)
        rows.append(
            {
                "group": group,
                **{f"share_{s}": row[s] for s in stages},
                "stage_peak": peak,
                "stage_peak_at": stages[values.index(peak)],
                "whole_session_share": static_share,
                "flattening": peak - static_share,
            }
        )
    return pl.DataFrame(rows).sort("flattening", descending=True)
