"""Layer 2 — attributions on the sequence model (protocol Sec. 11.2, H4).

Provides the second, independent attribution paradigm that H4 compares against
TreeSHAP. Two entry points:

* :func:`timeshap_feature_attribution` — the protocol's named tool (feedzai
  TimeSHAP), used when it is installed.
* :func:`permutation_feature_attribution` — a model-agnostic fallback that
  measures the same quantity (per-feature contribution over the whole prefix)
  by permuting one feature across all timesteps and observing the score change.

The fallback exists because H4 is a claim about *paradigms* — recurrent vs.
tree — not about a particular library. If TimeSHAP cannot be installed, the
comparison is still meaningful with a permutation-based recurrent attribution,
provided the paper says which was used. **It must say which.** Reporting a
convergence result while implying TimeSHAP produced it would misrepresent the
method.

Both return attribution aggregated to the *feature* level, because that is the
only granularity at which the two paradigms are comparable: TreeSHAP attributes
to prefix-aggregate features, TimeSHAP additionally to events and timesteps,
and there is no tree-side counterpart to a per-timestep value.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SequenceAttribution:
    method: str
    feature_names: list[str]
    mean_abs: np.ndarray
    mean_signed: np.ndarray
    n_explained: int

    def ranking(self) -> list[str]:
        return [self.feature_names[i] for i in np.argsort(-self.mean_abs)]

    def summary(self) -> str:
        top = ", ".join(self.ranking()[:3])
        return f"{self.method} on {self.n_explained} sequences; top: {top}"


def _shim_shap_kernel_alias() -> None:
    """Restore the ``Kernel`` alias TimeSHAP imports from shap.

    TimeSHAP 1.0.4 does ``from shap.explainers._kernel import Kernel``. shap
    renamed that class to ``KernelExplainer`` and dropped the alias, so the
    import fails against the pinned shap 0.45 (Sec. 4.1). This is the library
    gap Sec. 4.2 anticipated for the TimeSHAP arm.

    Re-adding the alias is a one-symbol shim on a pure rename, not a behaviour
    change: TimeSHAP subclasses the same object under either name. It is applied
    lazily and only here, so nothing else in the codebase depends on a patched
    shap. The alternative — downgrading shap — would break the pinned
    environment for the whole tabular arm to accommodate one module.
    """
    import shap.explainers._kernel as kernel_module

    if not hasattr(kernel_module, "Kernel"):
        kernel_module.Kernel = kernel_module.KernelExplainer


def timeshap_available() -> bool:
    """Whether TimeSHAP can actually be used, not merely imported.

    Checks the submodule the attribution path needs. A bare ``import timeshap``
    succeeds even when ``timeshap.explainer`` fails, which would report the
    library as available and then crash mid-run.
    """
    try:
        _shim_shap_kernel_alias()
        from timeshap.explainer import local_feat  # noqa: F401
    except Exception:
        return False
    return True


def timeshap_feature_attribution(
    model,
    X: np.ndarray,
    background: np.ndarray,
    feature_names: list[str],
    *,
    n_explain: int = 200,
    seed: int = 42,
) -> SequenceAttribution:
    """Feature-level attribution via TimeSHAP (Sec. 11.2)."""
    _shim_shap_kernel_alias()
    from timeshap.explainer import local_feat

    rng = np.random.default_rng(seed)
    pick = rng.choice(len(X), size=min(n_explain, len(X)), replace=False)
    average_event = background.mean(axis=(0, 1), keepdims=True)

    def f(sequence: np.ndarray) -> np.ndarray:
        return np.asarray(model(sequence)).reshape(-1, 1)

    contributions = []
    for i in pick:
        sequence = X[i : i + 1]
        result = local_feat(
            f,
            sequence,
            {"rs": seed, "nsamples": 320},
            entity_uuid=None,
            entity_col=None,
            baseline=average_event,
        )
        values = result.sort_values("Feature")["Shapley Value"].to_numpy()
        contributions.append(values[: len(feature_names)])

    stacked = np.vstack(contributions)
    return SequenceAttribution(
        method="TimeSHAP",
        feature_names=list(feature_names),
        mean_abs=np.abs(stacked).mean(axis=0),
        mean_signed=stacked.mean(axis=0),
        n_explained=len(pick),
    )


def permutation_feature_attribution(
    predict,
    X: np.ndarray,
    feature_names: list[str],
    *,
    n_repeats: int = 10,
    seed: int = 42,
) -> SequenceAttribution:
    """Model-agnostic recurrent attribution by permuting a feature across time.

    For each feature, its values are shuffled *across sessions but within its
    own timestep position*, so the temporal structure of the other features is
    preserved and only that feature's information is destroyed. Shuffling within
    a session instead would leave the feature's distribution intact and measure
    only order sensitivity, which is a different question.
    """
    rng = np.random.default_rng(seed)
    base = np.asarray(predict(X), dtype=float)

    mean_abs = np.zeros(len(feature_names))
    mean_signed = np.zeros(len(feature_names))

    for j in range(len(feature_names)):
        deltas = []
        for _ in range(n_repeats):
            permuted = X.copy()
            order = rng.permutation(len(X))
            permuted[:, :, j] = permuted[order, :, j]
            deltas.append(base - np.asarray(predict(permuted), dtype=float))
        stacked = np.vstack(deltas)
        mean_abs[j] = float(np.abs(stacked).mean())
        mean_signed[j] = float(stacked.mean())

    return SequenceAttribution(
        method="permutation (recurrent)",
        feature_names=list(feature_names),
        mean_abs=mean_abs,
        mean_signed=mean_signed,
        n_explained=len(X),
    )


#: Map from sequence-level feature names to the tabular features that carry the
#: same information, so H4 compares like with like. Anything absent from this
#: map has no counterpart and is excluded from the comparison rather than
#: silently matched to zero.
SEQUENCE_TO_TABULAR: dict[str, tuple[str, ...]] = {
    "event_type_view": ("n_views",),
    "event_type_cart": (),
    "event_type_remove_from_cart": (),
    "log_gap_s": ("inter_event_mean_s", "last_gap_s", "prefix_duration_s"),
    "log_price": ("price_mean", "price_max"),
    "is_new_product": ("n_unique_products", "product_revisit_rate"),
    "is_new_category": ("n_unique_categories", "category_revisit_rate"),
}


def align_for_h4(
    sequence: SequenceAttribution,
    tabular_importance: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Put both paradigms on a common footing for the H4 rank correlation.

    Returns aligned importance vectors and the concept labels they refer to.
    Sequence features with no tabular counterpart are dropped, and the reason
    matters: comparing them against an implicit zero would manufacture
    disagreement out of a schema mismatch rather than a paradigm difference.
    """
    labels: list[str] = []
    seq_values: list[float] = []
    tab_values: list[float] = []

    index = {name: i for i, name in enumerate(sequence.feature_names)}
    for seq_name, tabular_names in SEQUENCE_TO_TABULAR.items():
        if seq_name not in index or not tabular_names:
            continue
        matched = [tabular_importance[t] for t in tabular_names if t in tabular_importance]
        if not matched:
            continue
        labels.append(seq_name)
        seq_values.append(float(sequence.mean_abs[index[seq_name]]))
        # Sum, matching how correlated tabular features are grouped elsewhere:
        # Shapley values are additive contributions.
        tab_values.append(float(sum(matched)))

    return np.asarray(seq_values), np.asarray(tab_values), labels
