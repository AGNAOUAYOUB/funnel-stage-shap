"""Layer 3 — explanation quality (protocol Sec. 11.3).

Sec. 2 calls this the differentiator: "most e-commerce XAI papers produce SHAP
plots but never validate them". Three families, with thresholds fixed before
running (Appendix A: faithfulness correlation > 0.5, cross-paradigm Spearman
> 0.6).

**Faithfulness.** If an attribution is honest, removing the features it credits
should move the prediction by roughly the amount it credited them. Implemented
as the Bhatt et al. (2020) faithfulness correlation — over random feature
subsets, correlate the summed attribution against the actual change in
predicted probability when those features are replaced by baseline values —
plus deletion and insertion curves.

Removal means **replacing with a background value, not with zero**. Zero is a
real, in-distribution value for most of these features (a count of nothing, a
gap of no time), so zeroing does not remove information — it substitutes a
different, often meaningful, input and measures the model's response to that
instead.

**Stability.** A local-Lipschitz-style estimate: perturb the input slightly,
recompute the attribution, and take the worst-case ratio of attribution change
to input change. Expensive, because every perturbation needs a fresh SHAP pass,
so it runs on a small sample by design.

**Consistency.** Spearman rank correlation of the global importance ordering
across seeds. This is what tells you whether the migration figure is a finding
or one draw.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr

#: Appendix A, fixed before running.
FAITHFULNESS_THRESHOLD = 0.5
CROSS_PARADIGM_SPEARMAN_THRESHOLD = 0.6


@dataclass(frozen=True)
class FaithfulnessResult:
    correlation_mean: float
    correlation_std: float
    n_instances: int
    n_subsets: int
    passes: bool

    def summary(self) -> str:
        verdict = "PASS" if self.passes else "FAIL"
        return (
            f"faithfulness correlation {self.correlation_mean:.3f} "
            f"(sd {self.correlation_std:.3f}, n={self.n_instances}) "
            f"vs threshold {FAITHFULNESS_THRESHOLD}: {verdict}"
        )


@dataclass(frozen=True)
class CurveResult:
    fractions: np.ndarray
    mean_prediction: np.ndarray
    auc: float
    monotone: bool

    def summary(self) -> str:
        return f"AUC {self.auc:.4f}, monotone={self.monotone}"


def _baseline_row(background: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A single draw from the background, used as the replacement value."""
    return background[rng.integers(0, background.shape[0])]


def faithfulness_correlation(
    predict: callable,
    X: np.ndarray,
    shap_values: np.ndarray,
    background: np.ndarray,
    *,
    subset_size: int | None = None,
    n_subsets: int = 50,
    seed: int = 42,
) -> FaithfulnessResult:
    """Bhatt et al. (2020) faithfulness correlation.

    For each instance, draw random feature subsets; correlate the summed
    attribution over each subset with the drop in predicted probability when
    those features are replaced by background values. A faithful explanation
    gives a high positive correlation.
    """
    X = np.asarray(X, dtype=float)
    shap_values = np.asarray(shap_values, dtype=float)
    background = np.asarray(background, dtype=float)
    rng = np.random.default_rng(seed)

    n_instances, n_features = X.shape
    if subset_size is None:
        subset_size = max(1, n_features // 4)

    base_predictions = predict(X)
    correlations: list[float] = []

    for i in range(n_instances):
        attributions = np.empty(n_subsets)
        deltas = np.empty(n_subsets)
        perturbed = np.repeat(X[i][None, :], n_subsets, axis=0)

        for s in range(n_subsets):
            subset = rng.choice(n_features, size=subset_size, replace=False)
            attributions[s] = shap_values[i, subset].sum()
            perturbed[s, subset] = _baseline_row(background, rng)[subset]

        deltas = base_predictions[i] - predict(perturbed)

        if np.std(attributions) < 1e-12 or np.std(deltas) < 1e-12:
            continue
        correlations.append(float(np.corrcoef(attributions, deltas)[0, 1]))

    if not correlations:
        raise RuntimeError("no instance produced a computable faithfulness correlation")

    values = np.asarray(correlations)
    mean = float(values.mean())
    return FaithfulnessResult(
        correlation_mean=mean,
        correlation_std=float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        n_instances=len(values),
        n_subsets=n_subsets,
        passes=mean > FAITHFULNESS_THRESHOLD,
    )


def deletion_curve(
    predict: callable,
    X: np.ndarray,
    shap_values: np.ndarray,
    background: np.ndarray,
    *,
    n_steps: int = 10,
    seed: int = 42,
) -> CurveResult:
    """Remove the most positively-attributed features first; the prediction should fall.

    Sec. 11.3 asks that "removing top features degrade predicted probability
    monotonically". That is a claim about *signed* attribution: the features to
    remove are the ones pushing the prediction up, ordered by SHAP value
    descending, not by magnitude.

    Ordering by |SHAP| instead — the more common implementation — mixes features
    that raise the prediction with features that lower it. Removing both sets
    together moves individual predictions in opposite directions, the average
    barely moves, and the deletion and insertion curves collapse onto each
    other. That is what this implementation did first, and it made the metric
    unable to distinguish a good explanation from a useless one.

    Ordering is per instance, since attribution is local.
    """
    return _curve(predict, X, shap_values, background, n_steps=n_steps, seed=seed, insert=False)


def insertion_curve(
    predict: callable,
    X: np.ndarray,
    shap_values: np.ndarray,
    background: np.ndarray,
    *,
    n_steps: int = 10,
    seed: int = 42,
) -> CurveResult:
    """Start from background and add the most-attributed features first."""
    return _curve(predict, X, shap_values, background, n_steps=n_steps, seed=seed, insert=True)


def _curve(
    predict: callable,
    X: np.ndarray,
    shap_values: np.ndarray,
    background: np.ndarray,
    *,
    n_steps: int,
    seed: int,
    insert: bool,
) -> CurveResult:
    X = np.asarray(X, dtype=float)
    shap_values = np.asarray(shap_values, dtype=float)
    background = np.asarray(background, dtype=float)
    rng = np.random.default_rng(seed)

    n_instances, n_features = X.shape
    # Signed, descending: the features pushing the prediction up come first.
    order = np.argsort(-shap_values, axis=1)
    baseline = np.stack([_baseline_row(background, rng) for _ in range(n_instances)])

    fractions = np.linspace(0.0, 1.0, n_steps + 1)
    means = np.empty(len(fractions))
    #: Mean signed attribution of the features moved at each step, used to
    #: bound where the monotonicity claim applies.
    step_attribution = np.zeros(len(fractions))

    previous_k = 0
    for step, fraction in enumerate(fractions):
        k = int(round(fraction * n_features))
        current = (baseline if insert else X).copy()
        if k > 0:
            rows = np.repeat(np.arange(n_instances), k)
            cols = order[:, :k].ravel()
            source = X if insert else baseline
            current[rows, cols] = source[rows, cols]
        means[step] = float(np.mean(predict(current)))

        if k > previous_k:
            newly = order[:, previous_k:k]
            step_attribution[step] = float(
                np.mean(np.take_along_axis(shap_values, newly, axis=1))
            )
        previous_k = k

    auc = float(np.trapz(means, fractions))

    # Sec. 11.3's monotonicity claim is about the *top* features. With signed
    # ordering, once the positively-attributed features are exhausted the curve
    # necessarily turns back toward the baseline as negative contributors are
    # removed -- a V shape, not a failure. Checking monotonicity over the whole
    # sweep would therefore report every honest explanation as non-monotone.
    # Step s moves the features between k(s-1) and k(s), so the change it
    # caused is means[s] - means[s-1]. Only steps whose moved features are
    # positively attributed are covered by the claim.
    positive_steps = [s for s in range(1, len(means)) if step_attribution[s] > 0]
    changes = np.array([means[s] - means[s - 1] for s in positive_steps])
    if changes.size == 0:
        monotone = True
    else:
        monotone = (
            bool(np.all(changes >= -1e-9)) if insert else bool(np.all(changes <= 1e-9))
        )

    return CurveResult(fractions=fractions, mean_prediction=means, auc=auc, monotone=monotone)


@dataclass(frozen=True)
class StabilityResult:
    max_ratio: float
    mean_ratio: float
    n_instances: int
    noise_scale: float

    def summary(self) -> str:
        return (
            f"local Lipschitz: max {self.max_ratio:.3f}, mean {self.mean_ratio:.3f} "
            f"(n={self.n_instances}, noise sd {self.noise_scale})"
        )


def local_lipschitz(
    explain_fn: callable,
    X: np.ndarray,
    *,
    n_perturbations: int = 5,
    noise_scale: float = 0.05,
    seed: int = 42,
) -> StabilityResult:
    """Worst-case attribution change per unit input change (Sec. 11.3).

    ``explain_fn`` maps a matrix to its SHAP values, so each perturbation costs
    a full explanation pass — run it on tens of instances, not thousands.
    Perturbations are scaled by each feature's own standard deviation, so a
    feature measured in seconds is not perturbed a thousand times harder than
    one measured in counts.
    """
    X = np.asarray(X, dtype=float)
    rng = np.random.default_rng(seed)
    scales = X.std(axis=0)
    scales[scales == 0] = 1.0

    base = np.asarray(explain_fn(X), dtype=float)
    ratios: list[float] = []

    for _ in range(n_perturbations):
        noise = rng.normal(scale=noise_scale * scales, size=X.shape)
        perturbed = X + noise
        attributions = np.asarray(explain_fn(perturbed), dtype=float)

        numerator = np.linalg.norm(attributions - base, axis=1)
        denominator = np.linalg.norm(noise, axis=1)
        valid = denominator > 1e-12
        ratios.extend((numerator[valid] / denominator[valid]).tolist())

    if not ratios:
        raise RuntimeError("no valid perturbation pairs")

    values = np.asarray(ratios)
    return StabilityResult(
        max_ratio=float(values.max()),
        mean_ratio=float(values.mean()),
        n_instances=X.shape[0],
        noise_scale=noise_scale,
    )


@dataclass(frozen=True)
class ConsistencyResult:
    mean_spearman: float
    min_spearman: float
    n_pairs: int
    threshold: float
    passes: bool

    def summary(self) -> str:
        verdict = "PASS" if self.passes else "FAIL"
        return (
            f"rank consistency: mean rho {self.mean_spearman:.3f}, "
            f"min {self.min_spearman:.3f} over {self.n_pairs} pairs "
            f"vs threshold {self.threshold}: {verdict}"
        )


def rank_consistency(
    importances: list[np.ndarray], *, threshold: float = CROSS_PARADIGM_SPEARMAN_THRESHOLD
) -> ConsistencyResult:
    """Spearman agreement of importance orderings across runs (Sec. 11.3).

    Used two ways: across seeds, which tells you whether the migration figure is
    a finding or one draw; and between TreeSHAP and TimeSHAP, which is H4.

    The *minimum* pairwise correlation is reported alongside the mean, because a
    single badly disagreeing pair is exactly what a mean hides, and it is the
    pair a reviewer will ask about.
    """
    if len(importances) < 2:
        raise ValueError("rank consistency needs at least two importance vectors")

    values = [np.asarray(v, dtype=float).ravel() for v in importances]
    if len({v.shape for v in values}) != 1:
        raise ValueError("importance vectors must be the same length")

    correlations: list[float] = []
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            rho, _ = spearmanr(values[i], values[j])
            correlations.append(float(np.nan_to_num(rho)))

    mean = float(np.mean(correlations))
    return ConsistencyResult(
        mean_spearman=mean,
        min_spearman=float(np.min(correlations)),
        n_pairs=len(correlations),
        threshold=threshold,
        passes=mean > threshold,
    )
