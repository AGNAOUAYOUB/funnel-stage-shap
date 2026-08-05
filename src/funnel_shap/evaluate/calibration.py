"""Probability calibration under prior-probability shift (protocol Sec. 9.6, 10).

Sec. 9.6 asks for calibration on "a calibration split" and for pre/post
reporting. Two things went wrong in the first pass and are corrected here.

**The calibration set must not be the validation month.** Dataset A's conversion
prevalence swings from 1.6% (Feb) to 25.4% (Nov) and back to 12.5% (Dec).
Fitting an isotonic calibrator on November and applying it to December left
every baseline over-predicting by a factor of ~1.94 -- almost exactly the
0.2535/0.1251 prevalence ratio -- in every reliability bin. A calibration slice
drawn from across the *training* period is far better matched, and it also stops
the validation split doing two jobs (calibration and threshold selection) whose
requirements conflict.

**Residual shift needs correcting without touching test labels.** Even a
well-drawn calibration split cannot anticipate an unseen future month's base
rate. The textbook fix -- rescale by the ratio of new to old prevalence --
requires knowing the new prevalence, and reading it off the test labels would be
peeking at the partition the protocol opens exactly once.

`estimate_prior_em` avoids that. It is the Saerens-Latinne-Decaestecker (2002)
EM procedure, which estimates the shifted prior from the model's scores on
*unlabelled* data alone. Deployment has the same information: you see the
traffic, not whether it converted. Using it here is legitimate; using the test
prevalence directly would not be.

    Saerens, M., Latinne, P., Decaestecker, C. (2002). Adjusting the outputs of a
    classifier to new a priori probabilities: a simple procedure. Neural
    Computation 14(1), 21-41.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_EPS = 1e-12


@dataclass(frozen=True)
class PriorShiftResult:
    prior_source: float
    prior_estimated: float
    n_iterations: bool | int
    converged: bool

    def summary(self) -> str:
        return (
            f"prior {self.prior_source:.4f} -> {self.prior_estimated:.4f} "
            f"({self.n_iterations} iters, converged={self.converged})"
        )


def adjust_for_prior(
    probabilities: np.ndarray, *, prior_source: float, prior_target: float
) -> np.ndarray:
    """Rescale calibrated probabilities from one class prior to another.

    The standard prior-shift correction: multiply the odds by the ratio of
    target to source prior and renormalise.
    """
    p = np.clip(np.asarray(probabilities, dtype=float).ravel(), _EPS, 1 - _EPS)
    prior_source = float(np.clip(prior_source, _EPS, 1 - _EPS))
    prior_target = float(np.clip(prior_target, _EPS, 1 - _EPS))

    pos = p * (prior_target / prior_source)
    neg = (1.0 - p) * ((1.0 - prior_target) / (1.0 - prior_source))
    return pos / (pos + neg)


def estimate_prior_em(
    probabilities: np.ndarray,
    *,
    prior_source: float,
    max_iterations: int = 1000,
    tolerance: float = 1e-8,
) -> PriorShiftResult:
    """Estimate the target class prior from unlabelled scores (Saerens et al. 2002).

    Uses only ``probabilities`` -- no labels -- so it may be applied to the test
    partition without violating the protocol's single-look rule.
    """
    p = np.clip(np.asarray(probabilities, dtype=float).ravel(), _EPS, 1 - _EPS)
    prior_source = float(np.clip(prior_source, _EPS, 1 - _EPS))

    prior = prior_source
    converged = False
    iteration = 0
    while iteration < max_iterations:
        iteration += 1
        adjusted = adjust_for_prior(p, prior_source=prior_source, prior_target=prior)
        updated = float(adjusted.mean())
        if abs(updated - prior) < tolerance:
            prior = updated
            converged = True
            break
        prior = updated

    return PriorShiftResult(
        prior_source=prior_source,
        prior_estimated=float(np.clip(prior, _EPS, 1 - _EPS)),
        n_iterations=iteration,
        converged=converged,
    )


def correct_prior_shift(
    probabilities: np.ndarray, *, prior_source: float
) -> tuple[np.ndarray, PriorShiftResult]:
    """Estimate the target prior from the scores, then rescale to it.

    Returns the corrected probabilities and the estimation diagnostics, which
    must be reported: a correction whose estimated prior is wildly off the
    eventual observed prevalence is a finding, not a silent fix.
    """
    result = estimate_prior_em(probabilities, prior_source=prior_source)
    corrected = adjust_for_prior(
        probabilities, prior_source=prior_source, prior_target=result.prior_estimated
    )
    return corrected, result
