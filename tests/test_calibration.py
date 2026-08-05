"""Tests for prior-shift correction (protocol Sec. 9.6, 10)."""

from __future__ import annotations

import numpy as np
import pytest

from funnel_shap.evaluate.calibration import (
    adjust_for_prior,
    correct_prior_shift,
    estimate_prior_em,
)
from funnel_shap.evaluate.metrics import expected_calibration_error


def _shifted_population(
    n: int = 40_000, prior: float = 0.125, separation: float = 1.0, seed: int = 0
):
    """A population with a known prior and *correctly calibrated* posteriors.

    Two Gaussians, ``x | y=1 ~ N(mu, 1)`` and ``x | y=0 ~ N(-mu, 1)``, for which
    the exact posterior is available in closed form:

        logit p(y=1|x) = log(prior / (1 - prior)) + 2 * mu * x

    Deriving it rather than passing a latent score through a sigmoid matters:
    a sigmoid of a normal is not a calibrated probability, its mean sits near
    0.5 regardless of the prior, and testing a prior-shift correction against
    scores that never encoded the prior would measure nothing.
    """
    rng = np.random.default_rng(seed)
    y = rng.binomial(1, prior, size=n)
    x = rng.normal(loc=np.where(y == 1, separation, -separation), scale=1.0)
    logit = np.log(prior / (1 - prior)) + 2 * separation * x
    return y, 1 / (1 + np.exp(-logit))


def test_fixture_is_actually_calibrated() -> None:
    """Guard on the guard: if the fixture drifts, every test below is vacuous."""
    y, p = _shifted_population(prior=0.125, seed=0)
    assert p.mean() == pytest.approx(y.mean(), abs=0.01)
    assert expected_calibration_error(y, p, strategy="quantile") < 0.02


# ---------------------------------------------------------------------------
# The rescaling itself
# ---------------------------------------------------------------------------


def test_identical_priors_leave_probabilities_untouched() -> None:
    p = np.array([0.01, 0.2, 0.5, 0.8, 0.99])
    np.testing.assert_allclose(
        adjust_for_prior(p, prior_source=0.15, prior_target=0.15), p, rtol=1e-10
    )


def test_lowering_the_prior_lowers_every_probability() -> None:
    p = np.array([0.05, 0.3, 0.6, 0.9])
    adjusted = adjust_for_prior(p, prior_source=0.25, prior_target=0.125)
    assert np.all(adjusted < p)
    assert np.all((adjusted > 0) & (adjusted < 1))


def test_adjustment_preserves_ranking() -> None:
    """Discrimination must be untouched: this fixes calibration, not ordering."""
    rng = np.random.default_rng(3)
    p = rng.uniform(0.001, 0.999, size=5000)
    adjusted = adjust_for_prior(p, prior_source=0.25, prior_target=0.10)
    assert np.array_equal(np.argsort(p), np.argsort(adjusted))


def test_adjustment_is_invertible() -> None:
    p = np.array([0.05, 0.3, 0.6, 0.9])
    there = adjust_for_prior(p, prior_source=0.25, prior_target=0.10)
    back = adjust_for_prior(there, prior_source=0.10, prior_target=0.25)
    np.testing.assert_allclose(back, p, rtol=1e-8)


def test_extreme_priors_do_not_produce_nans() -> None:
    p = np.array([0.0, 1.0, 0.5])
    adjusted = adjust_for_prior(p, prior_source=0.0, prior_target=1.0)
    assert np.all(np.isfinite(adjusted))


# ---------------------------------------------------------------------------
# EM prior estimation
# ---------------------------------------------------------------------------


def test_em_recovers_a_known_shifted_prior() -> None:
    """The whole point: recover the target prior from scores alone."""
    y, p = _shifted_population(prior=0.125, seed=1)
    result = estimate_prior_em(p, prior_source=0.125)

    assert result.converged
    assert result.prior_estimated == pytest.approx(y.mean(), abs=0.02)


def test_em_detects_a_halved_prior() -> None:
    """Mirrors the real case: calibrated on ~25%, deployed on ~12.5%."""
    y, p = _shifted_population(prior=0.125, seed=2)
    # Pretend the model was calibrated on a 25% population.
    miscalibrated = adjust_for_prior(p, prior_source=0.125, prior_target=0.25)

    result = estimate_prior_em(miscalibrated, prior_source=0.25)
    assert result.prior_estimated == pytest.approx(y.mean(), abs=0.03)


def test_em_uses_no_labels() -> None:
    """Signature guard: labels must not be reachable from the estimator."""
    import inspect

    params = set(inspect.signature(estimate_prior_em).parameters)
    assert not params & {"y", "y_true", "labels", "prior_target"}


def test_em_is_stable_when_there_is_no_shift() -> None:
    _, p = _shifted_population(prior=0.15, seed=4)
    result = estimate_prior_em(p, prior_source=float(p.mean()))
    assert result.prior_estimated == pytest.approx(p.mean(), abs=0.02)


def test_em_reports_non_convergence_rather_than_hiding_it() -> None:
    _, p = _shifted_population(seed=5)
    result = estimate_prior_em(p, prior_source=0.5, max_iterations=1, tolerance=1e-15)
    assert result.converged is False
    assert result.n_iterations == 1


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


def test_correction_reduces_ece_under_prior_shift() -> None:
    """The headline claim, on synthetic data with a known 2x shift."""
    y, p = _shifted_population(prior=0.125, seed=6)
    miscalibrated = adjust_for_prior(p, prior_source=0.125, prior_target=0.25)

    before = expected_calibration_error(y, miscalibrated, strategy="quantile")
    corrected, result = correct_prior_shift(miscalibrated, prior_source=0.25)
    after = expected_calibration_error(y, corrected, strategy="quantile")

    assert before > 0.05, "the synthetic shift should be clearly miscalibrated"
    assert after < before / 2, f"correction did not help: {before:.4f} -> {after:.4f}"
    assert result.prior_estimated == pytest.approx(y.mean(), abs=0.03)


def test_correction_does_not_change_discrimination() -> None:
    from sklearn.metrics import average_precision_score, roc_auc_score

    y, p = _shifted_population(prior=0.125, seed=7)
    miscalibrated = adjust_for_prior(p, prior_source=0.125, prior_target=0.25)
    corrected, _ = correct_prior_shift(miscalibrated, prior_source=0.25)

    assert roc_auc_score(y, corrected) == pytest.approx(roc_auc_score(y, miscalibrated))
    assert average_precision_score(y, corrected) == pytest.approx(
        average_precision_score(y, miscalibrated)
    )


def test_correction_is_harmless_when_already_calibrated() -> None:
    """It must not damage a model that needed no correction."""
    y, p = _shifted_population(prior=0.15, seed=8)
    prior = float(p.mean())

    before = expected_calibration_error(y, p, strategy="quantile")
    corrected, _ = correct_prior_shift(p, prior_source=prior)
    after = expected_calibration_error(y, corrected, strategy="quantile")

    assert after < before + 0.01
