"""Tests for the contact-budget sweep (Section 6.3, amendment A42).

The sweep is what replaced a single F1-selected operating point after that
threshold turned out to determine the paper's S3 conclusion, so its arithmetic
carries a headline claim and is tested directly rather than through a model run.
"""

from __future__ import annotations

import numpy as np
import pytest

from funnel_shap.models.flag_rate import (
    DEFAULT_RATES,
    incremental_precision,
    sweep_table,
)


@pytest.fixture
def perfect() -> tuple[np.ndarray, np.ndarray]:
    """A ranker that puts every positive above every negative."""
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
    return y, scores


def test_flagging_everything_is_exactly_the_blanket_rule(perfect) -> None:
    """The identity that anchors the whole table: at rate 1 the model adds nothing."""
    y, scores = perfect
    assert incremental_precision(y, scores, 1.0) == pytest.approx(0.0)


def test_a_perfect_ranker_scores_one_minus_prevalence_at_its_own_rate(perfect) -> None:
    y, scores = perfect
    # Two positives out of ten: flagging the top 20% is all-correct.
    assert incremental_precision(y, scores, 0.2) == pytest.approx(1.0 - 0.2)


def test_an_inverted_ranker_is_negative(perfect) -> None:
    """Worse than the blanket rule must read as worse, not as zero."""
    y, scores = perfect
    assert incremental_precision(y, -scores, 0.2) < 0


def test_rate_selects_at_least_one_session() -> None:
    """A budget too small to round up to a session must not select none."""
    y = np.array([1, 0, 0, 0])
    scores = np.array([0.9, 0.1, 0.1, 0.1])
    assert incremental_precision(y, scores, 0.01) == pytest.approx(1.0 - 0.25)


@pytest.mark.parametrize("rate", [0.0, -0.1, 1.5])
def test_rates_outside_the_unit_interval_are_rejected(perfect, rate) -> None:
    y, scores = perfect
    with pytest.raises(ValueError, match="flag rate"):
        incremental_precision(y, scores, rate)


def test_mismatched_labels_and_scores_are_rejected(perfect) -> None:
    y, scores = perfect
    with pytest.raises(ValueError, match="same shape"):
        incremental_precision(y, scores[:-1], 0.2)


def test_contacts_column_is_the_reciprocal_of_the_mean(perfect) -> None:
    y, scores = perfect
    table = sweep_table({"S1": [(y, scores)]}, rates=[0.5])
    row = table.to_dicts()[0]
    assert row["contacts_per_incremental"] == pytest.approx(
        round(1.0 / row["incremental_precision_mean"], 1), abs=0.05
    )


def test_non_positive_means_report_no_contact_requirement() -> None:
    """A reciprocal of ~zero would imply a precision the data do not support."""
    y = np.array([1, 1, 0, 0])
    table = sweep_table({"S1": [(y, np.array([0.1, 0.2, 0.9, 0.8]))]}, rates=[0.5])
    row = table.to_dicts()[0]
    assert row["incremental_precision_mean"] < 0
    assert row["contacts_per_incremental"] is None


def test_dispersion_is_reported_across_seeds_and_omitted_for_one(perfect) -> None:
    y, scores = perfect
    single = sweep_table({"S1": [(y, scores)]}, rates=[0.2]).to_dicts()[0]
    assert single["incremental_precision_sd"] is None
    assert single["n_seeds"] == 1

    paired = sweep_table(
        {"S1": [(y, scores), (y, scores[::-1])]}, rates=[0.2]
    ).to_dicts()[0]
    assert paired["incremental_precision_sd"] > 0
    assert paired["n_seeds"] == 2


def test_stage_order_is_respected_and_missing_stages_are_skipped(perfect) -> None:
    y, scores = perfect
    table = sweep_table(
        {"S3": [(y, scores)], "S1": [(y, scores)]},
        rates=[0.2],
        stage_order=["S1", "S2", "S3"],
    )
    assert table["stage"].to_list() == ["S1", "S3"]


def test_default_rates_end_at_the_blanket_rule() -> None:
    assert DEFAULT_RATES[-1] == 1.0
    assert all(0 < r <= 1 for r in DEFAULT_RATES)
