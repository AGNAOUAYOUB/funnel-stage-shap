"""Tests for the sequence arm and H4 (protocol Sec. 9.3, 11.2)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from funnel_shap.explain.sequence_shap import (
    SEQUENCE_TO_TABULAR,
    align_for_h4,
    permutation_feature_attribution,
    timeshap_available,
    timeshap_feature_attribution,
)
from funnel_shap.models.sequence import SEQUENCE_FEATURES, build_sequences

torch = pytest.importorskip("torch")


def _linear_last_step(weights: np.ndarray):
    """A model whose output depends only on the final timestep, linearly."""

    def predict(seq):
        seq = np.asarray(seq, dtype=np.float32)
        return 1 / (1 + np.exp(-(seq[:, -1, :] @ weights)))

    return predict


@pytest.fixture(scope="module")
def toy():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 6, 4)).astype(np.float32)
    weights = np.array([2.0, -1.0, 0.5, 0.0])
    return _linear_last_step(weights), X, ["f0", "f1", "f2", "f3"], weights


# ---------------------------------------------------------------------------
# Sequence construction
# ---------------------------------------------------------------------------


def _prefix_frame(n_events: int = 5) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "session_id": ["s"] * n_events,
            "event_idx": list(range(n_events)),
            "event_time": [f"2019-10-01 00:0{i}:00" for i in range(n_events)],
            "event_type": ["view"] * (n_events - 1) + ["cart"],
            "product_id": [1, 2, 1, 3, 3][:n_events],
            "category_id": [1, 1, 1, 2, 2][:n_events],
            "price": [10.0] * n_events,
            "label": [True] * n_events,
        }
    ).with_columns(
        pl.col("event_time").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S")
    )


def test_sequences_are_padded_and_masked() -> None:
    batch = build_sequences(_prefix_frame(5), max_len=8)

    assert batch.X.shape == (1, 8, len(SEQUENCE_FEATURES))
    assert batch.mask[0, :5].all()
    assert not batch.mask[0, 5:].any()
    assert batch.feature_names == list(SEQUENCE_FEATURES)


def test_long_prefixes_keep_the_events_nearest_the_decision() -> None:
    """Truncating from the front preserves the decision-relevant tail."""
    batch = build_sequences(_prefix_frame(5), max_len=3)

    assert batch.X.shape[1] == 3
    assert batch.mask[0].all()
    # The final event is a cart, so the cart indicator must survive truncation.
    cart_col = SEQUENCE_FEATURES.index("event_type_cart")
    assert batch.X[0, -1, cart_col] == 1.0


def test_no_purchase_event_can_appear_in_a_sequence() -> None:
    """The anti-leakage guarantee must hold for this arm too (Sec. 7.4)."""
    assert "purchase" not in " ".join(SEQUENCE_FEATURES)
    from funnel_shap.models.sequence import EVENT_TYPES

    assert "purchase" not in EVENT_TYPES


# ---------------------------------------------------------------------------
# Attribution
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not timeshap_available(), reason="TimeSHAP not installed")
def test_timeshap_recovers_a_known_ordering(toy) -> None:
    predict, X, names, weights = toy
    attribution = timeshap_feature_attribution(
        predict, X, X, names, n_explain=6, seed=7
    )

    ranking = attribution.ranking()
    assert ranking[0] == "f0", "the largest-weight feature should rank first"
    # The zero-weight feature carries no attribution at all.
    assert attribution.mean_abs[names.index("f3")] == pytest.approx(0.0, abs=1e-9)
    assert attribution.method == "TimeSHAP"


def test_permutation_fallback_recovers_the_same_ordering(toy) -> None:
    predict, X, names, _ = toy
    attribution = permutation_feature_attribution(predict, X, names, n_repeats=6, seed=7)

    assert attribution.ranking()[0] == "f0"
    assert attribution.mean_abs[names.index("f3")] == pytest.approx(0.0, abs=1e-9)
    assert "permutation" in attribution.method


def test_method_is_carried_so_it_cannot_be_reported_ambiguously(toy) -> None:
    """H4 is a paradigm claim; the paper must say which tool produced it."""
    predict, X, names, _ = toy
    permutation = permutation_feature_attribution(predict, X, names, n_repeats=2, seed=7)
    assert permutation.method in permutation.summary()


# ---------------------------------------------------------------------------
# H4 alignment
# ---------------------------------------------------------------------------


def test_alignment_drops_features_with_no_tabular_counterpart(toy) -> None:
    """Comparing against an implicit zero would fake a paradigm disagreement."""
    from funnel_shap.explain.sequence_shap import SequenceAttribution

    sequence = SequenceAttribution(
        method="test",
        feature_names=list(SEQUENCE_FEATURES),
        mean_abs=np.arange(len(SEQUENCE_FEATURES), dtype=float),
        mean_signed=np.zeros(len(SEQUENCE_FEATURES)),
        n_explained=10,
    )
    tabular = {"n_views": 1.0, "price_mean": 2.0, "price_max": 1.0}

    seq_vec, tab_vec, labels = align_for_h4(sequence, tabular)

    assert len(seq_vec) == len(tab_vec) == len(labels)
    # event_type_cart has no counterpart and must be excluded, not zero-matched.
    assert "event_type_cart" not in labels
    assert "log_price" in labels


def test_alignment_sums_matched_tabular_features() -> None:
    """Shapley values are additive, so a concept's importance is the sum."""
    from funnel_shap.explain.sequence_shap import SequenceAttribution

    sequence = SequenceAttribution(
        method="test",
        feature_names=["log_price"],
        mean_abs=np.array([3.0]),
        mean_signed=np.array([0.0]),
        n_explained=5,
    )
    seq_vec, tab_vec, labels = align_for_h4(sequence, {"price_mean": 2.0, "price_max": 1.5})

    assert labels == ["log_price"]
    assert tab_vec[0] == pytest.approx(3.5)


def test_mapping_covers_every_sequence_feature() -> None:
    assert set(SEQUENCE_TO_TABULAR) == set(SEQUENCE_FEATURES)
