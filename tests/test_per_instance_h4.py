"""Tests for the per-instance H4 comparison (amendment A22)."""

from __future__ import annotations

import numpy as np
import pytest

from funnel_shap.explain.per_instance_h4 import (
    compare_per_instance,
    per_instance_table,
    summarise,
)
from funnel_shap.models.sequence import SEQUENCE_FEATURES

TREE_FEATURES = [
    "n_views",
    "price_mean",
    "price_max",
    "inter_event_mean_s",
    "last_gap_s",
    "prefix_duration_s",
    "n_unique_products",
    "product_revisit_rate",
]


def _arrays(n: int = 300, agreement: float = 1.0, seed: int = 0):
    """Tree and sequence attributions with a controllable level of agreement."""
    rng = np.random.default_rng(seed)
    ids = [f"s{i}" for i in range(n)]

    tree = rng.normal(size=(n, len(TREE_FEATURES)))
    sequence = rng.normal(size=(n, len(SEQUENCE_FEATURES)))

    # Make the price concept agree to the requested degree: the sequence's
    # log_price column becomes a mix of the tree's summed price columns and noise.
    price_cols = [TREE_FEATURES.index("price_mean"), TREE_FEATURES.index("price_max")]
    tree_price = tree[:, price_cols].sum(axis=1)
    seq_price_col = list(SEQUENCE_FEATURES).index("log_price")
    sequence[:, seq_price_col] = (
        agreement * tree_price + (1 - agreement) * rng.normal(size=n)
    )
    return tree, ids, sequence, ids


def test_perfect_agreement_is_detected() -> None:
    tree, tree_ids, sequence, seq_ids = _arrays(agreement=1.0)
    results = compare_per_instance(
        tree, TREE_FEATURES, tree_ids, sequence, list(SEQUENCE_FEATURES), seq_ids,
        stage="S1",
    )
    price = next(r for r in results if r.concept == "log_price")
    assert price.spearman > 0.95
    assert price.n_sessions == 300


def test_no_agreement_is_detected() -> None:
    """The metric must be able to fail."""
    tree, tree_ids, sequence, seq_ids = _arrays(agreement=0.0, seed=3)
    results = compare_per_instance(
        tree, TREE_FEATURES, tree_ids, sequence, list(SEQUENCE_FEATURES), seq_ids,
        stage="S1",
    )
    price = next(r for r in results if r.concept == "log_price")
    assert abs(price.spearman) < 0.2


def test_only_shared_sessions_are_compared() -> None:
    """Comparing different sessions would measure sampling noise, not agreement."""
    tree, tree_ids, sequence, _ = _arrays(agreement=1.0)
    seq_ids = [f"s{i}" for i in range(150, 450)]  # half overlap

    results = compare_per_instance(
        tree, TREE_FEATURES, tree_ids, sequence, list(SEQUENCE_FEATURES), seq_ids,
        stage="S1",
    )
    assert results
    assert all(r.n_sessions == 150 for r in results)


def test_too_few_shared_sessions_returns_nothing() -> None:
    tree, tree_ids, sequence, _ = _arrays()
    results = compare_per_instance(
        tree, TREE_FEATURES, tree_ids, sequence, list(SEQUENCE_FEATURES),
        [f"s{i}" for i in range(295, 305)], stage="S1",
    )
    assert results == []


def test_concepts_without_a_tree_counterpart_are_skipped() -> None:
    tree, tree_ids, sequence, seq_ids = _arrays()
    results = compare_per_instance(
        tree, TREE_FEATURES, tree_ids, sequence, list(SEQUENCE_FEATURES), seq_ids,
        stage="S1",
    )
    concepts = {r.concept for r in results}
    assert "event_type_cart" not in concepts
    assert "log_price" in concepts


def test_power_advantage_over_the_aggregate_test() -> None:
    """Each correlation rests on hundreds of sessions, not five concepts."""
    tree, tree_ids, sequence, seq_ids = _arrays(n=500)
    results = compare_per_instance(
        tree, TREE_FEATURES, tree_ids, sequence, list(SEQUENCE_FEATURES), seq_ids,
        stage="S3",
    )
    assert all(r.n_sessions == 500 for r in results)

    table = per_instance_table(results)
    assert table["n_sessions"].min() >= 500
    assert "500" in summarise(table)


def test_empty_results_are_rejected() -> None:
    with pytest.raises(ValueError, match="no concept"):
        per_instance_table([])


# ---------------------------------------------------------------------------
# Seed aggregation
# ---------------------------------------------------------------------------


def _seed_table(rhos: list[float], stage: str = "S2"):
    import polars as pl

    return pl.DataFrame(
        {
            "stage": [stage] * len(rhos),
            "concept": [f"c{i}" for i in range(len(rhos))],
            "tree_features": ["x"] * len(rhos),
            "spearman": rhos,
            "pearson": rhos,
            "n_sessions": [600] * len(rhos),
            "passes": [False] * len(rhos),
        }
    )


def test_sign_consistency_separates_a_finding_from_noise() -> None:
    """A mean near zero with a flipping sign is noise; a stable sign is a result."""
    from funnel_shap.explain.per_instance_h4 import aggregate_over_seeds

    aggregated = aggregate_over_seeds(
        {
            7: _seed_table([-0.20, 0.10]),
            17: _seed_table([-0.30, -0.10]),
            42: _seed_table([-0.15, 0.20]),
        }
    )
    rows = {r["concept"]: r for r in aggregated.to_dicts()}

    assert rows["c0"]["sign_consistent"] is True
    assert rows["c0"]["rho_mean"] < 0
    assert rows["c1"]["sign_consistent"] is False


def test_aggregation_reports_spread_not_just_mean() -> None:
    from funnel_shap.explain.per_instance_h4 import aggregate_over_seeds

    aggregated = aggregate_over_seeds({7: _seed_table([0.1]), 17: _seed_table([0.5])})
    row = aggregated.to_dicts()[0]

    assert row["rho_min"] == pytest.approx(0.1)
    assert row["rho_max"] == pytest.approx(0.5)
    assert row["rho_std"] > 0
    assert row["n_seeds"] == 2


def test_empty_seed_set_is_rejected() -> None:
    from funnel_shap.explain.per_instance_h4 import aggregate_over_seeds

    with pytest.raises(ValueError, match="no seed results"):
        aggregate_over_seeds({})
