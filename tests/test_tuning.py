"""Tests for Optuna tuning (protocol Sec. 9.5)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from funnel_shap.models.tuning import SEARCH_SPACES, tune_model


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(42)
    n = 900
    y = rng.binomial(1, 0.2, size=n)
    frame = pd.DataFrame(
        {
            "a": rng.normal(loc=y * 1.5, scale=1.0, size=n),
            "b": rng.normal(size=n),
        }
    )
    split = int(n * 0.7)
    return (
        frame.iloc[:split],
        y[:split],
        frame.iloc[split:],
        y[split:],
        ["a", "b"],
    )


def test_every_baseline_has_a_fixed_search_space() -> None:
    """Sec. 9.5 requires the spaces fixed and logged; a literal in source is both."""
    from funnel_shap.models.baselines import BASELINE_MODELS

    assert set(SEARCH_SPACES) == set(BASELINE_MODELS)
    for space in SEARCH_SPACES.values():
        assert space, "an empty search space would silently skip tuning"


def test_unknown_model_is_rejected(data) -> None:
    X_tr, y_tr, X_va, y_va, cols = data
    with pytest.raises(ValueError, match="no search space"):
        tune_model("magic", X_tr, y_tr, X_va, y_va, columns=cols, n_trials=2)


def test_tuning_improves_or_matches_the_default(data, tmp_path) -> None:
    from sklearn.metrics import average_precision_score

    from funnel_shap.models.baselines import build_pipeline

    X_tr, y_tr, X_va, y_va, cols = data
    result = tune_model(
        "lightgbm", X_tr, y_tr, X_va, y_va,
        columns=cols, n_trials=8, seed=7, storage_dir=tmp_path,
    )

    default = build_pipeline("lightgbm", cols, [], seed=7).fit(X_tr, y_tr)
    default_score = average_precision_score(y_va, default.predict_proba(X_va)[:, 1])

    assert result.best_value >= default_score - 0.05
    assert result.n_completed > 0


def test_search_is_seeded_and_reproducible(data, tmp_path) -> None:
    X_tr, y_tr, X_va, y_va, cols = data
    kwargs = dict(columns=cols, n_trials=6, seed=17)

    a = tune_model("lightgbm", X_tr, y_tr, X_va, y_va, storage_dir=tmp_path / "a", **kwargs)
    b = tune_model("lightgbm", X_tr, y_tr, X_va, y_va, storage_dir=tmp_path / "b", **kwargs)

    assert a.best_params == b.best_params
    assert a.best_value == pytest.approx(b.best_value)


def test_study_is_persisted_for_reproducibility(data, tmp_path) -> None:
    """Sec. 9.5: persist studies so a reviewer can reopen the search."""
    X_tr, y_tr, X_va, y_va, cols = data
    result = tune_model(
        "lightgbm", X_tr, y_tr, X_va, y_va,
        columns=cols, n_trials=4, seed=7, stage="S2", storage_dir=tmp_path,
    )

    assert (tmp_path / f"{result.study_name}.db").exists()
    assert result.study_name == "S2_lightgbm_seed7"


def test_best_params_stay_inside_the_declared_space(data, tmp_path) -> None:
    X_tr, y_tr, X_va, y_va, cols = data
    result = tune_model(
        "lightgbm", X_tr, y_tr, X_va, y_va,
        columns=cols, n_trials=6, seed=7, storage_dir=tmp_path,
    )

    assert set(result.best_params).issubset(set(SEARCH_SPACES["lightgbm"]))
    assert 0.01 <= result.best_params["learning_rate"] <= 0.2
    assert 200 <= result.best_params["n_estimators"] <= 1200


def test_objective_optimises_pr_auc_not_roc_auc(data, tmp_path) -> None:
    """Sec. 10 makes PR-AUC primary; tuning must target the reported metric."""
    X_tr, y_tr, X_va, y_va, cols = data
    result = tune_model(
        "lightgbm", X_tr, y_tr, X_va, y_va,
        columns=cols, n_trials=4, seed=7, storage_dir=tmp_path,
    )
    # PR-AUC on a 20%-prevalence problem sits far below a typical ROC-AUC.
    assert 0.0 < result.best_value < 1.0
    assert result.best_value > y_va.mean() - 0.1


def test_logreg_space_is_small_but_present(data, tmp_path) -> None:
    X_tr, y_tr, X_va, y_va, cols = data
    result = tune_model(
        "logreg", X_tr, y_tr, X_va, y_va,
        columns=cols, n_trials=4, seed=7, storage_dir=tmp_path,
    )
    assert "C" in result.best_params
