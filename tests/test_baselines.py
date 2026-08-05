"""Tests for the baseline factory and preprocessing (protocol Sec. 7.5, 9.1, 9.4).

The leakage tests here matter as much as the ones in test_journey: fitting an
encoder or a resampler outside the training fold produces a model that scores
well and means nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

from funnel_shap.models.baselines import (
    BASELINE_MODELS,
    ModelError,
    build_model,
    build_pipeline,
    scale_pos_weight,
)


@pytest.fixture
def tabular():
    """Imbalanced frame with numeric + categorical columns, like Dataset A."""
    rng = np.random.default_rng(42)
    n = 900
    y = rng.binomial(1, 0.15, size=n)
    frame = pd.DataFrame(
        {
            "num_signal": rng.normal(loc=y * 1.5, scale=1.0, size=n),
            "num_noise": rng.normal(size=n),
            "cat_low": rng.choice(["a", "b", "c"], size=n),
            "cat_high": rng.choice([f"lvl{i}" for i in range(40)], size=n),
        }
    )
    return frame, y, ["num_signal", "num_noise"], ["cat_low", "cat_high"]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_type", BASELINE_MODELS)
def test_every_protocol_baseline_builds(model_type: str) -> None:
    model = build_model(model_type, seed=17)
    assert model is not None


@pytest.mark.parametrize("model_type", BASELINE_MODELS)
def test_seed_reaches_every_library(model_type: str) -> None:
    """Each library spells it differently; a missed one silently breaks Sec. 6.2."""
    model = build_model(model_type, seed=23)
    params = model.get_params()
    seed_value = params.get("random_state", params.get("random_seed"))
    assert seed_value == 23, f"{model_type} did not receive the seed"


def test_unknown_model_is_rejected() -> None:
    with pytest.raises(ModelError, match="unknown model type"):
        build_model("transformer_xl", seed=7)


def test_scale_pos_weight_matches_the_negative_positive_ratio() -> None:
    y = np.array([1] * 15 + [0] * 85)
    assert scale_pos_weight(y) == pytest.approx(85 / 15)


def test_scale_pos_weight_rejects_all_negative() -> None:
    with pytest.raises(ModelError, match="no positive samples"):
        scale_pos_weight(np.zeros(10))


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def _numeric_steps(pipeline) -> dict:
    """ColumnTransformer.transformers holds (name, transformer, columns) triples."""
    for name, transformer, _ in pipeline.named_steps["preprocess"].transformers:
        if name == "numeric":
            return transformer.named_steps
    raise AssertionError("no numeric transformer found")


def test_logreg_is_scaled_and_trees_are_not(tabular) -> None:
    _, _, numeric, categorical = tabular

    assert "scale" in _numeric_steps(build_pipeline("logreg", numeric, categorical, seed=7))
    assert "scale" not in _numeric_steps(
        build_pipeline("lightgbm", numeric, categorical, seed=7)
    )


def test_high_cardinality_is_capped_not_exploded(tabular) -> None:
    """40 levels one-hot encoded would add 40 sparse columns to a 900-row frame."""
    frame, y, numeric, categorical = tabular
    pipeline = build_pipeline("logreg", numeric, categorical, seed=7)
    pipeline.fit(frame, y)

    n_features = pipeline.named_steps["preprocess"].transform(frame).shape[1]
    assert n_features < 30


def test_unseen_category_at_predict_time_does_not_crash(tabular) -> None:
    frame, y, numeric, categorical = tabular
    pipeline = build_pipeline("logreg", numeric, categorical, seed=7)
    pipeline.fit(frame, y)

    novel = frame.head(5).copy()
    novel["cat_low"] = "never_seen_before"
    proba = pipeline.predict_proba(novel)[:, 1]
    assert np.all(np.isfinite(proba))


def test_missing_values_are_imputed(tabular) -> None:
    frame, y, numeric, categorical = tabular
    holed = frame.copy()
    holed.loc[:50, "num_signal"] = np.nan
    holed.loc[:50, "cat_low"] = None

    pipeline = build_pipeline("logreg", numeric, categorical, seed=7)
    pipeline.fit(holed, y)
    assert np.all(np.isfinite(pipeline.predict_proba(holed)[:, 1]))


# ---------------------------------------------------------------------------
# Leakage: everything fits inside the fold
# ---------------------------------------------------------------------------


def test_pipeline_cross_validates_without_leaking(tabular) -> None:
    """If a transformer were fitted outside the fold this would score too well."""
    frame, y, numeric, categorical = tabular
    pipeline = build_pipeline("logreg", numeric, categorical, seed=7)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, frame, y, cv=cv, scoring="average_precision")
    assert np.all(scores > 0)
    assert np.all(scores < 1.0)


def test_pure_noise_features_score_near_prevalence(tabular) -> None:
    """The canary: a model fitted on noise must not beat the base rate."""
    frame, y, numeric, categorical = tabular
    rng = np.random.default_rng(0)
    noise = frame.copy()
    noise["num_signal"] = rng.normal(size=len(noise))

    pipeline = build_pipeline("logreg", numeric, categorical, seed=7)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, noise, y, cv=cv, scoring="average_precision")

    # Chance level for PR-AUC is the prevalence.
    assert scores.mean() < y.mean() + 0.10


def test_smote_uses_an_imblearn_pipeline(tabular) -> None:
    """A plain sklearn Pipeline would resample validation folds too (Sec. 9.4)."""
    from imblearn.pipeline import Pipeline as ImbPipeline

    _, _, numeric, categorical = tabular
    pipeline = build_pipeline("logreg", numeric, categorical, seed=7, imbalance="smote")
    assert isinstance(pipeline, ImbPipeline)
    assert "resample" in pipeline.named_steps


def test_smote_does_not_resample_at_predict_time(tabular) -> None:
    frame, y, numeric, categorical = tabular
    pipeline = build_pipeline("logreg", numeric, categorical, seed=7, imbalance="smote")
    pipeline.fit(frame, y)

    # One prediction per input row: no synthetic rows leak into scoring.
    assert len(pipeline.predict_proba(frame)) == len(frame)


def test_unknown_imbalance_method_is_rejected(tabular) -> None:
    _, _, numeric, categorical = tabular
    with pytest.raises(ModelError, match="unknown imbalance method"):
        build_pipeline("logreg", numeric, categorical, seed=7, imbalance="adasyn")


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_type", ["logreg", "random_forest", "lightgbm"])
def test_same_seed_gives_identical_predictions(tabular, model_type: str) -> None:
    frame, y, numeric, categorical = tabular

    a = build_pipeline(model_type, numeric, categorical, seed=17).fit(frame, y)
    b = build_pipeline(model_type, numeric, categorical, seed=17).fit(frame, y)

    np.testing.assert_allclose(
        a.predict_proba(frame)[:, 1], b.predict_proba(frame)[:, 1], rtol=1e-12
    )


def test_a_signal_carrying_model_beats_prevalence(tabular) -> None:
    frame, y, numeric, categorical = tabular
    pipeline = build_pipeline("lightgbm", numeric, categorical, seed=7)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, frame, y, cv=cv, scoring="average_precision")
    assert scores.mean() > y.mean() + 0.10


def test_class_weight_reaches_each_library() -> None:
    assert build_model("logreg", seed=7).class_weight == "balanced"
    assert build_model("random_forest", seed=7).class_weight == "balanced_subsample"
    assert build_model("lightgbm", seed=7).is_unbalance is True
    assert build_model("logreg", seed=7, class_weight=False).class_weight is None


def test_predictions_are_probabilities(tabular) -> None:
    frame, y, numeric, categorical = tabular
    pipeline = build_pipeline("xgboost", numeric, categorical, seed=7).fit(frame, y)
    proba = pipeline.predict_proba(frame)[:, 1]

    assert np.all((proba >= 0) & (proba <= 1))
    assert average_precision_score(y, proba) > y.mean()
