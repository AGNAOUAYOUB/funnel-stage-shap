"""Baseline model factory and preprocessing (protocol Sec. 7.5, 9.1, 9.4).

Sec. 9.1 names five baselines: Logistic Regression as the interpretable
reference, Random Forest, and the three gradient-boosting libraries. The tree
ensembles double as the TreeSHAP backbone for Sec. 11.1.

Two protocol rules are structural here, not conventions:

* **Every transformer is fitted inside the pipeline** (Sec. 7.5), so when the
  pipeline is cross-validated the encoders and scalers see only the training
  fold. Fitting an encoder on the full frame before splitting is the most common
  silent leak in tabular work and it is impossible to commit through this API.
* **Resampling happens on the training fold only** (Sec. 9.4). SMOTE is wired
  through `imblearn`'s pipeline, which — unlike scikit-learn's — applies
  resamplers during `fit` and skips them during `predict`. Using a plain sklearn
  Pipeline here would resample the validation data too and produce meaningless,
  optimistic numbers.

Scaling is applied for Logistic Regression and the neural model and skipped for
trees, which are invariant to monotone feature transforms (Sec. 7.5).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

#: Sec. 9.1. Keys are the `model.type` values accepted by RunConfig.
BASELINE_MODELS: tuple[str, ...] = (
    "logreg",
    "random_forest",
    "xgboost",
    "lightgbm",
    "catboost",
)

#: Models that need standardised inputs (Sec. 7.5).
NEEDS_SCALING: frozenset[str] = frozenset({"logreg"})

#: Models whose native categorical handling or TreeSHAP support means one-hot
#: encoding only adds sparsity.
TREE_MODELS: frozenset[str] = frozenset({"random_forest", "xgboost", "lightgbm", "catboost"})

#: Above this many levels, one-hot is counterproductive (Sec. 7.5).
HIGH_CARDINALITY_THRESHOLD = 15


class ModelError(ValueError):
    """Raised when a model cannot be built as configured."""


def build_model(
    model_type: str,
    *,
    seed: int,
    params: dict[str, Any] | None = None,
    class_weight: bool = True,
) -> Any:
    """Instantiate a baseline with its seed and imbalance handling set.

    ``class_weight`` maps to each library's own idiom, since they disagree:
    scikit-learn takes ``class_weight='balanced'``, XGBoost takes a scalar
    ``scale_pos_weight``, LightGBM accepts ``is_unbalance``, CatBoost takes
    ``auto_class_weights``. Getting this wrong is invisible — the model trains
    fine and just underperforms on the minority class.
    """
    params = dict(params or {})

    if model_type == "logreg":
        return LogisticRegression(
            max_iter=params.pop("max_iter", 2000),
            C=params.pop("C", 1.0),
            class_weight="balanced" if class_weight else None,
            random_state=seed,
            n_jobs=None,
            **params,
        )

    if model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=params.pop("n_estimators", 500),
            max_depth=params.pop("max_depth", None),
            min_samples_leaf=params.pop("min_samples_leaf", 5),
            class_weight="balanced_subsample" if class_weight else None,
            random_state=seed,
            n_jobs=params.pop("n_jobs", -1),
            **params,
        )

    if model_type == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=params.pop("n_estimators", 600),
            learning_rate=params.pop("learning_rate", 0.05),
            max_depth=params.pop("max_depth", 6),
            subsample=params.pop("subsample", 0.9),
            colsample_bytree=params.pop("colsample_bytree", 0.9),
            eval_metric=params.pop("eval_metric", "aucpr"),
            tree_method=params.pop("tree_method", "hist"),
            random_state=seed,
            n_jobs=params.pop("n_jobs", -1),
            **params,
        )

    if model_type == "lightgbm":
        from lightgbm import LGBMClassifier

        # LightGBM ignores `subsample` unless `subsample_freq > 0`. Left at its
        # default of 0 the bagging fraction is silently inert, which removes the
        # main source of seed-to-seed variation and makes the reported +/- std
        # across seeds an understatement of real variability (Sec. 6.2).
        return LGBMClassifier(
            n_estimators=params.pop("n_estimators", 800),
            learning_rate=params.pop("learning_rate", 0.03),
            num_leaves=params.pop("num_leaves", 63),
            subsample=params.pop("subsample", 0.9),
            subsample_freq=params.pop("subsample_freq", 1),
            colsample_bytree=params.pop("colsample_bytree", 0.9),
            is_unbalance=class_weight,
            random_state=seed,
            n_jobs=params.pop("n_jobs", -1),
            verbose=params.pop("verbose", -1),
            **params,
        )

    if model_type == "catboost":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            iterations=params.pop("iterations", 800),
            learning_rate=params.pop("learning_rate", 0.05),
            depth=params.pop("depth", 6),
            auto_class_weights="Balanced" if class_weight else None,
            random_seed=seed,
            verbose=params.pop("verbose", False),
            allow_writing_files=False,
            **params,
        )

    raise ModelError(
        f"unknown model type {model_type!r}; baselines are {BASELINE_MODELS} (Sec. 9.1)"
    )


def scale_pos_weight(y: np.ndarray) -> float:
    """XGBoost's imbalance idiom: negatives per positive."""
    y = np.asarray(y).ravel()
    n_pos = int(y.sum())
    if n_pos == 0:
        raise ModelError("cannot compute scale_pos_weight with no positive samples")
    return float((len(y) - n_pos) / n_pos)


def build_preprocessor(
    numeric: list[str],
    categorical: list[str],
    *,
    scale: bool,
    high_cardinality_threshold: int = HIGH_CARDINALITY_THRESHOLD,
) -> ColumnTransformer:
    """Column transformer fitted inside the pipeline, never on the full frame.

    Imputation is median for numeric and most-frequent for categorical, per
    Sec. 7.1 step 2; both are fitted on the training fold only by construction.
    """
    numeric_steps: list[tuple[str, Any]] = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))

    transformers: list[tuple[str, Any, list[str]]] = [
        ("numeric", Pipeline(numeric_steps), numeric)
    ]

    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "encode",
                            OneHotEncoder(
                                handle_unknown="infrequent_if_exist",
                                max_categories=high_cardinality_threshold,
                                sparse_output=False,
                                min_frequency=0.01,
                            ),
                        ),
                    ]
                ),
                categorical,
            )
        )

    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def build_pipeline(
    model_type: str,
    numeric: list[str],
    categorical: list[str],
    *,
    seed: int,
    params: dict[str, Any] | None = None,
    imbalance: str = "class_weight",
) -> Pipeline:
    """Preprocessing + optional resampling + estimator, as one fittable object.

    Returns an imblearn Pipeline when SMOTE is requested so the resampler runs
    during `fit` and is bypassed at `predict` time (Sec. 9.4). Everything the
    protocol requires to be fitted on training folds only lives inside this
    object, so cross-validation cannot leak.
    """
    if imbalance not in {"class_weight", "smote", "none"}:
        raise ModelError(f"unknown imbalance method {imbalance!r}")

    preprocessor = build_preprocessor(
        numeric, categorical, scale=model_type in NEEDS_SCALING
    )
    estimator = build_model(
        model_type, seed=seed, params=params, class_weight=imbalance == "class_weight"
    )

    if imbalance == "smote":
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline as ImbPipeline

        return ImbPipeline(
            [
                ("preprocess", preprocessor),
                ("resample", SMOTE(random_state=seed)),
                ("model", estimator),
            ]
        )

    return Pipeline([("preprocess", preprocessor), ("model", estimator)])
