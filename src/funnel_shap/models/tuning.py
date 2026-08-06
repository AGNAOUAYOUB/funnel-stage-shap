"""Hyperparameter tuning with Optuna (protocol Sec. 9.5).

Sec. 9.5 offers two routes: nested cross-validation, or "a strictly held-out
validation set touched only during tuning". This module takes the second, for a
reason specific to Dataset B: the split is *temporal* (Sec. 7.6), and nested CV
would shuffle folds across the time boundary, which is exactly the leakage the
temporal protocol exists to prevent. Nested CV remains appropriate for the
static Dataset A benchmark, where there is no ordering to violate.

Search spaces are fixed here rather than passed in, so a run cannot quietly
widen them; Sec. 9.5 requires them logged, and a literal in source is the most
auditable form of logging. Studies persist to `experiments/optuna/` as SQLite
so a reviewer can reopen a completed search.

**Known tension, stated rather than hidden.** Sec. 9.5 wants validation
"touched only during tuning", but Sec. 10 also selects the operating threshold
on validation. Both cannot hold literally. Tuning selects among models on
PR-AUC, which is threshold-free, while thresholding is a downstream choice on
the already-selected model, so the interaction is weak — but it is a
double-use of the validation split and belongs in the limitations section
rather than in a footnote nobody reads.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score

from ..paths import OPTUNA
from ..seeds import set_global_seed
from .baselines import build_pipeline

#: Fixed per-model search spaces (Sec. 9.5). Each entry maps a parameter to a
#: callable taking an Optuna trial. Ranges are deliberately conservative: wide
#: enough to matter, narrow enough that 100 trials can cover them.
SEARCH_SPACES: dict[str, dict[str, Callable[[Any], Any]]] = {
    "lightgbm": {
        "n_estimators": lambda t: t.suggest_int("n_estimators", 200, 1200, step=100),
        "learning_rate": lambda t: t.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": lambda t: t.suggest_int("num_leaves", 15, 255, log=True),
        "min_child_samples": lambda t: t.suggest_int("min_child_samples", 5, 200, log=True),
        "subsample": lambda t: t.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": lambda t: t.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": lambda t: t.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    },
    "xgboost": {
        "n_estimators": lambda t: t.suggest_int("n_estimators", 200, 1200, step=100),
        "learning_rate": lambda t: t.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": lambda t: t.suggest_int("max_depth", 3, 12),
        "min_child_weight": lambda t: t.suggest_float("min_child_weight", 1e-2, 20.0, log=True),
        "subsample": lambda t: t.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": lambda t: t.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": lambda t: t.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    },
    "catboost": {
        "iterations": lambda t: t.suggest_int("iterations", 200, 1200, step=100),
        "learning_rate": lambda t: t.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "depth": lambda t: t.suggest_int("depth", 4, 10),
        "l2_leaf_reg": lambda t: t.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),
    },
    "random_forest": {
        "n_estimators": lambda t: t.suggest_int("n_estimators", 200, 1000, step=100),
        "max_depth": lambda t: t.suggest_int("max_depth", 4, 40),
        "min_samples_leaf": lambda t: t.suggest_int("min_samples_leaf", 1, 50, log=True),
        "max_features": lambda t: t.suggest_float("max_features", 0.2, 1.0),
    },
    "logreg": {
        "C": lambda t: t.suggest_float("C", 1e-3, 1e2, log=True),
    },
}


@dataclass
class TuningResult:
    model: str
    stage: str
    best_params: dict[str, Any]
    best_value: float
    n_trials: int
    n_completed: int
    n_pruned: int
    seed: int
    study_name: str

    def summary(self) -> str:
        return (
            f"{self.stage}/{self.model}: PR-AUC {self.best_value:.4f} "
            f"({self.n_completed} completed, {self.n_pruned} pruned) {self.best_params}"
        )


def tune_model(
    model_type: str,
    X_train,
    y_train: np.ndarray,
    X_val,
    y_val: np.ndarray,
    *,
    columns: list[str],
    stage: str = "static",
    n_trials: int = 100,
    seed: int = 42,
    imbalance: str = "class_weight",
    storage_dir: Path = OPTUNA,
    timeout: int | None = None,
) -> TuningResult:
    """TPE search over the fixed space, maximising validation PR-AUC.

    The objective is PR-AUC because Sec. 10 makes it the primary metric;
    optimising ROC-AUC and reporting PR-AUC would tune for a different target
    than the one being claimed.
    """
    import optuna

    if model_type not in SEARCH_SPACES:
        raise ValueError(
            f"no search space for {model_type!r}; spaces are fixed per model (Sec. 9.5)"
        )

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    space = SEARCH_SPACES[model_type]
    set_global_seed(seed)

    def objective(trial) -> float:
        params = {name: fn(trial) for name, fn in space.items()}
        pipeline = build_pipeline(
            model_type, columns, [], seed=seed, params=params, imbalance=imbalance
        )
        pipeline.fit(X_train, y_train)
        scores = pipeline.predict_proba(X_val)[:, 1]
        return float(average_precision_score(y_val, scores))

    storage_dir.mkdir(parents=True, exist_ok=True)
    study_name = f"{stage}_{model_type}_seed{seed}"
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
        storage=f"sqlite:///{(storage_dir / f'{study_name}.db').as_posix()}",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

    return _result_from_study(study, model_type=model_type, stage=stage, seed=seed)


def load_tuned_params(
    model_type: str,
    stage: str,
    *,
    seed: int = 42,
    storage_dir: Path = OPTUNA,
) -> dict[str, Any]:
    """Reopen a persisted study and return its best parameters (Sec. 9.5)."""
    import optuna

    study_name = f"{stage}_{model_type}_seed{seed}"
    path = storage_dir / f"{study_name}.db"
    if not path.exists():
        raise FileNotFoundError(f"no completed study at {path}; run tuning first")
    study = optuna.load_study(
        study_name=study_name, storage=f"sqlite:///{path.as_posix()}"
    )
    return dict(study.best_params)


def _result_from_study(study, *, model_type: str, stage: str, seed: int) -> TuningResult:
    states = [t.state.name for t in study.trials]
    return TuningResult(
        model=model_type,
        stage=stage,
        best_params=dict(study.best_params),
        best_value=float(study.best_value),
        n_trials=len(study.trials),
        n_completed=states.count("COMPLETE"),
        n_pruned=states.count("PRUNED"),
        seed=seed,
        study_name=study.study_name,
    )
