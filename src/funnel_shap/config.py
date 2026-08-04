"""Run configuration schema (protocol Appendix A).

One config object per run, validated on load and logged to MLflow together with
its content hash and the git commit. Anything the protocol fixes before freeze
is expressed as a constrained field here, so an out-of-protocol run fails at
load time rather than silently producing an unusable result.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .seeds import SEEDS

Stage = Literal["S1", "S2", "S3", "S4", "static"]
Dataset = Literal["A", "B"]
SplitProtocol = Literal["temporal", "grouped"]


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DataConfig(_Base):
    dataset: Dataset
    stage: Stage
    split: SplitProtocol = "temporal"
    sessionization_gap_min: Literal[15, 30, 60] = 30

    @field_validator("stage")
    @classmethod
    def _stage_matches_dataset(cls, v: Stage, info: Any) -> Stage:
        dataset = info.data.get("dataset")
        if dataset == "A" and v != "static":
            raise ValueError(
                "Dataset A is session-level aggregate; only stage='static' is defined for it "
                "(Sec. 5.2/7.4). Event-level staging requires Dataset B."
            )
        if dataset == "B" and v == "static":
            raise ValueError(
                "stage='static' on Dataset B would use whole-session aggregates and leak the "
                "label (Sec. 7.4). Use S1-S4."
            )
        return v


class ModelConfig(_Base):
    type: Literal[
        "logreg", "random_forest", "xgboost", "lightgbm", "catboost", "gru", "transformer"
    ]
    params: dict[str, Any] = Field(default_factory=dict)
    calibrate: Literal["isotonic", "sigmoid", "none"] = "isotonic"


class ImbalanceConfig(_Base):
    # Applied inside training folds only (Sec. 9.4).
    method: Literal["class_weight", "smote", "none"] = "class_weight"


class TuningConfig(_Base):
    optuna_trials: int = Field(100, ge=1)
    nested_cv_outer: int = Field(5, ge=2)
    nested_cv_inner: int = Field(3, ge=2)


class ExplainConfig(_Base):
    # Interventional, not path-dependent (Sec. 11.1).
    tree_shap_mode: Literal["interventional"] = "interventional"
    background_size: int = Field(2000, ge=100)
    faithfulness: list[Literal["deletion", "insertion", "ablation"]] = Field(
        default_factory=lambda: ["deletion", "insertion", "ablation"]
    )
    # Pass thresholds pre-specified before running (Sec. 11.3).
    faithfulness_threshold: float = 0.5
    cross_paradigm_spearman_threshold: float = 0.6


class EvalConfig(_Base):
    primary_metric: Literal["pr_auc"] = "pr_auc"
    bootstrap_resamples: int = Field(2000, ge=2000)
    alpha: float = Field(0.05, gt=0, lt=1)
    correction: Literal["holm", "fdr_bh"] = "holm"


class RunConfig(_Base):
    name: str
    seed: int
    data: DataConfig
    model: ModelConfig
    imbalance: ImbalanceConfig = Field(default_factory=ImbalanceConfig)
    tuning: TuningConfig = Field(default_factory=TuningConfig)
    explain: ExplainConfig = Field(default_factory=ExplainConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    git_commit: str | None = None

    @field_validator("seed")
    @classmethod
    def _seed_in_list(cls, v: int) -> int:
        if v not in SEEDS:
            raise ValueError(f"seed {v} is not in the frozen seed list {SEEDS} (Appendix B)")
        return v

    @property
    def config_hash(self) -> str:
        payload = self.model_dump(mode="json", exclude={"git_commit"})
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    def to_mlflow_params(self) -> dict[str, Any]:
        """Flatten to the dotted key/value pairs MLflow stores as params."""
        flat: dict[str, Any] = {}

        def walk(prefix: str, value: Any) -> None:
            if isinstance(value, dict):
                for k, v in value.items():
                    walk(f"{prefix}.{k}" if prefix else k, v)
            elif isinstance(value, list):
                flat[prefix] = ",".join(str(x) for x in value)
            else:
                flat[prefix] = value

        walk("", self.model_dump(mode="json"))
        flat["config_hash"] = self.config_hash
        return flat


def current_git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return out.stdout.strip() or None


def load_config(path: str | Path) -> RunConfig:
    """Load and validate a run YAML, stamping the current git commit."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if "run" in raw and len(raw) == 1:
        raw = raw["run"]
    if raw.get("git_commit") in (None, "<auto>"):
        raw["git_commit"] = current_git_commit()
    return RunConfig(**raw)
