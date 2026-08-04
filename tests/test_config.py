"""Tests for the run-config schema (protocol Appendix A, Sec. 6.2).

The point of validating configs is that an out-of-protocol run should fail at
load time, before it burns compute and before its numbers can end up in a table.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from funnel_shap.config import RunConfig, load_config
from funnel_shap.paths import CONFIGS
from funnel_shap.seeds import SEEDS, set_global_seed


def _minimal(**overrides) -> dict:
    base = {
        "name": "test",
        "seed": 17,
        "data": {"dataset": "B", "stage": "S3"},
        "model": {"type": "lightgbm"},
    }
    base.update(overrides)
    return base


def test_appendix_a_template_validates() -> None:
    config = load_config(CONFIGS / "base.yaml")
    assert config.name == "stageS3_lightgbm_seed17"
    assert config.seed == 17
    assert config.data.stage == "S3"
    assert config.explain.tree_shap_mode == "interventional"
    assert config.eval.bootstrap_resamples >= 2000


def test_off_protocol_seed_is_rejected() -> None:
    with pytest.raises(ValidationError, match="frozen seed list"):
        RunConfig(**_minimal(seed=1234))


def test_static_stage_on_dataset_b_is_rejected() -> None:
    """Whole-session aggregates on the event data would leak the label (Sec. 7.4)."""
    with pytest.raises(ValidationError, match="leak the label"):
        RunConfig(**_minimal(data={"dataset": "B", "stage": "static"}))


def test_event_stage_on_dataset_a_is_rejected() -> None:
    with pytest.raises(ValidationError, match="session-level aggregate"):
        RunConfig(**_minimal(data={"dataset": "A", "stage": "S2"}))


def test_dataset_a_static_is_accepted() -> None:
    config = RunConfig(**_minimal(data={"dataset": "A", "stage": "static"}))
    assert config.data.stage == "static"


def test_path_dependent_shap_is_rejected() -> None:
    """Sec. 11.1 fixes interventional TreeSHAP to limit correlation artefacts."""
    with pytest.raises(ValidationError):
        RunConfig(**_minimal(explain={"tree_shap_mode": "path_dependent"}))


def test_too_few_bootstrap_resamples_is_rejected() -> None:
    with pytest.raises(ValidationError):
        RunConfig(**_minimal(eval={"bootstrap_resamples": 500}))


def test_unknown_key_is_rejected() -> None:
    """extra='forbid' catches a typo'd key rather than silently ignoring it."""
    with pytest.raises(ValidationError):
        RunConfig(**_minimal(calibrat="isotonic"))


def test_invalid_sessionisation_gap_is_rejected() -> None:
    with pytest.raises(ValidationError):
        RunConfig(**_minimal(data={"dataset": "B", "stage": "S3", "sessionization_gap_min": 45}))


def test_config_hash_is_stable_and_ignores_git_commit() -> None:
    a = RunConfig(**_minimal(git_commit="abc123"))
    b = RunConfig(**_minimal(git_commit="def456"))
    c = RunConfig(**_minimal(seed=23))

    assert a.config_hash == b.config_hash
    assert a.config_hash != c.config_hash


def test_mlflow_params_are_flat_scalars() -> None:
    params = RunConfig(**_minimal()).to_mlflow_params()
    assert "data.stage" in params
    assert "config_hash" in params
    assert all(not isinstance(v, (dict, list)) for v in params.values())


@pytest.mark.parametrize("seed", SEEDS)
def test_every_frozen_seed_is_settable(seed: int) -> None:
    set_global_seed(seed)


def test_off_protocol_seed_cannot_be_set() -> None:
    with pytest.raises(ValueError, match="frozen seed list"):
        set_global_seed(999)
