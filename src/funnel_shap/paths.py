"""Canonical project paths (protocol Sec. 6.1).

Every module resolves filesystem locations through here so that a run can be
relocated by setting FUNNEL_SHAP_ROOT rather than by editing code.
"""

from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    env = os.environ.get("FUNNEL_SHAP_ROOT")
    if env:
        return Path(env).resolve()
    # src/funnel_shap/paths.py -> src/funnel_shap -> src -> root
    return Path(__file__).resolve().parents[2]


ROOT = project_root()

CONFIGS = ROOT / "configs"
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
SPLITS = PROCESSED / "splits"
EXPERIMENTS = ROOT / "experiments"
MLRUNS = EXPERIMENTS / "mlruns"
OPTUNA = EXPERIMENTS / "optuna"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
TABLES = REPORTS / "tables"

# Dataset A — UCI Online Shoppers Purchasing Intention (Sec. 5.2)
DATASET_A_RAW = RAW / "online_shoppers_intention.csv"

# Dataset B — REES46-class event log (Sec. 5.3). A directory because the source
# ships one CSV per month; the ingest layer globs it.
DATASET_B_RAW = RAW / "rees46"


def ensure_dirs() -> None:
    """Create the writable output directories if they are missing."""
    for d in (RAW, INTERIM, PROCESSED, SPLITS, MLRUNS, OPTUNA, FIGURES, TABLES):
        d.mkdir(parents=True, exist_ok=True)
