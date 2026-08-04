# Funnel-Stage SHAP

Reproducible implementation of the pre-registered protocol **"Dynamic Explainable AI for
Sequential Purchase Prediction: A Funnel-Stage SHAP Framework for Digital Consumer
Journeys"** (v1.0, Ayoub Agnaou).

The framework predicts session-level conversion at each funnel stage
(awareness → consideration → intent → conversion) using only the event prefix available
at that stage, then measures how SHAP attributions *migrate* across stages — and whether
those stage-conditioned explanations are faithful and stable.

## Protocol discipline

This repository is written for a pre-registered study. Three rules are enforced in code,
not just in prose:

1. **The test partition is opened exactly once**, at the final evaluation step. Splits are
   frozen to `data/processed/splits/` and every model and explanation reads the identical
   split.
2. **No feature may see events at or after its stage's cut-point** (protocol Sec. 7.4).
   `RunConfig` rejects `stage: static` on Dataset B for exactly this reason, and the prefix
   builder is unit-tested against leakage.
3. **No single-run numbers.** Every reported metric is a mean ± std over the frozen seed
   list `[7, 17, 23, 42, 101]` with bootstrapped 95% CIs.

Deviations after freeze go in `docs/AMENDMENTS.md` with a date, rationale, and the affected
protocol sections.

## Layout

```
funnel-shap/
├── pyproject.toml / requirements.lock   # pinned environment (Sec. 4)
├── dvc.yaml                             # pipeline stages, tracked outputs
├── configs/                             # one YAML per run, seeded (Appendix A)
├── data/
│   ├── raw/                             # DVC-tracked, read-only
│   ├── interim/                         # sessionised, cleaned
│   └── processed/                       # feature matrices + frozen splits
├── src/funnel_shap/
│   ├── data/                            # cleaning, sessionisation, journey reconstruction
│   ├── features/                        # stage prefix feature engineering
│   ├── models/                          # baselines, sequence model, tuning
│   ├── explain/                         # stage SHAP, TimeSHAP, faithfulness metrics
│   ├── evaluate/                        # metrics, statistical tests
│   └── stats/                           # vendored DeLong, bootstrap CI
├── experiments/                         # MLflow runs, Optuna studies
├── reports/figures/                     # publication figures
└── tests/                               # pytest unit tests
```

## Setup

```bash
uv venv --python 3.11 .venv
```

```bash
uv pip install --python .venv/Scripts/python.exe -e ".[dev]"
```

The sequence arm (GRU/Transformer + TimeSHAP) and the XAI evaluation layer are optional
extras so the tabular arm stays CPU-only:

```bash
uv pip install --python .venv/Scripts/python.exe -e ".[sequence,xai-eval]"
```

## Data

Neither dataset is committed. Dataset A is fetched on demand; Dataset B must be downloaded
by hand because its Kaggle licence has to be confirmed and recorded before use
(protocol Sec. 5.3).

| | Dataset A | Dataset B |
|---|---|---|
| Source | UCI Online Shoppers Purchasing Intention | REES46-class event log (Kaggle) |
| Grain | 12,330 session-level aggregates | millions of events → ≥52k sessions |
| Role | reproducible static benchmark, static-vs-stage SHAP contrast | headline stage-migration and TimeSHAP results |
| Staging | coarse aggregate proxy, flagged non-causal | true event-order cut-points |

See `data/raw/DATASETS.md` for licences, citations, and the honest variable-availability
table (protocol Sec. 5.1) — including the two variables, search-query text and scroll
depth, that no public dataset provides and which are reported as limitations rather than
proxied silently.

## Running

```bash
.venv/Scripts/python.exe -m pytest
```

Pipeline stages are DVC targets; see `dvc.yaml` and the run sheet in the protocol Sec. 13.

## Citation

If you use Dataset A, cite Sakar et al. (2019). If you use Dataset B, cite the REES46
dataset per its Kaggle terms.
