"""Command-line entry points for the pipeline (protocol Sec. 13 run sheet).

Notebooks are for exploration only; everything that produces a reported number
runs through here so it is scripted, seeded, and DVC-trackable.
"""

from __future__ import annotations

import contextlib
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import typer

from . import paths
from .config import load_config


@dataclass
class _CachedAttribution:
    """Minimal stand-in for StageAttribution when redrawing from cache.

    Only the fields the appendix figures read are carried, so a cached redraw
    cannot accidentally depend on state that was not persisted.
    """

    feature_names: list
    shap_values: np.ndarray

    def global_importance(self) -> pl.DataFrame:
        mean_abs = np.abs(self.shap_values).mean(axis=0)
        total = mean_abs.sum()
        return pl.DataFrame(
            {
                "feature": list(self.feature_names),
                "mean_abs_shap": mean_abs,
                "share": mean_abs / total if total else mean_abs,
            }
        ).sort("mean_abs_shap", descending=True)


@dataclass
class _CachedExplanation:
    attribution: _CachedAttribution
    explained_matrix: np.ndarray

# Polars renders tables with box-drawing characters, which a Windows console
# defaulting to cp1252 cannot encode -- the run completes and then dies on the
# print. Reconfiguring stdout is cheaper than stripping the output.
for _stream in (sys.stdout, sys.stderr):
    with contextlib.suppress(AttributeError, ValueError):
        _stream.reconfigure(encoding="utf-8", errors="replace")

app = typer.Typer(add_completion=False, help="Funnel-Stage SHAP pipeline")


@app.command()
def fetch_a(overwrite: bool = False) -> None:
    """Download Dataset A (UCI Online Shoppers) into data/raw/."""
    from .data.ingest import fetch_dataset_a

    dest = fetch_dataset_a(overwrite=overwrite)
    typer.echo(f"Dataset A at {dest}")


@app.command()
def register_b(path: Path) -> None:
    """Record provenance for a Dataset-B event log you downloaded yourself."""
    from .data.ingest import register_dataset_b

    prov = register_dataset_b(path)
    typer.echo(f"registered {prov.n_rows:,} events, sha256={prov.sha256[:16]}...")
    typer.secho(
        "Confirm the Kaggle licence and replace the placeholder in provenance_B.json "
        "before publication (Sec. 5.3).",
        fg=typer.colors.YELLOW,
    )


@app.command()
def tables() -> None:
    """Regenerate the two auditable protocol tables (Sec. 5.1, Sec. 8)."""
    paths.ensure_dirs()

    from .data.schema import availability_table
    from .features.dictionary import availability_by_stage

    variable_table = availability_table()
    variable_table.write_csv(paths.TABLES / "variable_availability.csv")

    stage_table = availability_by_stage()
    stage_table.write_csv(paths.TABLES / "feature_availability_by_stage.csv")

    typer.echo(f"variable availability   -> {paths.TABLES / 'variable_availability.csv'}")
    typer.echo(f"feature availability    -> {paths.TABLES / 'feature_availability_by_stage.csv'}")


@app.command()
def describe_a() -> None:
    """Print Dataset A's headline shape statistics (Sec. 5.2)."""
    from .data.dataset_a import describe
    from .data.ingest import load_dataset_a

    stats = describe(load_dataset_a())
    for key, value in stats.items():
        typer.echo(f"{key:>14}: {value:,.4f}" if key == "prevalence" else f"{key:>14}: {value:,.0f}")


_SOURCE_OPT = typer.Option(None, help="Event-log CSV/glob; omit to use data/raw/rees46/*.csv")
_SYNTHETIC_OPT = typer.Option(
    False, help="Run on the synthetic fixture instead of Dataset B (smoke test only)"
)
_SUFFIX_OPT = typer.Option("synthetic", help="Which sessionised artefact to read")


_SAMPLE_OPT = typer.Option(
    0, help="Keep only N users (0 = all). Seeded; Sec. 5.3 permits this for tractability."
)
_REUSE_OPT = typer.Option(
    False, help="Reuse an existing sessions_<suffix>.parquet instead of re-sessionising"
)


@app.command()
def sessionize(
    source: Path = _SOURCE_OPT,
    gap_minutes: int = 30,
    synthetic: bool = _SYNTHETIC_OPT,
    sample_users: int = _SAMPLE_OPT,
    reuse_sessions: bool = _REUSE_OPT,
) -> None:
    """Clean and sessionise the event stream (Sec. 7.1-7.2)."""
    paths.ensure_dirs()

    from .data.flow import DataFlow
    from .data.ingest import scan_dataset_b
    from .data.journey import stage_cutpoints, stage_prevalence_table
    from .data.sessionize import SessionizeConfig
    from .data.sessionize import sessionize as run_sessionize
    from .data.subsample import subsample_users
    from .data.synthetic import make_event_log

    base_suffix = "synthetic" if synthetic else f"gap{gap_minutes}"
    label = f"{'synthetic' if synthetic else 'Dataset B'} (gap={gap_minutes}min)"
    flow = DataFlow(label)

    source_sessions = paths.INTERIM / f"sessions_{base_suffix}.parquet"
    if reuse_sessions:
        if not source_sessions.exists():
            raise typer.BadParameter(f"no sessions file at {source_sessions}")
        typer.echo(f"reusing {source_sessions.name}")
        sessions = pl.read_parquet(source_sessions)
    else:
        lazy = make_event_log(n_users=500, seed=42).lazy() if synthetic else scan_dataset_b(source)
        sessions = run_sessionize(
            lazy, SessionizeConfig(gap_minutes=gap_minutes), flow
        ).collect()
        sessions.write_parquet(source_sessions)
        flow.write(paths.TABLES / f"data_flow_{base_suffix}.csv")
        typer.echo(flow.summary())

    suffix = base_suffix
    if sample_users > 0:
        sessions, report = subsample_users(sessions, n_users=sample_users, seed=42)
        suffix = f"{base_suffix}s{sample_users}"
        typer.echo("")
        typer.echo(report.summary())

    out = paths.INTERIM / f"sessions_{suffix}.parquet"
    if out != source_sessions:
        sessions.write_parquet(out)

    cuts = stage_cutpoints(sessions.lazy())
    cuts.write_parquet(paths.INTERIM / f"cutpoints_{suffix}.parquet")

    prevalence = stage_prevalence_table(cuts)
    prevalence.write_csv(paths.TABLES / f"stage_prevalence_{suffix}.csv")
    typer.echo("")
    typer.echo(prevalence.select(
        ["stage", "modelled", "n_sessions", "reach_rate", "prevalence"]
    ))
    typer.echo(f"\nsessions -> {out}")


@app.command()
def build_features(suffix: str = _SUFFIX_OPT) -> None:
    """Build the nested stage-prefix feature matrices (Sec. 7.4, Sec. 8)."""
    paths.ensure_dirs()

    from .data.journey import MODELLING_STAGES, prefix_events
    from .features.prefix_features import build_stage_features

    sessions = pl.read_parquet(paths.INTERIM / f"sessions_{suffix}.parquet")
    cuts = pl.read_parquet(paths.INTERIM / f"cutpoints_{suffix}.parquet")

    for stage in MODELLING_STAGES:
        prefix = prefix_events(sessions.lazy(), cuts, stage)
        frame = build_stage_features(prefix, stage)
        out = paths.PROCESSED / f"features_{suffix}_{stage}.parquet"
        frame.write_parquet(out)
        typer.echo(
            f"{stage}: n={frame.height:,}  positives={int(frame['label'].sum()):,}  "
            f"prevalence={frame['label'].mean():.4f}  features={frame.width - 2}  -> {out.name}"
        )


@app.command()
def prepare_b(source: Path, overwrite: bool = False) -> None:
    """Convert a Dataset-B month (.csv or .csv.gz) to Parquet and register provenance."""
    from .data.ingest import convert_dataset_b_to_parquet, register_dataset_b

    typer.echo(f"converting {source.name} ...")
    dest = convert_dataset_b_to_parquet(source, overwrite=overwrite)
    size_gb = dest.stat().st_size / 1024**3
    typer.echo(f"parquet -> {dest} ({size_gb:.2f} GB)")

    prov = register_dataset_b(dest)
    typer.echo(f"registered {prov.n_rows:,} events, sha256={prov.sha256[:16]}...")


@app.command("freeze-splits")
def freeze_splits_cmd(
    suffix: str = _SUFFIX_OPT,
    seed: int = 42,
    overwrite: bool = False,
) -> None:
    """Freeze the temporal and grouped splits to disk (Sec. 7.6, run-sheet step 6)."""
    paths.ensure_dirs()

    from .data.splits import freeze_splits

    sessions = pl.read_parquet(paths.INTERIM / f"sessions_{suffix}.parquet")
    cuts = pl.read_parquet(paths.INTERIM / f"cutpoints_{suffix}.parquet")

    reports = freeze_splits(
        sessions, cuts, suffix=suffix, seed=seed, overwrite=overwrite
    )
    for protocol, report in reports.items():
        typer.echo(report.summary())
        if protocol == "temporal":
            typer.echo(
                f"  boundary val/test: {report.boundary_val_test}   "
                f"straddling users dropped: {report.n_straddling_users:,}"
            )
    typer.secho(
        "Splits are frozen. Every model and explanation must read them; "
        "the test partition opens exactly once, at final evaluation.",
        fg=typer.colors.YELLOW,
    )


@app.command()
def baselines_a(
    models: str = typer.Option("", help="Comma-separated subset; default is all five"),
    seeds: str = typer.Option("", help="Comma-separated subset of the frozen seed list"),
    imbalance: str = "class_weight",
    track: bool = typer.Option(True, help="Log the run to MLflow (Sec. 6.2)"),
) -> None:
    """Run the Dataset A baselines (run-sheet step 7, Sec. 9-10)."""
    paths.ensure_dirs()

    from .data.ingest import load_dataset_a
    from .models.baselines import BASELINE_MODELS
    from .models.run_baselines import run_dataset_a_baselines, summary_table, write_results
    from .seeds import SEEDS

    chosen_models = tuple(m.strip() for m in models.split(",") if m.strip()) or BASELINE_MODELS
    chosen_seeds = tuple(int(s) for s in seeds.split(",") if s.strip()) or SEEDS

    typer.echo(f"models: {chosen_models}")
    typer.echo(f"seeds:  {chosen_seeds}")

    runs = run_dataset_a_baselines(
        load_dataset_a(), models=chosen_models, seeds=chosen_seeds, imbalance=imbalance
    )
    write_results(runs, paths.TABLES)

    summary = summary_table(runs)
    typer.echo("")
    typer.echo(
        summary.select(
            ["model", "n_seeds", "pr_auc_mean", "pr_auc_std", "roc_auc_mean", "ece_mean"]
        )
    )
    typer.secho(
        "Test partition opened. Any further change to features, models or thresholds "
        "invalidates these numbers (Sec. 6.2).",
        fg=typer.colors.YELLOW,
    )

    from .tracking import track_run

    with track_run(
        "baselines-a",
        experiment="baselines-a",
        params={
            "dataset": "A", "models": ",".join(chosen_models),
            "seeds": ",".join(str(s) for s in chosen_seeds), "imbalance": imbalance,
        },
        enabled=track,
    ) as run:
        metrics = {}
        for row in summary.to_dicts():
            model = row["model"]
            for key, value in row.items():
                if key != "model":
                    metrics[f"{key}.{model}"] = value
        run.log_metrics(metrics)
        for name in ("dataset_a_baselines_per_seed.csv", "dataset_a_baselines_summary.csv"):
            run.log_artifact(paths.TABLES / name)
        if run.active:
            typer.echo(f"\nlogged to MLflow: {paths.MLRUNS}")


@app.command()
def tune(
    suffix: str = _SUFFIX_OPT,
    protocol: str = "temporal",
    models: str = typer.Option("lightgbm", help="Comma-separated model types"),
    n_trials: int = typer.Option(100, help="Trials per (stage, model) study"),
    seed: int = 42,
    track: bool = typer.Option(True, help="Log the run to MLflow (Sec. 6.2)"),
) -> None:
    """Optuna tuning per stage on the frozen validation split (Sec. 9.5)."""
    paths.ensure_dirs()

    import json

    from .data.journey import MODELLING_STAGES
    from .data.splits import load_split
    from .features.dictionary import ALL_FEATURE_SETS, FEATURE_DICTIONARY, feature_names
    from .models.run_stages import _split_features
    from .models.tuning import tune_model

    split = load_split(suffix, protocol)
    chosen_models = tuple(m.strip() for m in models.split(",") if m.strip())
    groups = set(ALL_FEATURE_SETS["full"])

    rows = []
    for stage in MODELLING_STAGES:
        path = paths.PROCESSED / f"features_{suffix}_{stage}.parquet"
        if not path.exists():
            continue
        joined, y, partition = _split_features(pl.read_parquet(path), split)
        train = partition == "train"
        val = partition == "val"
        if not val.any() or y[val].sum() == 0:
            typer.echo(f"{stage}: no evaluable validation partition, skipped")
            continue

        available = feature_names(stage)
        columns = [
            spec.name
            for spec in FEATURE_DICTIONARY
            if spec.name in available and spec.ablation_group in groups
        ]
        X = joined.select(columns).to_pandas()

        for model_type in chosen_models:
            typer.echo(f"tuning {stage}/{model_type} ({n_trials} trials) ...")
            result = tune_model(
                model_type,
                X[train], y[train], X[val], y[val],
                columns=columns, stage=stage, n_trials=n_trials, seed=seed,
            )
            typer.echo("  " + result.summary())
            rows.append(
                {
                    "stage": stage,
                    "model": model_type,
                    "seed": seed,
                    "protocol": protocol,
                    "n_trials": result.n_trials,
                    "n_completed": result.n_completed,
                    "n_pruned": result.n_pruned,
                    "best_val_pr_auc": result.best_value,
                    "best_params": json.dumps(result.best_params, sort_keys=True),
                }
            )

    if not rows:
        raise typer.BadParameter(f"no stage could be tuned for suffix {suffix!r}")
    out = paths.TABLES / f"tuning_{suffix}.csv"
    pl.DataFrame(rows).write_csv(out)
    typer.echo(f"\n-> {out}")

    from .tracking import track_run

    with track_run(
        f"tune/{suffix}",
        experiment="tuning",
        params={
            "suffix": suffix, "protocol": protocol, "models": ",".join(chosen_models),
            "n_trials": n_trials, "seed": seed,
        },
        enabled=track,
    ) as run:
        for row in rows:
            key = f"{row['stage']}.{row['model']}"
            run.log_metrics({
                f"best_val_pr_auc.{key}": row["best_val_pr_auc"],
                f"n_completed.{key}": row["n_completed"],
                f"n_pruned.{key}": row["n_pruned"],
            })
            # The selected configuration is a parameter of everything downstream,
            # so it is logged as such rather than left only in the study file.
            run.log_params({f"best_params.{key}": row["best_params"]})
        run.log_artifact(out)


@app.command()
def stage_models(
    suffix: str = _SUFFIX_OPT,
    protocol: str = "temporal",
    models: str = typer.Option("lightgbm", help="Comma-separated model types"),
    seeds: str = typer.Option("", help="Comma-separated subset of the frozen seed list"),
    ablation: bool = typer.Option(False, help="Run the full ablation ladder (Sec. 10, H3)"),
    tuned: bool = typer.Option(
        False, help="Use Optuna best params from experiments/optuna/ (Sec. 9.5)"
    ),
    track: bool = typer.Option(True, help="Log the run to MLflow (Sec. 6.2)"),
) -> None:
    """Fit the Dataset B stage models (run-sheet step 8, Sec. 9.2)."""
    paths.ensure_dirs()

    from .data.journey import MODELLING_STAGES
    from .models.run_stages import (
        ablation_table,
        improvement_curve,
        run_stage_models,
        stage_results_table,
    )
    from .seeds import SEEDS

    chosen_models = tuple(m.strip() for m in models.split(",") if m.strip())
    chosen_seeds = tuple(int(s) for s in seeds.split(",") if s.strip()) or SEEDS
    # The nested ladder plus the direct H3 contrast, which the ladder does not
    # test: it adds entropy/velocity after temporal, whereas H3 compares them to
    # baseline alone.
    feature_sets = (
        ("baseline", "+temporal", "+entropy_velocity", "full", "baseline+entropy_velocity")
        if ablation
        else ("full",)
    )

    features = {}
    for stage in MODELLING_STAGES:
        path = paths.PROCESSED / f"features_{suffix}_{stage}.parquet"
        if path.exists():
            features[stage] = pl.read_parquet(path)
    if not features:
        raise typer.BadParameter(f"no feature matrices for suffix {suffix!r}")

    params_by_stage = None
    if tuned:
        from .models.tuning import load_tuned_params

        if len(chosen_models) != 1:
            raise typer.BadParameter("--tuned requires exactly one model type")
        params_by_stage = {}
        for stage in features:
            try:
                params_by_stage[stage] = load_tuned_params(chosen_models[0], stage)
            except FileNotFoundError:
                typer.secho(f"{stage}: no study found, using defaults", fg=typer.colors.YELLOW)
        typer.echo(f"tuned params loaded for: {sorted(params_by_stage)}")

    runs = run_stage_models(
        features,
        suffix=suffix,
        protocol=protocol,
        models=chosen_models,
        seeds=chosen_seeds,
        feature_sets=feature_sets,
        params_by_stage=params_by_stage,
    )
    if not runs:
        raise typer.BadParameter("no stage produced an evaluable model")

    # The headline temporal-default tables keep their historical names; any
    # other configuration must not overwrite them.
    tag = suffix if protocol == "temporal" else f"{suffix}_{protocol}"
    if tuned:
        tag = f"{tag}_tuned"
    per_seed_path = paths.TABLES / f"stage_models_{tag}_per_seed.csv"
    stage_results_table(runs).write_csv(per_seed_path)
    curve = improvement_curve(runs)
    curve_path = paths.TABLES / f"improvement_curve_{tag}.csv"
    curve.write_csv(curve_path)

    artifacts = [per_seed_path, curve_path]

    typer.echo("")
    typer.echo(curve)
    typer.secho(
        "PR-AUC is not comparable across stages: chance level equals the prevalence. "
        "Read pr_auc_lift_mean, and report N and reach alongside (Sec. 9.2 / amendment A2).",
        fg=typer.colors.YELLOW,
    )

    if ablation:
        table = ablation_table(runs)
        ablation_path = paths.TABLES / f"ablation_{tag}.csv"
        table.write_csv(ablation_path)
        artifacts.append(ablation_path)
        typer.echo("")
        typer.echo(table)

        # Sec. 12: the confirmatory family, corrected across all of it.
        from .evaluate.stage_tests import run_stage_comparisons, summarise

        comparisons = run_stage_comparisons(runs)
        comparisons_path = paths.TABLES / f"comparisons_{tag}.csv"
        comparisons.write_csv(comparisons_path)
        artifacts.append(comparisons_path)
        typer.echo("")
        typer.echo(summarise(comparisons))

    # Sec. 6.2: the run is logged last, so every table it produced is attached.
    from .tracking import stage_run_metrics, track_run

    with track_run(
        f"stage-models/{tag}",
        experiment="stage-models",
        params={
            "suffix": suffix, "protocol": protocol, "models": ",".join(chosen_models),
            "seeds": ",".join(str(s) for s in chosen_seeds), "ablation": ablation,
            "tuned": tuned, "feature_sets": ",".join(feature_sets),
        },
        enabled=track,
    ) as run:
        run.log_metrics(stage_run_metrics(runs))
        for path in artifacts:
            run.log_artifact(path)
        if run.active:
            typer.echo(f"\nlogged to MLflow: {paths.MLRUNS}")


@app.command()
def stage_shap(
    suffix: str = _SUFFIX_OPT,
    protocol: str = "temporal",
    model: str = "lightgbm",
    seed: int = 42,
    max_explain: int = 5000,
    background_size: int = 2000,
    track: bool = typer.Option(True, help="Log the run to MLflow (Sec. 6.2)"),
) -> None:
    """Layer 1: stage-conditioned SHAP and the RQ2 migration table (Sec. 11.1)."""
    paths.ensure_dirs()

    from .data.journey import MODELLING_STAGES
    from .explain.run_stage_shap import explain_stages, migration_table, stage_importance_table

    features = {}
    for stage in MODELLING_STAGES:
        path = paths.PROCESSED / f"features_{suffix}_{stage}.parquet"
        if path.exists():
            features[stage] = pl.read_parquet(path)
    if not features:
        raise typer.BadParameter(f"no feature matrices for suffix {suffix!r}")

    explanations = explain_stages(
        features, suffix=suffix, protocol=protocol, model_type=model,
        seed=seed, max_explain=max_explain, background_size=background_size,
    )
    if not explanations:
        raise typer.BadParameter("no stage could be explained")

    importance = stage_importance_table(explanations)
    importance_path = paths.TABLES / f"stage_importance_{suffix}.csv"
    importance.write_csv(importance_path)

    migration = migration_table(explanations)
    migration_path = paths.TABLES / f"attribution_migration_{suffix}.csv"
    migration.write_csv(migration_path)

    typer.echo("")
    for stage, explanation in explanations.items():
        top = (
            importance.filter(pl.col("stage") == stage)
            .head(5)
            .select(["group", "share"])
        )
        typer.echo(f"{stage}  (n_train={explanation.n_train:,}, "
                   f"background={explanation.attribution.background_size}, "
                   f"explained={explanation.attribution.n_explained})")
        for row in top.to_dicts():
            typer.echo(f"    {row['share']:6.3f}  {row['group']}")

    reversals = migration.filter(pl.col("reversed_sign"))
    typer.echo("")
    typer.echo(f"sign reversals across stages: {reversals['group'].n_unique()}")
    typer.echo(f"migration table -> {migration_path}")

    from .tracking import track_run

    with track_run(
        f"stage-shap/{suffix}",
        experiment="stage-shap",
        params={
            "suffix": suffix, "protocol": protocol, "model": model, "seed": seed,
            "max_explain": max_explain, "background_size": background_size,
        },
        enabled=track,
    ) as run:
        # The attribution shares are the RQ2 result; logging them makes the
        # migration reconstructible from the run record alone.
        run.log_metrics({
            f"share.{row['stage']}.{row['group']}": row["share"]
            for row in importance.to_dicts()
        })
        run.log_metrics({"sign_reversals": reversals["group"].n_unique()})
        run.log_artifact(importance_path)
        run.log_artifact(migration_path)


@app.command()
def explanation_quality(
    suffix: str = _SUFFIX_OPT,
    protocol: str = "temporal",
    model: str = "lightgbm",
    seed: int = 42,
    consistency: bool = typer.Option(True, help="Run the across-seed consistency arm"),
    stability: bool = typer.Option(True, help="Run the local-Lipschitz arm (slow)"),
    track: bool = typer.Option(True, help="Log the run to MLflow (Sec. 6.2)"),
) -> None:
    """Layer 3: faithfulness, stability and consistency (Sec. 11.3, RQ3)."""
    paths.ensure_dirs()

    from .data.journey import MODELLING_STAGES
    from .explain.run_quality import evaluate_stage_quality, seed_consistency
    from .explain.run_stage_shap import explain_stages

    features = {}
    for stage in MODELLING_STAGES:
        path = paths.PROCESSED / f"features_{suffix}_{stage}.parquet"
        if path.exists():
            features[stage] = pl.read_parquet(path)
    if not features:
        raise typer.BadParameter(f"no feature matrices for suffix {suffix!r}")

    explanations = explain_stages(
        features, suffix=suffix, protocol=protocol, model_type=model,
        seed=seed, background_size=300, max_explain=1000,
    )
    quality = evaluate_stage_quality(explanations, seed=seed, with_stability=stability)

    rows = []
    typer.echo("")
    for stage, q in quality.items():
        typer.echo(f"{stage}: {q.faithfulness.summary()}")
        typer.echo(
            f"     deletion AUC {q.deletion_auc:.4f} (monotone={q.deletion_monotone})  "
            f"insertion AUC {q.insertion_auc:.4f}"
        )
        if q.stability:
            typer.echo(f"     {q.stability.summary()}")
        rows.append(
            {
                "stage": stage,
                "faithfulness_corr": q.faithfulness.correlation_mean,
                "faithfulness_sd": q.faithfulness.correlation_std,
                "faithfulness_passes": q.faithfulness.passes,
                "deletion_auc": q.deletion_auc,
                "insertion_auc": q.insertion_auc,
                "deletion_monotone": q.deletion_monotone,
                "lipschitz_max": q.stability.max_ratio if q.stability else None,
                "lipschitz_mean": q.stability.mean_ratio if q.stability else None,
            }
        )

    if consistency:
        typer.echo("")
        results = seed_consistency(
            features, suffix=suffix, protocol=protocol, model_type=model
        )
        by_stage = {}
        for stage, result in results.items():
            typer.echo(f"{stage}: {result.summary()}")
            by_stage[stage] = result
        for row in rows:
            r = by_stage.get(row["stage"])
            if r is not None:
                row["seed_spearman_mean"] = r.mean_spearman
                row["seed_spearman_min"] = r.min_spearman
                row["seed_consistency_passes"] = r.passes

    table = pl.DataFrame(rows)
    out = paths.TABLES / f"explanation_quality_{suffix}.csv"
    table.write_csv(out)
    typer.echo("")
    typer.echo(f"-> {out}")

    from .tracking import track_run

    with track_run(
        f"explanation-quality/{suffix}",
        experiment="explanation-quality",
        params={
            "suffix": suffix, "protocol": protocol, "model": model, "seed": seed,
            "consistency": consistency, "stability": stability,
        },
        enabled=track,
    ) as run:
        for row in rows:
            stage = row["stage"]
            run.log_metrics({
                f"{key}.{stage}": value
                for key, value in row.items()
                if key != "stage"
            })
        run.log_artifact(out)


@app.command()
def sequence_arm(
    suffix: str = _SUFFIX_OPT,
    protocol: str = "temporal",
    seed: int = 42,
    epochs: int = typer.Option(80, help="Max epochs (early stopping may halt sooner)"),
    patience: int = typer.Option(10, help="Early-stopping patience on val PR-AUC"),
    hidden: int = typer.Option(128, help="GRU hidden state size"),
    num_layers: int = typer.Option(2, help="Number of stacked GRU layers"),
    dropout: float = typer.Option(0.2, help="Dropout between layers and before head"),
    max_sessions: int = 60000,
    timeshap: bool = typer.Option(True, help="Use TimeSHAP; falls back to permutation"),
) -> None:
    """Sequence arm: GRU + Layer 2 attribution, and the H4 comparison (Sec. 9.3, 11.2)."""
    paths.ensure_dirs()

    from .data.journey import MODELLING_STAGES
    from .explain.run_sequence import h4_table, run_sequence_arm
    from .explain.run_stage_shap import explain_stages, stage_importance_table

    gru_kwargs = dict(
        hidden=hidden, num_layers=num_layers, dropout=dropout,
        patience=patience,
    )

    sessions = pl.read_parquet(paths.INTERIM / f"sessions_{suffix}.parquet")
    cuts = pl.read_parquet(paths.INTERIM / f"cutpoints_{suffix}.parquet")

    features = {}
    for stage in MODELLING_STAGES:
        path = paths.PROCESSED / f"features_{suffix}_{stage}.parquet"
        if path.exists():
            features[stage] = pl.read_parquet(path)

    # Tree-side importance, ungrouped, so H4 can map individual features.
    explanations = explain_stages(
        features, suffix=suffix, protocol=protocol, seed=seed,
        background_size=300, max_explain=800,
    )
    tabular = {}
    for stage, explanation in explanations.items():
        table = explanation.attribution.global_importance()
        tabular[stage] = dict(zip(table["feature"], table["mean_abs_shap"], strict=True))

    results = run_sequence_arm(
        sessions, cuts, suffix=suffix, protocol=protocol, seed=seed,
        epochs=epochs, max_sessions=max_sessions,
        use_timeshap=timeshap, tabular_importance=tabular,
        **gru_kwargs,
    )
    if not results:
        raise typer.BadParameter("no stage produced a sequence model")

    table = h4_table(results)
    table.write_csv(paths.TABLES / f"h4_cross_paradigm_{suffix}.csv")

    typer.echo("")
    for stage, r in results.items():
        typer.echo(
            f"{stage}: GRU test PR-AUC {r.test_pr_auc:.4f} "
            f"(n_train={r.n_train:,}, n_test={r.n_test:,})"
        )
        typer.echo(f"     {r.attribution.summary()}")
        if r.h4_spearman is not None:
            verdict = "PASS" if r.h4_spearman > 0.6 else "FAIL"
            typer.echo(
                f"     H4 Spearman vs TreeSHAP {r.h4_spearman:+.3f} "
                f"over {r.h4_n_concepts} concepts: {verdict}"
            )
    typer.echo("")
    typer.echo(f"-> {paths.TABLES / f'h4_cross_paradigm_{suffix}.csv'}")

    _ = stage_importance_table


@app.command("tune-gru")
def tune_gru_cmd(
    suffix: str = _SUFFIX_OPT,
    protocol: str = "temporal",
    seed: int = 42,
    n_trials: int = typer.Option(30, help="Optuna trials per stage"),
    epochs: int = typer.Option(80, help="Max epochs per trial"),
    patience: int = typer.Option(10, help="Early-stopping patience per trial"),
    max_sessions: int = 60000,
    max_len: int = 32,
) -> None:
    """Optuna tuning for the GRU sequence model (amendment A30)."""
    paths.ensure_dirs()
    import json

    import numpy as np

    from .data.journey import MODELLING_STAGES, prefix_events
    from .data.splits import load_split
    from .models.sequence import build_sequences
    from .models.tuning import tune_gru

    sessions = pl.read_parquet(paths.INTERIM / f"sessions_{suffix}.parquet")
    cuts = pl.read_parquet(paths.INTERIM / f"cutpoints_{suffix}.parquet")
    split = load_split(suffix, protocol)
    rng = np.random.default_rng(seed)

    rows = []
    for stage in MODELLING_STAGES:
        prefix = prefix_events(sessions.lazy(), cuts, stage).collect()
        prefix = prefix.join(
            split.select(["session_id", "partition"]), on="session_id", how="inner"
        )
        if prefix.is_empty():
            continue

        ids = prefix["session_id"].unique()
        if ids.len() > max_sessions:
            keep = pl.Series("session_id", rng.choice(ids.to_numpy(), max_sessions, replace=False))
            prefix = prefix.filter(pl.col("session_id").is_in(keep))

        batch = build_sequences(prefix, max_len=max_len)
        partition = (
            prefix.group_by("session_id", maintain_order=True)
            .agg(pl.first("partition"))
            .join(pl.DataFrame({"session_id": batch.session_ids}), on="session_id", how="right")
        )["partition"].to_numpy()

        import numpy as np  # already imported above, but ensure availability
        idx = {name: np.flatnonzero(partition == name) for name in ("train", "val", "test")}
        if any(len(v) == 0 for v in idx.values()):
            continue

        typer.echo(f"tuning GRU for {stage} ({n_trials} trials) ...")
        result = tune_gru(
            batch, idx["train"], idx["val"],
            stage=stage, n_trials=n_trials, seed=seed,
            epochs=epochs, patience=patience,
        )
        typer.echo(f"  {result.summary()}")
        rows.append({
            "stage": stage, "model": "gru", "seed": seed,
            "protocol": protocol, "n_trials": result.n_trials,
            "n_completed": result.n_completed, "n_pruned": result.n_pruned,
            "best_val_pr_auc": result.best_value,
            "best_params": json.dumps(result.best_params, sort_keys=True),
        })

    if not rows:
        raise typer.BadParameter(f"no stage could be tuned for suffix {suffix!r}")
    out = paths.TABLES / f"tuning_gru_{suffix}.csv"
    pl.DataFrame(rows).write_csv(out)
    typer.echo(f"\n-> {out}")


@app.command()
def figures(suffix: str = _SUFFIX_OPT) -> None:
    """Build the publication figures (Sec. 15 deliverables)."""
    paths.ensure_dirs()

    from .report.figures import build_all

    built = build_all(suffix)

    # Figure 5 needs two calibration variants of the same model, so it is built
    # here rather than from a stored table: the pre-A18 scores exist only as a
    # deliberate re-run.
    from .data.ingest import load_dataset_a
    from .models.run_baselines import run_dataset_a_baselines
    from .report.figures import figure_5_reliability

    try:
        fixed = run_dataset_a_baselines(
            load_dataset_a(), models=("lightgbm",), seeds=(7,),
            calibration_source="train_slice", n_resamples=2000,
        )[0]
        old = run_dataset_a_baselines(
            load_dataset_a(), models=("lightgbm",), seeds=(7,),
            calibration_source="val", n_resamples=2000,
        )[0]
        built["fig5"] = figure_5_reliability(fixed.y_test, old.test_scores, fixed.test_scores)
    except FileNotFoundError:
        typer.secho("Dataset A not present; skipping Figure 5", fg=typer.colors.YELLOW)

    if not built:
        raise typer.BadParameter("no input tables found; run the analysis commands first")
    for name, written in built.items():
        typer.echo(f"{name}: " + ", ".join(p.name for p in written))
    typer.echo(f"\n-> {paths.FIGURES}")


@app.command("whole-session")
def whole_session_cmd(
    suffix: str = _SUFFIX_OPT,
    protocol: str = "temporal",
    seed: int = 42,
    max_explain: int = 4000,
    background_size: int = 300,
    track: bool = typer.Option(True, help="Log the run to MLflow (Sec. 6.2)"),
) -> None:
    """RQ4: an empirical whole-session model to contrast against the stages."""
    paths.ensure_dirs()

    from .explain.whole_session import (
        contrast_table,
        member_to_group,
        run_whole_session_contrast,
    )

    importance_path = paths.TABLES / f"stage_importance_{suffix}.csv"
    if not importance_path.exists():
        raise typer.BadParameter(
            f"no stage importance table at {importance_path}; run stage-shap first"
        )

    importance = pl.read_csv(importance_path)
    # The whole-session model must be aggregated under the stage models'
    # reference grouping, or the two share taxonomies do not align.
    grouping = member_to_group(importance["group"].unique().to_list())

    sessions = pl.read_parquet(paths.INTERIM / f"sessions_{suffix}.parquet")
    cuts = pl.read_parquet(paths.INTERIM / f"cutpoints_{suffix}.parquet")

    result = run_whole_session_contrast(
        sessions, cuts, suffix=suffix, protocol=protocol, seed=seed,
        max_explain=max_explain, background_size=background_size,
        grouping=grouping,
    )
    typer.echo(result.summary())

    table = contrast_table(importance, result)
    out = paths.TABLES / f"whole_session_contrast_{suffix}.csv"
    table.write_csv(out)

    typer.echo("")
    typer.echo(table.select(
        ["group", "stage_peak", "stage_peak_at", "whole_session_share", "flattening"]
    ))
    typer.echo("")
    typer.echo(f"-> {out}")

    from .tracking import track_run

    with track_run(
        f"whole-session/{suffix}",
        experiment="whole-session",
        params={"suffix": suffix, "protocol": protocol, "seed": seed},
        enabled=track,
    ) as run:
        run.log_metrics({
            "pr_auc": result.pr_auc, "roc_auc": result.roc_auc,
            "prevalence": result.prevalence, "n_test": result.n_test,
        })
        run.log_metrics({f"whole_share.{k}": v for k, v in result.shares.items()})
        run.log_artifact(out)


@app.command()
def static_contrast(suffix: str = _SUFFIX_OPT) -> None:
    """RQ4: what a static whole-session attribution would flatten (Sec. 11.1)."""
    paths.ensure_dirs()

    from .explain.static_contrast import static_contrast_table, summarise_contrast

    path = paths.TABLES / f"attribution_migration_{suffix}.csv"
    if not path.exists():
        raise typer.BadParameter(f"no migration table at {path}; run stage-shap first")

    table = static_contrast_table(pl.read_csv(path))
    table.write_csv(paths.TABLES / f"static_contrast_{suffix}.csv")
    typer.echo("")
    typer.echo(summarise_contrast(table))
    typer.echo("")
    typer.echo(f"-> {paths.TABLES / f'static_contrast_{suffix}.csv'}")


@app.command()
def per_instance_h4(
    suffix: str = _SUFFIX_OPT,
    protocol: str = "temporal",
    seeds: str = typer.Option("42", help="Comma-separated seeds from the frozen list"),
    epochs: int = typer.Option(40, help="Max epochs for each seed's GRU (early stopping applies)"),
    patience: int = typer.Option(8, help="Early-stopping patience"),
    hidden: int = typer.Option(128, help="GRU hidden state size"),
    num_layers: int = typer.Option(2, help="Number of stacked GRU layers"),
    dropout: float = typer.Option(0.2, help="Dropout rate"),
    max_sessions: int = 30000,
    n_explain: int = 400,
) -> None:
    """H4 per instance: do the paradigms agree about individual journeys? (A22)"""
    paths.ensure_dirs()

    from .data.journey import MODELLING_STAGES
    from .explain.per_instance_h4 import (
        aggregate_over_seeds,
        compare_per_instance,
        per_instance_table,
        summarise,
        summarise_seeds,
    )
    from .explain.run_sequence import run_sequence_arm
    from .explain.run_stage_shap import explain_stages
    from .explain.sequence_shap import timeshap_instance_attributions

    chosen = [int(s) for s in seeds.split(",") if s.strip()]
    sessions = pl.read_parquet(paths.INTERIM / f"sessions_{suffix}.parquet")
    cuts = pl.read_parquet(paths.INTERIM / f"cutpoints_{suffix}.parquet")

    features = {}
    for stage in MODELLING_STAGES:
        path = paths.PROCESSED / f"features_{suffix}_{stage}.parquet"
        if path.exists():
            features[stage] = pl.read_parquet(path)

    per_seed: dict[int, pl.DataFrame] = {}
    for seed in chosen:
        typer.echo(f"seed {seed} ...")
        explanations = explain_stages(
            features, suffix=suffix, protocol=protocol, seed=seed,
            background_size=300, max_explain=4000,
        )
        sequence = run_sequence_arm(
            sessions, cuts, suffix=suffix, protocol=protocol, seed=seed,
            epochs=epochs, patience=patience, max_sessions=max_sessions,
            n_explain=n_explain,
            hidden=hidden, num_layers=num_layers, dropout=dropout,
            restrict_to_sessions={
                stage: e.explained_session_ids for stage, e in explanations.items()
            },
        )

        results = []
        for stage, seq in sequence.items():
            if stage not in explanations or seq.explained_X is None:
                continue
            per_instance = timeshap_instance_attributions(
                seq.predict_fn, seq.explained_X, seq.train_X, seq.feature_names, seed=seed
            )
            tree = explanations[stage]
            results.extend(
                compare_per_instance(
                    tree.attribution.shap_values,
                    tree.attribution.feature_names,
                    tree.explained_session_ids,
                    per_instance,
                    seq.feature_names,
                    seq.explained_session_ids,
                    stage=stage,
                )
            )
        if results:
            per_seed[seed] = per_instance_table(results)

    if not per_seed:
        raise typer.BadParameter("no stage produced enough shared sessions to compare")

    if len(per_seed) == 1:
        table = next(iter(per_seed.values()))
        table.write_csv(paths.TABLES / f"per_instance_h4_{suffix}.csv")
        typer.echo("")
        typer.echo(summarise(table))
    else:
        aggregated = aggregate_over_seeds(per_seed)
        aggregated.write_csv(paths.TABLES / f"per_instance_h4_seeds_{suffix}.csv")
        typer.echo("")
        typer.echo(summarise_seeds(aggregated))
    typer.echo("")
    typer.echo(f"-> {paths.TABLES}")


@app.command()
def check_config(path: Path) -> None:
    """Validate a run YAML against the protocol schema (Appendix A)."""
    config = load_config(path)
    typer.secho(f"OK  {config.name}", fg=typer.colors.GREEN)
    typer.echo(f"config_hash: {config.config_hash}")
    typer.echo(f"git_commit:  {config.git_commit}")


@app.command()
def appendix(
    suffix: str = _SUFFIX_OPT,
    protocol: str = "temporal",
    model: str = "lightgbm",
    seed: int = 42,
    max_explain: int = 3000,
    background_size: int = 500,
    reuse: bool = typer.Option(
        False, help="Redraw from cached scores/SHAP instead of refitting (layout work)"
    ),
) -> None:
    """Build the appendix figure and table set (Sec. 15 deliverables).

    Diagnostic plots (ROC, PR, confusion, SHAP) are drawn at a single seed
    because a pooled beeswarm or ROC curve is not a well-defined object; the
    headline metrics they sit beside remain five-seed means. Runtimes are
    measured here and written to `runtime_appendix.csv` so the computational
    cost table reports observations rather than estimates.
    """
    paths.ensure_dirs()

    import time

    from .data.journey import MODELLING_STAGES
    from .explain.run_stage_shap import explain_stages
    from .models.run_stages import run_stage_models
    from .report import appendix as apx
    from .report import appendix_diagrams, appendix_tables

    features = {}
    for stage in MODELLING_STAGES:
        path = paths.PROCESSED / f"features_{suffix}_{stage}.parquet"
        if path.exists():
            features[stage] = pl.read_parquet(path)
    if not features:
        raise typer.BadParameter(f"no feature matrices for suffix {suffix!r}")

    timings: list[dict] = []
    written: dict[str, list] = {}

    # Cache of the expensive inputs. Refitting and re-explaining costs roughly
    # seventeen minutes on this dataset while redrawing a figure costs seconds,
    # and layout work needs many redraws.
    cache = paths.INTERIM / f"appendix_cache_{suffix}_seed{seed}.npz"

    if reuse:
        if not cache.exists():
            raise typer.BadParameter(f"no cache at {cache}; run once without --reuse")
        blob = np.load(cache, allow_pickle=True)
        stage_names = [str(s) for s in blob["stages"]]
        scores = {s: (blob[f"y_{s}"], blob[f"p_{s}"]) for s in stage_names}
        thresholds = dict(zip(stage_names, blob["thresholds"], strict=True))
        explanations = {
            s: _CachedExplanation(
                _CachedAttribution([str(n) for n in blob[f"names_{s}"]], blob[f"shap_{s}"]),
                blob[f"matrix_{s}"],
            )
            for s in stage_names
            if f"shap_{s}" in blob
        }
        typer.echo(f"redrawing from {cache.name}")
    else:
        typer.echo(f"fitting stage models at seed {seed} for diagnostics ...")
        start = time.perf_counter()
        # The protocol floors bootstrap resamples at 2000 (Sec. 4.2). These
        # diagnostics do not use the intervals, but the floor is a protocol
        # invariant and is respected rather than bypassed.
        runs = run_stage_models(
            features, suffix=suffix, protocol=protocol,
            models=(model,), seeds=(seed,), feature_sets=("full",), n_resamples=2000,
        )
        timings.append(
            {"command": "appendix-stage-fit", "seconds": time.perf_counter() - start}
        )
        if not runs:
            raise typer.BadParameter("no stage produced an evaluable model")

        scores = {r.stage: (r.y_test, r.test_scores) for r in runs}
        thresholds = {r.stage: r.threshold for r in runs}

        typer.echo("computing stage attributions ...")
        start = time.perf_counter()
        explanations = explain_stages(
            features, suffix=suffix, protocol=protocol, model_type=model,
            seed=seed, max_explain=max_explain, background_size=background_size,
        )
        timings.append(
            {"command": "appendix-shap", "seconds": time.perf_counter() - start}
        )

        payload: dict = {
            "stages": np.array(list(scores)),
            "thresholds": np.array([thresholds[s] for s in scores]),
        }
        for stage, (y_true, y_score) in scores.items():
            payload[f"y_{stage}"] = np.asarray(y_true)
            payload[f"p_{stage}"] = np.asarray(y_score)
        for stage, explanation in explanations.items():
            payload[f"shap_{stage}"] = np.asarray(explanation.attribution.shap_values)
            payload[f"matrix_{stage}"] = np.asarray(explanation.explained_matrix)
            payload[f"names_{stage}"] = np.array(
                list(explanation.attribution.feature_names)
            )
        np.savez_compressed(cache, **payload)
        typer.echo(f"cached inputs -> {cache.name}")

    written["figA1"] = apx.figure_roc_curves(scores)
    written["figA2"] = apx.figure_pr_curves(scores)
    written["figA3"] = apx.figure_confusion_matrices(scores, thresholds)

    if explanations:
        written["figA4"] = apx.figure_shap_summary(explanations)
        written["figA5"] = apx.figure_shap_dependence(explanations)
        written["figA6"] = apx.figure_feature_importance(explanations)

    baselines = paths.TABLES / "dataset_a_baselines_summary.csv"
    if baselines.exists():
        written["figA7"] = apx.figure_performance_comparison(pl.read_csv(baselines))
    else:
        typer.secho("no Dataset A summary; skipping Figure A7", fg=typer.colors.YELLOW)

    ablation_path = paths.TABLES / f"ablation_{suffix}.csv"
    if ablation_path.exists():
        written["figA8"] = apx.figure_ablation(pl.read_csv(ablation_path))
    else:
        typer.secho("no ablation table; skipping Figure A8", fg=typer.colors.YELLOW)

    written.update(appendix_diagrams.build_all())

    # Only write timings when something was actually timed. A redraw measures
    # nothing, and writing an empty frame here would overwrite the real
    # measurements from the run that produced the cache -- destroying the only
    # observed runtimes in the project.
    if timings:
        pl.DataFrame(timings).write_csv(paths.TABLES / "runtime_appendix.csv")

    prevalence_path = paths.TABLES / f"stage_prevalence_{suffix}.csv"
    if prevalence_path.exists():
        appendix_tables.dataset_statistics_table(prevalence_path).write_csv(
            paths.TABLES / f"appendix_dataset_statistics_{suffix}.csv"
        )
    appendix_tables.environment_table().write_csv(
        paths.TABLES / "appendix_environment.csv"
    )
    appendix_tables.computational_cost_table().write_csv(
        paths.TABLES / "appendix_computational_cost.csv"
    )
    appendix_tables.literature_comparison_table().write_csv(
        paths.TABLES / "appendix_literature_comparison.csv"
    )

    typer.echo("")
    for name, files in sorted(written.items()):
        typer.echo(f"{name}: " + ", ".join(p.name for p in files))
    typer.echo(f"\nfigures -> {paths.FIGURES}")
    typer.echo(f"tables  -> {paths.TABLES}")


if __name__ == "__main__":
    app()
