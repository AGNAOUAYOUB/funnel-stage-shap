"""Command-line entry points for the pipeline (protocol Sec. 13 run sheet).

Notebooks are for exploration only; everything that produces a reported number
runs through here so it is scripted, seeded, and DVC-trackable.
"""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path

import polars as pl
import typer

from . import paths
from .config import load_config

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


@app.command()
def stage_models(
    suffix: str = _SUFFIX_OPT,
    protocol: str = "temporal",
    models: str = typer.Option("lightgbm", help="Comma-separated model types"),
    seeds: str = typer.Option("", help="Comma-separated subset of the frozen seed list"),
    ablation: bool = typer.Option(False, help="Run the full ablation ladder (Sec. 10, H3)"),
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

    runs = run_stage_models(
        features,
        suffix=suffix,
        protocol=protocol,
        models=chosen_models,
        seeds=chosen_seeds,
        feature_sets=feature_sets,
    )
    if not runs:
        raise typer.BadParameter("no stage produced an evaluable model")

    stage_results_table(runs).write_csv(paths.TABLES / f"stage_models_{suffix}_per_seed.csv")
    curve = improvement_curve(runs)
    curve.write_csv(paths.TABLES / f"improvement_curve_{suffix}.csv")

    typer.echo("")
    typer.echo(curve)
    typer.secho(
        "PR-AUC is not comparable across stages: chance level equals the prevalence. "
        "Read pr_auc_lift_mean, and report N and reach alongside (Sec. 9.2 / amendment A2).",
        fg=typer.colors.YELLOW,
    )

    if ablation:
        table = ablation_table(runs)
        table.write_csv(paths.TABLES / f"ablation_{suffix}.csv")
        typer.echo("")
        typer.echo(table)


@app.command()
def check_config(path: Path) -> None:
    """Validate a run YAML against the protocol schema (Appendix A)."""
    config = load_config(path)
    typer.secho(f"OK  {config.name}", fg=typer.colors.GREEN)
    typer.echo(f"config_hash: {config.config_hash}")
    typer.echo(f"git_commit:  {config.git_commit}")


if __name__ == "__main__":
    app()
