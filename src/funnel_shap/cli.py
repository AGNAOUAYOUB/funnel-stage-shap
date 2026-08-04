"""Command-line entry points for the pipeline (protocol Sec. 13 run sheet).

Notebooks are for exploration only; everything that produces a reported number
runs through here so it is scripted, seeded, and DVC-trackable.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import typer

from . import paths
from .config import load_config

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


@app.command()
def sessionize(
    source: Path = _SOURCE_OPT,
    gap_minutes: int = 30,
    synthetic: bool = _SYNTHETIC_OPT,
) -> None:
    """Clean and sessionise the event stream (Sec. 7.1-7.2)."""
    paths.ensure_dirs()

    from .data.flow import DataFlow
    from .data.ingest import scan_dataset_b
    from .data.journey import stage_cutpoints, stage_prevalence_table
    from .data.sessionize import SessionizeConfig
    from .data.sessionize import sessionize as run_sessionize
    from .data.synthetic import make_event_log

    if synthetic:
        lazy = make_event_log(n_users=500, seed=42).lazy()
        label = f"synthetic (gap={gap_minutes}min)"
    else:
        lazy = scan_dataset_b(source)
        label = f"Dataset B (gap={gap_minutes}min)"

    flow = DataFlow(label)
    sessions = run_sessionize(lazy, SessionizeConfig(gap_minutes=gap_minutes), flow).collect()

    suffix = "synthetic" if synthetic else f"gap{gap_minutes}"
    out = paths.INTERIM / f"sessions_{suffix}.parquet"
    sessions.write_parquet(out)

    flow.write(paths.TABLES / f"data_flow_{suffix}.csv")
    typer.echo(flow.summary())

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
def check_config(path: Path) -> None:
    """Validate a run YAML against the protocol schema (Appendix A)."""
    config = load_config(path)
    typer.secho(f"OK  {config.name}", fg=typer.colors.GREEN)
    typer.echo(f"config_hash: {config.config_hash}")
    typer.echo(f"git_commit:  {config.git_commit}")


if __name__ == "__main__":
    app()
