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

        # Sec. 12: the confirmatory family, corrected across all of it.
        from .evaluate.stage_tests import run_stage_comparisons, summarise

        comparisons = run_stage_comparisons(runs)
        comparisons.write_csv(paths.TABLES / f"comparisons_{suffix}.csv")
        typer.echo("")
        typer.echo(summarise(comparisons))


@app.command()
def stage_shap(
    suffix: str = _SUFFIX_OPT,
    protocol: str = "temporal",
    model: str = "lightgbm",
    seed: int = 42,
    max_explain: int = 5000,
    background_size: int = 2000,
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
    importance.write_csv(paths.TABLES / f"stage_importance_{suffix}.csv")

    migration = migration_table(explanations)
    migration.write_csv(paths.TABLES / f"attribution_migration_{suffix}.csv")

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
    typer.echo(f"migration table -> {paths.TABLES / f'attribution_migration_{suffix}.csv'}")


@app.command()
def explanation_quality(
    suffix: str = _SUFFIX_OPT,
    protocol: str = "temporal",
    model: str = "lightgbm",
    seed: int = 42,
    consistency: bool = typer.Option(True, help="Run the across-seed consistency arm"),
    stability: bool = typer.Option(True, help="Run the local-Lipschitz arm (slow)"),
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
    table.write_csv(paths.TABLES / f"explanation_quality_{suffix}.csv")
    typer.echo("")
    typer.echo(f"-> {paths.TABLES / f'explanation_quality_{suffix}.csv'}")


@app.command()
def sequence_arm(
    suffix: str = _SUFFIX_OPT,
    protocol: str = "temporal",
    seed: int = 42,
    epochs: int = 6,
    max_sessions: int = 60000,
    timeshap: bool = typer.Option(True, help="Use TimeSHAP; falls back to permutation"),
) -> None:
    """Sequence arm: GRU + Layer 2 attribution, and the H4 comparison (Sec. 9.3, 11.2)."""
    paths.ensure_dirs()

    from .data.journey import MODELLING_STAGES
    from .explain.run_sequence import h4_table, run_sequence_arm
    from .explain.run_stage_shap import explain_stages, stage_importance_table

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
    epochs: int = 4,
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
            epochs=epochs, max_sessions=max_sessions, n_explain=n_explain,
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


if __name__ == "__main__":
    app()
