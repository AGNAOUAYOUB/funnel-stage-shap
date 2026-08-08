"""Appendix tables (protocol Sec. 15 deliverables).

Dataset statistics, computational cost and the literature comparison, plus a
LaTeX emitter so the appendix does not have to be hand-transcribed from CSV --
transcription is where table/text mismatches enter a manuscript.

**On computational cost.** Timings are *measured* by instrumenting a run, never
estimated. `computational_cost_table` reads whatever measurements exist in
`reports/tables/runtime_*.csv`; if a stage was never timed it is reported as
unmeasured rather than given a plausible-looking number. A fabricated runtime
column is worse than no runtime column, because a reader can act on it.
"""

from __future__ import annotations

import platform
from pathlib import Path

import polars as pl

from ..paths import TABLES


def dataset_statistics_table(
    prevalence_csv: Path,
    flow_csv: Path | None = None,
) -> pl.DataFrame:
    """Per-stage descriptive statistics for Dataset B."""
    prevalence = pl.read_csv(prevalence_csv)
    keep = [
        c
        for c in (
            "stage", "definition", "modelled", "n_sessions", "reach_rate",
            "n_positive", "prevalence", "mean_prefix_events",
        )
        if c in prevalence.columns
    ]
    table = prevalence.select(keep)
    if flow_csv is not None and Path(flow_csv).exists():
        flow = pl.read_csv(flow_csv)
        total = int(flow["n"].max()) if "n" in flow.columns else None
        if total:
            table = table.with_columns(
                (pl.col("n_sessions") / total).alias("share_of_corpus")
            )
    return table


def environment_table() -> pl.DataFrame:
    """The execution environment, recorded so timings can be interpreted."""
    try:
        import numpy
        numpy_version = numpy.__version__
    except Exception:  # noqa: BLE001
        numpy_version = "unavailable"
    return pl.DataFrame(
        [
            {"item": "python", "value": platform.python_version()},
            {"item": "platform", "value": platform.platform()},
            {"item": "processor", "value": platform.processor() or "unknown"},
            {"item": "numpy", "value": numpy_version},
        ]
    )


def computational_cost_table(directory: Path = TABLES) -> pl.DataFrame:
    """Collate measured runtimes; report gaps as gaps.

    Returns one row per pipeline step with `seconds` populated only where a
    measurement exists on disk. Steps with no measurement carry a null and the
    status ``not measured``, so the table can be published without implying a
    precision that was never obtained.
    """
    steps = [
        ("sessionise + subsample", "sessionize"),
        ("build stage features", "build-features"),
        ("freeze splits", "freeze-splits"),
        ("Dataset A baselines", "baselines-a"),
        ("stage models (5 seeds, ablation)", "stage-models"),
        ("Optuna tuning (300 trials)", "tune"),
        ("stage TreeSHAP", "stage-shap"),
        ("explanation quality (Layer 3)", "explanation-quality"),
        ("sequence arm (GRU + TimeSHAP)", "sequence-arm"),
        ("whole-session contrast", "whole-session"),
    ]

    measured: dict[str, float] = {}
    for path in sorted(Path(directory).glob("runtime_*.csv")):
        # An empty or malformed runtime file means "no measurement", not a
        # crash: the cost table must still be produceable, reporting the gap.
        try:
            frame = pl.read_csv(path)
        except Exception:  # noqa: BLE001 - polars raises several types here
            continue
        if {"command", "seconds"} <= set(frame.columns):
            for row in frame.to_dicts():
                if row.get("seconds") is not None:
                    measured[str(row["command"])] = float(row["seconds"])

    rows = []
    for label, command in steps:
        seconds = measured.get(command)
        rows.append(
            {
                "step": label,
                "command": command,
                "seconds": seconds,
                "minutes": round(seconds / 60.0, 2) if seconds is not None else None,
                "status": "measured" if seconds is not None else "not measured",
            }
        )
    return pl.DataFrame(rows)


def literature_comparison_table() -> pl.DataFrame:
    """Positioning against the closest sequence-aware explanation methods.

    Every row states a property of the cited method that is checkable from its
    paper; no performance numbers are compared, because the datasets and targets
    differ and a side-by-side metric column would invite a false comparison.
    """
    return pl.DataFrame(
        [
            {
                "method": "TimeSHAP (Bento et al., 2021)",
                "attribution_unit": "event / timestep / feature",
                "temporal_constraint": "none (full sequence)",
                "explanation_validation": "none reported",
                "domain": "generic sequential",
            },
            {
                "method": "WindowSHAP (Nayebi et al., 2023)",
                "attribution_unit": "time window",
                "temporal_constraint": "none (full series)",
                "explanation_validation": "none reported",
                "domain": "clinical time series",
            },
            {
                "method": "SurvSHAP(t) (Krzyzinski et al., 2022)",
                "attribution_unit": "feature over time",
                "temporal_constraint": "none (full trajectory)",
                "explanation_validation": "none reported",
                "domain": "survival analysis",
            },
            {
                "method": "Multi-level PPM explanations (Wickramanayake et al., 2023)",
                "attribution_unit": "event / case",
                "temporal_constraint": "prefix-constrained",
                "explanation_validation": "not across prefixes",
                "domain": "process monitoring",
            },
            {
                "method": "Funnel-Stage SHAP (this work)",
                "attribution_unit": "feature, per funnel stage",
                "temporal_constraint": "stage cut-point (prefix only)",
                "explanation_validation": "faithfulness, stability, seed and cross-paradigm",
                "domain": "e-commerce clickstream",
            },
        ]
    )


_ESCAPES = {
    "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
    "_": r"\_", "{": r"\{", "}": r"\}",
}


def _latex_escape(value: object) -> str:
    text = "" if value is None else str(value)
    return "".join(_ESCAPES.get(ch, ch) for ch in text)


def to_latex(
    frame: pl.DataFrame,
    *,
    caption: str,
    label: str,
    float_precision: int = 4,
    column_names: dict[str, str] | None = None,
) -> str:
    """Emit a booktabs table.

    Written here rather than transcribed by hand because a table retyped into
    the manuscript is a table that can silently disagree with its source.
    """
    names = column_names or {}
    headers = [names.get(c, c.replace("_", " ")) for c in frame.columns]
    align = "".join(
        "r" if frame[c].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32) else "l"
        for c in frame.columns
    )

    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\small",
        f"\\begin{{tabular}}{{@{{}}{align}@{{}}}}",
        "\\toprule",
        " & ".join(f"\\textbf{{{_latex_escape(h)}}}" for h in headers) + " \\\\",
        "\\midrule",
    ]
    for row in frame.to_dicts():
        cells = []
        for column in frame.columns:
            value = row[column]
            if value is None:
                cells.append("--")
            elif isinstance(value, float):
                cells.append(f"{value:.{float_precision}f}")
            elif isinstance(value, bool):
                cells.append("yes" if value else "no")
            else:
                cells.append(_latex_escape(value))
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)
