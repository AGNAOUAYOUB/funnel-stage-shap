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

import numpy as np
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


def error_analysis_table(
    scores_by_stage: dict[str, tuple],
    thresholds: dict[str, float],
) -> pl.DataFrame:
    """Confusion counts at the operating threshold, against a trivial rule.

    The comparator is the always-positive classifier, whose F1 is
    ``2*pi/(1+pi)`` at prevalence ``pi``. It is the right null for a decision
    system: a targeting rule that flags everyone costs nothing to build, so a
    model earns its place only by beating it. Reporting precision and recall
    without this reference makes a degenerate operating point look respectable
    -- an F1 of 0.69 reads well until one notices that flagging every session
    scores 0.685.

    `flag_rate` is the share of sessions the model would refer for
    intervention, and `fp_per_tp` the false positives incurred per true
    positive: the two quantities an operator actually budgets against.
    """
    rows = []
    for stage in sorted(scores_by_stage, key=lambda s: {"S1": 0, "S2": 1, "S3": 2}.get(s, 99)):
        y_true, y_score = scores_by_stage[stage]
        y_true = np.asarray(y_true).astype(int).ravel()
        y_score = np.asarray(y_score, dtype=float).ravel()
        threshold = float(thresholds.get(stage, 0.5))
        predicted = (y_score >= threshold).astype(int)

        tp = int(((predicted == 1) & (y_true == 1)).sum())
        fp = int(((predicted == 1) & (y_true == 0)).sum())
        fn = int(((predicted == 0) & (y_true == 1)).sum())
        tn = int(((predicted == 0) & (y_true == 0)).sum())

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        prevalence = float(y_true.mean())
        trivial_f1 = 2 * prevalence / (1 + prevalence) if prevalence else 0.0

        rows.append(
            {
                "stage": stage,
                "n_test": int(y_true.size),
                "prevalence": round(prevalence, 4),
                "threshold": round(threshold, 4),
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                # The margin is derived from the *rounded* components so the
                # published columns subtract correctly. Rounding the exact
                # difference independently can leave the printed table off by
                # one in the last place, which reads as an arithmetic error.
                "f1": round(f1, 4),
                "trivial_positive_f1": round(trivial_f1, 4),
                "f1_over_trivial": round(round(f1, 4) - round(trivial_f1, 4), 4),
                "flag_rate": round(float((predicted == 1).mean()), 4),
                "fp_per_tp": round(fp / tp, 2) if tp else None,
            }
        )
    return pl.DataFrame(rows)


def decision_economics_table(per_seed: pl.DataFrame) -> pl.DataFrame:
    """Contacts required per conversion *beyond* a blanket targeting rule.

    Computed over the full seed list rather than one seed, because the
    quantity is reported as a general property. That matters more here than
    elsewhere: the S3 numerator (precision minus prevalence) sits close to
    zero, so its reciprocal is unstable, and a single-seed point estimate
    would imply a precision the data do not support. The per-seed range is
    returned alongside the mean so the instability is visible rather than
    averaged away.
    """
    rows = []
    frame = per_seed.filter(pl.col("feature_set") == "full")
    for stage in sorted(
        frame["stage"].unique().to_list(), key=lambda s: {"S1": 0, "S2": 1, "S3": 2}.get(s, 99)
    ):
        sub = frame.filter(pl.col("stage") == stage)
        prevalence = sub["prevalence"].to_numpy()
        precision = sub["precision"].to_numpy()
        incremental = precision - prevalence

        positive = incremental[incremental > 0]
        contacts = 1.0 / positive if positive.size else np.array([])
        mean_inc = float(incremental.mean())

        rows.append(
            {
                "stage": stage,
                "n_seeds": int(sub.height),
                "prevalence": round(float(prevalence.mean()), 4),
                "precision_mean": round(float(precision.mean()), 4),
                "precision_sd": round(float(precision.std(ddof=1)), 4),
                "incremental_mean": round(mean_inc, 4),
                "incremental_sd": round(float(incremental.std(ddof=1)), 4),
                # Distance from zero in seed standard deviations: below about
                # two, the stage's targeting value is not established.
                "sd_from_zero": round(
                    mean_inc / float(incremental.std(ddof=1)), 2
                ) if incremental.std(ddof=1) > 0 else None,
                "contacts_per_incremental": round(1.0 / mean_inc, 1) if mean_inc > 0 else None,
                "contacts_min": round(float(contacts.min()), 1) if contacts.size else None,
                "contacts_max": round(float(contacts.max()), 1) if contacts.size else None,
                "seeds_at_or_below_chance": int((incremental <= 0).sum()),
            }
        )
    return pl.DataFrame(rows)


def entropy_availability_table(
    explanations: dict,
    feature: str = "category_entropy",
) -> pl.DataFrame:
    """Separate a definitional artefact from a behavioural signal.

    Category entropy is zero for a prefix confined to a single category, so a
    stage where more prefixes span several categories will mechanically show
    more entropy attribution. This reports, per stage, the share of explained
    prefixes for which the feature is non-degenerate, and the attribution
    share both overall and restricted to those prefixes. If the restricted
    share still peaks at the same stage, the behavioural reading survives the
    correction; if it does not, the peak was availability.
    """
    rows = []
    for stage in sorted(
        explanations, key=lambda s: {"S1": 0, "S2": 1, "S3": 2}.get(s, 99)
    ):
        explanation = explanations[stage]
        names = [str(n) for n in explanation.attribution.feature_names]
        if feature not in names:
            rows.append({"stage": stage, "feature_present": False})
            continue
        j = names.index(feature)
        matrix = np.asarray(explanation.explained_matrix, dtype=float)
        shap = np.asarray(explanation.attribution.shap_values, dtype=float)

        defined = matrix[:, j] > 0
        overall = np.abs(shap).mean(axis=0)
        share_overall = float(overall[j] / overall.sum()) if overall.sum() else 0.0

        if defined.any():
            restricted = np.abs(shap[defined]).mean(axis=0)
            share_defined = float(restricted[j] / restricted.sum()) if restricted.sum() else 0.0
        else:
            share_defined = None

        rows.append(
            {
                "stage": stage,
                "feature_present": True,
                "n_explained": int(matrix.shape[0]),
                "share_prefixes_defined": round(float(defined.mean()), 4),
                "attribution_share_overall": round(share_overall, 4),
                "attribution_share_when_defined": (
                    round(share_defined, 4) if share_defined is not None else None
                ),
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
