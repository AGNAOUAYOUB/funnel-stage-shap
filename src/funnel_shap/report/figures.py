"""Publication figures (protocol Sec. 15 deliverables).

Elsevier wants vector output at a legible size; every figure is written as both
PDF (for submission) and PNG (for drafts). Styling is deliberately plain —
journals reformat anyway, and a distinctive style is a liability when the
typesetter overrides half of it.

Figure 3, the attribution migration trajectory, is the paper's centrepiece and
gets the most care: it must show absence as a *break* in the line rather than a
value of zero, and it must state its grouping rule in the caption, because both
are load-bearing for how it is read.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import polars as pl

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ..paths import FIGURES, TABLES  # noqa: E402

#: Colour-blind safe (Okabe-Ito). Journals print in greyscale often enough that
#: line style carries the distinction too, not colour alone.
PALETTE = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000")
LINESTYLES = ("-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1)), (0, (1, 1)))

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def _save(fig, name: str, directory: Path = FIGURES) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix in (".pdf", ".png"):
        path = directory / f"{name}{suffix}"
        fig.savefig(path)
        written.append(path)
    plt.close(fig)
    return written


def figure_1_data_flow(flow_csv: Path, *, directory: Path = FIGURES) -> list[Path]:
    """CONSORT-style data flow (Sec. 7.1 step 3)."""
    flow = pl.read_csv(flow_csv)
    steps = flow["step"].to_list()
    counts = flow["n_events"].to_list()

    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    positions = np.arange(len(steps))
    ax.barh(positions, counts, color=PALETTE[0], height=0.55)
    ax.set_yticks(positions)
    ax.set_yticklabels([s.replace(" ", "\n", 1) for s in steps], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("events")
    ax.xaxis.set_major_formatter(lambda x, _: f"{x / 1e6:.0f}M")

    for pos, count, removed in zip(positions, counts, flow["events_removed"], strict=True):
        label = f"{count:,}" + (f"  (−{removed:,})" if removed else "")
        ax.text(count, pos, "  " + label, va="center", fontsize=7)

    ax.set_xlim(0, max(counts) * 1.35)
    ax.set_title("Data flow: raw events to analysed sessions", fontsize=10)
    return _save(fig, "fig1_data_flow", directory)


def figure_2_improvement_curve(curve_csv: Path, *, directory: Path = FIGURES) -> list[Path]:
    """Prediction improvement by stage, raw and prevalence-normalised (RQ1/H1).

    Both panels, always. The raw panel alone reads as H1 confirmed; the lift
    panel alone hides that raw PR-AUC does rise. Side by side they make the
    point that the rise is prevalence, which is the actual finding.
    """
    curve = pl.read_csv(curve_csv)
    stages = curve["stage"].to_list()
    x = np.arange(len(stages))

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))

    ax = axes[0]
    ax.errorbar(
        x, curve["pr_auc_mean"], yerr=curve["pr_auc_std"].fill_null(0),
        marker="o", color=PALETTE[0], capsize=3, label="PR-AUC",
    )
    ax.plot(x, curve["prevalence"], marker="s", ls="--", color=PALETTE[6],
            label="chance level (prevalence)")
    ax.set_xticks(x, stages)
    ax.set_ylabel("PR-AUC")
    ax.set_title("Raw PR-AUC rises...", fontsize=9)
    ax.legend(fontsize=7, frameon=False)

    ax = axes[1]
    ax.plot(x, curve["pr_auc_lift_mean"], marker="o", color=PALETTE[1], label="PR-AUC lift")
    ax.plot(x, curve["roc_auc_mean"], marker="^", ls="-.", color=PALETTE[2], label="ROC-AUC")
    ax.axhline(1.0, color="grey", lw=0.8, ls=":")
    ax.set_xticks(x, stages)
    ax.set_ylabel("lift over chance / ROC-AUC")
    ax.set_title("...but normalised predictability falls", fontsize=9)
    ax.legend(fontsize=7, frameon=False)

    fig.suptitle("Predictive performance across funnel stages", fontsize=10, y=1.06)
    fig.tight_layout()
    return _save(fig, "fig2_improvement_curve", directory)


def pretty_group(group: str) -> str:
    """A readable name for a correlation group.

    Groups are keyed by their members joined with '+', which is auditable but
    unreadable in a legend — the two largest groups here have six and seven
    members and truncate to ellipses. Naming a group by the feature family its
    members share keeps the figure legible while the underlying key stays exact
    in the CSV, so nothing is lost for a reader who wants the membership.
    """
    members = group.split("+")
    if len(members) == 1:
        return members[0]

    # Keyword rules, checked in order. Family sets alone are not enough: the
    # category group and the product group share the same two families
    # ({counts, entropy_velocity}) and would collide on one label.
    joined = " ".join(members)
    # Order matters and is not alphabetical: the engagement group contains
    # `dwell_mean_product_s`, so a "product" rule placed first would claim it.
    # The most specific signal wins.
    rules = (
        ("categor", "navigation & category"),
        ("dwell", "engagement & tempo"),
        ("velocity", "engagement & tempo"),
        ("duration", "engagement & tempo"),
        ("price", "price"),
        ("product", "products viewed"),
    )
    for keyword, label in rules:
        if keyword in joined:
            return f"{label} ({len(members)})"
    return f"{members[0]} (+{len(members) - 1})"


def figure_3_migration(migration_csv: Path, *, directory: Path = FIGURES) -> list[Path]:
    """Attribution migration trajectories (RQ2). The centrepiece.

    Unavailable stages break the line instead of dropping to zero: a feature
    that does not exist at a stage has no attribution there, and drawing it at
    the floor would read as "this driver stopped mattering".
    """
    table = pl.read_csv(migration_csv)
    stage_order = ["S1", "S2", "S3"]
    present = [s for s in stage_order if s in set(table["stage"])]
    x = np.arange(len(present))

    totals = (
        table.filter(pl.col("available"))
        .group_by("group")
        .agg(pl.col("share").max().alias("peak"))
        .sort("peak", descending=True)
    )
    groups = totals["group"].to_list()

    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    for i, group in enumerate(groups):
        sub = table.filter(pl.col("group") == group)
        shares, mask = [], []
        for stage in present:
            row = sub.filter(pl.col("stage") == stage)
            available = bool(row["available"][0]) if row.height else False
            shares.append(row["share"][0] if (row.height and available) else np.nan)
            mask.append(available)

        ax.plot(
            x, shares, marker="o", ms=4,
            color=PALETTE[i % len(PALETTE)],
            ls=LINESTYLES[i % len(LINESTYLES)],
            label=pretty_group(group),
        )

    ax.set_xticks(x, present)
    ax.set_xlabel("funnel stage")
    ax.set_ylabel("share of total mean |SHAP|")
    ax.set_title("Attribution migration across funnel stages", fontsize=10)
    ax.legend(fontsize=6.5, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    return _save(fig, "fig3_attribution_migration", directory)


def figure_4_explanation_quality(quality_csv: Path, *, directory: Path = FIGURES) -> list[Path]:
    """Faithfulness, deletion/insertion and seed consistency (RQ3)."""
    quality = pl.read_csv(quality_csv)
    stages = quality["stage"].to_list()
    x = np.arange(len(stages))
    width = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.7))

    ax = axes[0]
    ax.bar(x, quality["faithfulness_corr"], width * 1.6, color=PALETTE[0],
           yerr=quality["faithfulness_sd"], capsize=3)
    ax.axhline(0.5, color=PALETTE[1], ls="--", lw=1, label="threshold 0.5")
    ax.set_xticks(x, stages)
    ax.set_ylabel("faithfulness correlation")
    ax.set_title("Faithfulness", fontsize=9)
    ax.legend(fontsize=7, frameon=False)

    ax = axes[1]
    ax.bar(x - width / 2, quality["deletion_auc"], width, label="deletion", color=PALETTE[1])
    ax.bar(x + width / 2, quality["insertion_auc"], width, label="insertion", color=PALETTE[2])
    ax.set_xticks(x, stages)
    ax.set_ylabel("AUC")
    ax.set_title("Deletion vs insertion", fontsize=9)
    ax.legend(fontsize=7, frameon=False)

    ax = axes[2]
    if "seed_spearman_mean" in quality.columns:
        ax.bar(x - width / 2, quality["seed_spearman_mean"], width, label="mean",
               color=PALETTE[0])
        ax.bar(x + width / 2, quality["seed_spearman_min"], width, label="worst pair",
               color=PALETTE[4])
        ax.axhline(0.6, color=PALETTE[1], ls="--", lw=1, label="threshold 0.6")
        ax.legend(fontsize=7, frameon=False)
    ax.set_xticks(x, stages)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Spearman across seeds")
    ax.set_title("Seed consistency", fontsize=9)

    fig.suptitle("Explanation quality by stage", fontsize=10, y=1.04)
    fig.tight_layout()
    return _save(fig, "fig4_explanation_quality", directory)


def figure_5_reliability(
    y_true: np.ndarray,
    before: np.ndarray,
    after: np.ndarray,
    *,
    directory: Path = FIGURES,
    labels: tuple[str, str] = ("calibrated on Nov (25.4%)", "calibrated on train (12.2%)"),
) -> list[Path]:
    """Reliability under prior-probability shift, before and after the A18 fix."""
    from ..evaluate.metrics import expected_calibration_error, reliability_curve

    fig, ax = plt.subplots(figsize=(4.0, 3.6))
    ax.plot([0, 1], [0, 1], color="grey", lw=0.8, ls=":", label="perfect calibration")

    for scores, label, colour, marker in zip(
        (before, after), labels, (PALETTE[1], PALETTE[2]), ("o", "s"), strict=True
    ):
        curve = reliability_curve(y_true, scores, n_bins=8)
        ece = expected_calibration_error(y_true, scores, strategy="quantile")
        ax.plot(curve["mean_predicted"], curve["fraction_positive"],
                marker=marker, ms=4, color=colour, label=f"{label}\nECE {ece:.3f}")

    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed conversion rate")
    ax.set_title("Calibration under prevalence shift", fontsize=10)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    return _save(fig, "fig5_reliability", directory)


def build_all(suffix: str, *, directory: Path = FIGURES) -> dict[str, list[Path]]:
    """Build every figure whose input table exists."""
    built: dict[str, list[Path]] = {}
    candidates = {
        "fig1": (TABLES / "data_flow_gap30.csv", figure_1_data_flow),
        "fig2": (TABLES / f"improvement_curve_{suffix}.csv", figure_2_improvement_curve),
        "fig3": (TABLES / f"attribution_migration_{suffix}.csv", figure_3_migration),
        "fig4": (TABLES / f"explanation_quality_{suffix}.csv", figure_4_explanation_quality),
    }
    for name, (path, fn) in candidates.items():
        if path.exists():
            built[name] = fn(path, directory=directory)
    return built
