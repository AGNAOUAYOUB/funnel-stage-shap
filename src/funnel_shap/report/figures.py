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


# ---------------------------------------------------------------------------
# Figure 7 — stage-wise predictive performance
# ---------------------------------------------------------------------------


def figure_7_stage_performance(
    per_seed_csv: Path, *, directory: Path = FIGURES
) -> list[Path]:
    """PR-AUC, ROC-AUC and calibration across S1 -> S2 -> S3.

    Three panels because the three quantities behave differently and collapsing
    them onto one axis would hide that. PR-AUC is drawn against its own chance
    level, since chance-level PR-AUC *is* the prevalence and that rises sixfold
    across the stages; ROC-AUC is prevalence-independent and needs no such
    reference; calibration is on a different scale entirely and is shown as
    error, where lower is better.
    """
    table = pl.read_csv(per_seed_csv).filter(pl.col("feature_set") == "full")
    agg = (
        table.group_by("stage")
        .agg(
            pl.col("prevalence").first().alias("prevalence"),
            pl.col("pr_auc").mean().alias("pr_auc"),
            pl.col("pr_auc_ci_low").mean().alias("pr_lo"),
            pl.col("pr_auc_ci_high").mean().alias("pr_hi"),
            pl.col("pr_auc_lift").mean().alias("lift"),
            pl.col("roc_auc").mean().alias("roc_auc"),
            pl.col("roc_auc_ci_low").mean().alias("roc_lo"),
            pl.col("roc_auc_ci_high").mean().alias("roc_hi"),
            pl.col("ece").mean().alias("ece"),
            pl.col("brier").mean().alias("brier"),
        )
        .sort("stage")
    )

    stages = agg["stage"].to_list()
    x = np.arange(len(stages))
    fig, axes = plt.subplots(1, 3, figsize=(7.8, 3.0))

    ax = axes[0]
    ax.errorbar(
        x, agg["pr_auc"],
        yerr=[agg["pr_auc"] - agg["pr_lo"], agg["pr_hi"] - agg["pr_auc"]],
        marker="o", color=PALETTE[0], capsize=3, lw=1.6, label="PR-AUC",
    )
    ax.plot(x, agg["prevalence"], marker="s", ls="--", color=PALETTE[6],
            lw=1.2, label="chance (prevalence)")
    for xi, pr, lift in zip(x, agg["pr_auc"], agg["lift"], strict=True):
        ax.annotate(f"lift {lift:.2f}", (xi, pr), textcoords="offset points",
                    xytext=(4, 12), ha="left", fontsize=6.8, color=PALETTE[0])
    ax.set_xticks(x, stages)
    ax.set_xlim(-0.45, len(stages) - 0.35)
    ax.set_ylim(0, 0.78)
    ax.set_ylabel("PR-AUC")
    ax.set_title("(a) PR-AUC vs chance", fontsize=9)
    ax.legend(fontsize=6.4, frameon=False, loc="upper left")

    ax = axes[1]
    ax.errorbar(
        x, agg["roc_auc"],
        yerr=[agg["roc_auc"] - agg["roc_lo"], agg["roc_hi"] - agg["roc_auc"]],
        marker="^", color=PALETTE[2], capsize=3, lw=1.6,
    )
    ax.axhline(0.5, color="grey", lw=0.9, ls=":")
    ax.text(len(stages) - 1, 0.503, "chance", fontsize=6.4, color="grey",
            ha="right", va="bottom")
    ax.set_xticks(x, stages)
    ax.set_ylim(0.48, 0.70)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("(b) ROC-AUC (prevalence-free)", fontsize=9)

    # Panel (c): ECE only as bars. Brier is deliberately NOT plotted beside it:
    # it is a proper scoring rule combining calibration with refinement, and its
    # scale tracks the base rate (the reference Brier for a prevalence-only
    # forecast is p(1-p), which is 0.068 at S1 but 0.250 at S3). Drawn on a
    # shared axis, S3's Brier towers over the others for reasons that have
    # nothing to do with calibration quality. It is reported as text instead.
    ax = axes[2]
    ax.bar(x, agg["ece"], 0.5, color=PALETTE[1])
    for xi, e, b, p in zip(x, agg["ece"], agg["brier"], agg["prevalence"], strict=True):
        ax.annotate(f"{e:.3f}", (xi, e), textcoords="offset points",
                    xytext=(0, 3), ha="center", fontsize=7, color=PALETTE[1],
                    weight="bold")
        ax.annotate(f"Brier {b:.3f}\n(ref {p * (1 - p):.3f})", (xi, 0),
                    textcoords="offset points", xytext=(0, -26), ha="center",
                    fontsize=5.9, color="grey", annotation_clip=False,
                    linespacing=1.3)
    ax.set_xticks(x, stages)
    ax.set_ylim(0, max(agg["ece"]) * 1.45)
    ax.set_ylabel("Expected Calibration Error")
    ax.set_title("(c) Calibration (lower is better)", fontsize=9)

    # Panel (c) carries the Brier annotations below its axis, so it omits the
    # axis label rather than colliding with it; the tick labels already name the
    # stages and the other two panels supply the caption for the row.
    for ax in axes[:2]:
        ax.set_xlabel("funnel stage")

    fig.suptitle("Stage-wise predictive performance", fontsize=10.5, y=1.06)
    fig.tight_layout()
    return _save(fig, "fig7_stage_performance", directory)


# ---------------------------------------------------------------------------
# Figure 8 — alluvial view of attribution migration
# ---------------------------------------------------------------------------


def figure_8_migration_alluvial(
    migration_csv: Path, *, directory: Path = FIGURES
) -> list[Path]:
    """Alluvial (Sankey-style) view of how attribution mass shifts across stages.

    The same quantity as the trajectory figure, shown as flow rather than as
    lines. Shares sum to one at each stage, so the stacked column height is
    constant and a ribbon's changing thickness reads directly as a driver
    gaining or losing attribution mass.

    Ordering within each column is held fixed across stages. Sorting each column
    independently would make ribbons cross for cosmetic reasons and imply
    movement between drivers that does not exist -- attribution mass is not
    transferred from one feature to another, it is recomputed at each stage.
    """
    import matplotlib.patches as mpatches

    table = pl.read_csv(migration_csv).filter(pl.col("available"))
    stage_order = [s for s in ("S1", "S2", "S3") if s in set(table["stage"])]

    groups = (
        table.group_by("group").agg(pl.col("share").max().alias("peak"))
        .sort("peak", descending=True)["group"].to_list()
    )

    shares = {
        stage: {
            r["group"]: (r["share"] or 0.0)
            for r in table.filter(pl.col("stage") == stage).to_dicts()
        }
        for stage in stage_order
    }

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    # Generous side margins: the in-bar value labels are wider than the bars
    # themselves and ran off the canvas at tighter limits.
    ax.set_xlim(-0.34, len(stage_order) - 1 + 1.02)
    ax.set_ylim(-0.09, 1.05)
    ax.axis("off")

    bar_w = 0.10
    spans_by_stage: dict[str, list[tuple[float, float]]] = {}
    for si, stage in enumerate(stage_order):
        y = 0.0
        spans = []
        for gi, group in enumerate(groups):
            h = shares[stage].get(group, 0.0)
            spans.append((y, y + h))
            if h > 0:
                ax.add_patch(
                    mpatches.Rectangle(
                        (si - bar_w / 2, y), bar_w, h,
                        facecolor=PALETTE[gi % len(PALETTE)], edgecolor="white",
                        linewidth=0.7, zorder=3,
                    )
                )
                if h > 0.06:
                    ax.text(si, y + h / 2, f"{h:.2f}", ha="center", va="center",
                            fontsize=6.4, color="white", weight="bold", zorder=4)
            y += h
        spans_by_stage[stage] = spans
        ax.text(si, -0.045, stage, ha="center", va="top", fontsize=10.5, weight="bold")

    for si in range(len(stage_order) - 1):
        left, right = stage_order[si], stage_order[si + 1]
        for gi in range(len(groups)):
            l0, l1 = spans_by_stage[left][gi]
            r0, r1 = spans_by_stage[right][gi]
            if (l1 - l0) <= 0 and (r1 - r0) <= 0:
                continue
            x0, x1 = si + bar_w / 2, si + 1 - bar_w / 2
            t = np.linspace(0, 1, 120)
            smooth = 3 * t**2 - 2 * t**3
            xs = x0 + (x1 - x0) * t
            lower = l0 + (r0 - l0) * smooth
            upper = l1 + (r1 - l1) * smooth
            ax.fill_between(xs, lower, upper, color=PALETTE[gi % len(PALETTE)],
                            alpha=0.32, linewidth=0, zorder=1)

    last = stage_order[-1]
    for gi, group in enumerate(groups):
        r0, r1 = spans_by_stage[last][gi]
        if (r1 - r0) <= 0.015:
            continue
        ax.text(len(stage_order) - 1 + bar_w, (r0 + r1) / 2,
                "  " + pretty_group(group), va="center", ha="left", fontsize=7.4,
                color=PALETTE[gi % len(PALETTE)])

    ax.set_title(
        "Attribution migration across funnel stages\n"
        "(ribbon thickness = share of total mean |SHAP| at that stage)",
        fontsize=10, pad=10, linespacing=1.4,
    )
    return _save(fig, "fig8_migration_alluvial", directory)
