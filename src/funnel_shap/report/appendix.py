"""Appendix figures and tables (protocol Sec. 15 deliverables).

These are the diagnostic and supporting exhibits that sit behind the headline
figures: ROC and precision-recall curves, confusion matrices, SHAP summary and
dependence plots, ungrouped feature importance, the baseline comparison and the
ablation ladder.

**Single-seed diagnostics, stated plainly.** Curve and instance-level plots are
drawn from one seed (42, the protocol's config-template default), because a
beeswarm or an ROC curve averaged over five seeds is not a well-defined object
and pooling them would misrepresent the spread. Every headline *number* remains
a five-seed mean; these plots are illustrative of the fitted model, and the
captions say which seed they come from. Where a quantity is genuinely
seed-averaged (feature importance, ablation) the figure says so instead.

Styling matches `figures.py` exactly -- same Okabe-Ito palette, same 300 dpi
vector-first output -- so the appendix does not read as a different document.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import polars as pl

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from ..paths import FIGURES  # noqa: E402

PALETTE = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000")
LINESTYLES = ("-", "--", "-.", ":", (0, (3, 1, 1, 1)))
GREY = "#5A5A5A"

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
        fig.savefig(path, facecolor="white")
        written.append(path)
    plt.close(fig)
    return written


def _stage_order(stages) -> list[str]:
    order = {"S1": 0, "S2": 1, "S3": 2}
    return sorted(stages, key=lambda s: order.get(s, 99))


# ---------------------------------------------------------------------------
# Predictive diagnostics
# ---------------------------------------------------------------------------


def figure_roc_curves(
    scores_by_stage: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    directory: Path = FIGURES,
    name: str = "figA1_roc_curves",
) -> list[Path]:
    """ROC per stage on the test partition.

    ROC is prevalence-free, so unlike the PR curves these three are directly
    comparable to one another; that is the whole reason both are shown.
    """
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    for i, stage in enumerate(_stage_order(scores_by_stage)):
        y, s = scores_by_stage[stage]
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(
            fpr, tpr, color=PALETTE[i], ls=LINESTYLES[i % len(LINESTYLES)], lw=1.6,
            label=f"{stage}  (AUC {roc_auc_score(y, s):.3f})",
        )
    ax.plot([0, 1], [0, 1], color=GREY, ls=":", lw=1, label="chance")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("ROC by funnel stage", fontsize=10)
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return _save(fig, name, directory)


def figure_pr_curves(
    scores_by_stage: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    directory: Path = FIGURES,
    name: str = "figA2_pr_curves",
) -> list[Path]:
    """Precision-recall per stage, each against its own chance level.

    The dashed horizontal is the stage prevalence. Without it the panels invite
    exactly the misreading the paper is about: S3's curve sits highest while
    being closest to its own baseline.
    """
    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    for i, stage in enumerate(_stage_order(scores_by_stage)):
        y, s = scores_by_stage[stage]
        precision, recall, _ = precision_recall_curve(y, s)
        prevalence = float(np.mean(y))
        # Chance level goes in the legend rather than as a floating label: S1
        # and S2 prevalences differ by 0.014 here, so side-by-side annotations
        # overprint each other.
        ax.plot(
            recall, precision, color=PALETTE[i], ls=LINESTYLES[i % len(LINESTYLES)],
            lw=1.6,
            label=(
                f"{stage}  AP {average_precision_score(y, s):.3f}"
                f"   (chance {prevalence:.3f})"
            ),
        )
        ax.axhline(prevalence, color=PALETTE[i], ls=":", lw=1, alpha=0.7)
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title("Precision-recall by funnel stage", fontsize=10)
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return _save(fig, name, directory)


def figure_confusion_matrices(
    scores_by_stage: dict[str, tuple[np.ndarray, np.ndarray]],
    thresholds: dict[str, float],
    *,
    directory: Path = FIGURES,
    name: str = "figA3_confusion_matrices",
) -> list[Path]:
    """Confusion matrices at each stage's validation-selected F1 threshold.

    Row-normalised, with raw counts printed: the stages have very different N
    and prevalence, so raw counts alone are not comparable across panels and
    proportions alone hide how thin S3 is.
    """
    stages = _stage_order(scores_by_stage)
    fig, axes = plt.subplots(1, len(stages), figsize=(3.1 * len(stages), 3.1))
    if len(stages) == 1:
        axes = [axes]

    for ax, stage in zip(axes, stages, strict=True):
        y, s = scores_by_stage[stage]
        threshold = thresholds.get(stage, 0.5)
        cm = confusion_matrix(y, (s >= threshold).astype(int), labels=[0, 1])
        with np.errstate(invalid="ignore", divide="ignore"):
            norm = cm / cm.sum(axis=1, keepdims=True)
        norm = np.nan_to_num(norm)

        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for r in range(2):
            for c in range(2):
                ax.text(
                    c, r, f"{norm[r, c]:.2f}\n({cm[r, c]:,})",
                    ha="center", va="center", fontsize=8,
                    color="white" if norm[r, c] > 0.5 else "black",
                )
        ax.set_xticks([0, 1], ["pred 0", "pred 1"], fontsize=8)
        ax.set_yticks([0, 1], ["true 0", "true 1"], fontsize=8)
        ax.set_title(f"{stage}  (threshold {threshold:.3f})", fontsize=9)
        ax.grid(False)

    fig.suptitle("Confusion matrices at the operating threshold", fontsize=10, y=1.04)
    return _save(fig, name, directory)


def figure_performance_comparison(
    summary: pl.DataFrame,
    *,
    directory: Path = FIGURES,
    name: str = "figA7_performance_comparison",
) -> list[Path]:
    """Dataset A model-family comparison, the basis for carrying LightGBM forward."""
    table = summary.sort("pr_auc_mean", descending=True)
    models = table["model"].to_list()
    y = np.arange(len(models))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 0.55 * len(models) + 1.9))

    ax1.barh(
        y, table["pr_auc_mean"], xerr=table["pr_auc_std"],
        color=PALETTE[0], height=0.6, capsize=3,
    )
    ax1.set_yticks(y, models, fontsize=8)
    ax1.invert_yaxis()
    ax1.set_xlabel("PR-AUC (mean $\\pm$ sd over seeds)")
    ax1.set_title("Primary metric", fontsize=9)

    ax2.barh(y, table["roc_auc_mean"], color=PALETTE[2], height=0.6)
    ax2.set_yticks(y, [""] * len(models))
    ax2.set_xlim(0.5, 1.0)
    ax2.set_xlabel("ROC-AUC")
    ax2.set_title("Secondary metric", fontsize=9)

    fig.suptitle(
        "Dataset A model-family comparison (month-ordered split)", fontsize=10, y=1.02
    )
    return _save(fig, name, directory)


def figure_ablation(
    ablation: pl.DataFrame,
    *,
    directory: Path = FIGURES,
    name: str = "figA8_ablation",
) -> list[Path]:
    """Ablation ladder: PR-AUC by feature set, per stage.

    Plotted per stage rather than pooled because chance level differs six-fold
    across stages, so a shared axis would compare skill against prevalence.
    """
    stages = _stage_order(ablation["stage"].unique().to_list())
    # Rung labels are long and each panel orders them differently, so the
    # panels need real separation; at default spacing a panel's labels
    # overprint its neighbour's bars.
    fig, axes = plt.subplots(
        1, len(stages), figsize=(4.0 * len(stages), 3.4),
        gridspec_kw={"wspace": 0.75},
    )
    if len(stages) == 1:
        axes = [axes]

    for ax, stage in zip(axes, stages, strict=True):
        sub = ablation.filter(pl.col("stage") == stage).sort("pr_auc_mean")
        x = np.arange(sub.height)
        ax.barh(
            x, sub["pr_auc_mean"], xerr=sub["pr_auc_std"],
            color=PALETTE[0], height=0.6, capsize=2.5,
        )
        ax.set_yticks(
            x, [s.replace("+", " + ").strip() for s in sub["feature_set"]], fontsize=7
        )
        ax.set_xlabel("PR-AUC")
        ax.set_title(stage, fontsize=9)
        lo = float(sub["pr_auc_mean"].min())
        hi = float(sub["pr_auc_mean"].max())
        pad = max((hi - lo) * 0.35, 0.004)
        ax.set_xlim(max(lo - pad, 0), hi + pad)

    fig.suptitle(
        "Feature-family ablation by stage (mean $\\pm$ sd over five seeds)",
        fontsize=10, y=1.03,
    )
    return _save(fig, name, directory)


# ---------------------------------------------------------------------------
# Attribution diagnostics
# ---------------------------------------------------------------------------


def figure_shap_summary(
    explanations: dict,
    *,
    top_n: int = 12,
    directory: Path = FIGURES,
    name: str = "figA4_shap_summary",
) -> list[Path]:
    """Beeswarm-style SHAP summary per stage, ungrouped.

    Point colour encodes the (rank-normalised) feature value, so the direction
    of effect is readable: the paper's grouped migration figure deliberately
    sums correlated features and therefore cannot show this.
    """
    stages = _stage_order(explanations)
    # Feature names are long and each panel carries its own (the ranking
    # differs by stage), so the panels need real horizontal separation --
    # at the default spacing each panel's labels overprint its neighbour's data.
    fig, axes = plt.subplots(
        1, len(stages), figsize=(4.6 * len(stages), 4.4),
        gridspec_kw={"wspace": 0.62},
    )
    if len(stages) == 1:
        axes = [axes]

    rng = np.random.default_rng(0)
    scatter = None
    for ax, stage in zip(axes, stages, strict=True):
        attribution = explanations[stage].attribution
        values = np.asarray(attribution.shap_values, dtype=float)
        matrix = np.asarray(explanations[stage].explained_matrix, dtype=float)
        names = list(attribution.feature_names)

        order = np.argsort(np.abs(values).mean(axis=0))[::-1][:top_n][::-1]
        for row, j in enumerate(order):
            v = values[:, j]
            feature = matrix[:, j]
            # Rank-normalise: raw feature scales differ by orders of magnitude,
            # so a shared colour axis would saturate on the largest feature.
            ranks = feature.argsort().argsort().astype(float)
            colour = ranks / max(ranks.max(), 1.0)
            jitter = rng.uniform(-0.16, 0.16, size=v.shape[0])
            scatter = ax.scatter(
                v, row + jitter, c=colour, cmap="coolwarm", s=3.0,
                alpha=0.65, linewidths=0, vmin=0, vmax=1, rasterized=True,
            )
        ax.axvline(0, color=GREY, lw=0.9, ls="--")
        ax.set_yticks(range(len(order)), [names[j] for j in order], fontsize=6.5)
        ax.set_xlabel("SHAP value", fontsize=8)
        ax.set_title(stage, fontsize=9)
        ax.grid(axis="y", alpha=0.15)

    if scatter is not None:
        cbar = fig.colorbar(scatter, ax=axes, fraction=0.02, pad=0.015)
        cbar.set_label("feature value (rank-normalised)", fontsize=7.5)
        cbar.ax.tick_params(labelsize=7)

    fig.suptitle("SHAP summary by stage (seed 42)", fontsize=10, y=1.02)
    return _save(fig, name, directory)


def figure_shap_dependence(
    explanations: dict,
    *,
    n_features: int = 3,
    directory: Path = FIGURES,
    name: str = "figA5_shap_dependence",
) -> list[Path]:
    """Dependence plots for each stage's strongest features.

    One row per stage, so the same feature can be followed across stages where
    it survives into the top set -- which is the migration story at the level
    of an individual variable rather than a family.
    """
    stages = _stage_order(explanations)
    fig, axes = plt.subplots(
        len(stages), n_features, figsize=(2.7 * n_features, 2.5 * len(stages))
    )
    axes = np.atleast_2d(axes)

    for r, stage in enumerate(stages):
        attribution = explanations[stage].attribution
        values = np.asarray(attribution.shap_values, dtype=float)
        matrix = np.asarray(explanations[stage].explained_matrix, dtype=float)
        names = list(attribution.feature_names)
        order = np.argsort(np.abs(values).mean(axis=0))[::-1][:n_features]

        for c in range(n_features):
            ax = axes[r, c]
            if c >= len(order):
                ax.axis("off")
                continue
            j = order[c]
            ax.scatter(
                matrix[:, j], values[:, j], s=4, alpha=0.5,
                color=PALETTE[r % len(PALETTE)], linewidths=0, rasterized=True,
            )
            ax.axhline(0, color=GREY, lw=0.8, ls="--")
            ax.set_xlabel(names[j], fontsize=7)
            if c == 0:
                ax.set_ylabel(f"{stage}\nSHAP value", fontsize=8)
            ax.tick_params(labelsize=6.5)

    fig.suptitle(
        "SHAP dependence: strongest features per stage (seed 42)", fontsize=10, y=1.005
    )
    fig.tight_layout()
    return _save(fig, name, directory)


def figure_feature_importance(
    explanations: dict,
    *,
    top_n: int = 15,
    directory: Path = FIGURES,
    name: str = "figA6_feature_importance",
) -> list[Path]:
    """Ungrouped mean |SHAP| per feature and stage.

    The paper's Figure 3 reports *grouped* shares because correlated features
    split credit arbitrarily at S1; this is the raw view behind it, and the
    difference between the two is itself informative.
    """
    frames = []
    for stage, explanation in explanations.items():
        table = explanation.attribution.global_importance()
        frames.append(table.with_columns(pl.lit(stage).alias("stage")))
    combined = pl.concat(frames)

    ranked = (
        combined.group_by("feature")
        .agg(pl.col("mean_abs_shap").max().alias("peak"))
        .sort("peak", descending=True)
        .head(top_n)
    )
    features = ranked["feature"].to_list()[::-1]
    stages = _stage_order(combined["stage"].unique().to_list())

    y = np.arange(len(features))
    height = 0.8 / len(stages)
    fig, ax = plt.subplots(figsize=(6.2, 0.32 * len(features) + 1.8))

    for i, stage in enumerate(stages):
        sub = combined.filter(pl.col("stage") == stage)
        lookup = dict(zip(sub["feature"], sub["mean_abs_shap"], strict=True))
        ax.barh(
            y + (i - (len(stages) - 1) / 2) * height,
            [lookup.get(f, 0.0) for f in features],
            height=height, color=PALETTE[i], label=stage,
        )

    ax.set_yticks(y, features, fontsize=7)
    ax.set_xlabel("mean |SHAP|")
    ax.set_title("Ungrouped feature importance by stage (seed 42)", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    return _save(fig, name, directory)
