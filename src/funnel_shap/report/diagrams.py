"""Conceptual diagrams for the manuscript (protocol Sec. 15 deliverables).

These are schematics rather than data plots: they describe the framework, its
positioning, and its logic. They are drawn in matplotlib rather than TikZ for a
practical reason -- matplotlib output can be rendered and inspected during
authoring, whereas TikZ would have to be committed unseen without a LaTeX
engine available. Both produce vector PDF; the visual style deliberately matches
the result figures so the two sets read as one system.

Every diagram is written to PDF (submission) and PNG (drafts and review).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from ..paths import FIGURES  # noqa: E402

#: Okabe-Ito, matching the result figures.
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PINK = "#CC79A7"
YELLOW = "#E69F00"
SKY = "#56B4E9"
GREY = "#5A5A5A"
LIGHT = "#F2F2F2"

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 8.5,
        "axes.grid": False,
    }
)


def _save(fig, name: str, directory: Path) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix in (".pdf", ".png"):
        path = directory / f"{name}{suffix}"
        fig.savefig(path, facecolor="white")
        written.append(path)
    plt.close(fig)
    return written


def _canvas(width: float, height: float):
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    return fig, ax


def _box(
    ax, x, y, w, h, text, *, fc="white", ec=GREY, fontsize=8, weight="normal",
    tc="black", radius=1.6, lw=1.1, ls="-", zorder=2,
):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls, zorder=zorder,
        )
    )
    ax.text(
        x + w / 2, y + h / 2, text, ha="center", va="center",
        fontsize=fontsize, color=tc, weight=weight, zorder=zorder + 1, linespacing=1.35,
    )


def _arrow(ax, start, end, *, color=GREY, lw=1.2, style="-|>", ls="-", zorder=1, rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            start, end, arrowstyle=style, mutation_scale=11,
            color=color, linewidth=lw, linestyle=ls, zorder=zorder,
            connectionstyle=f"arc3,rad={rad}", shrinkA=1, shrinkB=1,
        )
    )


# ---------------------------------------------------------------------------
# D1 — Overall framework architecture
# ---------------------------------------------------------------------------


def diagram_architecture(directory: Path = FIGURES) -> list[Path]:
    """End-to-end pipeline, clickstream to validated explanations."""
    fig, ax = _canvas(9.2, 6.4)

    # Lane labels live in a left gutter (x < 9) so no arrow or box can collide
    # with them; the first attempt placed them inside the lane and they were
    # struck through by both the Layer 1 box and the feeder arrows.
    gutter = 9.0
    lanes = [
        (79, 96, "DATA", LIGHT, BLUE),
        (58, 77, "JOURNEY\nRECONSTRUCTION", "#EAF3F8", GREEN),
        (37, 56, "MODELLING", "#FDF0E7", ORANGE),
        (3, 35, "EXPLANATION\nAND VALIDATION", "#EAF6F1", PINK),
    ]
    for y0, y1, label, colour, tcol in lanes:
        ax.add_patch(
            mpatches.Rectangle((gutter, y0), 100 - gutter, y1 - y0,
                               facecolor=colour, edgecolor="none", zorder=0)
        )
        ax.text(gutter - 1.4, (y0 + y1) / 2, label, fontsize=6.8, color=tcol,
                weight="bold", va="center", ha="center", rotation=90, linespacing=1.25)

    # Data lane
    _box(ax, 11, 82, 20, 10, "Raw clickstream\n42.4M events", fc="white", ec=BLUE)
    _box(ax, 34, 82, 20, 10, "Clean + dedup\nschema validation", fc="white", ec=BLUE)
    _box(ax, 57, 82, 18, 10, "Sessionise\n30-min gap", fc="white", ec=BLUE)
    _box(ax, 78, 82, 20, 10, "Seeded subsample\n485,459 sessions", fc="white", ec=BLUE)
    for x in (31, 54, 75):
        _arrow(ax, (x, 87), (x + 3, 87), color=BLUE)

    # Journey lane
    _box(ax, 11, 60, 25, 11,
         "Stage cut-points\nS1 2nd interaction\nS2 2nd browsing signal\nS3 1st cart",
         fc="white", ec=GREEN, fontsize=7.0)
    _box(ax, 39, 60, 24, 11,
         "Nested prefixes\nS1 $\\subseteq$ S2 $\\subseteq$ S3\nno purchase event",
         fc="white", ec=GREEN, fontsize=7.0, weight="bold")
    _box(ax, 66, 60, 32, 11,
         "Prefix features per stage\ncounts · temporal · engagement\nentropy · velocity · price",
         fc="white", ec=GREEN, fontsize=7.0)
    _arrow(ax, (36, 65.5), (39, 65.5), color=GREEN)
    _arrow(ax, (63, 65.5), (66, 65.5), color=GREEN)
    _arrow(ax, (51, 82), (51, 71), color=BLUE)

    # Modelling lane
    _box(ax, 11, 39, 22, 11, "Frozen splits\ntemporal (headline)\ngrouped (robustness)",
         fc="white", ec=ORANGE, fontsize=7.0)
    _box(ax, 36, 39, 26, 11, "Stage models per S$_k$\nLightGBM · Optuna\n5 seeds · calibrated",
         fc="white", ec=ORANGE, fontsize=7.0)
    _box(ax, 65, 39, 33, 11, "Sequence model per S$_k$\nGRU on same prefixes",
         fc="white", ec=ORANGE, fontsize=7.0)
    _arrow(ax, (33, 44.5), (36, 44.5), color=ORANGE)
    _arrow(ax, (62, 44.5), (65, 44.5), color=ORANGE)
    _arrow(ax, (51, 60), (51, 50), color=GREEN)

    # Explanation lane. Layers sit directly beneath their sources so the feeder
    # arrows are short verticals rather than long diagonals across the lane.
    _box(ax, 36, 20, 26, 11, "Layer 1\nInterventional TreeSHAP\nper stage, grouped",
         fc="white", ec=PINK, fontsize=7.0)
    _box(ax, 65, 20, 33, 11, "Layer 2\nTimeSHAP on the GRU\nsame prefixes",
         fc="white", ec=PINK, fontsize=7.0)
    _box(ax, 11, 20, 22, 11, "Layer 3\nfaithfulness · stability\nseed · cross-paradigm",
         fc="white", ec=PINK, fontsize=7.0)
    _arrow(ax, (49, 39), (49, 31), color=ORANGE)
    _arrow(ax, (81, 39), (81, 31), color=ORANGE)
    # Layers 1 and 2 are parallel attribution sources, not a sequence: both
    # feed the validation layer. An arrow between them would imply a dependency
    # that does not exist.
    _arrow(ax, (36, 25.5), (33, 25.5), color=PINK)
    _arrow(ax, (65, 28), (33, 28.5), color=PINK, rad=-0.10)

    _box(ax, 22, 5, 56, 9,
         "Validated stage-conditioned explanations\nattribution trajectory + quality evidence",
         fc="#DCEFE7", ec=GREEN, fontsize=8.2, weight="bold")
    _arrow(ax, (22, 20), (35, 14), color=PINK, lw=1.5, rad=0.15)

    ax.text(50, 99, "Funnel-Stage SHAP: end-to-end pipeline",
            ha="center", va="top", fontsize=11, weight="bold")
    return _save(fig, "diag1_architecture", directory)


# ---------------------------------------------------------------------------
# D2 — Research gap and positioning
# ---------------------------------------------------------------------------


def diagram_gap(directory: Path = FIGURES) -> list[Path]:
    """Three-circle positioning: journey modelling, XAI, explanation validation."""
    fig, ax = _canvas(8.0, 7.6)
    ax.set_aspect("equal", adjustable="box")

    r = 24
    centres = {
        "journey": (36, 62),
        "xai": (64, 62),
        "validation": (50, 38),
    }
    colours = {"journey": BLUE, "xai": ORANGE, "validation": GREEN}
    for key, (cx, cy) in centres.items():
        ax.add_patch(
            plt.Circle((cx, cy), r, facecolor=colours[key], edgecolor=colours[key],
                       alpha=0.13, linewidth=1.6, zorder=1)
        )

    ax.text(22, 88, "Consumer journey\nmodelling", ha="center", fontsize=9,
            weight="bold", color=BLUE)
    ax.text(78, 88, "Shapley attribution\nfor sequences", ha="center", fontsize=9,
            weight="bold", color=ORANGE)
    ax.text(50, 6.5, "Explanation quality\nevaluation", ha="center", fontsize=9,
            weight="bold", color=GREEN)

    # Exemplars in the exclusive regions
    ax.text(22, 68, "Moe (2003)\nBucklin & Sismeiro (2003)\nLemon & Verhoef (2016)",
            ha="center", fontsize=6.8, color=GREY, linespacing=1.4)
    ax.text(78, 68, "Lundberg & Lee (2017)\nLundberg et al. (2020)\nBento et al. (2021)",
            ha="center", fontsize=6.8, color=GREY, linespacing=1.4)
    ax.text(50, 22, "Bhatt et al. (2020)\nPetsiuk et al. (2018)\nHedström et al. (2023)",
            ha="center", fontsize=6.8, color=GREY, linespacing=1.4)

    # Pairwise intersections
    ax.text(50, 74, "applied e-commerce XAI\n(static, unvalidated)", ha="center",
            fontsize=6.9, color=GREY, style="italic", linespacing=1.35)
    ax.text(30, 45, "staged prediction\nwithout attribution", ha="center",
            fontsize=6.9, color=GREY, style="italic", linespacing=1.35)
    ax.text(70, 45, "validated attribution,\nno journey structure", ha="center",
            fontsize=6.9, color=GREY, style="italic", linespacing=1.35)

    # The gap
    ax.add_patch(plt.Circle((50, 55), 8.2, facecolor="white", edgecolor=PINK,
                            linewidth=2.0, zorder=4))
    ax.text(50, 55, "THIS\nWORK", ha="center", va="center", fontsize=8.6,
            weight="bold", color=PINK, zorder=5, linespacing=1.3)

    ax.annotate(
        "Stage-conditioned attribution,\nleakage-free by construction,\nvalidated per stage",
        xy=(56.5, 52), xytext=(88, 30), fontsize=7.4, color=PINK,
        ha="center", linespacing=1.45,
        arrowprops=dict(arrowstyle="-|>", color=PINK, lw=1.2,
                        connectionstyle="arc3,rad=-0.25"),
    )

    ax.text(50, 97, "Research gap and positioning", ha="center", va="top",
            fontsize=11, weight="bold")
    return _save(fig, "diag2_gap", directory)


# ---------------------------------------------------------------------------
# D3 — Anti-leakage prefix protocol
# ---------------------------------------------------------------------------


def diagram_prefix(directory: Path = FIGURES) -> list[Path]:
    """Stage cut-points and nested prefix construction on one worked session.

    The example is chosen so a reader can *verify* each cut-point from the
    figure alone. Product and category are both shown, because S2's trigger is
    a repeat view or a category switch and neither is checkable from a product
    label by itself. Browsing signals are marked explicitly, so the claim that
    S2 closes on the second one can be audited rather than taken on trust.
    """
    fig, ax = _canvas(9.8, 6.0)

    # (kind, product, category, is_browsing_signal, why)
    events = [
        ("view", "P1", "X", False, ""),
        ("view", "P2", "X", False, ""),
        ("view", "P3", "Y", True, "cat.\nswitch"),
        ("view", "P1", "X", True, "repeat\n+ switch"),
        ("view", "P4", "Y", True, ""),
        ("cart", "P4", "Y", False, ""),
        ("view", "P5", "Y", False, ""),
        ("cart", "P5", "Y", False, ""),
        ("buy", "P5", "Y", False, ""),
    ]
    x0, dx, y = 9.0, 9.7, 62
    xs = [x0 + i * dx for i in range(len(events))]
    bw, bh = 5.4, 5.4

    ax.plot([x0 - 4, xs[-1] + 4], [y, y], color=GREY, lw=1.0, zorder=0)
    for i, ((kind, prod, cat, signal, why), x) in enumerate(zip(events, xs, strict=True)):
        forbidden = kind == "buy"
        colour = {"view": BLUE, "cart": ORANGE, "buy": PINK}[kind]
        _box(ax, x - bw / 2, y - bh / 2, bw, bh,
             {"view": "view", "cart": "cart", "buy": "buy"}[kind],
             fc="white" if forbidden else colour, ec=colour, lw=1.5,
             fontsize=6.2, tc=colour if forbidden else "white",
             weight="bold", radius=1.0, ls="--" if forbidden else "-", zorder=3)
        ax.text(x, y - 6.4, f"$e_{i}$", ha="center", fontsize=6.4, color=GREY)
        ax.text(x, y + 5.0, f"{prod}\ncat {cat}", ha="center", fontsize=6.2,
                color=GREY, linespacing=1.25)
        if signal:
            ax.text(x, y - 11.2, "browsing\nsignal", ha="center", fontsize=5.9,
                    color=ORANGE, weight="bold", linespacing=1.2)
            if why:
                ax.text(x, y - 16.4, why, ha="center", fontsize=5.6, color=GREY,
                        linespacing=1.2)

    ax.text(xs[-1] + 4.6, y, "purchase\n= the label", fontsize=7.0, color=PINK,
            va="center", weight="bold", linespacing=1.3)

    cuts = [(1, "S1", GREEN, "2nd product\ninteraction"),
            (3, "S2", ORANGE, "2nd browsing\nsignal"),
            (5, "S3", PINK, "1st cart\nevent")]
    for idx, name, colour, note in cuts:
        xc = xs[idx] + dx / 2
        ax.plot([xc, xc], [y - 8, y + 12], color=colour, lw=1.3, ls=(0, (4, 2)), zorder=1)
        ax.text(xc, y + 13.5, f"{name} cut", ha="center", fontsize=7.4,
                color=colour, weight="bold")
        ax.text(xc, y + 18.5, note, ha="center", fontsize=6.2, color=GREY, linespacing=1.3)

    bars = [(1, "S1", GREEN, 34), (3, "S2", ORANGE, 26), (5, "S3", PINK, 18)]
    for idx, name, colour, by in bars:
        left = x0 - 4.6
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (left, by), xs[idx] + dx / 2 - left, 5.6,
                boxstyle="round,pad=0,rounding_size=1.1",
                facecolor=colour, edgecolor="none", alpha=0.30, zorder=1,
            )
        )
        ax.text(left + 1.6, by + 2.8, f"{name} prefix", fontsize=7.2, color=colour,
                weight="bold", va="center")
        ax.text(xs[idx] + dx / 2 + 1.8, by + 2.8, f"{idx + 1} events", fontsize=6.6,
                color=GREY, va="center")

    ax.add_patch(
        mpatches.Rectangle((xs[-1] - 5.0, 16), 10, 54, facecolor=PINK, alpha=0.10,
                           edgecolor=PINK, linestyle="--", linewidth=1.2, zorder=0)
    )
    ax.text(xs[-1], 13.0, "never in\nany prefix", ha="center", fontsize=6.4,
            color=PINK, weight="bold", linespacing=1.25)

    ax.text(48, 6.5,
            "Features at stage S$_k$ use only the prefix up to S$_k$'s cut-point. Prefixes "
            "nest (S1 $\\subseteq$ S2 $\\subseteq$ S3) and are\nclipped strictly before the "
            "first purchase, so no feature can be computed from the outcome it predicts.",
            ha="center", va="center", fontsize=7.4, color="black", linespacing=1.6)

    ax.text(50, 99, "Anti-leakage prefix protocol", ha="center", va="top",
            fontsize=11, weight="bold")
    return _save(fig, "diag3_prefix", directory)


# ---------------------------------------------------------------------------
# D4 — Design science build–evaluate cycle
# ---------------------------------------------------------------------------


def diagram_dsr(directory: Path = FIGURES) -> list[Path]:
    """Hevner-style three-column DSR frame with the build-evaluate loop."""
    fig, ax = _canvas(9.2, 6.0)

    for x0, w, title, colour in [
        (2, 26, "ENVIRONMENT", BLUE),
        (33, 34, "DESIGN SCIENCE", ORANGE),
        (72, 26, "KNOWLEDGE BASE", GREEN),
    ]:
        ax.add_patch(
            mpatches.Rectangle((x0, 11), w, 75, facecolor="none", edgecolor=colour,
                               linewidth=1.4, zorder=1)
        )
        ax.text(x0 + w / 2, 88.5, title, ha="center", fontsize=9, weight="bold",
                color=colour)

    _box(ax, 4.5, 62, 21, 20,
         "People\nanalysts, marketers\n\nOrganisations\nonline retail\n\nTechnology\n"
         "clickstream logs",
         fc="white", ec=BLUE, fontsize=7.0)
    _box(ax, 4.5, 34, 21, 24,
         "Business need\n\nWhich behavioural\ndrivers matter,\n"
         "at which point\nin the journey,\nand can the\nanswer be trusted?",
         fc="#EAF3F8", ec=BLUE, fontsize=7.0)

    _box(ax, 35.5, 60, 29, 22,
         "BUILD\n\nFunnel-stage SHAP artefact\n"
         "· stage cut-points + prefixes\n· stage models per S$_k$\n"
         "· three explanation layers",
         fc="white", ec=ORANGE, fontsize=7.2, weight="normal")
    _box(ax, 35.5, 26, 29, 22,
         "EVALUATE\n\n· PR-AUC lift, ROC-AUC, ECE\n"
         "· faithfulness, stability\n· seed + cross-paradigm\n"
         "· bootstrap CIs, Holm",
         fc="white", ec=ORANGE, fontsize=7.2)

    _arrow(ax, (42, 60), (42, 48), color=ORANGE, lw=1.6, rad=0.32)
    _arrow(ax, (58, 48), (58, 60), color=ORANGE, lw=1.6, rad=0.32)
    ax.text(50, 54, "refine", ha="center", fontsize=7.4, color=ORANGE,
            style="italic", weight="bold")
    ax.text(50, 21.5, "23 logged amendments\n(3 cut-point redefinitions, 2 retractions)",
            ha="center", fontsize=6.9, color=GREY, linespacing=1.35)

    _box(ax, 74.5, 62, 21, 20,
         "Foundations\nShapley values\nTreeSHAP, TimeSHAP\njourney theory",
         fc="white", ec=GREEN, fontsize=7.0)
    _box(ax, 74.5, 34, 21, 24,
         "Methodologies\npre-registration\nbootstrap CIs\nDeLong, Holm\n"
         "faithfulness metrics",
         fc="#EAF6F1", ec=GREEN, fontsize=7.0)

    _arrow(ax, (25.5, 72), (35.5, 72), color=BLUE)
    ax.text(30.5, 74.5, "relevance", ha="center", fontsize=6.8, color=BLUE, style="italic")
    _arrow(ax, (74.5, 72), (64.5, 72), color=GREEN)
    ax.text(69.5, 74.5, "rigour", ha="center", fontsize=6.8, color=GREEN, style="italic")

    _arrow(ax, (35.5, 40), (25.5, 40), color=ORANGE, ls=(0, (3, 2)))
    ax.text(30.5, 42.5, "application", ha="center", fontsize=6.8, color=ORANGE,
            style="italic")
    _arrow(ax, (64.5, 40), (74.5, 40), color=ORANGE, ls=(0, (3, 2)))
    ax.text(69.5, 42.5, "additions", ha="center", fontsize=6.8, color=ORANGE,
            style="italic")

    _box(ax, 16, 1.0, 68, 6.5,
         "Contribution: stage conditioning is a modelling commitment, not a reporting "
         "convenience —\nand explanation quality must be validated per stratum, "
         "not per model",
         fc="#FDF0E7", ec=ORANGE, fontsize=7.6)

    ax.text(50, 97, "Design science build–evaluate cycle", ha="center", va="top",
            fontsize=11, weight="bold")
    return _save(fig, "diag4_dsr", directory)


# ---------------------------------------------------------------------------
# D5 — Explanation validation workflow
# ---------------------------------------------------------------------------


def diagram_validation(directory: Path = FIGURES) -> list[Path]:
    """The Layer 3 workflow with pre-registered thresholds and observed verdicts."""
    fig, ax = _canvas(9.4, 6.2)

    _box(ax, 4, 84, 30, 11,
         "Stage attributions — TreeSHAP per S$_k$\n(validation split)",
         fc="#FDEBF3", ec=PINK, fontsize=7.4)
    _box(ax, 66, 84, 30, 11,
         "Sequence attributions — TimeSHAP on GRU\n(same prefixes)",
         fc="#FDEBF3", ec=PINK, fontsize=7.4)

    # Four tests, each fed from the tabular attributions. Boxes carry the
    # method and threshold; observed verdicts sit below with clear separation,
    # since the first attempt let them collide with the box border.
    tests = [
        (3, 47, 22, 22, "FAITHFULNESS",
         "Bhatt correlation\nover random\nfeature subsets", "$\\rho > 0.5$",
         "S1 0.571 ✓\nS2 0.494 ✗\nS3 0.565 ✓", GREEN),
        (27.5, 47, 22, 22, "DELETION /\nINSERTION",
         "remove top signed\nattributions, replace\nwith background", "monotone\ndegradation",
         "deletion AUC\n$\\approx$ ½ insertion\nat every stage", BLUE),
        (52, 47, 22, 22, "STABILITY",
         "local Lipschitz\nunder small\nperturbation", "bounded\nratio",
         "reported\nper stage", YELLOW),
        (76.5, 47, 20, 22, "REPRODUCIBILITY",
         "Spearman across\n5 frozen seeds", "$\\rho > 0.6$",
         "1.000 / 0.966\n/ 0.943 ✓", ORANGE),
    ]
    for x, yb, w, h, title, method, thresh, result, colour in tests:
        # Everything lives inside the box, including the observed verdict.
        # Placing results outside left them exposed to the cross-paradigm
        # feeder arrows, which struck through them.
        yb, h = 40, 29
        _box(ax, x, yb, w, h, "", fc="white", ec=colour, lw=1.3)
        ax.text(x + w / 2, yb + h - 3.6, title, ha="center", fontsize=7.2,
                weight="bold", color=colour, linespacing=1.2, va="center")
        ax.text(x + w / 2, yb + h - 11.5, method, ha="center", fontsize=6.4,
                color=GREY, linespacing=1.35, va="center")
        ax.text(x + w / 2, yb + h - 18.5, thresh, ha="center", fontsize=6.8,
                color="black", weight="bold", linespacing=1.2, va="center")
        ax.plot([x + 2.5, x + w - 2.5], [yb + 7.6, yb + 7.6], color=colour,
                lw=0.7, alpha=0.5, zorder=3)
        ax.text(x + w / 2, yb + 4.0, result, ha="center", fontsize=6.4,
                color=colour, linespacing=1.3, va="center", zorder=3)
        # Feed each test from the tabular attribution box, landing on its top edge.
        _arrow(ax, (19, 84), (x + w / 2, yb + h), color=PINK, rad=-0.06)

    _box(ax, 20, 19, 60, 13,
         "CROSS-PARADIGM AGREEMENT\n"
         "aggregate rank $\\rho$ (5 concepts):  +0.200 / +0.500 / +0.600\n"
         "per instance, 5 seeds:  +0.179 / −0.018 / +0.046  —  3 of 14 sign-stable",
         fc="#EAF6F1", ec=GREEN, fontsize=7.2)
    # Both paradigms feed the agreement test, routed down the outer margins so
    # neither arrow crosses the row of test boxes or their verdicts.
    _arrow(ax, (4.5, 84), (20, 25.5), color=PINK, rad=0.30)
    _arrow(ax, (95.5, 84), (80, 25.5), color=PINK, rad=-0.30)

    ax.text(50, 13.5,
            "Aggregate agreement does not imply per-instance agreement.",
            ha="center", fontsize=8.0, weight="bold", color=GREEN)
    ax.text(50, 6.5,
            "All thresholds fixed before any result was computed. A verdict is reported "
            "against the stated bar,\nnot rounded toward it: S2 misses faithfulness at "
            "0.494 and is reported as a failure.",
            ha="center", fontsize=7.0, color=GREY, linespacing=1.5)

    ax.text(50, 98, "Explanation validation workflow (Layer 3)", ha="center",
            va="top", fontsize=11, weight="bold")
    return _save(fig, "diag5_validation", directory)


# ---------------------------------------------------------------------------
# D6 — Contribution map
# ---------------------------------------------------------------------------


def diagram_contributions(directory: Path = FIGURES) -> list[Path]:
    """Research question to method, evidence and theoretical contribution."""
    fig, ax = _canvas(10.2, 6.6)

    cols = [
        (3, 20, "RESEARCH\nQUESTION", GREY),
        (25, 22, "METHODOLOGICAL\nINNOVATION", BLUE),
        (49, 24, "EMPIRICAL\nFINDING", ORANGE),
        (75, 22, "THEORETICAL\nCONTRIBUTION", GREEN),
    ]
    for x, w, title, colour in cols:
        ax.text(x + w / 2, 90, title, ha="center", fontsize=8, weight="bold",
                color=colour, linespacing=1.25)

    rows = [
        (68, "RQ1\nPrediction\nby stage",
         "Prefix-only features;\nPR-AUC lift as the\ncross-stage metric",
         "Lift falls 1.79→1.09\nROC-AUC 0.641→0.569\nH1 contradicted",
         "Predictability declines\nas intent forms;\nsignal moves off-stream", BLUE),
        (48, "RQ2\nAttribution\nmigration",
         "Stage-conditioned\nTreeSHAP with fixed\ncorrelation grouping",
         "Entropy 0.035→0.224\n→0.074 (non-monotone)\nH2 partly supported",
         "Exploration is a\nstage-specific mode,\nnot a decaying trend", ORANGE),
        (28, "RQ3\nFaithfulness\nand stability",
         "Pre-registered thresholds;\nper-instance as well as\naggregate agreement",
         "Seed ρ ≥ 0.886 ✓\nS2 faithfulness 0.494 ✗\nH3, H4 not supported",
         "Explanation quality\ntracks predictive signal\n→ validate per stratum", GREEN),
        (8, "RQ4\nActionability",
         "Static counterfactual\nbuilt on the same data\n(stage-average share)",
         "Static reports 0.111 for\nnavigation; true peak\n0.224 at S2",
         "Intervention window is\nearly/mid funnel, not\nat the cart", PINK),
    ]

    for yb, rq, method, finding, theory, colour in rows:
        _box(ax, 3, yb, 20, 16, rq, fc="white", ec=colour, fontsize=7.4, weight="bold")
        _box(ax, 25, yb, 22, 16, method, fc="white", ec=colour, fontsize=6.8)
        _box(ax, 49, yb, 24, 16, finding, fc="#FAFAFA", ec=colour, fontsize=6.8)
        _box(ax, 75, yb, 22, 16, theory, fc="white", ec=colour, fontsize=6.8)
        for x_end in (25, 49, 75):
            _arrow(ax, (x_end - 2, yb + 8), (x_end, yb + 8), color=colour, lw=1.0)

    ax.text(50, 2.5,
            "Three of four pre-registered hypotheses were not supported; "
            "all four are reported as found.",
            ha="center", fontsize=7.6, color=GREY, style="italic")

    ax.text(50, 100, "Contribution map", ha="center", va="top",
            fontsize=11, weight="bold")
    return _save(fig, "diag6_contributions", directory)


DIAGRAMS = {
    "diag1": diagram_architecture,
    "diag2": diagram_gap,
    "diag3": diagram_prefix,
    "diag4": diagram_dsr,
    "diag5": diagram_validation,
    "diag6": diagram_contributions,
}


def build_all(directory: Path = FIGURES) -> dict[str, list[Path]]:
    return {name: fn(directory) for name, fn in DIAGRAMS.items()}
