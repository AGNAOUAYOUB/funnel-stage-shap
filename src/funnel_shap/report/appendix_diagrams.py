"""Appendix schematic diagrams (protocol Sec. 15 deliverables).

Four schematics the results figures cannot carry: the research workflow, the
training pipeline, the SHAP computation workflow, and the model architecture.
They are drawn rather than photographed from a whiteboard so they stay vector,
recolour with the rest of the figure set, and can be regenerated when the
pipeline changes.

Style is inherited from `diagrams.py` -- same Okabe-Ito palette, same rounded
box and arrow helpers -- so a reader moving between the body and the appendix
does not experience a change of voice.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from ..paths import FIGURES  # noqa: E402
from .diagrams import (  # noqa: E402
    BLUE,
    GREEN,
    GREY,
    LIGHT,
    ORANGE,
    PINK,
    SKY,
    YELLOW,
    _arrow,
    _box,
    _canvas,
    _save,
)


def diagram_research_workflow(directory: Path = FIGURES) -> list[Path]:
    """End-to-end study workflow, from protocol freeze to manuscript."""
    fig, ax = _canvas(9.2, 4.6)

    ax.text(50, 95, "Research workflow", ha="center", fontsize=11, weight="bold")

    stages = [
        ("Protocol v1.0\nfrozen", BLUE, "Hypotheses, metrics,\nseeds, thresholds fixed"),
        ("Data\nacquisition", SKY, "Dataset A (UCI)\nDataset B (REES46)"),
        ("Journey\nreconstruction", GREEN, "Sessionise, stage\ncut-points, prefixes"),
        ("Stage\nmodelling", YELLOW, "Per-stage models\n5 seeds, frozen splits"),
        ("Explanation\n+ validation", ORANGE, "TreeSHAP, TimeSHAP\nfaithfulness, stability"),
        ("Reporting", PINK, "Tables, figures,\namendment log"),
    ]

    width, gap = 13.5, 2.6
    x0 = (100 - (len(stages) * width + (len(stages) - 1) * gap)) / 2
    for i, (title, colour, detail) in enumerate(stages):
        x = x0 + i * (width + gap)
        _box(ax, x, 55, width, 17, title, fc=colour, tc="white",
             ec=colour, fontsize=8, weight="bold")
        _box(ax, x, 30, width, 20, detail, fc=LIGHT, ec=GREY, fontsize=6.6)
        if i:
            _arrow(ax, (x - gap - 0.4, 63.5), (x - 0.6, 63.5))

    _box(
        ax, x0, 8, len(stages) * width + (len(stages) - 1) * gap, 13,
        "The test partition is opened exactly once, at stage-model evaluation. "
        "Every deviation after freeze is recorded in the amendment log with its date and rationale.",
        fc="white", ec=GREY, fontsize=7.2, ls="--",
    )
    return _save(fig, "figA9_research_workflow", directory)


def diagram_training_pipeline(directory: Path = FIGURES) -> list[Path]:
    """Per-stage training pipeline, including where calibration is fitted."""
    fig, ax = _canvas(9.0, 5.2)
    ax.text(50, 96, "Training pipeline (per funnel stage)", ha="center",
            fontsize=11, weight="bold")

    _box(ax, 3, 74, 22, 14, "Stage prefix\nfeature matrix",
         fc=BLUE, tc="white", ec=BLUE, fontsize=8, weight="bold")
    _box(ax, 3, 52, 22, 15, "Frozen split\n(temporal)", fc=LIGHT, ec=GREY, fontsize=7.5)

    _box(ax, 31, 78, 20, 11, "train (fit)", fc=GREEN, tc="white", ec=GREEN, fontsize=7.5)
    _box(ax, 31, 64, 20, 11, "train (calibration\nslice, 20%)",
         fc=SKY, tc="white", ec=SKY, fontsize=7)
    _box(ax, 31, 50, 20, 11, "validation", fc=YELLOW, tc="white", ec=YELLOW, fontsize=7.5)
    _box(ax, 31, 36, 20, 11, "test", fc=GREY, tc="white", ec=GREY, fontsize=7.5)

    for y in (83.5, 69.5, 55.5, 41.5):
        _arrow(ax, (25.4, 60), (30.6, y), rad=0.12)

    _box(ax, 57, 78, 20, 11, "LightGBM fit\n(class weights)",
         fc="white", ec=GREEN, fontsize=7.2)
    _box(ax, 57, 64, 20, 11, "Isotonic\ncalibration", fc="white", ec=SKY, fontsize=7.2)
    _box(ax, 57, 50, 20, 11, "Threshold\nselection (F1)", fc="white", ec=YELLOW, fontsize=7.2)
    _box(ax, 57, 36, 20, 11, "Final evaluation\n+ bootstrap CIs",
         fc="white", ec=GREY, fontsize=7.2)

    for y in (83.5, 69.5, 55.5, 41.5):
        _arrow(ax, (51.4, y), (56.6, y))

    _arrow(ax, (67, 77.6), (67, 75.4), color=GREEN)
    _arrow(ax, (67, 63.6), (67, 61.4), color=SKY)
    _arrow(ax, (67, 49.6), (67, 47.4), color=YELLOW)

    _box(ax, 83, 50, 15, 39,
         "repeat\nover\nseeds\n\n7\n17\n23\n42\n101",
         fc=LIGHT, ec=GREY, fontsize=7)
    _arrow(ax, (77.4, 69), (82.6, 69))

    _box(ax, 3, 8, 95, 20,
         "Calibration is fitted on a held-out slice of the *training* period, not on validation.\n"
         "Under the prior shift present in these data, calibrating on validation raises expected\n"
         "calibration error by roughly an order of magnitude (Figure 5); the textbook choice is the\n"
         "wrong one here. The test partition enters only at the final evaluation box.",
         fc="white", ec=ORANGE, fontsize=7.2, ls="--")
    return _save(fig, "figA10_training_pipeline", directory)


def diagram_shap_workflow(directory: Path = FIGURES) -> list[Path]:
    """How an attribution is produced and then checked."""
    fig, ax = _canvas(9.2, 5.0)
    ax.text(50, 96, "SHAP computation and validation workflow", ha="center",
            fontsize=11, weight="bold")

    _box(ax, 4, 74, 21, 14, "Fitted stage\nmodel $f_k$",
         fc=GREEN, tc="white", ec=GREEN, fontsize=8, weight="bold")
    _box(ax, 4, 54, 21, 14, "Background sample\n(training rows only)",
         fc=LIGHT, ec=GREY, fontsize=7.2)
    _box(ax, 4, 34, 21, 14, "Explained rows\n(validation partition)",
         fc=LIGHT, ec=GREY, fontsize=7.2)

    _box(ax, 33, 54, 22, 34,
         "Interventional\nTreeSHAP\n\n$f_k(x) = \\phi_0 + \\sum_j \\phi_j$",
         fc=BLUE, tc="white", ec=BLUE, fontsize=8, weight="bold")
    for y in (81, 61, 41):
        _arrow(ax, (25.4, y), (32.6, 71), rad=0.10)

    _box(ax, 62, 74, 34, 14,
         "Correlation grouping\n(fixed at the earliest, coarsest stage)",
         fc=SKY, tc="white", ec=SKY, fontsize=7.2)
    _box(ax, 62, 56, 34, 14, "Attribution trajectory\nacross S1 $\\to$ S2 $\\to$ S3",
         fc=ORANGE, tc="white", ec=ORANGE, fontsize=7.5, weight="bold")
    _arrow(ax, (55.4, 71), (61.6, 81))
    _arrow(ax, (55.4, 71), (61.6, 63))
    _arrow(ax, (79, 73.6), (79, 70.4))

    _box(ax, 4, 8, 92, 20,
         "Validation layer (all four arms run before any trajectory is interpreted)\n\n"
         "faithfulness (Bhatt correlation)   |   deletion / insertion curves vs a random-order baseline\n"
         "stability (local Lipschitz)   |   consistency (across seeds, and against TimeSHAP)",
         fc="white", ec=PINK, fontsize=7.4)
    _arrow(ax, (79, 55.6), (79, 29), color=PINK)
    return _save(fig, "figA11_shap_workflow", directory)


def diagram_model_architecture(directory: Path = FIGURES) -> list[Path]:
    """The two modelling arms, side by side on identical prefixes."""
    fig, ax = _canvas(9.2, 5.0)
    ax.text(50, 96, "Model architecture: two arms on identical prefixes",
            ha="center", fontsize=11, weight="bold")

    _box(ax, 30, 82, 40, 11, "Stage prefix $P_k(S_i)$",
         fc=BLUE, tc="white", ec=BLUE, fontsize=8.5, weight="bold")

    # Tabular arm
    _box(ax, 4, 62, 42, 13, "Tabular arm", fc=GREEN, tc="white", ec=GREEN,
         fontsize=8, weight="bold")
    _box(ax, 4, 47, 42, 12, "Feature mapping $\\Phi_k$\n19--23 engineered features",
         fc="white", ec=GREEN, fontsize=7.2)
    _box(ax, 4, 32, 42, 12, "LightGBM\n+ isotonic calibration", fc="white", ec=GREEN,
         fontsize=7.2)
    _box(ax, 4, 17, 42, 12, "Interventional TreeSHAP\n(exact)", fc=GREEN, tc="white",
         ec=GREEN, fontsize=7.2)

    # Sequence arm
    _box(ax, 54, 62, 42, 13, "Sequence arm", fc=ORANGE, tc="white", ec=ORANGE,
         fontsize=8, weight="bold")
    _box(ax, 54, 47, 42, 12, "Event embedding\n(ordered, padded)", fc="white",
         ec=ORANGE, fontsize=7.2)
    _box(ax, 54, 32, 42, 12, "GRU\n+ calibration", fc="white", ec=ORANGE, fontsize=7.2)
    _box(ax, 54, 17, 42, 12, "TimeSHAP\n(sampling-based)", fc=ORANGE, tc="white",
         ec=ORANGE, fontsize=7.2)

    _arrow(ax, (45, 81.6), (25, 75.4), color=GREEN, rad=0.12)
    _arrow(ax, (55, 81.6), (75, 75.4), color=ORANGE, rad=-0.12)
    for x, colour in ((25, GREEN), (75, ORANGE)):
        _arrow(ax, (x, 61.6), (x, 59.4), color=colour)
        _arrow(ax, (x, 46.6), (x, 44.4), color=colour)
        _arrow(ax, (x, 31.6), (x, 29.4), color=colour)

    _box(ax, 12, 3, 76, 10,
         "Both arms consume the same prefix, so any disagreement is representational, "
         "not a difference in what each model was allowed to see.",
         fc=LIGHT, ec=GREY, fontsize=7.4)
    return _save(fig, "figA12_model_architecture", directory)


def diagram_decision_support(directory: Path = FIGURES) -> list[Path]:
    """From behavioural data to an intervention decision, and back.

    The loop matters as much as the chain: an intervention changes the very
    behaviour the next prediction is computed from, which is why the
    stage-conditioned design cannot be evaluated once and then left alone.
    """
    fig, ax = _canvas(9.6, 5.4)
    ax.text(50, 96, "Decision-support architecture", ha="center",
            fontsize=11, weight="bold")

    chain = [
        ("Behavioural\ndata", SKY),
        ("Sequential\nrepresentation", BLUE),
        ("Stage\nprediction", GREEN),
        ("Prefix-constrained\nSHAP", YELLOW),
        ("Stage\ninterpretation", ORANGE),
        ("Managerial\ndecision", PINK),
    ]
    # Boxes are sized to the longest label ("Prefix-constrained"), which
    # overflows at the spacing the shorter labels would allow.
    width, gap = 14.6, 1.9
    x0 = (100 - (len(chain) * width + (len(chain) - 1) * gap)) / 2
    for i, (label, colour) in enumerate(chain):
        x = x0 + i * (width + gap)
        _box(ax, x, 62, width, 15, label, fc=colour, tc="white", ec=colour,
             fontsize=6.9, weight="bold")
        if i:
            _arrow(ax, (x - gap - 0.4, 69.5), (x - 0.6, 69.5))

    # What each link contributes, and what it cannot supply on its own.
    notes = [
        "events, ordered",
        "prefix only:\nno future events",
        "$\\hat{p}$ and its\nuncertainty",
        "why, at this\nstage",
        "which lever,\nwhen",
        "act / withhold",
    ]
    for i, note in enumerate(notes):
        x = x0 + i * (width + gap)
        _box(ax, x, 44, width, 14, note, fc=LIGHT, ec=GREY, fontsize=6.4)
        _arrow(ax, (x + width / 2, 61.6), (x + width / 2, 58.4), color=GREY, lw=0.9)

    _box(ax, x0 + 4 * (width + gap), 26, width * 2 + gap, 12,
         "Intervention", fc=GREY, tc="white", ec=GREY, fontsize=8, weight="bold")
    _arrow(ax, (x0 + 5 * (width + gap) + width / 2, 43.6),
           (x0 + 5 * (width + gap) + width / 2, 38.4), color=PINK)

    # Feedback: the intervention perturbs the stream the next prediction reads.
    _arrow(ax, (x0 + 4 * (width + gap), 32), (x0 + width / 2, 32),
           color=ORANGE, ls="--", rad=-0.16)
    ax.text(50, 21, "outcome feedback: the intervention alters the behaviour "
                    "the next prediction is computed from",
            ha="center", fontsize=7, color=ORANGE, style="italic")

    _box(ax, x0, 4, len(chain) * width + (len(chain) - 1) * gap, 12,
         "Prediction alone stops at the third box. It yields a score without a reason, at a "
         "stage it cannot name,\nand so cannot say which lever to pull or when. The "
         "explanation and interpretation links are what convert\na ranked list into a "
         "decision -- and the evidence here is that their value is highest early, not at the cart.",
         fc="white", ec=BLUE, fontsize=7.2, ls="--")
    return _save(fig, "fig9_decision_support", directory)


def build_all(directory: Path = FIGURES) -> dict[str, list[Path]]:
    return {
        "figA9": diagram_research_workflow(directory),
        "figA10": diagram_training_pipeline(directory),
        "figA11": diagram_shap_workflow(directory),
        "figA12": diagram_model_architecture(directory),
        "fig9": diagram_decision_support(directory),
    }
