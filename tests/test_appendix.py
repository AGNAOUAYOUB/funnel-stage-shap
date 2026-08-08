"""Tests for the appendix figures and tables (protocol Sec. 15)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
import pytest

from funnel_shap.report import appendix, appendix_diagrams, appendix_tables


@pytest.fixture
def scores():
    rng = np.random.default_rng(7)
    out = {}
    for stage, prevalence in (("S1", 0.10), ("S2", 0.07), ("S3", 0.52)):
        n = 400
        y = rng.binomial(1, prevalence, size=n)
        s = np.clip(rng.normal(loc=0.35 + 0.3 * y, scale=0.2, size=n), 0, 1)
        out[stage] = (y, s)
    return out


@dataclass
class _Attribution:
    feature_names: list
    shap_values: np.ndarray

    def global_importance(self) -> pl.DataFrame:
        mean_abs = np.abs(self.shap_values).mean(axis=0)
        return pl.DataFrame(
            {
                "feature": self.feature_names,
                "mean_abs_shap": mean_abs,
                "share": mean_abs / mean_abs.sum(),
            }
        ).sort("mean_abs_shap", descending=True)


@dataclass
class _Explanation:
    attribution: _Attribution
    explained_matrix: np.ndarray


@pytest.fixture
def explanations():
    rng = np.random.default_rng(11)
    out = {}
    for i, stage in enumerate(("S1", "S2", "S3")):
        names = [f"feat_{j}" for j in range(6)]
        shap = rng.normal(scale=1.0 + i * 0.2, size=(120, 6))
        matrix = rng.normal(size=(120, 6))
        out[stage] = _Explanation(_Attribution(names, shap), matrix)
    return out


def test_roc_and_pr_curves_are_written_as_vector_and_raster(scores, tmp_path) -> None:
    roc = appendix.figure_roc_curves(scores, directory=tmp_path)
    pr = appendix.figure_pr_curves(scores, directory=tmp_path)

    for written in (roc, pr):
        suffixes = {p.suffix for p in written}
        assert suffixes == {".pdf", ".png"}, "journals need vector; drafts need raster"
        assert all(p.exists() and p.stat().st_size > 0 for p in written)


def test_confusion_matrices_use_the_supplied_thresholds(scores, tmp_path) -> None:
    written = appendix.figure_confusion_matrices(
        scores, {"S1": 0.4, "S2": 0.5, "S3": 0.6}, directory=tmp_path
    )
    assert all(p.exists() for p in written)


def test_confusion_matrices_survive_a_missing_threshold(scores, tmp_path) -> None:
    """A missing threshold must fall back, not raise mid-figure."""
    written = appendix.figure_confusion_matrices(scores, {}, directory=tmp_path)
    assert all(p.exists() for p in written)


def test_shap_summary_and_dependence_render(explanations, tmp_path) -> None:
    summary = appendix.figure_shap_summary(explanations, directory=tmp_path)
    dependence = appendix.figure_shap_dependence(explanations, directory=tmp_path)
    assert all(p.exists() for p in summary + dependence)


def test_feature_importance_covers_every_stage(explanations, tmp_path) -> None:
    written = appendix.figure_feature_importance(explanations, directory=tmp_path)
    assert all(p.exists() for p in written)


def test_stage_order_is_funnel_order_not_alphabetical() -> None:
    assert appendix._stage_order(["S3", "S1", "S2"]) == ["S1", "S2", "S3"]


def test_ablation_and_performance_figures_render(tmp_path) -> None:
    ablation = pl.DataFrame(
        {
            "stage": ["S1", "S1", "S2", "S2"],
            "feature_set": ["baseline", "full", "baseline", "full"],
            "pr_auc_mean": [0.124, 0.131, 0.072, 0.083],
            "pr_auc_std": [0.001, 0.001, 0.001, 0.002],
        }
    )
    summary = pl.DataFrame(
        {
            "model": ["catboost", "lightgbm"],
            "pr_auc_mean": [0.699, 0.677],
            "pr_auc_std": [0.006, 0.008],
            "roc_auc_mean": [0.905, 0.899],
        }
    )
    assert all(p.exists() for p in appendix.figure_ablation(ablation, directory=tmp_path))
    assert all(
        p.exists()
        for p in appendix.figure_performance_comparison(summary, directory=tmp_path)
    )


def test_all_appendix_diagrams_render(tmp_path) -> None:
    built = appendix_diagrams.build_all(tmp_path)
    assert set(built) == {"figA9", "figA10", "figA11", "figA12", "fig9"}
    for written in built.values():
        assert all(p.exists() and p.stat().st_size > 0 for p in written)


def test_computational_cost_reports_unmeasured_steps_as_unmeasured(tmp_path) -> None:
    """A fabricated runtime is worse than an absent one; absence must be visible."""
    table = appendix_tables.computational_cost_table(tmp_path)

    assert table.height > 0
    assert set(table["status"].unique()) == {"not measured"}
    assert table["seconds"].null_count() == table.height


def test_computational_cost_picks_up_real_measurements(tmp_path) -> None:
    pl.DataFrame({"command": ["stage-models"], "seconds": [123.5]}).write_csv(
        tmp_path / "runtime_stage_models.csv"
    )
    table = appendix_tables.computational_cost_table(tmp_path)
    row = table.filter(pl.col("command") == "stage-models").to_dicts()[0]

    assert row["status"] == "measured"
    assert row["seconds"] == pytest.approx(123.5)
    assert row["minutes"] == pytest.approx(2.06, abs=0.01)


def test_literature_comparison_includes_this_work_and_states_validation() -> None:
    table = appendix_tables.literature_comparison_table()
    assert any("this work" in m for m in table["method"])
    # The positioning claim is about validation; the column must be populated.
    assert table["explanation_validation"].null_count() == 0


def test_latex_emitter_escapes_and_formats() -> None:
    frame = pl.DataFrame({"name": ["a_b & c"], "value": [0.12345], "flag": [True]})
    latex = appendix_tables.to_latex(frame, caption="Cap", label="tab:x")

    assert r"a\_b \& c" in latex
    assert "0.1235" in latex
    assert "yes" in latex
    assert latex.startswith("\\begin{table}") and latex.rstrip().endswith("\\end{table}")


def test_latex_emitter_renders_nulls_as_dashes() -> None:
    frame = pl.DataFrame({"step": ["x"], "seconds": [None]}, schema={"step": pl.Utf8, "seconds": pl.Float64})
    latex = appendix_tables.to_latex(frame, caption="C", label="tab:y")
    assert "--" in latex


def test_dataset_statistics_table_selects_available_columns(tmp_path) -> None:
    csv = tmp_path / "prev.csv"
    pl.DataFrame(
        {
            "stage": ["S1", "S2"],
            "modelled": [True, True],
            "n_sessions": [475140, 233160],
            "reach_rate": [0.979, 0.480],
            "prevalence": [0.0885, 0.0683],
        }
    ).write_csv(csv)

    table = appendix_tables.dataset_statistics_table(csv)
    assert table.height == 2
    assert "reach_rate" in table.columns


def test_cost_table_tolerates_an_empty_runtime_file(tmp_path) -> None:
    """An empty runtime file means 'not measured', not a crash."""
    (tmp_path / "runtime_appendix.csv").write_text("", encoding="utf-8")
    table = appendix_tables.computational_cost_table(tmp_path)

    assert table.height > 0
    assert set(table["status"].unique()) == {"not measured"}


def test_cost_table_ignores_a_runtime_file_with_null_seconds(tmp_path) -> None:
    pl.DataFrame(
        {"command": ["stage-models"], "seconds": [None]},
        schema={"command": pl.Utf8, "seconds": pl.Float64},
    ).write_csv(tmp_path / "runtime_x.csv")
    table = appendix_tables.computational_cost_table(tmp_path)

    row = table.filter(pl.col("command") == "stage-models").to_dicts()[0]
    assert row["status"] == "not measured"


def test_error_analysis_compares_against_the_trivial_positive_rule(scores) -> None:
    """A targeting rule that flags everyone is free; a model must beat it."""
    thresholds = {"S1": 0.4, "S2": 0.4, "S3": 0.3}
    table = appendix_tables.error_analysis_table(scores, thresholds)

    assert table.height == 3
    for row in table.to_dicts():
        pi = row["prevalence"]
        assert row["trivial_positive_f1"] == pytest.approx(2 * pi / (1 + pi), abs=1e-4)
        assert row["tp"] + row["fp"] + row["fn"] + row["tn"] == row["n_test"]
        assert row["f1_over_trivial"] == pytest.approx(
            row["f1"] - row["trivial_positive_f1"], abs=1e-4
        )


def test_error_analysis_flags_a_degenerate_always_positive_model() -> None:
    """At a threshold below every score the model IS the trivial rule."""
    y = np.array([0, 1, 1, 0, 1, 1])
    p = np.full(6, 0.9)
    table = appendix_tables.error_analysis_table({"S3": (y, p)}, {"S3": 0.1})
    row = table.to_dicts()[0]

    assert row["flag_rate"] == pytest.approx(1.0)
    assert row["recall"] == pytest.approx(1.0)
    assert row["f1_over_trivial"] == pytest.approx(0.0, abs=1e-4)


def test_error_analysis_reports_fp_per_tp_and_handles_no_positives() -> None:
    y = np.array([0, 0, 0, 0])
    p = np.array([0.9, 0.9, 0.1, 0.1])
    row = appendix_tables.error_analysis_table({"S1": (y, p)}, {"S1": 0.5}).to_dicts()[0]

    assert row["tp"] == 0
    assert row["fp"] == 2
    assert row["fp_per_tp"] is None, "no true positives means the ratio is undefined"
