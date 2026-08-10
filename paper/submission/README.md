# Funnel-Stage SHAP — Overleaf project

Upload this ZIP to Overleaf (New Project → Upload Project). Main file: `main.tex`.
Compiler: pdfLaTeX. The class is `cas-sc` (the Elsevier CAS single-column class,
available on Overleaf by default); bibliography style `cas-model2-names` with
natbib author–year citations, which renders APA-style `(Author, year)` in text.

## Contents

- `main.tex` — the manuscript (submission draft). Compiles on its own.
- `supplement.tex` — twelve supporting figures (schematics and per-stage
  diagnostics), compiled separately as supplementary material. Shares
  `figures/` and `references.bib` with the manuscript. Figures are numbered
  S1–S12; no claim in the article depends on one.
- `references.bib` — 71 unique entries. Bibliographic fields come from the
  verified project bibliography and the authors' reference-manager export.
- `figures/` — 26 PDF figures (14 in the article, 12 in the supplement), all
  generated from the frozen analysis pipeline.

In the repository the figures are not duplicated here; they live in
`paper/figures/` and are staged into the package by `build_zip.ps1`, which
regenerates `paper/funnel_shap_overleaf.zip`. Regenerate the figures themselves
with `funnel-shap figures --suffix gap30s200000`.

## Before submission — REQUIRED author actions

1. **Affiliations and second author.** Confirm the affiliation block and supply
   S. Ailli's given name; add ORCIDs to the `[orcid=]` fields in the front
   matter.
2. **Citation verification.** Bibliographic fields come from the authors'
   reference-manager export; the remaining check is substantive — confirm each
   source actually supports the claim it is cited for, particularly in the
   related-work section.
3. **Repository DOI.** Appendix D carries one `[DATA REQUIRED]` marker for the
   public repository DOI or URL, to be inserted on acceptance.
4. **REES46 licence.** Dataset B is publicly distributed under its own terms;
   confirm the wording in the data-availability statement matches them.

## Provenance of numbers

Every numerical result comes from `reports/tables/` in the analysis repository,
produced from frozen splits and the seed list {7, 17, 23, 42, 101}. Headline
stage models use default LightGBM hyperparameters; the Optuna study
(100 trials/stage) is reported as a robustness analysis, and the persisted
studies are reopenable from `experiments/optuna/`. No number in the manuscript
is estimated, interpolated, or reported from memory.

Three analyses were added after the protocol was frozen and are labelled
exploratory in the text and in Appendix A: the fixed-cohort evaluation
(`common_cohort_gap30s200000.csv`), the flag-rate sweep
(`flag_rate_sweep_gap30s200000.csv`), and the purchase-excluded whole-session
comparator (`whole_session_contrast_gap30s200000_nobuy.csv`).
