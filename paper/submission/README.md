# Funnel-Stage SHAP — Overleaf project

Upload this ZIP to Overleaf (New Project → Upload Project). Main file: `main.tex`.
Compiler: pdfLaTeX. The class is `elsarticle` (available on Overleaf by default);
bibliography style `elsarticle-harv` with natbib author–year citations.

## Contents

- `main.tex` — the manuscript (submission draft).
- `references.bib` — 62 entries: 34 from the verified project bibliography plus 28
  from the authors' curated reference export (keys renamed to match the
  manuscript; full author lists, volumes, issues and DOIs as exported).
- `figures/` — 13 PDF figures (7 results figures, 6 conceptual diagrams), all
  generated from the frozen analysis pipeline.

In the repository the figures are not duplicated here; they live in
`paper/figures/` and are staged into the package by `build_zip.ps1`, which
regenerates `paper/funnel_shap_overleaf.zip`. Regenerate the figures themselves
with `funnel-shap figures --suffix gap30s200000`.

## Before submission — REQUIRED author actions

1. **Affiliations and second author.** The affiliation block and S. Ailli's given
   name are placeholders (marked `NOTE TO AUTHORS` in `main.tex`).
2. **Citation verification.** Bibliographic fields now come from the authors'
   reference-manager export; the remaining check is substantive — confirm each
   source actually supports the claim it is cited for, particularly in the
   related-work section.
3. **Generative-AI declaration.** Complete the placeholder section per the target
   journal's policy.
4. **Journal name.** `\journal{Decision Support Systems}` is a working target;
   change as appropriate.

## Provenance of numbers

Every numerical result in Sections 4–5 comes from `reports/tables/` in the
analysis repository, produced from frozen splits and the seed list {7, 17, 29,
42, 87}. Headline stage models use default LightGBM hyperparameters; the Optuna
study (100 trials/stage) is reported as a robustness analysis, and the persisted
studies are reopenable from `experiments/optuna/`. No number in the manuscript
is estimated, interpolated, or reported from memory.
