# Funnel-Stage SHAP — Decision Support Systems submission package

Upload to Overleaf via New Project → Upload Project. Main file: `main.tex`.
Compiler: pdfLaTeX. `titlepage.tex` compiles separately.

Formatting follows the Elsevier *Decision Support Systems* guide for authors,
checked against the live guide rather than from memory. The requirements that
shaped these files are quoted below with the file that satisfies each.

## Contents

| File | Purpose |
|---|---|
| `main.tex` | The **anonymized** manuscript. Contains no author-identifying content. |
| `titlepage.tex` | Uploaded as a separate file: names, affiliations, corresponding author, CRediT, funding, competing-interest and generative-AI declarations. |
| `highlights.txt` | 5 bullets, each ≤85 characters. Uploaded as its own file with "highlights" in the name. |
| `references.bib` | 71 unique entries. |
| `figures/` | Only the figures the manuscript cites; `build_zip.ps1` stages them and fails if one is missing. |

## Requirements this package implements

- **Double anonymized review.** "This journal follows a double anonymized
  review process." `main.tex` is verified to contain zero identifying strings;
  everything identifying is in `titlepage.tex`.
- **References.** "Indicate references by adding a number within square
  brackets in the text… Number references in the order they appear in your
  article." Implemented with `natbib` (`numbers`) and `unsrtnat`, so the
  existing `\citep` calls render as `[n]` in first-appearance order.
- **Page format.** "Submitted papers, unless formally approved by the editor,
  must not be more than 34 pages, double spaced throughout and using at least
  11.5 point font size. The 34 pages must include all materials — Abstract,
  text, figures/tables, references, and appendices. Page margins should be set
  as 1 inch on all sides." Implemented with `geometry` (1 in), `setspace`
  (`\doublespacing`) and `scrextend` (`\changefontsizes{11.5pt}`).
- **Abstract.** "does not exceed 250 words" — currently 245.
- **Keywords.** "1 to 7 keywords… avoid keywords consisting of multiple words"
  — 7 supplied.
- **Highlights.** "3 to 5 bullet points, each a maximum of 85 characters,
  including spaces."
- **Sections.** "Divide your manuscript into clearly defined and numbered
  sections. Number subsections 1.1 (then 1.1.1, 1.1.2, …)."

## Supplementary material: not used, deliberately

> "The Journal does not accept electronic supplementary material without
> express consent of the Editor-in-Chief. Any supplementary material will be
> limited to high resolution images, datasets, and necessary sound or video
> images. **Material critical to the understanding and analysis of the
> submitted manuscript must be in the body of the submitted manuscript.**"

An earlier draft of this package moved supporting exhibits to a supplement to
meet the page limit. That is not available here. Every exhibit carrying a
hypothesis verdict or a number the text reports was returned to the body, and
the remaining exhibits — schematic diagrams and per-stage diagnostic plots —
were **cut from the submission rather than relocated**, since a pointer to
material the reader cannot obtain is worse than no pointer. The figure-
generation code for them remains in the analysis repository.

## Outstanding before submission

1. **Page count.** The manuscript currently exceeds 34 pages. Reducing it is a
   prose compression, not a further exhibit cut — the exhibits are now at the
   minimum the argument supports.
2. **ORCIDs** for both authors (`titlepage.tex`).
3. **Funding statement** (`titlepage.tex`), or the explicit "no specific grant"
   sentence the guide supplies.
4. **Repository DOI** (`titlepage.tex` and the data-availability statement).
5. **Substantive citation check** — confirm each source supports the claim it is
   cited for.

## Provenance of numbers

Every numerical result comes from `reports/tables/` in the analysis repository,
produced from frozen splits and the seed list {7, 17, 23, 42, 101}. Headline
stage models use default LightGBM hyperparameters; the Optuna study
(100 trials/stage) is a robustness analysis. No number in the manuscript is
estimated, interpolated, or reported from memory.

Analyses added after the protocol was frozen are labelled exploratory at every
point of use: the normalised PR-gain metric; the whole-session comparator and
its purchase-excluded revision; the error analysis against a trivial rule and
the decision economics from it; the entropy-availability test; the fixed-cohort
evaluation; the contact-budget sweep; and the permutation null for attribution.
