# Manuscript build

```
paper/
├── paper.tex          # the manuscript
├── references.bib     # 34 entries, all cited
├── figures/           # 5 vector PDFs, copied from ../reports/figures/
└── README.md
```

## Compiling

No LaTeX engine was available in the environment where this was written, so
**the file has not been compiled**. It has been checked mechanically instead:
every `\cite` key resolves to a bib entry, every bib entry is cited, every
`\includegraphics` target exists, every `\ref` has a `\label`, and all float
environments balance. Expect the usual first-compile fixes (float placement,
column overflow) rather than errors.

```bash
latexmk -pdf paper.tex
```

or, without latexmk:

```bash
pdflatex paper && bibtex paper && pdflatex paper && pdflatex paper
```

**Overleaf** works without local installation: upload `paper.tex`,
`references.bib` and the `figures/` directory, and set the compiler to pdfLaTeX.

## If `elsarticle` is not installed

The manuscript uses Elsevier's class because that is the submission target.
Nothing else in the file depends on it. To fall back to the standard `article`
class, swap the three marked lines at the top of `paper.tex`:

```latex
% \documentclass[review,3p,times,authoryear]{elsarticle}
\documentclass[11pt,a4paper]{article}
\usepackage[authoryear,round]{natbib}
```

and change the bibliography style at the end of the file:

```latex
\bibliographystyle{apalike}   % instead of elsarticle-harv
```

Both styles produce APA-style author–year citations: `\citep{key}` renders as
(Author, year) and `\citet{key}` as Author (year).

## Before submission

Three items in the manuscript are deliberately marked in red and must be
resolved. They are not placeholders for polish; each is a substantive gap.

1. **Verify every reference.** The entries are works I am confident exist and
   whose content matches the use made of them, but volume, issue and page
   numbers are the least reliable fields and were not checked against the
   publications themselves. The study protocol named a method called
   "GroupSegment-SHAP"; I could not verify that it exists, so it is **absent**
   rather than approximated. If it is real, add it to §2.2.

2. **Resolve the `[UNVERIFIED CLAIM]` in §2.3.** The assertion that applied
   e-commerce XAI studies rarely validate their explanations is the paper's
   central gap claim and currently rests on informal reading. It needs a
   documented search before it can stand as written — a reviewer who disputes it
   undermines the contribution.

3. **Complete the generative-AI declaration honestly.** AI was used
   substantially in implementing the analysis code and drafting the manuscript.
   The required wording is journal-specific; do not understate it.

Also unresolved: **Dataset B's licence**. No formal licence text is published.
The data-availability statement says so rather than asserting a grant. Obtain
written confirmation from REES46 and replace that paragraph.

## Regenerating the numbers

Every figure and table comes from `../reports/`. To rebuild from the frozen
splits:

```bash
python -m funnel_shap.cli stage-models --suffix gap30s200000 --ablation
```

```bash
python -m funnel_shap.cli stage-shap --suffix gap30s200000 --background-size 500 --max-explain 2000
```

```bash
python -m funnel_shap.cli explanation-quality --suffix gap30s200000
```

```bash
python -m funnel_shap.cli figures --suffix gap30s200000
```

Then copy the refreshed PDFs into `paper/figures/`.
