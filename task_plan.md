# Task Plan: Funnel-Stage SHAP Revisions & Manuscript Finalization

## Objectives & Success Criteria
1. Complete all technical and code revisions as required by reviewers and documented in `REVISION_PLAN.md`.
2. Generate all empirical outputs (nested cohort $C_{123}$, empirical whole-session contrast for RQ4, PR-Gain/Lift dual metrics, TreeSHAP seed agreement, insertion/deletion random baselines, Holm-adjusted CIs).
3. Regenerate all figures (`reports/figures/*.pdf`, `reports/figures/*.png`) and tables (`reports/tables/*.csv`).
4. Update manuscript (`paper/submission/main.tex` and `paper/submission/references.bib`) with full academic rigor, resolving all 29 amendment references, metric definitions, empirical baseline findings, and table annotations.
5. Build and verify the submission package via `python paper/submission/build_zip.py`.

---

## Phases & Status

### Phase 1: Code Audit & Implementation of Revisions
- [ ] 1.1 Metric additions (Normalised PR-Gain, Fraction of Attainable Lift $\eta$, simultaneous Holm CIs) in `evaluate/metrics.py`
- [ ] 1.2 Empirical Whole-Session comparator & leakage cost in `explain/static_contrast.py` & models
- [ ] 1.3 Nested Cohort $C_{123}$ verification & pipeline integration
- [ ] 1.4 TreeSHAP within-model seed control evaluation vs cross-paradigm
- [ ] 1.5 Insertion/Deletion random baseline computation
- [ ] 1.6 Update `figures.py` and `diagrams.py` for all updated plots and 29-amendment consistency

### Phase 2: Execution & Output Generation
- [ ] 2.1 Run analytical pipeline and regenerate tables in `reports/tables/`
- [ ] 2.2 Run figure & diagram generation to produce updated assets in `reports/figures/`

### Phase 3: Comprehensive Manuscript Revision (`main.tex`)
- [ ] 3.1 Synchronize abstract and introduction (42.4M events, 5.36M sessions, 485,459 analytical sample; 29 amendments; formalize RQ4)
- [ ] 3.2 Update methodology (§3 & §4): dual metrics (PR-Lift, PR-Gain, $\eta$), nested cohort $C_{123}$, empirical whole-session formulation, Holm CIs
- [ ] 3.3 Update empirical results (§5): updated Table 4 & Table 9, empirical RQ4 contrast in §5.4, faithfulness reframing, seed control, prior-shift calibration
- [ ] 3.4 Ensure proper figure and table placement throughout sections
- [ ] 3.5 Reconcile all internal numbers across text, tables, and appendices

### Phase 4: Overleaf Package Compilation & Verification
- [ ] 4.1 Run `python paper/submission/build_zip.py`
- [ ] 4.2 Verify zip package contents and structure
- [ ] 4.3 Final validation and report summary
