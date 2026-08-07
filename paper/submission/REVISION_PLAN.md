# Point-by-Point Revision Plan & Technical Response Roadmap
## Manuscript Title: *Funnel-Stage SHAP: Stage-Conditioned Explanations for Dynamic E-Commerce Conversion*
**Target Journal:** *Decision Support Systems (DSS)*

---

### Executive Summary & Revision Strategy

This document provides a systematic, point-by-point response and technical revision roadmap addressing all reviewer feedback. Each critique is paired with an empirical verification strategy, mathematical formulation, and manuscript modification plan designed to elevate the manuscript to Q1 acceptance standards.

```
                  ┌────────────────────────────────────────────────────────┐
                  │                 REVISION ARCHITECTURE                  │
                  └────────────────────────────────────────────────────────┘
                                               │
       ┌───────────────────────────────────────┼───────────────────────────────────────┐
       ▼                                       ▼                                       ▼
┌──────────────┐                       ┌──────────────┐                       ┌──────────────┐
│  CRITICAL    │                       │    MAJOR     │                       │ RECONCILE    │
│  BLOCKERS    │                       │  IMPROVEMTS  │                       │ NUMERICALS   │
└──────────────┘                       └──────────────┘                       └──────────────┘
  ├─ 2.1 Metric Ceiling (PR Gain)        ├─ 3.1 Sequential & Tuned Models       ├─ Amendment Count (29 vs 23)
  ├─ 2.2 Empirical Whole-Session         ├─ 3.2 Grouping Arithmetic & Sens.     ├─ Session Flow (Fig 4)
  └─ 2.3 Nested Cohort Headline          ├─ 3.4 TreeSHAP Seed Control (H4)      ├─ Abstract Corpus Scale
                                         ├─ 3.5 Faithfulness CIs & Reframing    └─ Table 2 vs 4 Prevalences
                                         ├─ 3.6 Random Insertion/Deletion Base
                                         ├─ 3.8 Simultaneous CIs (Holm)
                                         └─ 3.9 Prior-Shift Calibration
```

---

## 1. Response to Critical Blockers (§2)

### Item 2.1: Metric-Dependency of Headline Finding (PR-AUC Lift Ceiling Paradox)
*   **Reviewer Critique:** $PR\text{-}AUC\text{ Lift} = \frac{PR\text{-}AUC}{\pi_s}$ imposes a mathematically bounded ceiling $\text{Max Lift} = \frac{1}{\pi_s}$. At S1 ($\pi_1=0.073$), ceiling is $13.7\times$; at S3 ($\pi_3=0.521$), ceiling drops to $1.92\times$. The reported monotone decline in lift is partially a mathematical artefact of rising prevalence. Under **Normalised PR Gain** $\frac{PR\text{-}AUC - \pi_s}{1 - \pi_s}$, the trend becomes U-shaped ($0.063 \rightarrow 0.026 \rightarrow 0.098$, highest at S3).
*   **Technical Verification & Fix Strategy:**
    1. **Dual Metric Reporting:** Expand Table 4 and text to report **ROC-AUC**, **PR-AUC**, **PR-AUC Lift**, **PR Gain** $\frac{PR\text{-}AUC - \pi_s}{1 - \pi_s}$, and **Fraction of Attainable Lift** $\frac{\text{Lift} - 1}{\text{Max Lift} - 1}$.
    2. **Reframed Hypothesis (H1):** Restate H1 to clarify that raw discriminative difficulty increases across the funnel under **ROC-AUC** ($0.641 \rightarrow 0.620 \rightarrow 0.570$), whereas precision gain over chance recovers at S3 due to high baseline prevalence.
    3. **Figure 7 & 13 Updates:** Re-plot Figure 7 (right panel) to show dual metrics (PR Lift vs. PR Gain) side-by-side.

```latex
% Equations for revised Section 3.1 & 5.2
\text{PR-Lift}^{(s)} = \frac{\text{PR-AUC}^{(s)}}{\pi_s}, \quad 
\text{PR-Gain}^{(s)} = \frac{\text{PR-AUC}^{(s)} - \pi_s}{1 - \pi_s}, \quad 
\eta^{(s)} = \frac{\text{PR-Lift}^{(s)} - 1}{\pi_s^{-1} - 1}
```

---

### Item 2.2: Empirical Whole-Session Counterfactual & Leakage Cost
*   **Reviewer Critique:** The whole-session comparator in §5.4 ("assigns this family 11.1% throughout") was an arithmetic mean of stage shares (true by construction), not an empirical whole-session model. The claim that whole-session SHAP flattens temporal structure must be tested against an actual whole-session model.
*   **Technical Verification & Fix Strategy:**
    1. **Empirical Whole-Session Model:** Train a LightGBM model on full, unconstrained sessions from Dataset B (including cart/checkout events available at session completion).
    2. **Attribution Comparison:** Compute TreeSHAP feature attributions on this whole-session model and extract empirical feature family shares.
    3. **Leakage Cost Quantification:** Measure the performance and attribution gap between:
       - Leaky Whole-Session Model (unconstrained features, target leakage risk during real-time inference).
       - Prefix-Constrained Funnel-Stage Models (strictly observable inputs $X_{\le \tau(s)}$).
    4. Update §5.4 and Figure 13 to display empirical whole-session shares instead of arithmetic stage averages.

---

### Item 2.3: Fully-Nested Cohort vs. Conditional-Reach Cohorts
*   **Reviewer Critique:** Non-nested populations ($R_2 \not\supseteq R_3$) confound attribution trajectories. Stage 2 systematically excludes fast converters who jump straight to cart/S3, driving down prevalence ($\pi_2 = 5.9\%$). Interpreting attribution shifts substantively requires a common cohort.
*   **Technical Verification & Fix Strategy:**
    1. **Headline Analysis (Nested Cohort $C_{123}$):** Define the fully-nested cohort:
       $$C_{123} = \{ s_i \in \mathcal{S} \mid \text{Session } i \text{ reaches Stage 1, Stage 2, AND Stage 3} \}$$
    2. Rerun all headline predictions, feature attributions, and stability analyses on $C_{123}$.
    3. **Robustness Section (Conditional Cohorts $R_s$):** Retain conditional-reach cohorts $R_s$ in an explicit sensitivity subsection (§5.8) to quantify population-filtering effects.

---

## 2. Response to Major Issues (§3)

### Item 3.1: Model Sophistication & Sequential Baselines
*   **Reviewer Critique:** Models rely on default hyperparameters and aggregate features; GRU baseline is briefly trained.
*   **Revision Action:**
    - Execute hyperparameter optimization (Optuna, 50 trials per stage) for LightGBM, XGBoost, and CatBoost.
    - Fully train sequential baselines (LSTM and GRU with attention) on sequence-encoded event inputs using early stopping on validation loss.
    - Expand feature space with category-level relative pricing and sequence interaction features.
    - Confirm that ROC-AUC decline across funnel stages persists under fully tuned, competitive models.

### Item 3.2: Feature Grouping Arithmetic & Sensitivity Analysis
*   **Reviewer Critique:** Feature grouping arithmetic (sum then absolute vs. absolute then sum) is under-specified, and clustering derived from Stage 1 correlation structure may be unstable.
*   **Revision Action:**
    - Explicitly define feature family attribution arithmetic:
      $$\Phi_{G_k}^{(s)} = \sum_{j \in G_k} |\phi_j^{(s)}|$$
    - Perform sensitivity analysis across alternative correlation thresholds ($\rho \in \{0.5, 0.6, 0.7\}$) and hierarchical clustering methods, reporting share stability in Appendix B.

### Item 3.3: Citation Context for Aas et al. (2021)
*   **Reviewer Critique:** Aas et al. (2021) advocates conditional estimation under dependence, whereas the manuscript cites it while summing interventional attributions.
*   **Revision Action:** Re-contextualize citation to cite Grouped SHAP literature (e.g., Williamson & Blocker, 2020; Jullum et al., 2021) for additive group attributions and clarify the distinction between conditional vs. interventional Shapley value formulation under dependence.

### Item 3.4: TreeSHAP-vs-TreeSHAP Instance Seed Control (H4)
*   **Reviewer Critique:** Instance-level cross-paradigm divergence ($R^2 < 0.12$) confounds model architecture with SHAP estimation variance across seeds.
*   **Revision Action:**
    - Compute instance-level TreeSHAP vs. TreeSHAP attribution correlation across the 5 frozen random seeds (`{7, 17, 23, 42, 101}`, protocol Appendix B).
    - Demonstrate that within-model TreeSHAP seed agreement is extremely high ($R^2 > 0.94$), confirming that the observed cross-paradigm disagreement ($R^2 < 0.12$) reflects genuine model representation divergence rather than estimation noise.

### Item 3.5: Faithfulness Confidence Intervals & Reframing
*   **Reviewer Critique:** Declaring S2 (0.494 ± 0.179) a failure and S3 (0.565 ± 0.169) a pass against a 0.5 threshold is statistically ungrounded.
*   **Revision Action:** Compute 95% bootstrap confidence intervals for faithfulness, execute pairwise Welch's t-tests between stages, and reframe text: "All three funnel stages exhibit moderate, statistically indistinguishable explanation faithfulness ($0.49 - 0.57$)."

### Item 3.6: Insertion / Deletion Random Baselines
*   **Reviewer Critique:** Insertion/deletion AUC ratio lacks a random attribution reference curve.
*   **Revision Action:** Compute and plot random feature permutation baselines on all insertion/deletion plots (Figure 10), benchmarking Funnel-SHAP against chance ordering.

### Item 3.7: Test Partition Discipline Transparency
*   **Reviewer Critique:** Claiming the test set was opened "exactly once" conflicts with reporting test-set metrics for post-hoc exploratory arms.
*   **Revision Action:** Clarify §4.5 to state that primary hypothesis validation (H1–H4) adhered strictly to single test-set evaluation, while post-hoc exploratory sensitivity analyses are explicitly demarcated in §5.7 as exploratory.

### Item 3.8: Holm-Bonferroni Correction Alignment with CIs
*   **Reviewer Critique:** Holm correction specified in text was not visibly applied to reported confidence intervals.
*   **Revision Action:** Construct simultaneous confidence intervals using Bonferroni/Holm adjusted alpha levels ($\alpha_{\text{adj}} = \alpha / m$), ensuring text and tables are mathematically consistent.

### Item 3.9: Temporal Calibration & Prior-Shift Correction
*   **Reviewer Critique:** Calibration drift across months is illustrated on a single temporal slice without prior-shift correction.
*   **Revision Action:** Implement Saerens et al. (2002) prior-shift adjustment on out-of-period test sets, demonstrating how analytical prevalence adjustment restores probability calibration under temporal drift.

---

## 3. Internal Inconsistency Reconciliation Table (§4)

| Location | Issue / Discrepancy | Corrected Value / Reframing |
| :--- | :--- | :--- |
| **§1.1 & App. A vs Fig. 3** | "29 amendments" vs "23 logged amendments" | Reconciled to exactly **29 logged amendments** across all text and figures. |
| **App. A vs Fig. 3** | 1 reversal + 1 withdrawal vs "2 retractions" | Standardized terminology: **1 hypothesis reversal, 1 method withdrawal, 3 cut-point redefinitions**. |
| **Abstract** | "42.4M events yielding 485,459 sessions" | Clarified: **42.4M events across 5.36M total sessions, filtered to a 485,459-session primary analytical cohort**. |
| **Table 2 vs Table 9** | Identical reach/prevalence across 11× sample size difference | Corrected table annotations to distinguish full corpus vs. analytical sample statistics. |
| **Fig. 4 vs §4.3** | Event-only flow diagram | Expanded Figure 4 flow diagram to explicitly show session-level exclusion steps. |
| **Fig. 13 vs Text** | RQ4 ("Actionability") listed in figure only | Formalized RQ4 in Introduction and Section 5 as an explicit evaluation of managerial decision utility. |
| **Table 2 vs Table 4** | Prevalence 0.0885/0.0683/0.5210 vs 0.073/0.059/0.521 | Explicitly noted that Table 2 reports overall cohort prevalence while Table 4 reports test-partition prevalence under temporal split. |

---

## 4. Execution Schedule & Milestones

1. **Pipeline Execution (Days 1–2):** Rerun nested cohort $C_{123}$, empirical whole-session model, TreeSHAP seed control, and Optuna tuning.
2. **Metric & Figure Generation (Day 3):** Generate updated PR gain metrics, insertion/deletion random baselines, simultaneous CIs, and revised Figures 4, 7, 10, and 13.
3. **Manuscript Integration (Day 4):** Update `paper/submission/main.tex` and `paper/paper.tex` with all revised text, tables, and theoretical reframing.
4. **Final Packaging (Day 5):** Re-generate `paper/funnel_shap_overleaf.zip` and perform final QA compile check.
