# Key Research Findings & Revision Discoveries

## 1. Metric Dependencies & Mathematical Formulation
- $PR\text{-Lift} = \frac{PR\text{-}AUC}{\pi_s}$, $\text{Max Lift} = \frac{1}{\pi_s}$.
- $\text{PR-Gain} = \frac{PR\text{-}AUC - \pi_s}{1 - \pi_s}$, normalizes across varying base rates $\pi_s \in \{0.073, 0.059, 0.521\}$.
- Fraction of Attainable Lift $\eta = \frac{\text{PR-Lift} - 1}{\pi_s^{-1} - 1} = \text{PR-Gain}$.
- Under ROC-AUC, discriminative performance declines monotonically ($0.641 \rightarrow 0.620 \rightarrow 0.570$). Under PR-Lift, it drops ($1.79\times \rightarrow 1.45\times \rightarrow 1.09\times$). Under PR-Gain, it is U-shaped ($0.063 \rightarrow 0.026 \rightarrow 0.098$).

## 2. Empirical Whole-Session Baseline (RQ4)
- Rather than an arithmetic average of stage shares ($11.1\%$), an empirical LightGBM trained on unconstrained whole sessions yields empirical baseline attribution shares.
- The temporal trajectory shows dynamic navigation peaking at S2 ($22.4\%$), whereas the static whole-session model distributes attribution without stage sensitivity, confirming the temporal flattening hypothesis empirically.

## 3. Nested Cohort ($C_{123}$) vs Conditional Cohorts ($R_s$)
- Nested cohort $C_{123}$ ensures identical session population across all stages, eliminating survival and composition bias.
- Conditional reach cohorts ($R_s$) are preserved in sensitivity analysis (§5.8) to demonstrate the robustness of findings across filtering definitions.

## 4. TreeSHAP Seed Control & Faithfulness
- Within-model TreeSHAP seed agreement across 5 seeds: $R^2 > 0.94$.
- Cross-paradigm (TreeSHAP vs TimeSHAP GRU): $R^2 < 0.12$, confirming that divergence is representational, not estimation noise.
- Faithfulness: moderate across all stages ($0.49 - 0.57$), statistically indistinguishable with overlapping 95% bootstrap CIs.
