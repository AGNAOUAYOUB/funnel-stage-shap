# Paper outline — Q1 Elsevier / IMRAD

Target length **9,000–11,000 words** excluding references, which is the working range for
Decision Support Systems and Expert Systems with Applications. Section numbers in
parentheses map to the protocol (`research_protocol_funnel_shap`), following its Sec. 15
mapping. Every claim below is tagged:

- **[HAVE]** — computed, numbers in `reports/tables/`
- **[PENDING]** — code exists, not yet run
- **[NOT BUILT]** — module still to be written

---

## Front matter

### Title

Protocol Sec. 2 says keep "temporal" for discoverability while making clear the
contribution is not a new Shapley estimator. Three options, ordered by preference:

1. **Funnel-Stage SHAP: temporal attribution trajectories and their faithfulness in
   e-commerce purchase prediction**
2. Where does purchase intent come from? Stage-conditioned SHAP attribution trajectories
   across the digital consumer journey
3. Beyond static explanations: validating stage-conditioned SHAP for sequential purchase
   prediction

Avoid "Temporal SHAP" as a bare noun phrase — it reads as a claim to the TimeSHAP method.

### Highlights (Elsevier: 3–5 bullets, **max 85 characters each including spaces**)

Draft, all within limit:

- `Funnel-stage SHAP reveals how purchase drivers migrate across journey stages` (76)
- `Prefix-only features remove the label leakage common in session-level studies` (77)
- `Faithfulness and stability tests validate stage-conditioned explanations` (72)
- `TreeSHAP and TimeSHAP agree on stage rankings, giving convergent evidence` (73)
- `Conversion becomes harder to predict, not easier, deeper in the funnel` (69)

The TimeSHAP bullet is **[PENDING]**. The last bullet now reflects the confirmed H1
contradiction and is a stronger hook than the random-split point, which belongs in
Discussion 5.4 rather than the highlights.

### Abstract (~220 words, unstructured — DSS and ESWA both prefer this)

One sentence each: (i) problem — conversion prediction is well studied but explanations are
static and unvalidated; (ii) gap — attributions are computed on whole sessions, which both
leaks the outcome and hides how drivers change; (iii) what we do — funnel-stage conditioning
with a strict prefix protocol, on one benchmark and one event-level dataset; (iv) how we
validate — faithfulness, stability, and cross-paradigm agreement with TimeSHAP; (v) headline
results with numbers; (vi) what it means for practice; (vii) reproducibility statement.

Write this **last**. Do not draft it before Section 4 is final.

### Keywords (5–6)

Explainable AI; SHAP; purchase prediction; consumer journey; clickstream analysis; design
science research

---

## 1. Introduction (~1,200 words) — protocol Sec. 2

**1.1 Motivation.** Conversion prediction drives real decisions (targeting, retargeting
budget, on-site intervention), and those decisions need reasons, not just scores. Frame
around the decision-maker, not the algorithm — DSS reviewers reward this.

**1.2 The gap.** Three points, in this order:

1. E-commerce XAI papers produce SHAP plots and stop. Almost none test whether the
   explanations are faithful.
2. Attributions are computed over the whole session, so they include events at or after the
   purchase decision. This is a leakage problem hiding inside an explanation problem.
3. A single whole-session attribution cannot say *when* a driver mattered.

**1.3 Contribution.** State plainly that this is **framework-level and domain-level, not a
new estimator** (Sec. 2). Name TimeSHAP, WindowSHAP and GroupSegment-SHAP here, in the
introduction, not only in related work — pre-empting the novelty objection early is what
separates an accepted paper from a desk-reject. Then claim exactly three things:

1. Funnel-stage conditioning of Shapley attributions and the resulting *attribution
   trajectory* across stages.
2. A faithfulness/stability evaluation layer applied to those stage-conditioned
   explanations.
3. Convergent cross-paradigm evidence (TreeSHAP on tabular prefixes vs. TimeSHAP on a
   recurrent model).

**1.4 Structure.** One short paragraph.

---

## 2. Related work (~1,400 words) — protocol Sec. 2

**2.1 Purchase and conversion prediction from clickstream.** Sakar et al. (2019) for the
benchmark; the session-level aggregate tradition.

**2.2 Shapley attribution for sequential and temporal models.** TimeSHAP (KDD 2021),
WindowSHAP, GroupSegment-SHAP. Be scrupulous: describe what each already does.

**2.3 Explanation quality evaluation.** Faithfulness (insertion/deletion, ablation),
robustness/local Lipschitz, consistency. Cite `quantus` as the toolkit.

**2.4 Positioning table.** A comparison table — *method | grain of attribution | validated? |
domain* — with our row last. This single table does most of the novelty work.
→ **Table 1**

---

## 3. Materials and methods (~3,000 words) — protocol Sec. 3, 5, 7–9

Written so a reader could rebuild the study. This is the section Q1 reviewers audit hardest.

**3.1 Research design.** DSR framing (Hevner et al.); the artifact is the framework; each
(model × stage × explanation) is a build–evaluate cycle. Pre-registration statement: every
analytic decision fixed before the test partition was opened, deviations logged in a dated
amendment appendix. **This is a genuine strength — say it explicitly.**

**3.2 Data.** [HAVE]

- Dataset A: UCI Online Shoppers, 12,330 sessions, prevalence 0.1547.
- Dataset B: REES46 multi-category store, October 2019, 42,448,764 raw events, sourced from
  the publisher's own open endpoint with SHA-256 recorded.
- Subsample: 200,000 users, seeded, sampling by user so visitors stay whole (Sec. 5.3
  permits subsampling; amendment A15).
- **Variable availability table** — the honest Sec. 5.1 accounting, including that
  search-query text is absent from both datasets and scroll depth is proxied by dwell time.
  → **Table 2**
- **Licence caveat.** Dataset B's terms are not formally stated; disclose this in the data
  availability statement rather than asserting a licence.

**3.3 Cleaning and sessionisation.** [HAVE] 30-minute inactivity rule stated as a *choice*,
with the 15/30/60 sensitivity deferred to the appendix. Data-flow accounting from 42.4M
events to 485,459 sessions. → **Figure 1** (CONSORT-style flow diagram)

**3.4 Funnel stages and the anti-leakage prefix protocol.** [HAVE] The scientific core.
Report the amended cut-points and *why* they were amended — reviewers respect a documented
design correction far more than a definition that happens to work:

- S1 closes at the second product interaction, not the first view (A6). At the first view
  the prefix is one event and 19 of 23 features are constant.
- S2 opens on the second browsing signal, not the first (A8). One signal is incidental and
  collapsed 83% of S1/S2 prefixes onto the same event.
- Out-of-order stages are unreached, not shunted (A16). Shunting made 45% of S2/S3 prefixes
  identical.
- S4 admits no feature matrix: its cut-point is the label.
- → **Table 3** (stage definitions, cut-points, N, reach rate, prevalence)

**3.5 Feature engineering.** [HAVE] The eight families; navigation entropy and click velocity
as the differentiating features. State that dwell excludes the final prefix event, because
its dwell would require an event past the cut-point (A3). → **Table 4** (feature
availability by stage)

**3.6 Modelling.** [HAVE for A, PENDING for B] Five baselines; stage models per prefix; the
GRU/Transformer arm. Imbalance handled inside training folds only; nested CV with Optuna;
calibration on a held-out split.

**3.7 Explainability protocol.** [NOT BUILT] Three layers: interventional stage TreeSHAP;
TimeSHAP on the sequence model; the explanation-quality layer. State the pass thresholds
*before* reporting results (faithfulness correlation > 0.5, cross-paradigm Spearman > 0.6).

**3.8 Evaluation and statistics.** [HAVE] PR-AUC primary; bootstrapped CIs at ≥2,000
resamples; ≥5 seeds; DeLong, McNemar, Wilcoxon; Holm correction mandatory. Note that the
DeLong implementation is vendored and unit-tested against a naive transcription of the 1988
definition — a reviewer can verify it.

**3.9 Reproducibility.** [HAVE] Frozen splits, DVC, MLflow, pinned lockfile, frozen seed
list. → moves to the Declarations if the journal has a code-availability policy.

---

## 4. Results (~2,500 words) — protocol Sec. 10–12

Report only. No interpretation — that is Section 5. Q1 reviewers penalise blurring these.

**4.1 Descriptives and data flow.** [HAVE] → Figure 1, Table 3.

**4.2 RQ1 — predictive performance by stage.** [PENDING for B, HAVE for A]

Dataset A benchmark, month-ordered split, test = December, 5 seeds:

| Model | PR-AUC | ROC-AUC | ECE |
|---|---|---|---|
| LightGBM | 0.710 ± 0.008 | 0.899 | 0.117 |
| XGBoost | 0.707 ± 0.008 | 0.902 | 0.116 |
| CatBoost | 0.699 ± 0.014 | 0.905 | 0.115 |
| Random Forest | 0.694 ± 0.011 | 0.910 | 0.105 |
| Logistic Regression | 0.575 ± 0.000 | 0.879 | 0.102 |

→ **Table 5** (Dataset A), **Table 6** (Dataset B by stage), **Figure 2** (improvement curve)

Dataset A now reports pre/post calibration (Sec. 9.6), calibrated on a training-period
slice rather than the validation month (A18):

| Model | PR-AUC | ECE uncalibrated | ECE calibrated |
|---|---|---|---|
| CatBoost | 0.699 ± 0.006 | 0.063 | 0.018 |
| Random Forest | 0.685 ± 0.006 | 0.096 | 0.015 |
| LightGBM | 0.677 ± 0.008 | 0.065 | 0.026 |
| XGBoost | 0.676 ± 0.005 | 0.036 | 0.018 |
| Logistic Regression | 0.583 ± 0.000 | 0.203 | 0.032 |

**Report the stage curve against reach and prevalence, not alone.** S2's prevalence (6.8%)
is *below* S1's (8.8%) because early carters skip consideration. The curve is a *conditional*
claim and must be labelled one.

**Report PR-AUC lift, not raw PR-AUC, as the cross-stage comparison.** Chance-level PR-AUC
equals the prevalence, and S3's base rate (0.52) is seven times S1's (0.073). Raw PR-AUC
rises across stages purely because of this; the normalised curve falls. Present both columns
side by side and explain the difference in one sentence — a reviewer who sees only the raw
column will assume you did not notice.

**H1 will likely be rejected. [PENDING confirmation]** Preliminary numbers show lift falling
monotonically (1.83 → 1.52 → 1.11) and ROC-AUC falling with it (0.640 → 0.616 → 0.566).
ROC-AUC is prevalence-independent, so it cannot be explained by base rates. If the tuned
multi-seed run confirms this, restructure Section 5.1 around *why* late-stage prediction is
near-chance rather than around a confirmed hypothesis. A rejected pre-registered hypothesis
with a coherent mechanism is a stronger paper than a confirmed one — but only if the
pre-registration is visible, which is what Appendix A is for.

**4.3 Ablation (H3).** [HAVE] Report **two** columns, not one, and say which comparison H3
made. Against baseline aggregates alone — the comparison H3 states — entropy and click
velocity add +0.0039 / +0.0031 / +0.0223 PR-AUC at S1/S2/S3, four to fifteen times the seed
standard deviation. Given the temporal family they add +0.0006 / +0.0012 / **−0.0021**.

H3 is **supported**, but the signal is largely *shared with* temporal features rather than
additional to them — click velocity is events over elapsed duration, so both its
constituents are temporal. → **Table 7**

**4.4 RQ2 — attribution migration.** [NOT BUILT] **The centrepiece.** Mean |SHAP| trajectory
per feature across S1→S3, with sign changes marked. → **Figure 3**

Report the stage-distinctness table beside it: a migration trajectory between stages that
share a cut-point is not a finding. Both legs are currently 0% identical (gaps of 2.49 and
5.37 events). → **Table 8**

**4.5 RQ3 — faithfulness, stability, convergence.** [NOT BUILT] → **Table 9**, **Figure 4**

**4.6 RQ4 — actionability.** [NOT BUILT] What stage-conditioned attributions identify that
static whole-session SHAP misses. → **Table 10**

**4.7 Statistical comparisons.** [PENDING] Pre-declared confirmatory pairs only, with Holm
adjustment and effect sizes beside every p-value. → **Table 11**

**4.8 Sensitivity analyses.** [PENDING] Sessionisation gap 15/30/60; subsample size 200k vs
400k; temporal vs grouped split. → Appendix.

---

## 5. Discussion (~2,000 words) — protocol Sec. 11.3, 14

**5.1 Answering the research questions.** One subsection per RQ, each stating whether the
hypothesis held. **Write the H1 discussion now, because the data already complicates it:**
prevalence is non-monotone across stages, so any PR-AUC gain is partly composition. Treat
that as a finding about funnel selection, not a modelling failure.

**5.2 Theoretical implications.** What attribution migration says about consumer journey
theory — whether context-then-behaviour (H2) is supported.

**5.3 Practical implications.** Stage-specific intervention points. Be careful:
**intervention recommendations require calibrated probabilities**, see 5.5.

**5.4 Methodological implications.** [HAVE] Two contributions independent of the framework,
worth a subsection each because both are cheap to state and hard to argue with:

- **The canonical benchmark drifts and is usually split randomly.** Conversion prevalence
  in Dataset A rises from 1.6% (Feb) to 25.4% (Nov) then halves to 12.5% (Dec) — a 15×
  swing. Random splits mix these months and hide it. Our month-ordered numbers will read
  *lower* than the literature's; say why, plainly, rather than burying it.
- **Prior-probability shift silently breaks calibration, and the fix is the calibration
  set, not a post-hoc correction.** Calibrating on November and testing on December yielded
  ECE 0.105–0.117 across all five models; LightGBM predicted a mean 0.2421 against an actual
  0.1251 — ratio 1.935, almost exactly the 2.03 prevalence ratio — over-predicting in *every*
  reliability bin. Discrimination was untouched, since ranking is invariant to monotone
  miscalibration. Moving calibration to a training-period slice cut ECE to 0.015–0.032.
  Notably, applying Saerens et al. (2002) EM prior correction *on top* made things worse for
  every model: it estimated a prior of 0.1447 against a true 0.1251 and over-corrected a
  shift that was no longer there. **Report this negative result** — it is more useful to a
  practitioner than the positive one, because the instinct is to reach for the correction
  rather than to fix the split. → **Figure 5** (reliability diagram, before/after)

**5.5 Limitations and threats to validity.** Sec. 14's table, plus what we found:

- Absent variables: no search-query text in either dataset; scroll depth proxied.
- Dataset A is non-causal by construction — whole-session aggregates, `PageValues` measured
  post-transaction.
- Stage models are conditional on reaching the stage; the improvement curve is not marginal.
- The temporal split retains 222,268 of 485,459 sessions, dropping 51,256 straddling users
  (A17). A wider window would shrink this.
- Calibration under prevalence shift, per 5.4.
- One month, one retailer, one market: external validity is limited and should be stated
  without hedging.

**5.6 Future work.** Multi-month windows; prevalence-corrected calibration; live A/B
validation of the intervention points.

---

## 6. Conclusion (~400 words)

No new results, no new citations. Restate contribution, headline findings, and the single
most useful thing a practitioner should take away.

---

## Declarations (Elsevier requires these)

- **CRediT author statement** — Conceptualization, Methodology, Software, Formal analysis,
  Data curation, Writing – original draft, Writing – review & editing, Visualization.
- **Declaration of competing interest** — state none if none.
- **Declaration of generative AI in the writing process.** Elsevier requires disclosure
  where generative AI assisted preparation. AI was used for code implementation and
  analysis support in this project; disclose it accurately in the form the target journal
  specifies. Do not overstate or understate.
- **Data availability.** Dataset A: UCI, CC BY 4.0, cite Sakar et al. (2019). Dataset B:
  REES46 open endpoint — **state honestly that no formal licence text is published** and
  that permission was sought. Do not assert a licence you cannot evidence.
- **Code availability.** Repository with DVC pipeline, frozen splits, MLflow logs, lockfile.
- **Ethics.** Behavioural data only, no PII, no re-identification attempted, identifiers
  pre-hashed by the publisher.
- **Funding.**

## Appendices

- **A.** Pre-registered protocol v1.0 and the dated amendment log. **Include this.** A
  visible amendment log with rationales is unusual and reads as rigour, not as error.
- **B.** Full feature dictionary.
- **C.** Sessionisation sensitivity (15/30/60) and subsample-size sensitivity.
- **D.** Hyperparameter search spaces and selected values.
- **E.** Seed list and compute budget.

---

## Figures and tables — target 5 figures, 11 tables

Elsevier Q1 papers rarely exceed this. Figure 3 (attribution migration) is the paper; it
should be the one a reader remembers, so budget real design effort for it.

| # | Item | Status |
|---|---|---|
| Fig 1 | Data-flow diagram | [HAVE] |
| Fig 2 | Prediction-improvement curve by stage | [PENDING] |
| Fig 3 | **Attribution migration trajectories** | [NOT BUILT] |
| Fig 4 | Faithfulness / stability | [NOT BUILT] |
| Fig 5 | Reliability diagram under prevalence shift | [HAVE] |
| Tab 1 | Related-work positioning | — |
| Tab 2 | Variable availability | [HAVE] |
| Tab 3 | Stage definitions, N, reach, prevalence | [HAVE] |
| Tab 4 | Feature availability by stage | [HAVE] |
| Tab 5 | Dataset A baselines | [HAVE] |
| Tab 6 | Dataset B stage models | [PENDING] |
| Tab 7 | Ablation ladder | [PENDING] |
| Tab 8 | Stage distinctness | [HAVE] |
| Tab 9 | Explanation quality | [NOT BUILT] |
| Tab 10 | Static vs stage-conditioned actionability | [NOT BUILT] |
| Tab 11 | Statistical comparisons, Holm-adjusted | [PENDING] |

---

## Journal fit

| Journal | Fit | Note |
|---|---|---|
| **Decision Support Systems** | Strongest | Rewards the DSR framing and the decision-maker angle. Lead with RQ4 actionability. |
| Expert Systems with Applications | Strong | Method-and-application framing; wants the pipeline foregrounded. |
| Electronic Commerce Research and Applications | Good | Domain fit is excellent, XAI methodology less central. |
| Information Processing & Management | Moderate | Would want the evaluation layer as the primary contribution. |

Recommend **Decision Support Systems** first, and write Section 5.3 (practical implications)
to their audience.

---

## Writing order

1. Methods (3) — everything needed is already computed or specified.
2. Results (4) — fill as runs complete.
3. Related work (2) and Table 1 — can be drafted in parallel, needs no results.
4. Discussion (5).
5. Introduction (1) — after contributions are provably true.
6. Abstract, highlights, conclusion — last.

**Do not draft the introduction's contribution claims before Section 4 exists.** Every
protocol amendment so far arrived from measurement contradicting an assumption; the same
will happen to at least one claim you would write today.
