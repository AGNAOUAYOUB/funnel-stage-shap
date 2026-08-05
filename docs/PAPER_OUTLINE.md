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
- `Navigation entropy peaks mid-funnel, which static explanations average away` (75)
- `Conversion becomes harder to predict, not easier, deeper in the funnel` (69)
- `Faithfulness and stability tests validate stage-conditioned explanations` (72)

All within Elsevier's 85-character limit, and all five are now **[HAVE]** — the faithfulness
bullet is supported by Layer 3 (seed consistency ρ ≥ 0.94, deletion AUC roughly half
insertion at every stage), with the S2 threshold miss reported in 4.5. The TimeSHAP
convergence bullet was dropped: it competes with stronger claims for five slots and belongs
in the abstract instead.

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
made. All differences below carry paired-bootstrap 95% CIs, Holm-corrected across the
nine-contrast family.

| Stage | +temporal vs baseline | entropy/velocity vs baseline (H3) | entropy/velocity given temporal |
|---|---|---|---|
| S1 | +0.0102 [+0.0045, +0.0167] | +0.0038 [−0.0002, +0.0084] | −0.0006 [−0.0017, +0.0005] |
| S2 | +0.0120 [+0.0057, +0.0197] | +0.0035 [−0.0011, +0.0085] | +0.0007 [−0.0017, +0.0031] |
| S3 | +0.0139 [−0.0074, +0.0370] | +0.0156 [−0.0027, +0.0346] | +0.0018 [−0.0045, +0.0085] |

**H3 is not supported**: the CI covers zero at every stage. The temporal family is supported
at S1 and S2 and undecided at S3, where 2,973 test sessions leave every interval about ±0.02
wide. The robust finding is the **redundancy**: given temporal, entropy and velocity
contribute nothing, and those are the only intervals narrow enough to distinguish "no
effect" from "cannot tell". → **Table 7**

**Do not report mean ± seed-std as an error bar for a between-model difference.** Seed
spread measures refit variability on a fixed test sample; it is roughly an order of
magnitude tighter than the sampling uncertainty of the metric, and using it as an interval
turns a null into a positive. One sentence in Methods 3.8 covering this is worth including —
it is a common error in the applied literature.

**4.4 RQ2 — attribution migration.** [HAVE, single seed] **The centrepiece.** Share of total
mean-|SHAP| per correlation group across stages, interventional TreeSHAP, background 500,
2,000 validation sessions explained per stage:

| Group | S1 | S2 | S3 | Δ S1→S3 | sign reversal |
|---|---|---|---|---|---|
| Price (2 features) | **0.437** | 0.239 | 0.379 | −0.058 | |
| Engagement / tempo (7) | 0.169 | **0.368** | 0.338 | **+0.169** | |
| Products viewed (2) | 0.192 | 0.033 | 0.057 | **−0.136** | yes |
| Navigation / entropy (6) | 0.035 | **0.224** | 0.074 | +0.039 | yes |
| Hour of day | 0.157 | 0.125 | 0.133 | −0.024 | |
| Weekend | 0.010 | 0.011 | 0.020 | +0.009 | yes |

Three things to report, in this order:

1. **Behavioural intensity gains attribution mass across the funnel** — engagement/tempo
   doubles from 0.169 to 0.338, the largest single shift. This is H2's behaviour arm, and it
   is supported.
2. **Navigation entropy peaks at S2 and collapses again** (0.035 → 0.224 → 0.074). It is
   not monotone, and a whole-session attribution would average it to a middling constant.
   This is the clearest single piece of evidence for RQ4 and should be called out as such
   rather than buried in the trajectory table.
3. **Product-breadth attribution collapses** after awareness (0.192 → 0.033), with a sign
   reversal.

**H2's context arm cannot be tested on Dataset B.** Traffic source and device are absent
from the REES46 schema (Table 2), so "context dominates early" rests on Dataset A. Price is
the nearest available product-context proxy and does lead at S1 (0.437) before declining,
which is weakly consistent, but it is a product attribute rather than an acquisition
channel and must not be presented as one.

Report the stage-distinctness table beside the figure: a trajectory between stages sharing a
cut-point is not a finding. Both legs are 0% identical (gaps of 2.49 and 5.37 events).
→ **Table 8**

**Caption must state the grouping rule.** Correlation groups are fixed across stages and
taken from S1, the coarsest, because a two-event S1 prefix makes seven engagement and
temporal features numerically identical — their individual S1 attributions are arbitrary
splits of one quantity. The cost is that migration *within* a merged group is invisible at
later stages, where those features do separate.

**[PENDING] Single seed (42), single model.** RQ3 requires rank correlation across seeds
before this table is publishable.

**4.5 RQ3 — faithfulness, stability, convergence.** [HAVE except TimeSHAP]
→ **Table 9**, **Figure 4**

| Stage | Faithfulness ρ | vs 0.5 | Deletion AUC | Insertion AUC | Monotone | Seed ρ (mean / min) | vs 0.6 |
|---|---|---|---|---|---|---|---|
| S1 | 0.571 ± 0.177 | **pass** | 0.2365 | 0.5039 | yes | 1.000 / 1.000 | **pass** |
| S2 | 0.494 ± 0.179 | *fail* | 0.1440 | 0.3820 | yes | 0.966 / 0.943 | **pass** |
| S3 | 0.565 ± 0.169 | **pass** | 0.3495 | 0.5898 | yes | 0.943 / 0.886 | **pass** |

Three claims, in this order:

1. **Seed consistency passes decisively** (ρ ≥ 0.943 mean, ≥ 0.886 worst pair, against a
   pre-registered 0.6). This is what licenses Figure 3: the migration trajectory survives
   reseeding and is not one draw. Report the *minimum* pairwise correlation alongside the
   mean — a reviewer will ask about the worst pair, not the average.
2. **Deletion AUC is roughly half insertion AUC at every stage**, and removing top
   positively-attributed features degrades the prediction monotonically. The explanations
   point at features the model genuinely uses.
3. **Faithfulness is stage-dependent, and S2 misses the pre-registered threshold** (0.494
   against 0.5). Report this as a failure against the stated bar rather than rounding it up
   — the whole point of fixing the threshold in advance was to make a near-miss reportable.

**Frame the S2 miss with the H1 result rather than as an isolated defect.** S2 is also where
predictive lift is weakest relative to its base rate. A model with less signal to explain
produces attributions that are harder to validate; the two findings are consistent, and
saying so is stronger than treating the faithfulness miss as noise.

**H4 (cross-paradigm).** [HAVE] GRU per stage, TimeSHAP attribution, ranking correlated
against TreeSHAP:

| Stage | GRU PR-AUC | Tree PR-AUC | Spearman | Concepts compared |
|---|---|---|---|---|
| S1 | 0.1237 | 0.1312 | +0.200 | 4 |
| S2 | 0.0593 | 0.0834 | +0.500 | 5 |
| S3 | 0.5646 | 0.5681 | +0.600 | 5 |

**H4 is not supported at the pre-registered threshold, and the test is underpowered.**
Report both, and lead with the second. The comparison rests on four or five mappable
concepts — with four items Spearman can only take ±1.0, ±0.8, ±0.6, ±0.4, ±0.2 or 0, so the
0.6 threshold is being applied to something that is barely a statistic. State plainly that
H4 was pre-registered without anticipating how few concepts the two paradigms would share.

Note that the GRU is *competitive* (0.124 vs 0.131 at S1, 0.565 vs 0.568 at S3), so this is
not a failed convergence test caused by one incompetent arm.

**Agreement rises monotonically down the funnel** (0.200 → 0.500 → 0.600). Combined with S2
having both the weakest lift (A19) and the only faithfulness miss (4.5), three independent
measurements say the same thing: **where signal is weak, explanations are less faithful and
less reproducible across paradigms.** That deserves its own paragraph in 5.1 — it is a
finding about when to trust an explanation, which is the paper's subject.

**Per-instance convergence.** [HAVE] Report this beside the pre-registered version — the gap
between them is the finding.

Five seeds, mean across concepts:

| Stage | Aggregate ρ (5 concepts) | **Per-instance mean ρ (5 seeds)** | Concepts sign-stable |
|---|---|---|---|
| S1 | +0.200 | **+0.179** | 1 of 4 |
| S2 | +0.500 | **−0.018** | 0 of 5 |
| S3 | +0.600 | **+0.046** | 2 of 5 |

Zero of fourteen (stage, concept) pairs clear 0.6, and only 3 of 14 hold a consistent sign
across seeds.

**Aggregate convergent validity does not imply agreement about individual cases.** The same
models yield aggregate correlations of +0.200 to +0.600 and per-instance agreement
indistinguishable from zero. For anyone acting on a specific visitor's explanation, only the
second matters. This is a claim about how convergent validity should be tested in XAI
generally, not only about this dataset, and belongs in 5.4 as well as here.

**Report sign stability, not only the mean.** With seed standard deviations of 0.2–0.4 on
these correlations, a mean alone cannot distinguish a weak effect from noise. A
single-seed run of this analysis produced an apparent −0.151 at S2 that did not survive
reseeding (A23b); the paper must not repeat that claim.

**4.6 RQ4 — actionability.** [PARTIAL] What stage-conditioned attributions identify that
static whole-session SHAP misses. → **Table 10**

The strongest exhibit already exists: **navigation entropy is non-monotone across stages**
(0.035 → 0.224 → 0.074). Any whole-session attribution collapses this to a single middling
number and would rank it as a minor, uninteresting driver. The stage-conditioned view says
something a practitioner can act on — wandering behaviour matters specifically during
consideration, so that is where a recommendation or filter intervention has purchase, and it
is close to irrelevant at cart.

Pair this with the A19 finding for the managerial argument: late-stage prediction is near
chance (S3 lift 1.09), so the actionable window is **early and mid funnel**, which is the
opposite of where cart-abandonment practice concentrates spend. Together these are the
paper's practical contribution.

**The formal contrast.** [HAVE] `static_contrast` builds the counterfactual a static analysis
would report — the stage-average attribution share — and measures how far each driver
actually travels. → **Table 10**

| Driver | Static share | Range | Peak | Shape |
|---|---|---|---|---|
| engagement & tempo | 0.292 | 0.199 | 0.368 at S2 | non-monotone |
| price | 0.352 | 0.198 | 0.437 at S1 | non-monotone |
| navigation & category | 0.111 | 0.189 | 0.224 at S2 | non-monotone |

**Navigation is the strongest case in relative terms.** A static analysis reports 0.111 and
would rank it a minor driver; it actually reaches **0.224 at S2** — twice the static figure —
and falls to 0.074 by S3. The static number is not merely imprecise, it is a value the
driver never takes at any stage.

Note the static baseline is built on the *same* data by pooling stages, not by switching to
Dataset A. Comparing across datasets would confound the question with a change of schema and
grain, which is a different claim.

**4.7 Statistical comparisons.** [PENDING] Pre-declared confirmatory pairs only, with Holm
adjustment and effect sizes beside every p-value. → **Table 11**

**4.8 Sensitivity analyses.** → Appendix C.

**Sessionisation gap (Sec. 7.2 requirement).** [HAVE] The 30-minute rule is a convention, and
the appendix should say plainly that it barely matters here:

| Gap | Sessions (full) | S1 reach / prev | S2 reach / prev | S3 reach / prev |
|---|---|---|---|---|
| 15 min | 5,544,851 | 0.978 / 0.0864 | 0.471 / 0.0652 | 0.092 / 0.5160 |
| 30 min | 5,366,181 | 0.979 / 0.0885 | 0.480 / 0.0683 | 0.093 / 0.5210 |
| 60 min | 5,219,040 | 0.979 / 0.0897 | 0.487 / 0.0705 | 0.093 / 0.5248 |

Halving or doubling the threshold moves the session count by ±3%, stage reach by under one
percentage point, and stage prevalence by well under half a point — all far smaller than the
between-stage differences the paper's claims rest on. **Say this explicitly.** The
literature's caution about the reflexive 30-minute rule is well founded in general, and the
useful contribution here is showing it is not load-bearing *for these conclusions*, rather
than repeating the caution and moving on.

**Subsample size (amendment A15).** [HAVE] Doubling to 400,000 users:

| Stage | PR-AUC 200k | PR-AUC 400k | Δ | Lift 200k | Lift 400k | ROC-AUC 200k | ROC-AUC 400k |
|---|---|---|---|---|---|---|---|
| S1 | 0.1312 | 0.1346 | +0.0034 | 1.79 | 1.84 | 0.6406 | 0.6482 |
| S2 | 0.0834 | 0.0873 | +0.0040 | 1.41 | 1.49 | 0.6198 | 0.6341 |
| S3 | 0.5681 | 0.5766 | +0.0086 | 1.09 | 1.10 | 0.5695 | 0.5699 |

Every metric improves slightly with more data — expected, and evidence the 200k sample is
mildly conservative rather than distorting. **The headline conclusion is unchanged:** lift
still falls monotonically (1.84 → 1.49 → 1.10) and ROC-AUC with it (0.648 → 0.634 → 0.570).
Stage reach and prevalence are near-identical (S1 0.0886 vs 0.0885, S3 0.5213 vs 0.5210).

Because the sample is nested at a fixed seed, the 400k set is a strict superset of the
200k one, so these differences are attributable to the added users rather than to a
different draw. State that — it is what makes the comparison interpretable.

**[PENDING]** temporal vs grouped split comparison.

---

## 5. Discussion (~2,000 words) — protocol Sec. 11.3, 14

**5.0 Explanation quality varies by stage.** Three independent measurements converge on
consideration (S2) being where this framework's explanations are least trustworthy: weakest
predictive lift (1.41), the only faithfulness threshold miss (0.494 vs 0.5), and the lowest
aggregate cross-paradigm agreement. A fourth signal — negative per-instance agreement — was
reported from a single seed and **withdrawn** after reseeding (A23b); do not use it.

The transferable claim is **explanation quality tracks predictive signal, so validate
explanations per stage rather than per model.** Reporting stage-averaged explanation quality
would hide all three signals.

**5.1 Answering the research questions.** One subsection per RQ, each stating whether the
hypothesis held. **Write the H1 discussion now, because the data already complicates it:**
prevalence is non-monotone across stages, so any PR-AUC gain is partly composition. Treat
that as a finding about funnel selection, not a modelling failure.

**5.2 Theoretical implications.** What attribution migration says about consumer journey
theory. H2's **behaviour arm is supported** — engagement and tempo attribution doubles from
S1 to S3 (0.169 → 0.338). Its **context arm cannot be tested on Dataset B**, since traffic
source and device are absent from the schema, so that half rests on Dataset A.

The theoretically interesting result is the one H2 did not predict: **navigation entropy is
non-monotone**, peaking during consideration (0.035 → 0.224 → 0.074). Journey theory tends
to treat exploration as decaying steadily as intent forms; the data says exploratory
breadth is a *stage-specific* signal that matters where the visitor is choosing between
alternatives and stops mattering once they have chosen. A monotone framing of the funnel
cannot express that, which is an argument for stage conditioning as a modelling choice
rather than merely a reporting convenience.

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

- **S3 is underpowered.** With 2,973 test sessions every intent-stage interval is about
  ±0.02 wide, admitting both a meaningful gain and a meaningful loss, so no S3 contrast can
  be decided. A sample-size limitation, not a null result; it argues for raising the
  Dataset B subsample or widening the window beyond one month.
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
| Fig 2 | Prediction-improvement curve by stage | [HAVE] |
| Fig 3 | **Attribution migration trajectories** | [HAVE] |
| Fig 4 | Faithfulness / deletion-insertion curves | [HAVE] |
| Fig 5 | Reliability diagram under prevalence shift | [HAVE] |
| Tab 1 | Related-work positioning | — |
| Tab 2 | Variable availability | [HAVE] |
| Tab 3 | Stage definitions, N, reach, prevalence | [HAVE] |
| Tab 4 | Feature availability by stage | [HAVE] |
| Tab 5 | Dataset A baselines | [HAVE] |
| Tab 6 | Dataset B stage models | [HAVE] |
| Tab 7 | Ablation ladder | [HAVE] |
| Tab 8 | Stage distinctness | [HAVE] |
| Tab 9 | Explanation quality | [HAVE] |
| Tab 10 | Static vs stage-conditioned actionability | [HAVE] |
| Tab 11 | Statistical comparisons, Holm-adjusted | [HAVE] |

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
