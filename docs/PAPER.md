# Funnel-Stage SHAP: temporal attribution trajectories and their faithfulness in e-commerce purchase prediction

**Ayoub Agnaou**

---

> ## ⚠ CITATION VERIFICATION REQUIRED BEFORE SUBMISSION
>
> The reference list contains works I am confident exist and whose content matches the use
> made of them here. **Every citation must still be verified against the actual publication
> before submission** — volume, issue, page numbers and, above all, that each source says
> what it is cited as saying. Three specific cautions:
>
> 1. **Page numbers and volumes are the least reliable elements below.** Check all of them.
> 2. **The protocol referenced "GroupSegment-SHAP".** I could not verify that a method by
>    that name exists. It is *not* cited here. If it does exist, add it to §2.2; if it does
>    not, the related-work positioning is unaffected.
> 3. **Sections marked `[UNVERIFIED CLAIM]`** assert something about the state of the
>    literature that I have not systematically checked (e.g. "most e-commerce XAI studies do
>    not validate their explanations"). These need a documented search before they can stand
>    as written.
>
> All *numerical results* in §4 come from `reports/tables/` in the accompanying repository
> and are reproducible from the frozen splits and seed list.

---

## Highlights

- Funnel-stage SHAP reveals how purchase drivers migrate across journey stages
- Prefix-only features remove the label leakage common in session-level studies
- Navigation entropy peaks mid-funnel, which static explanations average away
- Conversion becomes harder to predict, not easier, deeper in the funnel
- Faithfulness and stability tests validate stage-conditioned explanations

## Abstract

Conversion prediction from clickstream data is well established, but the explanations
attached to such models are typically computed once over an entire session and are almost
never validated. Both practices are problematic: a whole-session attribution includes events
at or after the purchase decision, and it cannot express *when* a driver mattered. We propose
a funnel-stage SHAP framework in which Shapley attributions are conditioned on journey stage
and computed strictly on the event prefix available at each stage's cut-point, and we pair it
with an evaluation layer that tests whether the resulting explanations are faithful, stable
and reproducible. The framework is applied to a canonical session-level benchmark (12,330
sessions) and to an event-level clickstream of 42.4 million events yielding 485,459 sessions.
Three results run against expectation. First, once normalised by its own chance level,
predictive performance *declines* across the funnel (lift 1.79 to 1.09; ROC-AUC 0.641 to
0.570), so conversion becomes harder, not easier, to predict as intent forms. Second,
attribution is strongly non-monotone: navigation entropy carries 3.5% of attribution mass at
awareness, 22.4% at consideration and 7.4% at intent, a pattern any whole-session analysis
averages into a single uninformative value. Third, cross-paradigm agreement between tree- and
recurrent-model attributions holds at the level of aggregate rankings but disappears at the
level of individual journeys. We argue that explanation quality tracks predictive signal and
should therefore be validated per stage rather than per model.

**Keywords:** Explainable AI; SHAP; purchase prediction; consumer journey; clickstream
analysis; design science research

---

## 1. Introduction

Online retailers routinely predict whether a browsing session will end in a purchase, and
they act on those predictions: allocating retargeting budget, triggering on-site
interventions, prioritising service contacts. Decisions of that kind require reasons, not
only scores. A model that ranks sessions accurately but cannot say *why* offers little
guidance about what to change (Rudin, 2019), and the marketing literature has long framed the
online journey as a sequence of qualitatively distinct phases rather than a single
undifferentiated visit (Bucklin & Sismeiro, 2003; Lemon & Verhoef, 2016; Moe, 2003).

Shapley additive explanations (Lundberg & Lee, 2017), and their exact and efficient
tree-model variant (Lundberg et al., 2020), have become the default vehicle for supplying
those reasons. In the e-commerce setting they are typically applied in a particular way: a
single model is fitted to session-level aggregates, and a single attribution is computed per
feature over the whole session. We argue that this common practice carries two defects, one
methodological and one substantive.

**The methodological defect is leakage inside the explanation.** When features are aggregated
over an entire session, they necessarily include events at or after the moment of purchase.
The canonical benchmark for this task illustrates the problem directly: its strongest
predictor, `PageValues`, is the average value of pages on which a transaction was completed
and is therefore partly a consequence of conversion rather than a cause of it (Sakar et al.,
2019). A model built on such features may predict well and explain nothing, because the
explanation describes the outcome rather than the behaviour that preceded it.

**The substantive defect is temporal flattening.** A single attribution per feature cannot
express that a driver mattered at one point in the journey and not at another. If the
importance of a behavioural signal genuinely changes as intent forms — which the staged view
of the consumer journey predicts — then a whole-session attribution reports its average, a
number that may correspond to no phase of the journey at all.

### 1.1 Positioning relative to existing methods

Sequence-aware Shapley estimation already exists, and this paper does not propose a new one.
TimeSHAP extends KernelSHAP to recurrent models and produces event-, timestep- and
feature-level attributions (Bento et al., 2021). WindowSHAP computes Shapley values over
time-series windows rather than individual timesteps (Nayebi et al., 2023). Our contribution
is **not** a Shapley estimator. It is framework-level and domain-level, and consists of three
parts:

1. **Funnel-stage conditioning of attributions.** We compute Shapley attributions separately
   at each funnel stage, on the event prefix available at that stage, and characterise the
   resulting *attribution trajectory* across stages rather than reporting one whole-session
   attribution.
2. **A validation layer for stage-conditioned explanations.** We test faithfulness, stability
   under perturbation, and reproducibility across random seeds, with thresholds fixed before
   any result was computed.
3. **Cross-paradigm evidence, tested two ways.** We compare tree-based stage attributions
   against recurrent-model attributions both as aggregate rankings and, critically, at the
   level of individual journeys.

The study is pre-registered. Every analytic decision was fixed before the test partition was
opened, and each subsequent deviation is recorded with a date and rationale in a public
amendment log (Appendix A). Twenty-six amendments were logged, several of which reverse an
initial decision on the basis of measurement; we regard this record as part of the
contribution rather than an embarrassment to be hidden.

### 1.2 Structure

Section 2 positions the work against sequential Shapley methods and explanation-quality
evaluation. Section 3 specifies the data, the funnel-stage cut-points, the anti-leakage prefix
protocol and the modelling and evaluation procedures. Section 4 reports results against four
pre-registered hypotheses. Section 5 discusses implications, limitations and threats to
validity. Section 6 concludes.

---

## 2. Related work

### 2.1 Purchase prediction from clickstream data

Modelling online browsing and purchase behaviour has an established tradition in marketing
science, from path analysis of within-site navigation (Montgomery et al., 2004) to models of
purchase incidence at individual sites (Sismeiro & Bucklin, 2004; Van den Poel & Buckinx,
2005). Moe (2003) argued that visits are heterogeneous in purpose — directed buying,
search/deliberation, hedonic browsing, knowledge building — which implies that a single model
of "the session" conflates behaviourally distinct processes. The customer-journey literature
makes the same point at a higher level of abstraction, treating pre-purchase, purchase and
post-purchase as separate stages with distinct drivers (Lemon & Verhoef, 2016).

The machine-learning literature on the same task has largely proceeded on session-level
aggregates. The UCI Online Shoppers dataset (Sakar et al., 2019) is the canonical benchmark
and is dominated by gradient-boosted tree ensembles (Chen & Guestrin, 2016; Ke et al., 2017;
Prokhorenkova et al., 2018) and random forests (Breiman, 2001).

### 2.2 Shapley attribution for sequential models

Shapley values provide an additive attribution with desirable axiomatic properties (Lundberg
& Lee, 2017), and TreeSHAP makes their computation tractable and exact for tree ensembles
(Lundberg et al., 2020). Two caveats matter here. First, attributions computed under the
path-dependent estimator can assign credit to features merely correlated with those the model
uses; the interventional estimator, computed against an explicit background distribution,
weakens that dependence (Lundberg et al., 2020). Second, Shapley values are unreliable under
strong feature dependence more generally (Aas et al., 2021), which motivates grouping
correlated features before ranking them.

For sequential data, TimeSHAP (Bento et al., 2021) adapts KernelSHAP to recurrent models,
adding temporal-coalition pruning to keep computation tractable and producing attributions at
event, timestep and feature level. WindowSHAP (Nayebi et al., 2023) attributes to time
windows. Both address *how to compute* Shapley values on sequences. Neither addresses which
*decision point* the attribution should be conditioned on, which is the question this paper
takes up.

### 2.3 Evaluating explanations

That an explanation is produced does not make it faithful to the model. Bhatt et al. (2020)
formalise faithfulness as the correlation between the attribution assigned to a feature
subset and the change in prediction when that subset is removed. Deletion and insertion
curves offer a complementary view, measuring how quickly a prediction degrades as
highly-attributed features are removed (Petsiuk et al., 2018). Stability is typically
operationalised as a local-Lipschitz bound on attribution change under small input
perturbation (Alvarez-Melis & Jaakkola, 2018), and toolkits now exist to standardise such
measurements (Hedström et al., 2023). Explanations are also known to be manipulable and
therefore worth verifying rather than trusting (Slack et al., 2020).

`[UNVERIFIED CLAIM]` Our impression is that applied e-commerce XAI studies rarely apply any
of these tests, reporting SHAP summary plots without validation. **This claim requires a
systematic review before it can be stated as written**; it currently rests on informal
reading and is the single most important gap-claim in the paper.

### 2.4 Positioning

**Table 1** places this work against the methods above.

| Work | Attribution grain | Decision point | Validated? | Domain |
|---|---|---|---|---|
| Lundberg & Lee (2017) | feature, per instance | not applicable | axioms only | general |
| Lundberg et al. (2020) | feature, per instance | not applicable | axioms only | general (trees) |
| Bento et al. (2021) | event / timestep / feature | whole sequence | partial | general (recurrent) |
| Nayebi et al. (2023) | time window | whole sequence | partial | time series |
| Typical applied e-commerce XAI | feature, whole session | whole session | rarely | e-commerce |
| **This work** | **feature, per funnel stage** | **stage cut-point (prefix only)** | **faithfulness, stability, seed and cross-paradigm consistency** | **e-commerce** |

---

## 3. Materials and methods

### 3.1 Research design

The study follows a design-science frame (Hevner et al., 2004): the artefact is the
funnel-stage SHAP framework, and each (model × stage × explanation) combination constitutes a
build–evaluate cycle assessed on predictive utility (§3.8), explanation quality (§3.7) and
statistical rigour (§3.8).

The analysis is pre-registered. All analytic decisions — stage definitions, feature families,
model families, metrics, statistical tests and pass thresholds — were fixed in a protocol
document before the test partition was opened. The test partition was read once, at final
evaluation. Deviations after freeze are logged with date, rationale and affected sections
(Appendix A).

### 3.2 Data

Two datasets are used, at deliberately different grains.

**Dataset A** is the UCI Online Shoppers Purchasing Intention dataset (Sakar et al., 2019):
12,330 sessions described by 10 numeric and 7 categorical session-level aggregates, with a
binary `Revenue` target at 15.47% prevalence. It serves as a reproducible benchmark and as the
static counterpart against which stage-conditioning is contrasted. Because its features are
whole-session aggregates, models fitted to it are flagged **non-causal** throughout.

**Dataset B** is an event-level clickstream published by the REES46 Marketing Platform,
covering a multi-category online store. We use October 2019: 42,448,764 raw events with
fields `event_time`, `event_type` ∈ {view, cart, remove_from_cart, purchase}, `product_id`,
`category_id`, `brand`, `price`, `user_id` and `user_session`. After cleaning and
sessionisation this yields 485,459 sessions from a seeded sample of 200,000 users (§3.3).

**Table 2** reports variable availability honestly. Of ten idealised journey variables, two
are absent from both datasets. Search-query text does not appear in either. Scroll depth is
proxied by product-page dwell time and is never described as scroll depth in results. Device
and traffic source are present in Dataset A and absent from Dataset B, which constrains the
hypotheses testable on each (§4.4).

| Variable | Dataset A | Dataset B | Note |
|---|---|---|---|
| Page views | present | present | A: page-type counts; B: view events |
| Session duration | present | present | B: last minus first event time |
| Device | present | **absent** | B has no device field |
| Traffic source | present | **absent** | Constrains H2 on Dataset B |
| Time of day | proxy | present | A: month and weekend only |
| Bounce | present | proxy | B: single-event sessions |
| Cart events | **absent** | present | Core to stage S3 |
| Search queries | **absent** | **absent** | Reported as a limitation; no proxy claimed |
| Scroll depth | proxy | proxy | Product-page dwell time |
| Purchase | present | present | The label |

### 3.3 Cleaning, sessionisation and subsampling

Events with null values in `user_session`, `event_time` or `event_type`, unparseable
timestamps, or event types outside the expected vocabulary were removed, as were exact
duplicates on (`user_id`, `event_time`, `event_type`, `product_id`). Sessions were derived by
cutting each user's time-ordered event stream on a 30-minute inactivity gap. We treat the
30-minute rule as a *choice* rather than a standard and report a 15/30/60-minute sensitivity
analysis in §4.7. Sessions of fewer than two events were dropped, as were sessions above the
99.5th percentile of length or duration. **Figure 1** presents the resulting data flow.

For tractability the analysis uses a seeded random sample of 200,000 users, retaining every
session belonging to a sampled user. Sampling is by *user*, not by session, because both
splitting protocols (§3.6) keep visitors whole; sampling sessions independently would
fragment users before the splitter saw them. The sample is a deterministic hash of (seed,
user id), so it is stable across reruns and nested across sample sizes — a property used in
the sample-size sensitivity analysis (§4.7).

### 3.4 Funnel stages and the anti-leakage prefix protocol

This is the scientific core of the design. Four stages are defined on the event sequence, each
with an explicit cut-point, and **features at stage S_k are computed only on the event prefix
up to that cut-point**. The prefixes therefore nest, S1 ⊆ S2 ⊆ S3, and no prefix may contain
a purchase event.

Three cut-point definitions were amended during development, each on the basis of
measurement. We report the amendments because the reasoning is instructive and because a
definition that merely happens to work invites less confidence than one that was corrected.

- **S1 (awareness)** closes at the *second* product interaction. The original definition —
  the first view — produced a single-event prefix in which 19 of 23 features were constant,
  which would have made any improvement curve an artefact of S1 having no features (A6).
- **S2 (consideration)** opens on the *second* browsing signal (a repeat product view or a
  category switch on a view event). Requiring only the first signal placed S2's cut-point on
  the same event as S1's for 82.8% of sessions (A8).
- **S3 (intent)** opens at the first cart event. A stage whose trigger fires out of funnel
  order is treated as *unreached* rather than advanced to a later position; the alternative
  made 45.1% of S2 and S3 prefixes identical on real data (A16).
- **S4 (conversion)** is defined for descriptive purposes only. Its cut-point is the purchase
  event, which is the label, so no S4 feature matrix can exist without leakage.

**Table 3** reports the resulting stage populations.

| Stage | Definition | Cut-point | N | Reach | Prevalence |
|---|---|---|---|---|---|
| S1 | Entry to initial catalogue contact | 2nd product interaction | 475,140 | 97.9% | 8.85% |
| S2 | Product and category browsing | 2nd browsing signal | 233,160 | 48.0% | 6.83% |
| S3 | Cart activity begins | 1st cart event | 45,031 | 9.3% | 52.10% |
| S4 | Checkout window (descriptive only) | purchase or session end | 485,459 | 100% | 10.78% |

Two consequences of this design must travel with every result derived from it. First, the
stage models are **conditional on reaching the stage**: a session with no cart has no S3
prefix. Second, prevalence differs sharply by stage, which makes raw PR-AUC incomparable
across stages (§3.8).

### 3.5 Feature engineering

Features are grouped into families: behavioural counts, temporal features, engagement,
navigation entropy, click velocity, category transitions, a purchase-intent composite with
documented (not fitted) weights, and price context. Navigation entropy is the Shannon entropy
of the category distribution within the prefix, capturing focus versus wandering.

Two definitional details are load-bearing. **Dwell time** on an event is the interval to the
following event, and is defined only when that following event lies inside the same prefix;
the final event of a prefix contributes no dwell, since computing it would require an event
past the cut-point. **Cart-count features are excluded entirely**: because S3's cut-point is
the *first* cart event, its prefix contains exactly one cart event by construction, so cart
count, removal count and cart recency are constants on every dataset (A8). **Table 4** gives
feature availability by stage.

### 3.6 Splitting

Two protocols are used. The **temporal split** (headline) assigns each user to a period by
their first session and cuts on that boundary, so it is simultaneously ordered in time and
identity-clean. This is stricter than the protocol required: a purely date-based cut would
still allow a returning visitor to appear on both sides. Users whose activity straddles a
boundary are dropped and counted rather than reassigned. The **grouped split** (robustness)
partitions by user at random. A naive random split by session is never used.

For Dataset A neither protocol applies — there is no user identifier, and the only temporal
field is a month name — so sessions are partitioned on month boundaries chosen by exhaustive
search for the split closest to 70/15/15 (§4.7 reports the consequences).

### 3.7 Explanation protocol

**Layer 1 — stage-conditioned TreeSHAP.** Interventional TreeSHAP (Lundberg et al., 2020) is
computed per stage against a background sample drawn from training rows only. Correlated
features are clustered by absolute Spearman correlation and their attributions summed, since
Shapley values are additive (Aas et al., 2021). The clustering is fixed across stages and
taken from the earliest stage, which is the coarsest: at S1 a two-event prefix renders seven
engagement and temporal features numerically identical, so their individual attributions
there are arbitrary partitions of a single quantity. Attributions are computed on the
validation partition, not the test partition, which is reserved for predictive evaluation.

**Layer 2 — recurrent attribution.** A GRU is trained per stage on the *same prefixes* as the
tree models and explained with TimeSHAP (Bento et al., 2021). Training the recurrent model on
whole sessions would have let it see the purchase event, breaking the prefix protocol for
that arm alone and rendering any comparison meaningless.

**Layer 3 — explanation quality.** Faithfulness is measured by the Bhatt et al. (2020)
correlation and by deletion and insertion curves (Petsiuk et al., 2018), with removal
implemented as replacement by a background value rather than by zero — zero is a valid
in-distribution value for most of these features and does not remove information. Deletion
orders features by *signed* attribution, since the claim being tested is that removing
features which push the prediction up causes it to fall. Stability is a local-Lipschitz
estimate (Alvarez-Melis & Jaakkola, 2018). Consistency is Spearman rank correlation of
importance orderings across the five seeds and between the two paradigms. Pass thresholds
were fixed in advance: faithfulness correlation > 0.5, cross-paradigm Spearman > 0.6.

### 3.8 Modelling and evaluation

Five model families were evaluated on Dataset A — logistic regression, random forest (Breiman,
2001), XGBoost (Chen & Guestrin, 2016), LightGBM (Ke et al., 2017) and CatBoost
(Prokhorenkova et al., 2018) — and LightGBM was carried forward to the stage models.
Headline stage models use the libraries' default hyperparameters. A tuning study — Optuna
(Akiba et al., 2019) TPE search, 100 trials per stage over fixed search spaces, scored on a
held-out validation set rather than by nested cross-validation because the temporal split
forbids shuffling folds across the time boundary — is reported as a robustness analysis
(§4.7). It improves S3 and changes no conclusion; keeping defaults as the headline keeps the
explanation layers attached to the exact models they explain. Tuning and threshold selection
both touch the validation partition, a double use noted in §5.5. Class imbalance was handled by class
weighting inside training folds only; SMOTE (Chawla et al., 2002) was available but not used
in the reported runs.

**PR-AUC is the primary metric**, as appropriate under class imbalance (Davis & Goadrich,
2006; Saito & Rehmsmeier, 2015). Because chance-level PR-AUC equals the prevalence, and
prevalence differs by a factor of six across stages (Table 3), we report **PR-AUC lift** —
PR-AUC divided by prevalence — as the cross-stage comparison, alongside the prevalence-free
ROC-AUC. Reporting raw PR-AUC across stages would report prevalence as though it were skill.

Probability calibration follows Niculescu-Mizil and Caruana (2005), fitted isotonically on a
dedicated calibration split carved from the *training* period. Calibration quality is reported
as Brier score and Expected Calibration Error (Guo et al., 2017). Every metric is reported as
a mean over five frozen seeds; between-model differences carry paired bootstrap confidence
intervals over 2,000 stratified resamples. Paired ROC-AUC comparisons use the DeLong test
(DeLong et al., 1988) in the fast formulation of Sun and Xu (2014), vendored and unit-tested
against a naive transcription of the original definition. Multiple comparisons are corrected
by the Holm procedure (Holm, 1979) across the entire pre-declared family; Benjamini–Hochberg
(1995) is available as an alternative.

---

## 4. Results

### 4.1 Data flow and descriptive statistics

Cleaning and sessionisation reduced 42,448,764 raw events to 37,368,894 events in 5,366,181
sessions, from which the seeded 200,000-user sample retains 485,459 sessions (Figure 1).
Stage populations and prevalences are given in Table 3.

The stage structure itself carries a finding. Conversion prevalence is **not monotone** across
the funnel: 8.85% at S1, 6.83% at S2, 52.10% at S3. Consideration selects *against*
conversion, because visitors who cart early are, by the ordering rule, recorded as skipping
consideration, and those visitors convert at much higher rates.

### 4.2 Predictive performance (RQ1, H1)

Dataset A results are reported in **Table 5**, on the month-ordered split with December held
out, over five seeds.

| Model | PR-AUC | ROC-AUC | ECE (uncalibrated) | ECE (calibrated) |
|---|---|---|---|---|
| CatBoost | 0.699 ± 0.006 | 0.905 | 0.063 | 0.018 |
| Random forest | 0.685 ± 0.006 | 0.910 | 0.096 | 0.015 |
| LightGBM | 0.677 ± 0.008 | 0.899 | 0.065 | 0.026 |
| XGBoost | 0.676 ± 0.005 | 0.902 | 0.036 | 0.018 |
| Logistic regression | 0.583 ± 0.000 | 0.879 | 0.203 | 0.032 |

Logistic regression's zero standard deviation is correct rather than anomalous: with a fixed
split and a convex objective the fit is deterministic.

Dataset B stage models are reported in **Table 6** and **Figure 2**.

| Stage | N (test) | Prevalence | PR-AUC | **PR-AUC lift** | ROC-AUC | ECE |
|---|---|---|---|---|---|---|
| S1 | 38,059 | 0.073 | 0.1312 ± 0.0009 | **1.79** | 0.6406 | 0.010 |
| S2 | 17,884 | 0.059 | 0.0834 ± 0.0015 | **1.41** | 0.6198 | 0.011 |
| S3 | 2,973 | 0.521 | 0.5681 ± 0.0041 | **1.09** | 0.5695 | 0.017 |

**H1 is not supported; the direction of the effect is reversed.** H1 predicted that PR-AUC
would rise monotonically from awareness to intent. Raw PR-AUC does rise, from 0.131 to 0.568
— but chance-level PR-AUC rises with it, from 0.073 to 0.521. Normalised, the curve runs the
other way: lift falls monotonically from 1.79 to 1.09, and ROC-AUC, which is
prevalence-independent and therefore cannot be explained by base rates, falls with it from
0.641 to 0.570. Seed standard deviations of 0.0009–0.0041 are one to two orders of magnitude
smaller than the between-stage differences.

By the intent stage the model is barely above its own base rate. We interpret this in §5.1.

### 4.3 Feature-family ablation (H3)

**Table 7** reports the ablation, with paired bootstrap 95% confidence intervals over the
Holm-corrected family of nine contrasts.

| Stage | +temporal vs baseline | Entropy/velocity vs baseline (H3) | Entropy/velocity given temporal |
|---|---|---|---|
| S1 | **+0.0102** [+0.0045, +0.0167] | +0.0038 [−0.0002, +0.0084] | −0.0006 [−0.0017, +0.0005] |
| S2 | **+0.0120** [+0.0057, +0.0197] | +0.0035 [−0.0011, +0.0085] | +0.0007 [−0.0017, +0.0031] |
| S3 | +0.0139 [−0.0074, +0.0370] | +0.0156 [−0.0027, +0.0346] | +0.0018 [−0.0045, +0.0085] |

**H3 is not supported.** The interval for entropy and click-velocity features against
baseline aggregates covers zero at every stage. The temporal family, by contrast, contributes
reliably at S1 and S2.

The robust result here is the **redundancy**: given the temporal family, entropy and velocity
add nothing, and those are the only intervals narrow enough to distinguish "no effect" from
"cannot tell". This is mechanically unsurprising — click velocity is events divided by
elapsed prefix duration, so both of its constituents are temporal features.

No S3 contrast is decidable. With 2,973 test sessions every intent-stage interval spans
roughly ±0.02, admitting both a meaningful gain and a meaningful loss. This is a power
limitation, not a null result.

### 4.4 Attribution migration (RQ2, H2)

**Figure 3** presents the attribution trajectories; **Table 8** gives the underlying shares of
total mean |SHAP| by correlation group.

| Driver group | S1 | S2 | S3 | Δ (S1→S3) |
|---|---|---|---|---|
| Price (2 features) | **0.437** | 0.239 | 0.379 | −0.058 |
| Engagement and tempo (7) | 0.169 | **0.368** | 0.338 | **+0.169** |
| Products viewed (2) | 0.192 | 0.033 | 0.057 | **−0.136** |
| Navigation and category (6) | 0.035 | **0.224** | 0.074 | +0.039 |
| Hour of day | 0.157 | 0.125 | 0.133 | −0.024 |
| Weekend | 0.010 | 0.011 | 0.020 | +0.009 |

Both stage pairs are fully distinct — no session shares a cut-point between S1 and S2 or
between S2 and S3 (mean gaps 2.49 and 5.37 events) — so the trajectories compare genuinely
different prefixes.

**H2 is partially supported.** Its behavioural arm holds: attribution to engagement and tempo
doubles from 0.169 at awareness to 0.338 at intent, the largest single shift in the table. Its
context arm **cannot be tested on Dataset B**, because traffic source and device are absent
from the schema (Table 2). Price leads at S1 and is the nearest available product-context
proxy, but it is a product attribute rather than an acquisition channel and we do not present
it as one.

The more interesting result is one H2 did not anticipate. **Navigation entropy is strongly
non-monotone**, rising from 3.5% of attribution mass at awareness to 22.4% at consideration
before falling to 7.4% at intent. Exploratory breadth is a *stage-specific* signal: it matters
where the visitor is choosing among alternatives and ceases to matter once they have chosen.

### 4.5 Explanation quality (RQ3)

**Table 9** and **Figure 4** report Layer 3.

| Stage | Faithfulness ρ | vs 0.5 | Deletion AUC | Insertion AUC | Lipschitz (max / mean) | Seed ρ (mean / worst pair) | vs 0.6 |
|---|---|---|---|---|---|---|---|
| S1 | 0.571 ± 0.177 | **pass** | 0.2365 | 0.5039 | 0.048 / 0.015 | 1.000 / 1.000 | **pass** |
| S2 | 0.494 ± 0.179 | *fail* | 0.1440 | 0.3820 | 0.073 / 0.013 | 0.966 / 0.943 | **pass** |
| S3 | 0.565 ± 0.169 | **pass** | 0.3495 | 0.5898 | 0.028 / 0.010 | 0.943 / 0.886 | **pass** |

**Seed consistency passes decisively** at every stage, with a worst-pair correlation of 0.886
against a threshold of 0.6. This is what licenses Figure 3: the migration trajectory survives
reseeding.

**Deletion AUC is roughly half insertion AUC at every stage**, and removing top
positively-attributed features degrades the prediction monotonically. The explanations point
at features the model genuinely uses.

**Faithfulness is stage-dependent, and S2 misses the pre-registered threshold** (0.494 against
0.5). We report this as a failure against the stated bar rather than rounding it up.

**Stability passes at every stage.** The local-Lipschitz estimate (25 instances per stage,
Gaussian input noise of sd 0.05) bounds the attribution change at a maximum ratio of 0.073
(S2), with means near 0.01: perturbing the input moves the explanation by at most a small
fraction of the perturbation itself. Notably, S2 — weakest on faithfulness — is not unstable;
its explanations are consistent and robust, they are just less tethered to the model's
actual sensitivity than at the other stages.

### 4.6 Cross-paradigm agreement (H4)

The GRU is competitive with the tree models — test PR-AUC 0.124 against 0.131 at S1 and 0.565
against 0.568 at S3 — so any disagreement below is not attributable to one incompetent arm.

At the level of **aggregate feature rankings**, Spearman correlations between TimeSHAP and
TreeSHAP are +0.200 (S1), +0.500 (S2) and +0.600 (S3), against a pre-registered threshold of
0.6. **H4 is not supported.** However, this test is severely underpowered: the two paradigms
share only four or five mappable concepts, and with four items Spearman can take only a
handful of discrete values. The pre-registration did not anticipate how few concepts the
schemas would share.

We therefore also computed agreement **per instance**: for each concept, the tree and
sequence attributions were correlated across 350+ shared sessions, over five seeds.

| Stage | Aggregate ρ (5 concepts) | **Per-instance mean ρ (5 seeds)** | Concepts sign-stable |
|---|---|---|---|
| S1 | +0.200 | +0.179 | 1 of 4 |
| S2 | +0.500 | −0.018 | 0 of 5 |
| S3 | +0.600 | +0.046 | 2 of 5 |

Per-instance agreement is **indistinguishable from zero at every stage**, and only 3 of 14
(stage, concept) pairs hold a consistent sign across seeds. **Aggregate convergent validity
does not imply agreement about individual cases.** The same models produce aggregate
correlations up to +0.600 and per-instance agreement of approximately zero.

We note explicitly that a single-seed version of this analysis produced an apparent −0.151 at
S2 that did **not** survive reseeding (Appendix A, A23b). With seed standard deviations of
0.2–0.4 on these correlations, sign stability rather than the mean is the reportable quantity.

### 4.7 Sensitivity analyses

**Sessionisation gap.** Halving or doubling the 30-minute inactivity threshold changes the
session count by ±3%, stage reach by under one percentage point and stage prevalence by well
under half a point — all far smaller than the between-stage differences on which the paper's
claims rest.

| Gap | Sessions | S1 reach / prev | S2 reach / prev | S3 reach / prev |
|---|---|---|---|---|
| 15 min | 5,544,851 | 0.978 / 0.0864 | 0.471 / 0.0652 | 0.092 / 0.5160 |
| 30 min | 5,366,181 | 0.979 / 0.0885 | 0.480 / 0.0683 | 0.093 / 0.5210 |
| 60 min | 5,219,040 | 0.979 / 0.0897 | 0.487 / 0.0705 | 0.093 / 0.5248 |

**Sample size.** Doubling the subsample to 400,000 users changes PR-AUC by +0.003 to +0.009
and leaves every conclusion intact: lift still falls monotonically (1.84, 1.49, 1.10) and
ROC-AUC with it (0.648, 0.634, 0.570). Because the sample is nested at a fixed seed, the
400,000-user set is a strict superset of the 200,000-user set, so these differences are
attributable to the added users rather than to a different draw.

**Hyperparameter tuning.** Refitting the stage models with Optuna-selected hyperparameters
(§3.8) moves PR-AUC by +0.002 (S1) and +0.003 (S2) — within one to two seed standard
deviations — and by +0.032 at S3 (0.568 → 0.600, roughly eight seed standard deviations).
The lift ordering is unchanged: 1.81, 1.46, 1.15 tuned against 1.79, 1.41, 1.09 at defaults,
still monotonically falling. All three searches selected the minimum tree count in the space
(200) with strong regularisation — shallow trees at S1 and S3, a slow learning rate at S2 —
consistent with a drift-prone temporal split rewarding conservative models. That the largest
gain lands at S3 fits the same reading: the underpowered stage benefits most from
regularisation, and even tuned it remains the weakest stage.

**Split protocol.** On the grouped split, which partitions users at random and is
identity-clean but not time-ordered, lift falls 1.98 → 1.53 → 1.15. The monotone decline on
which H1's rejection rests is therefore not an artefact of the temporal protocol. Grouped
performance sits above temporal at every stage, which is what temporal drift
predicts: a model tested on its own period's peers faces an easier problem than one tested
on the future.

**Calibration set.** Fitting the calibrator on the validation month rather than on a slice of
the training period raises Expected Calibration Error from 0.025 to 0.118 (**Figure 5**), for
reasons developed in §5.4.

---

## 5. Discussion

### 5.1 Interpreting the reversed improvement curve

The clearest result of the study contradicts its own pre-registered hypothesis: normalised
predictive performance *falls* across the funnel. We offer a substantive rather than a
technical interpretation.

Once a visitor has placed an item in a cart, they convert 52% of the time. What separates the
converters from the abandoners at that point is largely **not observable in a clickstream**:
checkout friction, payment failure, unexpected delivery cost, external interruption. Browsing
behaviour discriminates well early, when it is the only thing that differs between visitors,
and poorly late, when the population has become behaviourally homogeneous and the decisive
factors have moved off-stream. Supporting this reading, S1 achieves the highest lift with the
*fewest* features (19 against 23), so the decline is not a feature-availability artefact.

This has a direct implication for practice, developed in §5.3: the window in which
behavioural data supports intervention is **early and mid funnel**, not at the cart.

### 5.2 Attribution migration and journey theory

The behavioural arm of H2 is supported, and this is consistent with a staged view of the
consumer journey in which context gives way to demonstrated behaviour as intent forms (Lemon
& Verhoef, 2016; Moe, 2003).

The result that H2 did not predict is more interesting. Navigation entropy peaks at
consideration and collapses at intent. Journey theory tends to treat exploration as decaying
steadily as intent forms; these data suggest instead that exploratory breadth is a
*stage-specific diagnostic* — informative exactly where the visitor is choosing among
alternatives, uninformative once the choice is made. A monotone framing of the funnel cannot
express that pattern, which is an argument for stage conditioning as a modelling commitment
rather than a reporting convenience.

### 5.3 Implications for practice

Two implications follow, and they point in the same direction.

First, **intervention should be concentrated early**. If late-stage prediction is close to
chance relative to its base rate (lift 1.09 at S3), then models cannot usefully discriminate
among cart-holders, and effort spent scoring them is largely wasted. This runs against the
concentration of cart-abandonment tooling in the industry.

Second, **the actionable signal is stage-specific**. Navigation entropy is the clearest case:
a whole-session analysis reports a share of 0.111 and would rank it a minor driver, where it
in fact reaches 0.224 at consideration — a value the static figure never takes at any stage
(**Table 10**). An intervention targeting wandering behaviour has purchase during
consideration and close to none at cart.

We note the necessary precondition: acting on predicted probabilities requires them to be
calibrated, which §5.4 shows is easy to get wrong.

### 5.4 Methodological implications

Three findings hold independently of the framework and may transfer more readily than the
framework itself.

**A canonical benchmark drifts, and is usually split randomly.** Conversion prevalence in
Dataset A rises almost monotonically from 1.6% in February to 25.4% in November before halving
to 12.5% in December — a fifteen-fold range. Random splitting mixes these months and conceals
the drift. Our month-ordered results are consequently lower than those commonly reported on
this benchmark, and we regard that as a correction rather than an underperformance.

**Prior-probability shift silently destroys calibration, and the remedy is the calibration
set, not a post-hoc correction.** Fitting the calibrator on November (25.4% prevalence) and
applying it to December (12.5%) produced a mean predicted probability of 0.2421 against an
observed rate of 0.1251 — a ratio of 1.935, almost exactly the prevalence ratio of 2.03 — with
over-prediction in every reliability bin (Figure 5). Discrimination was unaffected, since
ranking is invariant to monotone miscalibration. Notably, applying the EM prior-shift
correction of Saerens et al. (2002) *on top of* a correctly drawn calibration set made
calibration **worse** for every model, since it estimated a prior of 0.1447 against a true
0.1251 and over-corrected a shift that was no longer present. The instinct is to reach for the
correction; the answer is to fix the split.

**Aggregate convergent validity does not imply per-instance agreement.** Two attribution
paradigms produced aggregate rank correlations up to +0.600 on the same models while agreeing
approximately not at all about which individual journeys each driver mattered for (§4.6). Any
study claiming convergent validity between explanation methods should report per-instance
agreement, since aggregate agreement is both easier to achieve and less relevant to a
practitioner reasoning about a specific case.

### 5.5 Limitations and threats to validity

**Absent variables.** Search-query text is unavailable in both datasets and scroll depth is
proxied by dwell time (Table 2). Device and traffic source are absent from Dataset B, so H2's
context arm is untestable there.

**Dataset A is non-causal by construction.** Its features are whole-session aggregates and
`PageValues` is measured post-transaction. It functions as a benchmark and as the static
comparator, not as a causal model.

**Stage models are conditional on reaching the stage.** The improvement curve is a conditional
claim, and Table 3 reports the reach rates and prevalences needed to read it as such.

**The intent stage is underpowered.** With 2,973 test sessions, no S3 ablation contrast is
decidable. This is a sample-size limitation and argues for a wider observation window.

**The temporal split discards data.** Keeping visitors whole *and* respecting time order costs
51,256 of 200,000 users as boundary-straddlers, retaining 222,268 of 485,459 sessions. A
window longer than one month would reduce this fraction substantially.

**The recurrent arm is small.** The GRU is trained briefly on a subsample, and its
attributions carry seed standard deviations of 0.2–0.4. Conclusions from it are correspondingly
weak, and one single-seed conclusion had to be withdrawn (A23b).

**The validation partition is used twice.** Hyperparameter tuning selects among models on
validation PR-AUC, and the operating threshold is subsequently chosen on the same partition.
The interaction is weak — tuning optimises a threshold-free metric and thresholding is a
downstream choice on the already-selected model — but it is a double use, and a stricter
design would reserve a separate partition for each.

**External validity.** One retailer, one market, one month. The framework is portable; these
specific attribution trajectories are not claimed to be.

### 5.6 Future work

Priorities, in order: a multi-month observation window, which addresses both the S3 power
limitation and the temporal-split attrition; prevalence-corrected calibration for deployment
under seasonal drift; a larger sequence model to test whether the per-instance disagreement
is a property of the paradigms or of an under-trained GRU; and live A/B validation of the
early-funnel intervention implied by §5.3.

---

## 6. Conclusion

We proposed a funnel-stage SHAP framework that conditions Shapley attributions on journey
stage, computes them strictly on the event prefix available at each stage, and validates them
for faithfulness, stability and reproducibility. Applied to 42.4 million clickstream events
and a canonical session-level benchmark, it produced three results that a whole-session
analysis could not have produced.

Normalised predictive performance falls rather than rises across the funnel, so conversion
becomes harder to predict as intent forms, and the useful intervention window is earlier than
industry practice assumes. Attribution is strongly non-monotone, with navigation entropy
mattering at consideration and nowhere else — a pattern any single whole-session attribution
averages away. And convergent validity between explanation paradigms, which holds at the level
of aggregate rankings, disappears at the level of individual journeys.

Three of the four pre-registered hypotheses were not supported. We report them as such. The
practical recommendation for researchers applying SHAP to consumer-journey data is narrower
than the framework itself: **validate explanations per stage rather than per model**, because
explanation quality tracks predictive signal, and a stage-averaged quality metric conceals
exactly the stages where the explanation should not be trusted.

---

## Declarations

**CRediT author statement.** Ayoub Agnaou: Conceptualization, Methodology, Software, Formal
analysis, Data curation, Writing – original draft, Writing – review & editing, Visualization.

**Declaration of competing interest.** [To complete.]

**Declaration of generative AI in the writing process.** [To complete — and to complete
accurately. Generative AI was used substantially in implementing the analysis code and in
drafting this manuscript. State this in the form the target journal specifies; do not
understate it.]

**Data availability.** Dataset A is publicly available from the UCI Machine Learning
Repository under CC BY 4.0 (Sakar et al., 2019). Dataset B is published openly by the REES46
Marketing Platform. **No formal licence text is published for Dataset B**; the files are
distributed without a click-through licence and are widely used in published work, but a
specific grant of rights for research publication could not be located. This position is
stated here rather than a licence asserted. [Obtain written confirmation from REES46 before
submission and replace this paragraph.]

**Code availability.** The complete analysis pipeline, frozen data splits, seed list,
environment lockfile and amendment log are available at [repository URL].

**Ethics.** Behavioural data only; no personally identifying information. Identifiers were
pre-hashed by the publisher and no re-identification was attempted. Explanations are reported
at cohort level.

**Funding.** [To complete.]

---

## Appendices

- **Appendix A** — Pre-registered protocol (v1.0) and the dated amendment log (26 entries).
- **Appendix B** — Full feature dictionary.
- **Appendix C** — Sessionisation gap and sample-size sensitivity analyses.
- **Appendix D** — Hyperparameter search spaces and selected values.
- **Appendix E** — Seed list and compute budget.

---

## References

Aas, K., Jullum, M., & Løland, A. (2021). Explaining individual predictions when features are
dependent: More accurate approximations to Shapley values. *Artificial Intelligence, 298*,
103502.

Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A next-generation
hyperparameter optimization framework. In *Proceedings of the 25th ACM SIGKDD International
Conference on Knowledge Discovery & Data Mining* (pp. 2623–2631).

Alvarez-Melis, D., & Jaakkola, T. S. (2018). On the robustness of interpretability methods.
*arXiv preprint arXiv:1806.08049*.

Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: A practical and
powerful approach to multiple testing. *Journal of the Royal Statistical Society: Series B,
57*(1), 289–300.

Bento, J., Saleiro, P., Cruz, A. F., Figueiredo, M. A. T., & Bizarro, P. (2021). TimeSHAP:
Explaining recurrent models through sequence perturbations. In *Proceedings of the 27th ACM
SIGKDD Conference on Knowledge Discovery & Data Mining* (pp. 2565–2573).

Bhatt, U., Weller, A., & Moura, J. M. F. (2020). Evaluating and aggregating feature-based
model explanations. In *Proceedings of the 29th International Joint Conference on Artificial
Intelligence* (pp. 3016–3022).

Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5–32.

Bucklin, R. E., & Sismeiro, C. (2003). A model of web site browsing behavior estimated on
clickstream data. *Journal of Marketing Research, 40*(3), 249–267.

Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic
minority over-sampling technique. *Journal of Artificial Intelligence Research, 16*, 321–357.

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In *Proceedings of
the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp.
785–794).

Davis, J., & Goadrich, M. (2006). The relationship between precision-recall and ROC curves. In
*Proceedings of the 23rd International Conference on Machine Learning* (pp. 233–240).

DeLong, E. R., DeLong, D. M., & Clarke-Pearson, D. L. (1988). Comparing the areas under two or
more correlated receiver operating characteristic curves: A nonparametric approach.
*Biometrics, 44*(3), 837–845.

Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural
networks. In *Proceedings of the 34th International Conference on Machine Learning* (pp.
1321–1330).

Hedström, A., Weber, L., Krakowczyk, D., Bareeva, D., Motzkus, F., Samek, W., Lapuschkin, S.,
& Höhne, M. M.-C. (2023). Quantus: An explainable AI toolkit for responsible evaluation of
neural network explanations and beyond. *Journal of Machine Learning Research, 24*(34), 1–11.

Hevner, A. R., March, S. T., Park, J., & Ram, S. (2004). Design science in information systems
research. *MIS Quarterly, 28*(1), 75–105.

Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian
Journal of Statistics, 6*(2), 65–70.

Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017).
LightGBM: A highly efficient gradient boosting decision tree. In *Advances in Neural
Information Processing Systems 30* (pp. 3146–3154).

Lemon, K. N., & Verhoef, P. C. (2016). Understanding customer experience throughout the
customer journey. *Journal of Marketing, 80*(6), 69–96.

Lundberg, S. M., Erion, G., Chen, H., DeGrave, A., Prutkin, J. M., Nair, B., Katz, R.,
Himmelfarb, J., Bansal, N., & Lee, S.-I. (2020). From local explanations to global
understanding with explainable AI for trees. *Nature Machine Intelligence, 2*(1), 56–67.

Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. In
*Advances in Neural Information Processing Systems 30* (pp. 4765–4774).

Moe, W. W. (2003). Buying, searching, or browsing: Differentiating between online shoppers
using in-store navigational clickstream. *Journal of Consumer Psychology, 13*(1–2), 29–39.

Montgomery, A. L., Li, S., Srinivasan, K., & Liechty, J. C. (2004). Modeling online browsing
and path analysis using clickstream data. *Marketing Science, 23*(4), 579–595.

Nayebi, A., Tipirneni, S., Reddy, C. K., Foreman, B., & Subbian, V. (2023). WindowSHAP: An
efficient framework for explaining time-series classifiers based on Shapley values. *Journal
of Biomedical Informatics, 144*, 104438.

Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good probabilities with supervised
learning. In *Proceedings of the 22nd International Conference on Machine Learning* (pp.
625–632).

Petsiuk, V., Das, A., & Saenko, K. (2018). RISE: Randomized input sampling for explanation of
black-box models. In *Proceedings of the British Machine Vision Conference*.

Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A. (2018). CatBoost:
Unbiased boosting with categorical features. In *Advances in Neural Information Processing
Systems 31* (pp. 6638–6648).

Rudin, C. (2019). Stop explaining black box machine learning models for high stakes decisions
and use interpretable models instead. *Nature Machine Intelligence, 1*(5), 206–215.

Saerens, M., Latinne, P., & Decaestecker, C. (2002). Adjusting the outputs of a classifier to
new a priori probabilities: A simple procedure. *Neural Computation, 14*(1), 21–41.

Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC
plot when evaluating binary classifiers on imbalanced datasets. *PLOS ONE, 10*(3), e0118432.

Sakar, C. O., Polat, S. O., Katircioglu, M., & Kastro, Y. (2019). Real-time prediction of
online shoppers' purchasing intention using multilayer perceptron and LSTM recurrent neural
networks. *Neural Computing and Applications, 31*(10), 6893–6908.

Sismeiro, C., & Bucklin, R. E. (2004). Modeling purchase behavior at an e-commerce web site: A
task-completion approach. *Journal of Marketing Research, 41*(3), 306–323.

Slack, D., Hilgard, S., Jia, E., Singh, S., & Lakkaraju, H. (2020). Fooling LIME and SHAP:
Adversarial attacks on post hoc explanation methods. In *Proceedings of the AAAI/ACM
Conference on AI, Ethics, and Society* (pp. 180–186).

Sun, X., & Xu, W. (2014). Fast implementation of DeLong's algorithm for comparing the areas
under correlated receiver operating characteristic curves. *IEEE Signal Processing Letters,
21*(11), 1389–1393.

Van den Poel, D., & Buckinx, W. (2005). Predicting online-purchasing behaviour. *European
Journal of Operational Research, 166*(2), 557–575.
