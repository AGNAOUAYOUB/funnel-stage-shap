# Amendment log

Protocol v1.0 states: "Any deviation after freeze is logged in a dated amendment appendix."
Each entry records the date, the affected protocol sections, what changed, and why.

The protocol is **not yet frozen**: the environment is pinned and the data/feature layer is
built and tested, but no split has been written to `data/processed/splits/` and the test
partition has never been read. Entries below are pre-freeze decisions and open questions,
recorded so the eventual freeze is auditable.

---

## 2026-08-04 — Pre-freeze implementation decisions

### A1. S4 admits no feature matrix (Sec. 7.4, 9.2)

**Decision.** `S4` is defined for descriptive journey statistics but raises an error if a
feature matrix or model is requested for it. Modelling stages are `{S1, S2, S3}`.

**Why.** S4's cut-point is the purchase event, which *is* the label. Any S4 prefix would
contain the outcome. Sec. 9.2 already implies this by naming `{S1, S2, S3}` as the set that
yields the improvement curve; this makes it enforced rather than conventional.

### A2. Stage models are conditional on reaching the stage (Sec. 9.2, 10, H1)

**Decision.** The Sk model is fitted on sessions that reached Sk. `stage_prevalence_table`
reports N, reach rate and prevalence per stage, and the paper must present the H1 curve
alongside them.

**Why.** This follows necessarily from "features use only the prefix up to Sk's cut-point"
— a session with no cart event has no S3 prefix. But it means the H1 improvement curve is
partly a **composition effect**: the S3 population is small and heavily selected toward
converters. On the synthetic fixture, S1 prevalence is 0.13 against S3's 0.42 at a 32% reach
rate; on real data the gap will be at least as large. Reporting PR-AUC gains across stages
without this caveat would overstate H1. The curve is a *conditional* claim: given a visitor
has reached this stage, how well can we predict conversion.

**Open question for the analysis phase.** A marginal complement — evaluating every stage
model on the full session population, scoring non-reachers at the last stage they did reach
— would make the stages directly comparable. This is not in protocol v1.0. Recommend adding
it as a secondary analysis rather than substituting it for the pre-registered one.

### A3. Dwell excludes the final prefix event (Sec. 8)

**Decision.** Dwell on an event is the gap to the next event, defined only when that next
event is inside the same prefix. The last prefix event contributes no dwell.

**Why.** The obvious implementation — dwell = gap to next event, full stop — reads one
event past the cut-point for the final event of every prefix, which is exactly the leak
Sec. 7.4 forbids. The chosen definition is slightly conservative, which is the correct
direction for a leakage-sensitive quantity.

### A4. Session IDs are re-derived rather than taken from `user_session` (Sec. 7.2)

**Decision.** Sessions are cut from `user_id` + `event_time` on the configured inactivity
gap. The vendor's `user_session` column is retained only to compute an agreement statistic.

**Why.** Sec. 7.2 requires a 15/30/60-minute sensitivity check. That is only meaningful if
the gap parameter actually determines the sessions; trusting the vendor's column would
apply *its* undocumented timeout at every setting and make the sensitivity appendix vacuous.
The agreement statistic quantifies how far the choice reshapes the unit of analysis.

### A5. Dataset A provenance is a local copy (Sec. 5.2, 6.2)

**Decision.** `data/raw/online_shoppers_intention.csv` was copied from an existing working
directory on the research machine, not fetched from UCI. SHA-256 and row/prevalence checks
are recorded in `provenance_A.json`; the file matches the protocol's expected 12,330 rows
and ~15% prevalence exactly.

**Why.** The file was already present, so no download was needed. **Action before
submission:** re-fetch from the canonical UCI endpoint via `funnel-shap fetch-a` and confirm
the hash, so provenance does not rest on a working copy of unknown history.

### A6. RESOLVED — S1's cut-point moves to the second product interaction (Sec. 7.3)

**Decision (2026-08-04, author).** S1 closes at the *second* product interaction rather than
the first view. Sec. 7.3's S1 row is amended to: "session entry → initial catalogue contact;
cut-point = second product interaction".

**Effect, measured.** Constant S1 features fell from **19 of 23 to 4 of 23**. The minimum S1
prefix is now two events, which is what makes dwell, inter-event gap and category entropy
measurable at all. The four that remain constant (`n_events`, `n_views`,
`inter_event_std_s`, `purchase_intent_score`) are constant by construction of a two-event
prefix; A8 subsumes them.

**Original problem, retained for the record.** Sec. 7.3 defined S1 as "session entry → first
product interaction" with cut-point "first view". Sessions in an event log almost always
*open* with a view, so `cut_S1 = 0` and the S1 prefix was a single event: mean S1 prefix
length exactly 1.0, with 19 of 23 features constant, because a one-event prefix has no
second event to measure against. Only `hour_of_day`, `is_weekend`, `price_mean` and
`price_max` varied. Consequences had it stood: RQ1/H1 would have compared a near-null S1
model against genuine S2/S3 models, so the improvement curve would have measured S1's lack
of features rather than the journey becoming more predictable; and RQ2/H2 — "attribution
mass migrates from context to behaviour" — would have been circular at S1, since context
features were the only non-constant ones there.

Candidates considered: (1) cut at the second product interaction — **chosen**; (2) define S1
as a time or event window; (3) drop S1 from the modelling set and report the curve over
S2/S3 only.

### A8. RESOLVED — degenerate stage features and S1/S2 collapse (Sec. 7.3, 7.4, 8)

**Decision (2026-08-04, author approved fixing the issue; the mechanism below differs
from what was originally proposed — see "Why the proposed fix was rejected").**

Two changes, both keeping cut-points at the *first* occurrence of a trigger:

1. **S2 opens on the second browsing signal**, not the first. Sec. 7.3's S2 row is amended
   to "product/category browsing; cut-point = second repeat view or category switch".
2. **The three cart-count features are removed from the dictionary entirely**
   (`n_cart_adds`, `n_cart_removes`, `time_since_last_cart_s`), along with four features
   that a fixed two-interaction S1 prefix cannot vary (`n_events`, `n_views`,
   `inter_event_std_s`, `purchase_intent_score` at S1 only).

**Effect, measured on the fixture.**

| | before A8 | after A8 |
|---|---|---|
| Constant features at S1 | 4 of 23 | **0 of 19** |
| Constant features at S3 | 3 of 26 | **0 of 23** |
| S1→S2 sessions with identical cut-point | **82.8%** | **0%** |
| S1→S2 mean extra events | 0.21 | 1.41 |
| S2→S3 mean extra events | 5.48 | 4.87 |

S2's reach rate falls from 97.7% to 84.8%, which is the intended consequence: sessions
whose only browsing signal was incidental no longer count as having reached consideration.

**Why the proposed fix was rejected.** The recommendation was to let each stage's prefix run
*through* the stage, ending just before the next stage's trigger. Implementing it showed
that it introduces an outcome-dependent truncation that is worse than the problem it solves.
Under that scheme a prefix ends either at the next stage's trigger *or*, for sessions that
end mid-stage, at session end. Whether truncation happens is then a function of the outcome:
at S3, a purchasing session's prefix stops at the purchase while a non-purchasing session's
runs to session end, so purchasers systematically get *shorter* prefixes. `n_events` and
every duration feature would become predictive through the truncation rule rather than
through behaviour — and would collect large SHAP mass at exactly the stage where the paper
claims cart-proximity features dominate (H2). That is a spurious finding waiting to happen.

Cutting at a stage's *opening* is the only future-independent choice available, so it is
kept. The narrow real defects — degenerate features and colliding triggers — are fixed
directly instead.

**Residual issue for the Sec. 11.1 explanation layer.** At S1 the two-event prefix makes
`prefix_duration_s`, `inter_event_mean_s`, `last_gap_s` and `dwell_total_s` numerically
identical, with `click_velocity` and `events_per_active_minute` deterministic functions of
them. They are not constant, so they pass the guard, but they are perfectly collinear.
Sec. 11.1 already requires clustering correlated features and reporting grouped
attributions; S1 is the stage where that requirement binds hardest, and the migration figure
must use the grouped form there or the attribution will split arbitrarily across six
redundant columns.

**Original problem, retained for the record:**

Sec. 7.4 says features use "the prefix of events up to Sk's cut-point", and Sec. 7.3 sets
each cut-point at the *first* occurrence of the stage's trigger. Combining the two means a
stage's prefix ends exactly when that stage opens — so the stage's own behaviour is excluded
from its own feature vector. Two measured consequences:

**S3's cart features are constant by construction.** S3 cuts at the first cart event, so the
prefix contains exactly one cart event, always. `n_cart_adds` = 1, `n_cart_removes` = 0 (a
removal cannot precede the first add), `time_since_last_cart_s` = 0. All three features that
make S3 the *intent* stage carry zero information. This is structural, not a fixture
artefact — it will hold identically on REES46.

**S1 and S2 are nearly the same stage.** With S1 at the second interaction and S2 at the
first repeat view or category switch, **82.8%** of sessions reaching both have an identical
cut-point (mean 0.21 extra events between them). For most sessions the S1 and S2 models are
the same model fitted on the same rows, so the S1→S2 leg of the attribution-migration
trajectory — the paper's centrepiece figure — is trivially empty. S2→S3 is by contrast
cleanly distinct (0% identical, mean 5.5 extra events).

`stage_distinctness_table` computes both figures and must be reported alongside the
migration figure: a migration trajectory is only interpretable between stages that are
actually distinct.

The resolution originally proposed here — prefixes running *through* each stage, ending
just before the next stage's trigger — was implemented, measured, and rejected for the
reason given above. The adopted fix keeps cut-points at stage openings.

### A10. Dataset B sourced from REES46 directly, not Kaggle (Sec. 5.3)

**Decision.** `data/raw/rees46/2019-Oct.csv.gz` was downloaded from
`https://data.rees46.com/datasets/marketplace/2019-Oct.csv.gz`, REES46's own open endpoint,
rather than the Kaggle mirror Sec. 5.3 names first. October 2019, one month, per Sec. 5.3's
allowance to "subsample a fixed window/month for tractability". SHA-256 recorded in
`DATASETS.md` and `provenance_B.json`.

**Why.** Sec. 5.3 permits "an equivalent open event log". The Kaggle copy requires API
credentials that do not exist on this machine; REES46 serves the identical files without
authentication. Taking them from the originating publisher is a stronger provenance chain
than a third-party mirror, not a weaker one.

**Unresolved: the licence.** Sec. 5.3 requires confirming the licence permits research
publication. It could not be confirmed. Kaggle's metadata field says "Data files © Original
Authors" — a reservation of rights, not a grant — while REES46 publishes the files as "free
datasets ... for your neural network" and links an IEEE paper built on them. No formal
licence text exists anywhere I could find. Written confirmation from REES46 should be
obtained and recorded before submission; see the licence finding in `DATASETS.md`.

### A11. The temporal split is grouped by user as well (Sec. 7.6)

**Decision.** The temporal split assigns each *user* to a period by their first session and
cuts on that, rather than cutting sessions on a date alone. Users whose activity straddles a
boundary are dropped and counted in the split report.

**Why.** Sec. 7.6 presents temporal and grouped as alternative protocols, with temporal as
the headline. But a purely date-based cut still lets a returning visitor appear on both
sides of the boundary — exactly the identity leakage the grouped protocol exists to prevent.
Rather than pick which leak to accept in the headline result, the temporal split is both:
ordered in time *and* identity-clean. The grouped split remains as the separate robustness
arm Sec. 7.6 asks for. The cost is the dropped straddling band, which is reported rather
than hidden.

### A12. Dataset A is split by month, and it drifts hard (Sec. 5.2, 7.6, 9.6, 10)

**Decision.** Dataset A uses a month-ordered split: train = Feb–Oct (7,605), val = Nov
(2,998), test = Dec (1,727). Boundaries are chosen by exhaustive search over the ten months
present for the pair whose cumulative shares come closest to 70/15/15.

**Why not the Sec. 7.6 protocols.** Neither transfers. Dataset A has no `user_id`, so a
grouped split is undefined, and no timestamp finer than a month *name* (no year, no day, no
clock), so a true temporal cut is impossible. Sec. 7.6 nonetheless forbids a naive random
split, correctly: sessions in the same month share promotions, stock and seasonality.

**The achievable split is not 70/15/15.** Month sizes are wildly uneven — May (3,364) and
November (2,998) are 52% of the file between them, while February is 184 sessions. The
closest month-boundary split is 61.7 / 24.3 / 14.0. Row-count boundaries were tried first
and produced an *empty validation partition*, because both the 70% and 85% marks land
inside November.

**The finding that matters: conversion prevalence drifts by 15x across the file.**

| Feb | Mar | May | June | Jul | Aug | Sep | Oct | Nov | Dec |
|---|---|---|---|---|---|---|---|---|---|
| 1.6% | 10.1% | 10.9% | 10.1% | 15.3% | 17.6% | 19.2% | 20.9% | 25.4% | 12.5% |

Prevalence rises almost monotonically from February to November, then halves in December.
Two consequences:

1. **Most published results on this benchmark use a random split**, which mixes these months
   and hides the drift. Reporting a month-ordered result alongside is a genuine, cheap
   contribution — and it means our Dataset A numbers will look *worse* than the literature's
   for a good reason that must be stated plainly, not buried.
2. **Calibration and threshold selection are compromised as currently specified.** Sec. 9.6
   calibrates on a held-out split and Sec. 10 fixes the threshold before test. Doing either
   on November (25.4% prevalence) and then evaluating on December (12.5%) would
   systematically over-predict — the calibration curve would be fitted to roughly double the
   base rate it is applied to.

**Open question for the modelling phase.** Options: (a) accept it and report the prevalence
shift with prevalence-corrected calibration; (b) move the boundary so validation is more
representative, at the cost of a smaller or oddly-shaped test period; (c) calibrate on a
month-stratified subsample of the training period rather than on the validation month.
Recommend (a) plus reporting the shift explicitly — it is an honest property of the data,
and hiding it by re-cutting until the partitions match would be exactly the kind of
post-hoc tuning the protocol freeze exists to prevent.

**CONFIRMED EMPIRICALLY (run-sheet step 7).** The predicted harm is real and large. Across
all five baselines at five seeds, test-set ECE is 0.105–0.117 — an order of magnitude worse
than a well-calibrated model. For LightGBM at seed 7 on the December test month:

- mean predicted probability **0.2421** against an actual prevalence of **0.1251**
- ratio **1.935**, almost exactly the Nov/Dec prevalence ratio of 0.2535 / 0.1251 = **2.03**
- the model **over-predicts in every single reliability bin**, not just on average

This is textbook prior-probability shift: the isotonic calibrator was fitted on November's
25.4% base rate and applied to December's 12.5%. The discrimination metrics are unaffected
(PR-AUC 0.710 ± 0.008 for LightGBM, ROC-AUC 0.899), because ranking is invariant to a
monotone miscalibration — but every probability the model emits is roughly double the truth,
which makes the Sec. 10 threshold and any RQ4 cost-based intervention argument unusable as
they stand. Option (a) is no longer a preference; some prevalence correction is required.

**Dataset A baseline results (test = December, month-ordered split, 5 seeds):**

| Model | PR-AUC | ROC-AUC | ECE |
|---|---|---|---|
| LightGBM | 0.710 ± 0.008 | 0.899 | 0.117 |
| XGBoost | 0.707 ± 0.008 | 0.902 | 0.116 |
| CatBoost | 0.699 ± 0.014 | 0.905 | 0.115 |
| Random Forest | 0.694 ± 0.011 | 0.910 | 0.105 |
| Logistic Regression | 0.575 ± 0.000 | 0.879 | 0.102 |

Logistic regression's zero standard deviation is correct, not a bug: with a fixed split and
a convex objective it is deterministic, so the seed changes nothing.

### A13. LightGBM must be imported before scikit-learn on Windows (Sec. 4, 6.2)

**Decision.** `funnel_shap/__init__.py` imports LightGBM eagerly on Windows, before anything
can pull in scikit-learn.

**Why.** scikit-learn ships its own `sklearn/.libs/vcomp140.dll` for machines lacking the
Visual C++ redistributable and loads it by absolute path. LightGBM's `lib_lightgbm.dll`
links against the system `vcomp140.dll`. When scikit-learn wins the race, both OpenMP
runtimes sit in the process and the first LightGBM `fit` dies with

    OSError: exception: access violation reading 0x0000000000000000

inside `LGBM_DatasetSetField` — a native crash with no Python-level cause. It reproduces
after merely *using* scikit-learn, or importing XGBoost, CatBoost, SHAP or imbalanced-learn
(all of which import scikit-learn). Preloading `vcomp140.dll` by name via ctypes does not
help: the absolute-path load still creates a second module. Import order is the fix.

Worth recording because it is invisible: it does not surface until a LightGBM baseline is
fitted, and on a different machine — one where the redistributable placement differs — it
may not surface at all, which makes it exactly the kind of environment-dependent failure
Sec. 6.2's reproducibility requirements exist to pin down.

### A14. `stage_cutpoints` computed cut-points as window expressions (Sec. 7.3)

**Not a protocol change — a performance defect and a latent leak, both now fixed.**

The first implementation computed each stage's cut-point with
`pl.when(...).min().over(session)` and then took `first()` per group. That is five
group-and-broadcast passes over the event frame *before* the aggregation. On the synthetic
fixture it was imperceptible; on Dataset B's 37.4M events and 5.37M sessions it burned 2.6
CPU-hours without finishing. Folding the five into a single `group_by().agg()` makes it
complete, with identical semantics — the existing journey tests pass unchanged, which is
what licenses the rewrite.

**The latent leak.** Rewriting the within-session index in terms of `cum_count()` introduced
an unsigned integer: `cum_count` returns `UInt32`, so for a session whose *first* event is a
purchase, the anti-leakage clip `first_purchase - 1` wrapped to 4,294,967,295 instead of
-1. Every cut-point would then have compared as admissible, and prefixes could have included
the purchase event itself — silently defeating Sec. 7.4 for exactly the sessions where it
matters most. `test_purchase_only_session_yields_no_admissible_prefix` caught it on the
first run. The index is now cast to `Int64` before the subtraction.

This is the clearest argument for why the leakage invariants are unit tests rather than
review comments: the bug arrived as a *performance* change, in a different function, and
nothing about it looked like a leak.

### A15. Dataset B is subsampled to 200,000 users, seeded (Sec. 5.3, Appendix C)

**Decision.** Analysis runs on a seeded random sample of 200,000 users, keeping every
session those users have. Seed 42. The full month stays on disk and its data-flow table is
reported, so the sampling fraction is auditable.

**Why.** Sec. 5.3 sets the requirement at ">=52k sessions" and explicitly permits
subsampling "for tractability, documented and seeded". October 2019 alone yields 5,366,181
sessions — about 100x the floor — and Appendix C budgets TimeSHAP at roughly 1–5k
sequences. The binding constraint is compute, not evidence. Concretely: cut-point
computation on the full month did not finish after 1.9 CPU-hours even with the A14
optimisation, because grouping 37.4M events by a 5.4M-cardinality string key dominates
everything else.

**Sampling is by user, never by session.** Both Sec. 7.6 split protocols keep a visitor
whole. Sampling sessions independently would tear users apart before the splitter ever saw
them, silently defeating the identity-leakage guarantee the grouped split exists to provide.

**The sample is a hash of (seed, user id)**, not a function of iteration order, so it is
stable across reruns, Polars versions and row orderings. It is also *nested*: at a fixed
seed a larger sample is a strict superset of a smaller one, which makes a sample-size
sensitivity check cheap — rerun at 400k and the 200k results are a subset, so any
difference is attributable to the added users rather than to a different draw.

**To report in the paper.** The sampling fraction, the seed, and a sample-size sensitivity
check (200k vs 400k users) alongside the Sec. 7.2 gap sensitivity. If headline conclusions
move between sample sizes, the sample is too small and must be raised.

### A16. Out-of-order stages are unreached, not shunted (Sec. 7.3)

**Decision.** A stage whose trigger fires out of funnel order is marked *unreached* rather
than pushed forward. Concretely: a visitor who carts before producing a second browsing
signal skipped consideration, so S2 is null for them and S3 sits on the cart. The
`_monotone` helper that pushed later cut-points past earlier ones is gone.

**Why.** With shunting, S3 was dragged onto S2's cut-point whenever the cart came first, and
on real REES46 data **45.1% of S2/S3 pairs had identical cut-points** — the same collapse
A8 fixed on the S1→S2 leg, reappearing on S2→S3. After the change both legs are 0% identical,
with mean gaps of 2.49 and 5.37 events.

**Also fixed: cart events were counting as browsing signals.** A cart in a different category
from the preceding view satisfied "category switch", so it could open S2 and S3 on the same
event. S2 is defined as "product/category browsing", so its triggers are now restricted to
view events. The synthetic fixture never exposed this, because there carts always inherit
the preceding view's category — a reminder that the fixture pins semantics, not realism.

**Resulting Dataset B stage table** (200k-user subsample, 485,459 sessions):

| Stage | N | Reach | Prevalence |
|---|---|---|---|
| S1 | 475,140 | 97.9% | 8.8% |
| S2 | 233,160 | 48.0% | 6.8% |
| S3 | 45,031 | 9.3% | 52.1% |

**This complicates H1.** H1 predicts PR-AUC rising monotonically from awareness to intent.
But S2's prevalence (6.8%) is *lower* than S1's (8.8%), because the visitors who cart early
— the ones most likely to convert — skip consideration and are excluded from S2 entirely.
The population that lingers in consideration is genuinely less likely to buy. Expect the
improvement curve to dip at S2 and jump at S3, and be prepared to report that as a finding
about the funnel rather than as a failure of the model.

### A17. The temporal split costs 54% of Dataset B's sessions (Sec. 7.6)

**Observed, not decided.** On the 200k-user subsample the temporal split retains 222,268 of
485,459 sessions: train 154,661, val 28,892, test 38,715, with **51,256 of 200,000 users
dropped as straddlers**. The grouped split retains all 485,459.

**Why.** The window is a single month and visitors return within it, so a user assigned to
the training period by their first session very often remains active past the train/val
boundary. Keeping them would either leak identity across the boundary or break the temporal
ordering; A11 chose to drop them and count them.

**Judgement.** The retained 222k sessions are still 4x the Sec. 5.3 floor, so the temporal
split remains viable as the headline. But the loss must be reported, and it is an argument
for widening Dataset B to two or three months: with a longer window the straddling band is a
much smaller fraction of the whole. Recommend reporting both splits side by side, with the
retention figures visible, rather than quietly showing only the temporal result.

### A18. RESOLVED — calibration moves to a training-period split (Sec. 9.6)

**Decision.** Calibration is fitted on a 20% label-stratified slice carved from the
*training* period, not on the validation month. Validation now does one job (threshold
selection) instead of two.

**Why.** Sec. 9.6 asks for "a calibration split", and the first pass reused validation for
it. Validation is November, whose 25.4% prevalence is double December's, and A12 showed
every model over-predicting by almost exactly that ratio. The training-period slice has
prevalence 0.1223 against December's 0.1251 — a near-perfect match — because it is drawn
across all training months rather than one extreme one.

**Effect, measured across five models at five seeds:**

| Model | PR-AUC | ECE uncalibrated | ECE calibrated | ECE + prior correction |
|---|---|---|---|---|
| CatBoost | 0.699 ± 0.006 | 0.063 | **0.018** | 0.033 |
| Random Forest | 0.685 ± 0.006 | 0.096 | **0.015** | 0.016 |
| LightGBM | 0.677 ± 0.008 | 0.065 | **0.026** | 0.039 |
| XGBoost | 0.676 ± 0.005 | 0.036 | **0.018** | 0.021 |
| Logistic Regression | 0.583 ± 0.000 | 0.203 | **0.032** | 0.052 |

ECE falls from the 0.105–0.117 band under November calibration to **0.015–0.032** — roughly
a sixfold improvement. PR-AUC drops slightly (LightGBM 0.710 to 0.677) because 20% of the
training period is now held out; that is the price of a calibration set and it is worth
paying.

**The EM prior correction is implemented but not applied by default, and this is a
finding.** `evaluate/calibration.py` implements Saerens et al. (2002), which estimates the
shifted prior from unlabelled scores — legitimate, since it never touches test labels.
Applied *on top of* the corrected calibration split it makes ECE **worse** for every model
(e.g. CatBoost 0.018 to 0.033), because it estimates a prior of 0.1447 against a true 0.1251
and over-corrects a shift that is no longer there. Report it as a negative result: once the
calibration set is drawn from a representative period, post-hoc prior correction is
unnecessary and harmful. It is retained in the codebase for the Dataset B arm, where the
temporal boundary may induce genuine shift.

### A19. CONFIRMED ACROSS SEEDS — the stage improvement curve contradicts H1 (Sec. 9.2, 10, RQ1)

**Still untuned (Sec. 9.5 Optuna pending) and the stage models are uncalibrated, so treat
the absolute levels as provisional. The *direction* is now robust: seed-to-seed standard
deviation is 0.0002–0.0030, one to two orders of magnitude smaller than the between-stage
differences.**

Dataset B (200k-user subsample, temporal split, LightGBM, five seeds):

Calibrated, five seeds (ECE now 0.010–0.017, down from 0.103–0.327 uncalibrated — the A18
correction transfers to the stage models):

| Stage | N test | Prevalence | PR-AUC | **PR-AUC lift** | ROC-AUC | ECE |
|---|---|---|---|---|---|---|
| S1 | 38,059 | 0.073 | 0.1312 ± 0.0009 | **1.79** | 0.6406 | 0.010 |
| S2 | 17,884 | 0.059 | 0.0834 ± 0.0015 | **1.41** | 0.6198 | 0.011 |
| S3 | 2,973 | 0.521 | 0.5681 ± 0.0041 | **1.09** | 0.5695 | 0.017 |

H1 predicts PR-AUC rising monotonically from awareness to intent. Raw PR-AUC does rise
(0.134 to 0.581), but **that is entirely prevalence**: chance-level PR-AUC equals the base
rate, so S3 starts from a floor of 0.52 where S1 starts from 0.073.

Normalised, the curve **runs the other way**. Lift falls monotonically, 1.83 to 1.11, and
ROC-AUC — which is prevalence-independent and so cannot be explained away this
way — falls with it, 0.640 to 0.566. By S3 the model is barely above chance relative to its
own base rate.

This is coherent rather than anomalous. Once a visitor has carted, they convert 52% of the
time, and what separates the converters from the abandoners is largely *unobserved* in a
clickstream: checkout friction, payment failure, delivery cost, distraction. Browsing
behaviour discriminates well early, when it is the only signal that differs, and poorly
late, when everyone looks alike and the decisive factors are off-stream. S1 achieves the
highest lift with the *fewest* features (19 against 23), which strengthens the reading.

**Consequence for the paper.** H1 as stated will likely be rejected, and that is a result
worth reporting rather than a failure to explain away. It also sharpens RQ4: if late-stage
prediction is near-chance, the actionable intervention window is *early*, which is the
opposite of where cart-abandonment practice concentrates its effort.

**Before claiming any of this**: run the Optuna-tuned models (Sec. 9.5) and confirm the
ROC-AUC decline survives tuning. Also calibrate the stage models — current stage ECE runs
0.11–0.33, so their probabilities are not yet usable for RQ4.

### A20b. CORRECTION — H3 is **not** supported once proper CIs are used

**This supersedes the verdict in A20 below. A20 is retained to show the error.**

A20 concluded H3 was supported because the entropy/velocity gain over baseline was "four to
fifteen times the seed standard deviation". **That comparison was invalid.** Seed-to-seed
standard deviation measures how much the *same model* moves when refitted on the *same*
data with a different random seed. It says nothing about sampling uncertainty of the metric
on a finite test set, which is what a difference between two models has to be judged
against — and which is far larger. Using it as an error bar understated uncertainty by
roughly an order of magnitude.

Run through the Sec. 12 machinery — paired bootstrap on the test set, 2,000 stratified
resamples, seed-averaged scores, Holm-corrected across the nine-contrast family:

| Stage | Contrast | Δ PR-AUC | 95% CI | Verdict |
|---|---|---|---|---|
| S1 | +temporal vs baseline | +0.0102 | [+0.0045, +0.0167] | **excludes 0** |
| S1 | baseline+entropy/velocity vs baseline | +0.0038 | [−0.0002, +0.0084] | includes 0 |
| S1 | +entropy/velocity vs +temporal | −0.0006 | [−0.0017, +0.0005] | includes 0 |
| S2 | +temporal vs baseline | +0.0120 | [+0.0057, +0.0197] | **excludes 0** |
| S2 | baseline+entropy/velocity vs baseline | +0.0035 | [−0.0011, +0.0085] | includes 0 |
| S2 | +entropy/velocity vs +temporal | +0.0007 | [−0.0017, +0.0031] | includes 0 |
| S3 | +temporal vs baseline | +0.0139 | [−0.0074, +0.0370] | includes 0 |
| S3 | baseline+entropy/velocity vs baseline | +0.0156 | [−0.0027, +0.0346] | includes 0 |
| S3 | +entropy/velocity vs +temporal | +0.0018 | [−0.0045, +0.0085] | includes 0 |

**Verdicts:**

- **H3 is not supported.** The entropy/velocity gain over baseline has a CI covering zero at
  every stage. S1 comes closest (+0.0038, upper bound of the negative side −0.0002) but does
  not clear it.
- **The temporal family is supported at S1 and S2**, cleanly, and not at S3 — where the test
  set is 2,973 sessions and the interval is correspondingly wide (±0.02).
- **The redundancy finding survives and is the robust one.** Given temporal, entropy/velocity
  contributes −0.0006 / +0.0007 / +0.0018 with intervals tight around zero. This is the
  strongest evidence in the family, because it is the one place the CIs are narrow enough to
  say "nothing here" rather than "cannot tell".

**Why S3 cannot decide anything.** Every S3 interval is wide enough to admit both a
meaningful gain and a meaningful loss. With 2,973 test sessions the study is simply
underpowered at the intent stage. That is a sample-size limitation, not a null result, and
must be reported as such — an argument for raising the Dataset B subsample (A15) or widening
the window beyond one month (A17).

**Methodological note worth a sentence in the paper.** Reporting mean ± seed-std invites
exactly the error made in A20: the spread looks like an error bar and is not one. Where a
difference between models is claimed, the uncertainty must come from resampling the
evaluation data, not from reseeding the fit.

### A20. SUPERSEDED by A20b — H3 holds as written, but the features are redundant with temporal

**The redundancy half of this survived; the "H3 supported" half did not. See A20b.**

**The nuance is the finding. Reporting either half alone would misrepresent it.**

Ablation on Dataset B, five seeds, LightGBM, bagging fixed (A21):

| Stage | baseline | +temporal | +entropy/velocity | full | baseline+entropy/velocity |
|---|---|---|---|---|---|
| S1 | 0.1241 | 0.1355 | 0.1361 | 0.1361 | 0.1281 |
| S2 | 0.0774 | 0.0876 | 0.0888 | 0.0882 | 0.0806 |
| S3 | 0.5618 | 0.5821 | 0.5800 | 0.5822 | 0.5842 |

The two comparisons that matter:

| Stage | entropy/velocity **vs baseline** (H3 as written) | entropy/velocity **given temporal** (the ladder) |
|---|---|---|
| S1 | **+0.0039** | +0.0006 |
| S2 | **+0.0031** | +0.0012 |
| S3 | **+0.0223** | −0.0021 |

**H3 as written is supported.** Against baseline aggregates alone, navigation entropy and
click velocity add 0.0039 / 0.0031 / 0.0223 PR-AUC — between roughly four and fifteen times
the seed standard deviation (0.0003–0.0016), so not noise.

**But the information is almost entirely redundant with the temporal family.** Given
temporal, the same features add 0.0006 / 0.0012 / −0.0021 — nothing, and at S3 slightly
negative. S3 is the sharpest case: entropy/velocity is the single largest gain over baseline
of any family (+0.0223), and is worth *less than zero* once temporal features are present.

This is mechanically sensible rather than surprising. Click velocity is events divided by
elapsed prefix duration; both of its constituents are temporal features. It re-expresses
information the temporal family already carries rather than adding a new channel.

**How to report it.** State H3 as supported, then immediately state the redundancy, with
both columns side by side. The honest claim is *"entropy and velocity features carry real
predictive signal beyond aggregate counts, but that signal is substantially shared with
simpler temporal features rather than additional to them"* — a more precise and more useful
result than either "supported" or "rejected".

**Still to do before the paper**: back this with the Sec. 12 machinery — paired bootstrap
PR-AUC differences with CIs and Holm correction — rather than comparing means to standard
deviations by eye.

### A21. LightGBM's `subsample` was silently inert (Sec. 6.2)

**Fixed.** `subsample=0.9` has no effect unless `subsample_freq > 0`, and it defaults to 0.
Bagging was therefore disabled, removing the main source of seed-to-seed variation: S1's
baseline rung reported a standard deviation of exactly 0.0000 across five seeds.

This understated the reported ± std, which Sec. 6.2 relies on to represent run-to-run
variability. `subsample_freq=1` is now set alongside `subsample`. All Dataset B stage
numbers are being regenerated; results predating this fix have artificially tight spreads.

### A22. H4 is not supported, but the test as designed is underpowered (Sec. 11.2, RQ3)

**Report both halves. The second is the more important one.**

GRU per stage, trained on the same prefixes as the tree models, explained with TimeSHAP,
feature rankings correlated against TreeSHAP:

| Stage | GRU test PR-AUC | Tree test PR-AUC | H4 Spearman | Concepts | vs 0.6 |
|---|---|---|---|---|---|
| S1 | 0.1237 | 0.1312 | +0.200 | 4 | fail |
| S2 | 0.0593 | 0.0834 | +0.500 | 5 | fail |
| S3 | 0.5646 | 0.5681 | +0.600 | 5 | fail (threshold is strict) |

**The sequence model is not broken**, which matters — a failed convergence test would be
uninformative if one paradigm were simply incompetent. The GRU matches the tree models at S1
(0.124 vs 0.131) and S3 (0.565 vs 0.568) and trails at S2, so both arms are genuinely
modelling the same signal.

**The comparison rests on four or five concepts.** That is the finding. Spearman on n = 4
takes only a handful of discrete values — with four items the achievable correlations are
±1.0, ±0.8, ±0.6, ±0.4, ±0.2, 0 — so a "correlation" here is barely a statistic, and the
pre-registered 0.6 threshold is being applied to a quantity that cannot land near it by
chance in any meaningful sense. The concept count is small because Dataset B lacks device
and traffic-source fields, the per-timestep feature set is deliberately compact, and two
event-type indicators have no prefix-aggregate counterpart at all.

H4 was pre-registered without anticipating that the two paradigms would share so few
mappable concepts. Recording that honestly is worth more than a verdict computed on five
points.

**A better test exists and should be run before submission.** Rather than correlating
aggregate importance across ~5 concepts, correlate **per-instance attributions** for the
concepts that do map: thousands of paired observations instead of five, and a direct answer
to whether the paradigms agree about *individual journeys* rather than about a global
ranking. That is a stronger reading of "convergent validity" than the pre-registered one and
should be reported alongside it, not instead of it.

**One genuine signal in the current numbers.** Agreement rises monotonically down the funnel
(0.200, 0.500, 0.600). The paradigms converge where the signal concentrates. That is
consistent with the Layer 3 result — S2 is also where faithfulness missed its threshold —
and with A19: stages with weaker signal produce explanations that are both less faithful and
less reproducible across paradigms. Three independent measurements pointing the same way is
worth a paragraph in the discussion.

### A23b. CORRECTION — the S2 negative agreement does not survive reseeding

**A23 below reported per-instance agreement from a single seed and drew a stage-specific
conclusion from it. Five seeds retract that conclusion. A23 is kept so the error is visible.**

| Stage | Single seed (42) | **Five seeds, mean** | Concepts sign-stable |
|---|---|---|---|
| S1 | +0.100 | **+0.179** | 1 of 4 |
| S2 | **−0.151** | **−0.018** | **0 of 5** |
| S3 | +0.276 | **+0.046** | 2 of 5 |

**Only 3 of 14 (stage, concept) pairs hold a consistent sign across seeds.** S2's apparent
negative agreement was a single-seed artefact: across five GRU initialisations the mean is
−0.018 and *no* concept keeps its sign. The claim that the paradigms "actively disagree" at
S2 is withdrawn.

**What survives, and it is still the substantive result.** Per-instance agreement is
near zero at every stage (+0.179, −0.018, +0.046) while the aggregate test reported +0.200,
+0.500 and +0.600 on the same models. Aggregate convergent validity does not imply agreement
about individual cases — that finding is unchanged and is now properly seeded. It is simply
a uniform absence of per-instance agreement rather than a stage-varying pattern.

**Consequence for the S2 narrative.** A23 claimed four independent measurements flagged S2.
Three do: weakest predictive lift (A19), the only faithfulness threshold miss (0.494), and
the lowest aggregate cross-paradigm agreement. The fourth does not. The subsection proposed
for the paper should be rewritten around three convergent signals, or dropped in favour of
the uniform per-instance finding, which is cleaner and better supported.

**Why this happened.** The GRU is small, trained for four epochs on a subsample, and its
attributions are correspondingly unstable — seed standard deviations of 0.2–0.4 on the very
correlations being compared. The lesson is the same one A20b taught on the tabular side:
a quantity computed once is not evidence, and this codebase now defaults to the frozen seed
list everywhere it reports one.

### A23. SUPERSEDED by A23b — Per-instance H4 from a single seed

A22 flagged the pre-registered H4 as underpowered — a Spearman over four or five concepts.
The per-instance version asks instead: *for a given driver, do the two paradigms agree about
which journeys it mattered for?* Each correlation is computed across 534–600 shared sessions
rather than across five concepts.

| Stage | Aggregate H4 (5 concepts) | **Per-instance mean ρ** | Sessions |
|---|---|---|---|
| S1 | +0.200 | **+0.100** | 534 |
| S2 | +0.500 | **−0.151** | 600 |
| S3 | +0.600 | **+0.276** | 600 |

**Zero of fourteen (stage, concept) pairs clear the 0.6 threshold.** Overall mean ρ = 0.073.

**The finding is the divergence between the two rows.** At S2 the aggregate test reports
+0.500 — respectable-looking, near the threshold — while per-instance agreement is
*negative*. The two paradigms produce broadly similar average rankings there while actively
disagreeing about which individual visitors each driver mattered for. Aggregate convergent
validity does not imply agreement about individual cases, and for anyone acting on a
specific visitor's explanation, only the second matters.

This reframes H4 rather than merely failing it. The pre-registered version asked the easier
question, and would have been passed at S3 (+0.600, exactly at threshold) on evidence that
does not survive contact with individual journeys.

**S2 is now flagged by four independent measurements**: weakest predictive lift (A19), the
only faithfulness threshold miss (0.494 vs 0.5), lowest per-instance cross-paradigm
agreement, and here the only *negative* one. Consideration is the stage where this
framework's explanations are least trustworthy, and the paper should say so plainly rather
than reporting stage-averaged explanation quality.

**Methodological note.** The first run had an incidental overlap of 49 sessions at S1
because both arms subsampled validation independently. The sequence arm now preferentially
explains sessions the tree arm already explained, making the intersection deliberate. The
conclusion was unchanged (mean ρ 0.085 then, 0.073 now) but 49 sessions would not have been
reportable.

### A9. Python 3.13 is present on the machine; the project pins 3.11 (Sec. 4)

**Decision.** The project venv is CPython 3.11.15, provisioned by `uv`, independent of the
system 3.13.

**Why.** The protocol pins 3.11 and several pinned libraries (notably `numpy<2`,
`scikit-learn 1.4.x`) have no 3.13 wheels at those versions. Building on the system
interpreter would have forced the versions off their pins.

---

## 2026-08-06 — Post-freeze completion of three protocol arms

### A24. Hyperparameter tuning ran after the headline results; defaults stay headline (Sec. 9.5)

**Decision.** The Sec. 9.5 Optuna search (TPE, 100 trials per stage, fixed spaces, lightgbm)
was executed against the frozen temporal validation split, after the default-hyperparameter
stage models had already opened the test partition. Tuned models were then evaluated on test
over the same five seeds. The tuned numbers are reported as a sensitivity analysis
(paper §4.7); the default-hyperparameter results remain the headline.

**Why headline is unchanged.** Every explanation-layer artefact — SHAP attributions,
migration, faithfulness, the sequence-arm comparison — was computed against the default
models. Promoting tuned numbers to headline would detach the explanations from the models
they explain, or force a full re-run of Layers 1–3 against retuned models. Tuning moved S1
and S2 by +0.002/+0.003 (within seed noise) and S3 by +0.032; the lift ordering
(1.81 > 1.46 > 1.15) is unchanged, so no conclusion depends on the choice.

**What tuning selected.** All three stages chose the minimum tree count in the space (200)
with heavy regularisation. The search pushed away from complexity, consistent with temporal
drift penalising models that fit their own period too well.

**Studies persist** in `experiments/optuna/` (SQLite, reopenable), summarised in
`reports/tables/tuning_gap30s200000.csv`; tuned per-seed results in
`stage_models_gap30s200000_tuned_per_seed.csv`.

### A25. Grouped-split robustness arm executed (Sec. 7.6)

**Decision.** The grouped protocol (user-partitioned, identity-clean, not time-ordered) was
run with the headline configuration: lightgbm, defaults, five seeds, full feature set.
Results to `stage_models_gap30s200000_grouped_per_seed.csv` and
`improvement_curve_gap30s200000_grouped.csv`; CLI table names are now protocol-tagged so
non-temporal runs cannot overwrite the headline tables.

**Result.** Lift 1.98 → 1.53 → 1.15, monotone as on the temporal split, and uniformly higher
— the direction temporal drift predicts. The H1 reversal is not an artefact of the temporal
protocol.

### A26. Stability arm of Layer 3 executed; previous run had claimed the column (Sec. 11.3)

**Decision.** The published explanation-quality run used `--no-stability` and left the
Lipschitz columns null while the paper text described stability as reported per stage. The
arm has now been run (25 instances per stage, noise sd 0.05) and the table regenerated.

**Result.** Max Lipschitz ratio 0.048 (S1), 0.073 (S2), 0.028 (S3); means 0.010–0.015. All
stages stable. Faithfulness, deletion/insertion and seed-consistency values reproduced
exactly, so the regeneration changed nothing previously reported. S2 remains the anomalous
stage on faithfulness but is not unstable.

---

## 2026-08-06 (later) — Literature claim substantiated

### A27. The `[UNVERIFIED CLAIM]` about explanation validation is now evidenced (Sec. 2.3)

**What was flagged.** The draft asserted, on impression rather than evidence, that
applied e-commerce XAI studies rarely validate their explanations. It was marked
`[UNVERIFIED CLAIM]` because no systematic search had been run, and it is load-bearing:
the paper's validation layer is motivated by it.

**Search performed (2026-08-06).** Targeted search for quantitative reviews of XAI
*evaluation practice*, as opposed to XAI method surveys. Query themes: proportion of
XAI application papers that evaluate explanation quality; systematic/scoping reviews of
SHAP application practice. Four reviews with explicit counts were located and their
bibliographic details verified against publisher records.

**What the evidence supports.**

- Saarela & Podgorelec (2024), *Applied Sciences* 14(19):8884 — PRISMA review of 512
  XAI application articles: most give no quantitative evaluation of explanation
  quality; ~19% use quantitative metrics; SHAP/LIME predominate.
- Mainali & Weber (2023), arXiv:2307.09673 — scoping review of 187 application papers
  self-describing as "explainable model": **81% conduct no evaluation** of the XAI
  method; 64% of the full set use SHAP or LIME with no evaluation.
- Nauta et al. (2023), *ACM Comput. Surv.* 55(13s):295 — 312 papers that *introduce*
  XAI methods: one in three evaluate exclusively with anecdotal evidence.
- Adadi & Berrada (2018), *IEEE Access* 6:52138--52160 — 5% of 381 papers focused
  explicitly on evaluation.

**Decision.** The claim is retained and strengthened, now stated with figures and
citations rather than as an impression. Two qualifications are recorded honestly:
(i) none of the four reviews is specific to e-commerce, so the manuscript claims the
gap for *applied XAI* generally rather than for e-commerce in particular, which is the
weaker and supportable form; (ii) the abstract's "almost never validated" was reduced
to "rarely validated", since 81% no-evaluation implies roughly one in five do evaluate.

**Also softened.** A separate, still-unevidenced assertion that applied studies "rarely
draw the distinction" between predictive contribution and explanatory attribution was
rewritten as a statement about what this paper reports, not about what others omit.
