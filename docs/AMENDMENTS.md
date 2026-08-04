# Amendment log

Protocol v1.0 states: "Any deviation after freeze is logged in a dated amendment appendix."
Each entry records the date, the affected protocol sections, what changed, and why.

The protocol is **not yet frozen**: the environment is pinned and the data/feature layer is
built and tested, but no split has been written to `data/processed/splits/` and the test
partition has never been read. Entries below are pre-freeze implementation decisions that
the protocol left open, recorded here so the eventual freeze is auditable.

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

### A6. OPEN ISSUE — S1 as defined yields a one-event prefix (Sec. 7.3, RQ1, H1)

**Not a decision. A protocol question that needs answering before freeze.**

Sec. 7.3 defines S1 as "session entry → first product interaction" with cut-point "first
view". Sessions in an event log almost always *open* with a view, so `cut_S1 = 0` and the
S1 prefix is a single event. Measured on the synthetic fixture: mean S1 prefix length is
exactly 1.0 event, and **19 of 23 S1 features are constant across all sessions** — every
count, duration, gap, entropy, velocity and transition feature is degenerate because a
one-event prefix has no second event to measure against.

The only features that vary at S1 are `hour_of_day`, `is_weekend`, `price_mean` and
`price_max` — i.e. time-of-day plus the price of the single product viewed.

This is not an implementation artefact; it follows directly from the stated cut-point. Its
consequences:

- **RQ1/H1** would compare a near-null S1 model against genuine S2/S3 models. The
  improvement curve would be dominated by S1 having almost no features, not by the journey
  becoming more predictive.
- **RQ2/H2** — "attribution mass migrates from context to behaviour" — is close to
  guaranteed at S1 by construction, since only context features are non-constant there.
  Confirming H2 on this definition would be circular.

Three candidate resolutions, in order of preference:

1. **Redefine S1's cut-point as the first *repeat or second* product interaction** — i.e.
   awareness spans entry through the visitor's first engagement signal, not the entry event
   itself. Keeps four honest stages and gives S1 real features.
2. **Define S1 by elapsed time or event budget** (e.g. the first 60 seconds, or first 3
   events), making awareness a window rather than a single instant.
3. **Drop S1 from the modelling set** and report the curve over S2/S3 only, stating that
   awareness carries no behavioural signal by construction.

Option 1 or 2 requires a Sec. 7.3 amendment before freeze; option 3 requires amending
Sec. 9.2 and H1. Deciding this *after* seeing test-set results would not be defensible, so
it must be settled now.

### A7. Python 3.13 is present on the machine; the project pins 3.11 (Sec. 4)

**Decision.** The project venv is CPython 3.11.15, provisioned by `uv`, independent of the
system 3.13.

**Why.** The protocol pins 3.11 and several pinned libraries (notably `numpy<2`,
`scikit-learn 1.4.x`) have no 3.13 wheels at those versions. Building on the system
interpreter would have forced the versions off their pins.
