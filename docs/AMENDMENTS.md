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

### A9. Python 3.13 is present on the machine; the project pins 3.11 (Sec. 4)

**Decision.** The project venv is CPython 3.11.15, provisioned by `uv`, independent of the
system 3.13.

**Why.** The protocol pins 3.11 and several pinned libraries (notably `numpy<2`,
`scikit-learn 1.4.x`) have no 3.13 wheels at those versions. Building on the system
interpreter would have forced the versions off their pins.
