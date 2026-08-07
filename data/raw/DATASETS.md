# Datasets: access, licensing, ethics

Protocol Sec. 5. This file is the human-readable companion to the machine-readable
`provenance_*.json` records written by the ingest layer.

## Dataset A — UCI Online Shoppers Purchasing Intention

| | |
|---|---|
| Role | Static benchmark; reproducible baseline, static-vs-stage SHAP contrast, sanity check (Sec. 5.2) |
| Shape | 12,330 sessions; 10 numeric + 7 categorical features; binary target `Revenue` |
| Prevalence | 0.1547 (1,908 positives) — matches the ~15% the protocol expects |
| Licence | CC BY 4.0, UCI Machine Learning Repository |
| Citation | Sakar, C.O., Polat, S.O., Katircioglu, M., Kastro, Y. (2019). *Real-time prediction of online shoppers' purchasing intention using multilayer perceptron and LSTM recurrent neural networks.* Neural Computing and Applications 31, 6893–6908. |
| Status | **Present** in `data/raw/online_shoppers_intention.csv` |

**Provenance status (amendment A5, partially resolved).** The local file
`data/raw/online_shoppers_intention.csv` has SHA-256
`b3055ee355f59134d851d32641183cb4a8b45def7124d2f50442a042f358e0d9`, which matches the
value recorded in `provenance_A.json`, and its shape and prevalence match the protocol's
expectation exactly (12,330 rows, ~15.47%).

What that establishes and what it does not: the file is stable and every reported result
is traceable to this exact content. It does **not** establish that the file was fetched
from the canonical UCI endpoint
(`https://archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip`),
because that endpoint serves a ZIP archive whose hash necessarily differs from the CSV it
contains, so the two cannot be compared directly. Confirming the chain requires
downloading the archive, extracting the CSV and comparing *that* hash to the value above —
`funnel-shap fetch-a` performs the download step. Until then the provenance is
self-consistent but not independently corroborated.


**Non-causal flag (Sec. 7.4).** Dataset A carries whole-session aggregates with no event
order. Every feature is measured over the entire session, including whatever happened at
the moment of purchase. `PageValues` is the clearest case: it is the average value of pages
the visitor completed a transaction on, so it is partly a *consequence* of conversion. Any
model fitted here is a static benchmark and must be labelled non-causal in the paper.
Comparing its performance to Dataset B's prefix models measures what leakage buys, not
which predictor is better.

## Dataset B — REES46-class event log

| | |
|---|---|
| Role | Event-level primary; genuine journey reconstruction, stage-migration attribution, TimeSHAP arm (Sec. 5.3) |
| Source | `https://data.rees46.com/datasets/marketplace/2019-Oct.csv.gz` — REES46's own open endpoint |
| Window | October 2019, one month, as Sec. 5.3 permits ("subsample a fixed window/month for tractability, documented and seeded") |
| Size | 1,741,928,540 bytes compressed |
| SHA-256 | `8ebca1ad741295297368f2cf0315e3f36853a1a11768fb16babf8c9b83838147` |
| Retrieved | 2026-08-04 |
| Licence | **See the licence finding below — unresolved, action required** |
| Status | **Present** in `data/raw/rees46/` |

Columns as delivered, matching the Sec. 5.3 spec exactly: `event_time`, `event_type` ∈
{view, cart, remove_from_cart, purchase}, `product_id`, `category_id`, `category_code`,
`brand`, `price`, `user_id`, `user_session`.

### Why not Kaggle

Sec. 5.3 names the Kaggle mirror "or an equivalent open event log". The Kaggle copy sits
behind an authenticated API and no credentials exist on this machine. REES46 publishes the
identical monthly files from their own domain with no authentication, which is a *better*
provenance chain than a third-party mirror: it is the originating publisher. A Hugging Face
mirror also exists but was not used, for the same reason.

### Licence finding — needs your decision before submission

Sec. 5.3 requires the licence to be confirmed as permitting research publication and
recorded. It could not be confirmed, and the evidence points in two directions:

- The **Kaggle metadata field states "Data files © Original Authors"** — that is a
  reservation of rights, not a grant. It does not, on its face, permit redistribution or
  publication.
- **REES46 publishes them as "Free datasets with eCommerce behavior data ... for your
  neural network"**, serves them openly without a click-through, and links an IEEE paper
  built on these datasets from the same page. Intent and precedent clearly point to
  academic use being welcome.

No formal licence text (CC BY, ODbL, or similar) is stated anywhere I could find. "Freely
downloadable and widely used in published work" is not the same as "licensed for
publication", and a Q1 reviewer or a journal's data-availability check may ask.

**Recommended action:** email REES46 for written confirmation that academic use and
publication are permitted, and record the reply verbatim in `provenance_B.json`. Until
then, cite the dataset as REES46 / Michael Kechinov and state the position honestly in the
data-availability statement rather than asserting a licence that is not documented.

### Reproducing the download

```bash
curl -L -o data/raw/rees46/2019-Oct.csv.gz https://data.rees46.com/datasets/marketplace/2019-Oct.csv.gz
```

Then convert to Parquet and register provenance. Polars' lazy CSV scan cannot stream a
gzip member, and decompressing the month to plain CSV costs several times the disk that
Parquet does, so the conversion happens once up front:

```bash
.venv/Scripts/python.exe -m funnel_shap.cli prepare-b data/raw/rees46/2019-Oct.csv.gz
```

### The synthetic fixture

`funnel_shap.data.synthetic` generates an event log with the same schema, used by the test
suite and by `funnel-shap sessionize --synthetic`. It exists so the leakage invariants and
feature definitions could be pinned before the real data arrived, and it remains the
fixture for CI. **No result in the paper may come from it.**

## Variable availability (Sec. 5.1)

Sec. 5.1 requires honest reporting of which of the ten idealised journey variables actually
exist. The machine-readable table is generated by `funnel-shap tables` into
`reports/tables/variable_availability.csv`. Headlines:

- **Absent in both datasets:** search-query text. Reported as a limitation; no proxy is
  claimed.
- **Proxied in both:** scroll depth, via product-page dwell time. Always named as a proxy
  in results, never as "scroll depth".
- **Absent in B:** device and traffic source. This matters for H2, whose "context features
  dominate early stages" arm therefore rests on Dataset A.

## Ethics and governance (Sec. 5.4)

Behavioural data only; no PII. Identifiers in both datasets arrive already
hashed/obfuscated and no re-identification is attempted. Explanations are reported at the
cohort level; the representative local waterfall plots (Sec. 11.1) use synthetic or
aggregated exemplars so no individual user's journey is exposed. If the institution
requires it, obtain an ethics/IRB exemption for secondary use of public anonymised data and
cite it in the paper.
