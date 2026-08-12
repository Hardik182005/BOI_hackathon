# Judge Criteria Scorecard

Master prompt §35. The five prototype judging criteria, each answered with a
**verifiable artefact** rather than a claim.

Every number in this document was measured on this machine and is reproducible
from the listed file or command. Where a figure is unflattering, it is printed
unflattering.

---

## Scorecard at a glance

| Criterion | Our strongest single piece of evidence |
|---|---|
| **Innovation** | The **Dual-Evidence ProofGraph** — the only system in the seven repos we verified that also argues *against* its own alert |
| **Technical Feasibility** | Release gate **PASS, 23/23**, 196 tests, and an 8-variant organiser dry-run on the real workbook |
| **Business Potential** | **Top 25 alerts → 25 mules, 0 false positives** on out-of-fold data |
| **Scalability** | **688 rows/s** batch scoring, model loaded once, chunked HTTP path |
| **User Experience** | 11 white-background pages; every alert answers *"why flagged?"* **and** *"why might this be wrong?"* |

---

## 1. Innovation

### 1.1 Dual-Evidence ProofGraph — the main USP

> Every alert must prove why it was raised — and show why it might be wrong.

`src/muleguard/explain/proofgraph.py` · `docs/PROOFGRAPH_DESIGN.md`

Standard SHAP explanations are one-sided: they list what pushed the score up. In
a queue where **most alerts are legitimate customers**, a one-sided explanation
is not just incomplete — it is an active driver of false positives, because the
reviewer has no structured reason to close a case.

The ProofGraph builds both sides. Live example `CASE-18A744455E`: 13 nodes, 12
edges, 5 prosecution nodes, 1 defence node, 1 uncertainty node. Evidence balance
0.9156 — overwhelming — and the verdict was still downgraded from
`ENHANCED_REVIEW_RECOMMENDED` to `REVIEW_RECOMMENDED` because an uncertainty node
was present. **The doubt mechanically outranks the evidence.**

Six structural doubts are measured, not narrated: model-family disagreement,
non-decisive conformal set, out-of-distribution input, unremarkable anomaly
percentile, missing deciding inputs, verifier declining to confirm.

**Judge check:** open any case → the defence column is on the same screen, in the
same format, with the same weight.

### 1.2 Counterfactual Twin

For each alerted account we find a **real, labelled-legitimate row** that is
nearest on the deciding features, and show the feature-by-feature difference on
an IQR scale. Not a perturbed synthetic point — an actual customer.

"This account looks like a mule; here is a genuine customer who looks almost
identical, and here is exactly what separates them."

### 1.3 Feature Availability Firewall

`src/muleguard/features/firewall.py` · `docs/FEATURE_AVAILABILITY_AUDIT.md`

One question asked of every one of 3,924 columns: **was this value knowable when
the analyst had to decide?** 13 columns fail and are quarantined.

This is worth more than it sounds. `F3912` = `FRAUD_SUSPECTED` is a *resolution
status* — it records the answer. Measured: admitting it moves PR-AUC from
**0.6142 to 0.9419**. A team that leaves it in reports a spectacular number and
fails the hidden validation set completely.

### 1.4 Validation Compatibility Score

`docs/VALIDATION_LAB_REPORT.md` §5. One auditable 0–100 number telling the
organiser how much like our training data their file is — schema completeness
(40), distribution compatibility (30), missingness consistency (15), value-range
coverage (15). Weights fixed before any file was ever scored. **It reports; it
never modifies a prediction.**

### 1.5 Merchant false-positive safeguard

`docs/MERCHANT_LEGITIMACY_VERIFIER.md`. The common approach is `score × 0.70` for
merchants — unmeasurable, decalibrating, invisible. We refused it and trained a
model instead: OOF PR-AUC 0.63604, lift 72.19×, and **zero mules among the 1,452
accounts it places in `STRONG_BUSINESS_EVIDENCE`** against a 0.881 % book rate.

It excludes `TOTAL_ALL_RAILS` **even though including it would raise its own
score**, because exculpatory evidence is worthless if it is derived from the
accusation. It adjusts escalation *confidence*, never the risk probability.

### 1.6 Sealed Validation Protocol

*"We didn't peek at the labels"* is unfalsifiable. We made it falsifiable:
predictions are written and SHA-256-hashed **before** the target column is
readable; `reveal_metrics` recomputes the hash and refuses to score if it moved.
Rehearsed live — sealed `10:59:04.490653Z`, revealed `10:59:38.459657Z`,
`seal_verified: true`.

---

## 2. Technical Feasibility

| Demonstrated | Evidence |
|---|---|
| **The real `DataSet.xlsx` works** | 9,082 × 3,925 loaded, SHA-256 verified unmodified by the release gate. Both workbooks are read-only and never overwritten |
| **`Description.xlsx` semantic registry** | 3,924 column records parsed into a queryable registry with availability class, family, window and transform. `docs/FEATURE_DICTIONARY_REPORT.md` |
| **No fake graph** | Graph adapter reports `UNAVAILABLE` by default and says why, in the product. Gate check `graph_never_fabricates_edges` PASS |
| **Batch scoring** | 1,818-row locked test scored end-to-end at 688 rows/s; HTTP path chunked at 500 rows |
| **Hidden-validation upload** | 8 file variants (xlsx, csv, shuffled columns, extra columns, quarantined columns present, dropped columns, unseen category) — **8/8 scored** |
| **One-command local run** | `./run.sh` — health-gated startup, ~8–12 s to ready |
| **Offline core scoring** | No internet, no API keys, no MCP, no browser agent. Gate check `scoring_survives_ollama_outage` PASS |

### The release gate

`docs/FINAL_RELEASE_GATE.md`, regenerated 2026-08-12T11:34:10Z:

> **PASS — 23 / 23 checks · 196 tests passed in 143.44 s**

Checks that would block a release include: no target or `F3912` leakage; no
split overlap; locked-test touch log; LLM cannot alter a score; no auto-freeze
path; scoring survives an Ollama outage; robustness grade not hand-picked; the
label audit changed nothing; the merchant verifier cannot lower risk; no
forbidden verdict vocabulary in shipped source; organiser dry-run passed.

### The honesty test

The gate check `robustness_grade_not_hand_picked` publishes our badge as
**`LOW`**, limited by `prediction_rank_stability` = 0.3694, read off thresholds
fixed before the measurement. We could have redefined the metric. The check
exists specifically to stop us.

---

## 3. Business Potential

### 3.1 The alert budget is the business case

Prevalence is 0.891 %. A bank does not get to review 9,082 accounts — it gets to
review as many as its team can handle. Measured on out-of-fold predictions:

| Alerts reviewed | Mules found | Recall | Precision |
|---:|---:|---:|---:|
| **25** | **25** | 0.391 | **1.000** |
| 50 | 44 | 0.688 | 0.880 |
| 100 | 53 | 0.828 | 0.530 |

**Twenty-five alerts, twenty-five mules, zero wasted reviews.** Locked test:
PR-AUC 0.7263, **lift 77.7× over prevalence**, Recall@100 0.824.

### 3.2 Reduced false-positive review burden

At the top-100 budget the model hands the team 47 legitimate customers. Five
independent layers exist to let a reviewer dispose of those quickly and safely —
no accusatory language, a defence panel, the merchant verifier, conformal
abstention on 4.7 % of cases, and calibrated tiers. `docs/FALSE_POSITIVE_CONTROL_REPORT.md`.

### 3.3 Evidence packet and auditability

Every case exports a packet containing the score, the calibrated probability, the
tier, the prosecution and defence nodes with the **source column named for each**,
the counterfactual twin, the conformal status and the model fingerprint. A node
with no provenance cannot be serialised.

Bundle SHA-256 `d12914de5abee99a…`, runtime fingerprint `afd0dc1d8fc02eb9`,
verified byte-identical before the first upload and after the last.

### 3.4 Human workflow, and its limits

Five verdicts, all describing work for a person: `REVIEW_RECOMMENDED`,
`ENHANCED_REVIEW_RECOMMENDED`, `MONITOR_ONLY`,
`INSUFFICIENT_EVIDENCE_TO_ESCALATE`, `NO_ACTION_INDICATED`. **No account is ever
frozen automatically** — no such code path exists. High-impact actions require a
named approver.

### 3.5 Bank API integration

FastAPI with a published OpenAPI schema; scoring, batch, ProofGraph, validation
and graph routes are all HTTP-first. Nothing in the scoring path depends on the
frontend.

---

## 4. Scalability

| Demonstrated | Measured |
|---|---|
| Batch API | 688 rows/s in-process on the 1,818-row locked test; HTTP chunked at 500 rows with progress events |
| Model in memory | loaded **once** at startup; latency stable across repeated calls |
| Fast tree ensemble | XGBoost inference is single-digit ms per row, CPU-only, no GPU required |
| Upload jobs | chunked with audit events; concurrency tested at 5 and 10 parallel requests, all 200 OK, outputs deterministic |
| Optional real-transaction graph adapter | complete and tested, bounded (`max_depth` 8, `max_nodes` 60, cycle length ≤ 6), switched on by one edge-file upload with no retraining |
| Production roadmap | documented in `docs/DEPLOYMENT_GUIDE.md` as an extension, not a claim |

**Scale honesty.** These are single-node CPU figures on an i5-1135G7 with 16 GB.
Multi-worker deployment, PostgreSQL and GPU challengers are a documented roadmap,
not something we are claiming to have benchmarked.

---

## 5. User Experience

Eleven pages, all on a **white background with black text and light grey
borders** — no dark theme, no gradients, no neon, readable at 1366×768:

`Overview` · `AlertQueue` · `CaseDetail` · `ProofGraph` · `ValidationLab` ·
`GraphLab` · `FeatureIntelligence` · `ModelPerformance` · `ModelCard` ·
`DriftMonitoring` · `BusinessValue`

| Requirement | Where |
|---|---|
| Simple white UI | every page; enforced by frontend tests |
| Alert queue | `AlertQueue` — sorted by calibrated risk, tiered, filterable |
| ProofGraph | `ProofGraph` — the dual-evidence view, prosecution left, defence right |
| **"Why flagged?"** | prosecution column: named features, values, and the source column for each |
| **"Why might this be a false positive?"** | defence column: negative-SHAP features plus the six measured structural doubts. **This is a first-class panel, not a tooltip** |
| Validation Lab | `ValidationLab` — drop a file, get schema integrity, shield report, compatibility score, sealed predictions |
| Downloadable evidence | per-case evidence packet and batch CSV, with formula-injection protection on export |

**The Graph Lab is deliberately empty.** It renders the `UNAVAILABLE` message
explaining that the competition extract has no counterparties and that we will
not fabricate them — rather than a placeholder network. A judge who opens it is
told, in the product, why.

---

## 6. What this scorecard does not claim

- We do **not** claim state-of-the-art accuracy. Our OOF PR-AUC is
  0.76904 ± 0.02663 and the measured **seed-noise floor on this data is 0.0905** —
  most headline differences between teams here are inside the noise.
- We do **not** claim the robustness badge is good. It is **LOW**.
- We do **not** claim 22 mule typologies are detectable. Six are
  `NOT_AVAILABLE` and are published as such.
- We do **not** claim proxy-discrimination leakage is bounded. Measuring it is a
  stated open gap in `docs/FAIRNESS_AND_SENSITIVE_FEATURE_AUDIT.md`.
- We do **not** claim production scale. The numbers above are one laptop.

Every one of these could have been quietly dropped. Each is printed because the
hidden validation set will not be impressed by a claim it can disprove.
