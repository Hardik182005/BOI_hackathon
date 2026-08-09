# Pattern Availability Matrix

Master prompt §12. Which known mule typologies this dataset can actually
support — and, just as importantly, which it cannot.

Implementation: `src/muleguard/features/meta_features.py` (13 meta-features,
`META_DESCRIPTIONS`), `src/muleguard/explain/proofgraph.py` (`PATTERN` nodes).

---

## 1. Why this matrix exists

It is easy to write a mule-detection pitch that lists eight famous typologies —
fan-in/fan-out, layering chains, smurfing rings, dormant reactivation — and imply
the system detects all of them. It is much harder to say which of them the
supplied data can support.

This dataset is **9,082 account-level rows of aggregate features**. There is no
transaction table, no counterparty identifier, no timestamp sequence, and no edge
list. Every typology that is fundamentally *relational* — that is defined by who
paid whom — is **not detectable here**, and saying otherwise would mean
fabricating a graph from aggregates, which addendum UPDATE 8 forbids in exactly
those words:

> Do NOT derive fake sender/receiver relationships from F-columns.

So each typology below is graded against what the file genuinely carries.

---

## 2. Availability grades

| Grade | Meaning |
|---|---|
| **DIRECT** | the behaviour is measured by named columns; a feature encodes it directly |
| **PROXY** | the behaviour leaves an account-level fingerprint the features capture, but the pattern itself is not observed |
| **PARTIAL** | one side or one aspect is observable; the full typology is not |
| **NOT_AVAILABLE** | requires data this extract does not contain — recorded as absent, never simulated |

---

## 3. The matrix

| # | Typology | Grade | Evidence in this dataset | Feature / meta-feature |
|---|---|---|---|---|
| 1 | **Pass-through / rapid layering** — funds leave about as fast as they arrive | **DIRECT** | total debit vs total credit amount, 7d and 31d | `MG_PASSTHROUGH_7D`, `MG_PASSTHROUGH_31D` |
| 2 | **Low retention** — incoming funds are not held | **DIRECT** | average 7-day balance vs 7-day credit amount | `MG_RETENTION_RATIO` |
| 3 | **Dormant reactivation / burst** — an account that was quiet accelerates | **DIRECT** | share of 31-day activity landing in the last 7 days, against the 7/31 = 0.226 steady-state baseline | `MG_BURST_7_31`, `MG_BURST_AMT_7_31` |
| 4 | **Rail fragmentation / layering across channels** | **DIRECT** | material debit activity across 11 rails in 7 days | `MG_RAIL_FRAGMENTATION` |
| 5 | **Cash-out pressure** — value converted to cash quickly | **DIRECT** | cash + ATM debit over 7-day credit | `MG_CASHOUT_PRESSURE` |
| 6 | **Balance drain / throughput beyond capacity** | **DIRECT** | 7-day debit vs average 7-day balance | `MG_BALANCE_DRAIN` |
| 7 | **New-account abuse** — a young account moving large electronic value | **DIRECT** | electronic + UPI debit scaled by account tenure bucket | `MG_NEW_ACCOUNT_ACTIVITY` |
| 8 | **Peer-group anomaly** — behaviour unlike comparable customers | **DIRECT** | bank-supplied balance deviations against occupation and segment peer groups | `MG_PROFILE_MISMATCH` |
| 9 | **Multi-signal convergence** — several independent alert types fire together | **DIRECT** | 11 pre-decision alert-description flags, all resolved | `MG_ALERT_CONVERGENCE` |
| 10 | **Odd-hour activity** | **DIRECT** | night alerts as a share of all alerts | `MG_ODD_HOUR_ALERT_RATIO` |
| 11 | **Fan-in — funds received from many users** | **PROXY** | the alert flag `RCVING_FUNDS_FROM_MULITPLE_USERS` fires at account level; the counterparties themselves are not in the file | `MG_ALERT_CONVERGENCE` component |
| 12 | **Fan-out — one-to-many disbursement** | **PROXY** | `ONE_TO_MANY_UPI_PAYMENTS`, `MULTI_DBS_FROM_ACCOUNT`, `MULTI_UPI_DB_TXNS` fire at account level | `MG_ALERT_CONVERGENCE` component |
| 13 | **Credential-takeover pattern** — password change followed by large transfers | **PROXY** | `PWD_CHANGED_LARGE_FUND_XFERS` flag | alert flag |
| 14 | **Cross-border / risky-jurisdiction exposure** | **PROXY** | `RISKY_COUNTRY_TXNS` flag | alert flag |
| 15 | **Payment-gateway abuse** | **PROXY** | `MULTI_PG_TXNS`, `FAILED_UPI_TXNS` flags | alert flag |
| 16 | **Merchant legitimacy (exculpatory)** — a real business, not a mule | **DIRECT** | POS activity, GST activity, tenure ≥ 3, healthy retention | `MG_MERCHANT_LEGITIMACY` |
| 17 | **Smurfing ring / structuring across a group of accounts** | **NOT_AVAILABLE** | requires linking multiple accounts; there is no shared identifier and no counterparty column | — |
| 18 | **Mule network / community detection** | **NOT_AVAILABLE** | requires an edge list | Graph Adapter, edge-file only (§18) |
| 19 | **Multi-hop layering chain (A → B → C)** | **NOT_AVAILABLE** | requires directed transaction records | Graph Adapter, edge-file only |
| 20 | **Temporal sequence / velocity between individual transactions** | **NOT_AVAILABLE** | the file is pre-aggregated over L7D/L14D/L31D windows; individual transactions and their ordering are not present | — |
| 21 | **Device / IP / geolocation clustering** | **NOT_AVAILABLE** | no device, IP or location columns | — |
| 22 | **Out-of-time drift patterns** | **NOT_AVAILABLE** | the snapshot-month column `F2230` deterministically separates the classes and is quarantined; see `HIDDEN_VALIDATION_STRATEGY.md` §4 | — |

**Summary: 11 DIRECT, 5 PROXY, 6 NOT_AVAILABLE.**

---

## 4. Measured coverage

Every meta-feature was checked for actual non-null coverage on the full
9,082-row extract, rather than assumed to resolve:

| Meta-feature | Non-null coverage |
|---|---:|
| `MG_PASSTHROUGH_7D` | 1.0000 |
| `MG_PASSTHROUGH_31D` | 1.0000 |
| `MG_RETENTION_RATIO` | 1.0000 |
| `MG_BURST_7_31` | 1.0000 |
| `MG_BURST_AMT_7_31` | 1.0000 |
| `MG_RAIL_FRAGMENTATION` | 1.0000 |
| `MG_CASHOUT_PRESSURE` | 1.0000 |
| `MG_BALANCE_DRAIN` | 1.0000 |
| `MG_NEW_ACCOUNT_ACTIVITY` | 1.0000 |
| `MG_PROFILE_MISMATCH` | 1.0000 |
| `MG_ALERT_CONVERGENCE` | 1.0000 |
| `MG_ODD_HOUR_ALERT_RATIO` | 1.0000 |
| `MG_MERCHANT_LEGITIMACY` | 1.0000 |

All 11 rails resolve (`UPI`, `ELEC_XFER`, `NET_BNKING`, `MBNKING`, `CASH`,
`ATM`, `POS_PYMT`, `BBPS`, `NON_CASH_CHQ`, `CHQ`, `GST`) and all 11 alert flags
resolve (`HIGH_VALUE_UPI_DB_TXNS`, `MULTI_DBS_FROM_ACCOUNT`, `MULTI_PG_TXNS`,
`PWD_CHANGED_LARGE_FUND_XFERS`, `RCVING_FUNDS_FROM_MULITPLE_USERS`,
`RISKY_COUNTRY_TXNS`, `STATUS_CHANGE_AFTER_WD`, `TXN_AT_UNUSUAL_TIME`,
`MULTI_UPI_DB_TXNS`, `FAILED_UPI_TXNS`, `ONE_TO_MANY_UPI_PAYMENTS`).

Coverage is not an assumption anywhere in the pipeline: `MetaFeatureBuilder`
resolves inputs by *variable name* through the registry, and a meta-feature whose
inputs are absent **degrades to null, not to zero**. An uploaded validation file
with a different schema therefore produces honest missingness rather than a fake
0 that the model would read as a real measurement.

---

## 5. Which of these survived selection

Three meta-features entered the champion's 120:

| Feature | Typology it encodes |
|---|---|
| `MG_PASSTHROUGH_7D` | #1 — pass-through intensity, the defining mule behaviour |
| `MG_RAIL_FRAGMENTATION` | #4 — layering spread across payment rails |
| `MG_ALERT_CONVERGENCE` | #9, and by extension the fan-in / fan-out proxies #11–#12 |

They were not hand-placed. Stability selection, refitted independently inside
every training fold, chose them out of a pool of 3,925 candidates. The fact that
the three survivors are the three most domain-defensible of the thirteen is a
result, not a design.

The other ten remain computed and available in the ProofGraph as context, and are
retained for the Merchant Legitimacy Verifier (`MG_MERCHANT_LEGITIMACY`, view
`E_profile_merchant`) and for future feature views.

---

## 6. Discipline: 13 meta-features, not 300

The set is deliberately small. Every one of them:

- is built **only** from firewall-admitted columns, so a meta-feature cannot
  smuggle post-resolution information back in through the side door;
- resolves inputs by variable name through the registry, so the code reads as
  banking logic (`UPI_AMT_DB_L7D`) while the matrix stays keyed on F-numbers;
- is computable at inference **from a single account row** — no group statistics,
  no target, no cross-row aggregation, therefore nothing to fit inside a fold and
  nothing that can leak between train and validation;
- carries a plain-language meaning defensible from the data dictionary alone,
  because that string is what an analyst reads in the ProofGraph.

Generating hundreds of automatic ratios would almost certainly have raised the
local OOF number. It would also have produced features nobody could explain in a
review, and — given 81 positives — a large fraction of them would be noise that
happened to fit. Addendum UPDATE 7 rules out that class of complexity
explicitly.

---

## 7. How patterns appear in an alert

`build_proofgraph(..., patterns=[...])` emits `PATTERN` nodes, each carrying:

- a `source` (`pattern_card`) — no node without provenance is ever serialised,
- `supporting_features` — the columns that fired it, which are then linked by
  `SUPPORTS` edges to the corresponding prosecution nodes,
- a confidence weight.

A reviewer therefore sees the typology name *and* the columns behind it, and can
follow the edge from the pattern down to the individual feature values. A
pattern that cannot name its supporting features does not render.

---

## 8. The honest bottom line

Six of the twenty-two typologies in this matrix are **not detectable** from the
supplied data, and four of those six are the ones most commonly shown in mule
detection demos — networks, chains, rings, and transaction-level velocity.

We publish them as `NOT_AVAILABLE`. The Transaction Graph Adapter (§18) exists so
that the moment a real edge file is provided, those rows become live — and until
one is, the graph views stay empty rather than being filled with relationships
inferred from aggregate columns.
