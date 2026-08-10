# Trinetra Dual-Evidence ProofGraph — Design

> **Every alert must prove why it was raised — and show why it might be wrong.**

Master prompt §17. Implementation: `src/muleguard/explain/proofgraph.py`,
served at `GET /v1/proofgraph/{case_id}` and `GET /v1/proofgraph/{case_id}/twin`,
rendered by `frontend/src/components/ProofGraph.tsx`.

---

## 1. The problem this exists to solve

A standard SHAP panel answers exactly one question: *what pushed this score up?*
It is a prosecution brief. The reviewer receives a ranked list of incriminating
numbers and is given nothing to argue back with — no column that lowered the
score, no note that the model families disagreed, no indication that four of the
six deciding inputs were missing for this account.

That is not a small aesthetic complaint. In a portfolio where prevalence is
**0.89 %**, most alerts a reviewer opens are false positives. A reviewer who is
handed only incriminating evidence, on a queue that is mostly wrong, learns to
stop reading and start rubber-stamping. The explanation format is itself a
false-positive control.

So the ProofGraph never produces a one-sided artefact. It assembles two sides
from the **same verified model output** and lets them be weighed against each
other in the open.

---

## 2. Structure

Nine node types, each of which must name its origin:

| Node type | What it asserts | Where its `source` points |
|---|---|---|
| `ACCOUNT` | the case root: calibrated risk and review tier | `account_reference` |
| `EVIDENCE_FOR` | a feature whose SHAP contribution **raised** the score | a dataset column (`F1702`, `MG_PASSTHROUGH_7D`) |
| `EVIDENCE_AGAINST` | a feature whose SHAP contribution **lowered** it | a dataset column |
| `UNCERTAINTY` | a structural doubt the pipeline measured | a pipeline metric (`model_agreement`, `conformal_status`, `ood_status`, `input_missingness`) |
| `MODEL_VOTE` | one family's independent probability | `raw_scores.<family>` |
| `PATTERN` | a matched behavioural pattern card | `pattern_card` |
| `COUNTERFACTUAL` | model sensitivity: "if this feature were typical…" | the column being moved |
| `COUNTERFACTUAL_TWIN` | the closest **real** non-escalated account | `counterfactual_twin` |
| `DECISION` | the courtroom recommendation | `action.policy` |

Edges are typed, not decorative: `RAISED_BY`, `MITIGATED_BY`, `QUALIFIED_BY`,
`SCORED_BY`, `MATCHES_PATTERN`, `SUPPORTS`, `WOULD_CHANGE_IF`, `COMPARED_WITH`,
`RESULTS_IN`. The relation is what tells a reviewer whether a node is an
accusation, a mitigation, or a caveat, so it is carried in the data rather than
in the styling.

### A live example

`GET /v1/proofgraph/CASE-18A744455E` — calibrated risk 1.000, tier
`CRITICAL_REVIEW`:

```
13 nodes, 12 edges
evidence_counts: { prosecution: 5, defence: 1, uncertainty: 1 }
disagreement:    MODEL_CONSENSUS
courtroom:       verdict REVIEW_RECOMMENDED, evidence_balance 0.9156
```

Note what happened there. Evidence balance is 0.92 — overwhelmingly
prosecution — and the risk is 1.000, and the verdict is still the *milder*
`REVIEW_RECOMMENDED` rather than `ENHANCED_REVIEW_RECOMMENDED`. One uncertainty
node was present, which sets `contested`, and a contested case does not get the
stronger recommendation regardless of how lopsided the evidence looks. That is
§17 working: the doubt is not decoration, it changes the output.

---

## 3. The defence side is not a courtesy — it is computed

Two sources feed it.

**Signed SHAP.** `_evidence_nodes` splits the verified reason list on
`direction`. Features with `INCREASES_RISK` become prosecution; everything else
becomes defence. Both sides carry the same payload — column, registry variable
name, description, the account's value, the legitimate-cohort median, and the
percentile of that value within the legitimate cohort — so a defence node is
exactly as checkable as a prosecution node.

**Structural doubt** (`_structural_doubt_nodes`), promoted from small print to
first-class evidence:

| Condition | Node | Why it belongs on the defence side |
|---|---|---|
| `model_agreement < 0.85` | `doubt:model_agreement` | the boosting families do not converge on this account |
| `conformal_status` not a singleton | `doubt:conformal` | at the configured confidence level, the prediction set does not exclude the legitimate label |
| `ood_status != IN_DISTRIBUTION` | `doubt:ood` | the score describes an input unlike anything in training |
| `anomaly_percentile < 80` | `doubt:anomaly` | a detector that never saw the labels finds this account unremarkable |
| `verifier_confirms_risk is False` | `doubt:verifier` | the hard-negative verifier declines to confirm |
| ≥ ⅓ of deciding features null | `doubt:missing_inputs` | the case rests on less data than it appears to |

The last one is the one reviewers ask for once and then never stop wanting. If
five of the six features behind a `CRITICAL_REVIEW` are absent for this account,
the alert is materially weaker than the risk number suggests, and nothing in a
conventional SHAP panel would ever say so.

Structural doubts are split by type when the courtroom weighs them:
`EVIDENCE_AGAINST` ones (verifier, anomaly) are genuine defence; `UNCERTAINTY`
ones (agreement, conformal, OOD, missingness) are caveats. Both reach the
courtroom — an earlier revision kept only the `UNCERTAINTY` type and silently
dropped the strongest defence points.

---

## 4. The Model Courtroom (addendum UPDATE 10)

`model_courtroom()` is pure and deterministic. It reads nodes that were already
built from verified pipeline output and arranges them into two arguments:

```
p_strength = Σ weight(prosecution)
d_strength = Σ weight(defence) + Σ weight(uncertainty)
balance    = p_strength / (p_strength + d_strength)
contested  = disagreement == MODEL_DISAGREEMENT  or  any uncertainty node exists
```

| Condition | Verdict |
|---|---|
| tier `MONITOR`/`NO_ACTION`, or risk < 0.2 | `MONITOR_ONLY` (or `NO_ACTION_INDICATED` below 0.05) |
| no prosecution evidence | `INSUFFICIENT_EVIDENCE_TO_ESCALATE` |
| balance ≥ 0.65 **and not contested** | `ENHANCED_REVIEW_RECOMMENDED` |
| balance ≥ 0.50 | `REVIEW_RECOMMENDED` |
| otherwise | `INSUFFICIENT_EVIDENCE_TO_ESCALATE` |

Every verdict names **work for a person to do**. None of them names what the
account holder is. That is not phrasing preference — `assert_language_safe`
walks the entire serialised payload and raises `UnsafeLanguage` on
`guilty`, `criminal`, `fraudster`, `permanently_safe`, `certified_clean`,
`auto_freeze`, `confirmed mule`, `proven mule`. A graph containing any of them
never reaches the frontend.

**UPDATE 10 compliance is architectural, not promised.** The bullets in
`prosecution` and `defence` are computed here, in Python, before any language
model is consulted. Ollama may rephrase these strings for an analyst; it cannot
add a bullet, because there is no code path by which a bullet enters the list
other than this function. `evidence_policy` states it in the payload itself:

> Both columns are built only from verified model output. No language model
> contributed evidence to either side.

---

## 5. The counterfactual twin

`TwinIndex` holds **real legitimate development rows** — never a synthesised
point. The question a reviewer actually has is *"is this account unusual, or does
half my book look like this?"*, and the only answer that survives challenge is a
real account they can go and look at.

- Distance is computed **only over the deciding features** (`focus_idx`), not
  over all 3,924 columns, where the signal would be buried in noise.
- Per-feature scale is the legitimate cohort's IQR, so a feature measured in
  rupees does not dominate one measured in counts.
- A missing-vs-present mismatch costs one scale unit rather than zero, so
  "this account has no data where the twin does" is a difference, not a match.
- Row labels are opaque references. No customer identifier enters the index.

`twin_differences()` then returns the top-k columns where the case departs from
its twin, each with the registry description attached. The reviewer's sentence
becomes: *"here is an account that looks like this one and was not escalated —
and here are the three numbers where they part company."*

---

## 6. What the graph refuses to do

| Rule | Enforcement |
|---|---|
| No fabricated evidence | `assert_evidence_traceable` raises `UntraceableEvidence` on any node without a `source`, and on any edge referencing a node that does not exist |
| No invented relationships (UPDATE 8) | nodes describe **one account's own aggregate features**. Counterparty edges exist only when a real edge file was uploaded, and that lives in `muleguard.graph.adapter`, not here |
| Disagreement is uncertainty, never risk (UPDATE 6) | `disagreement_profile` returns families, mean, std, rank_std, max−min and a status — and carries its own `interpretation` string into the payload: *"It widens the review band; it does not raise the risk score."* No code path lets dispersion increase `calibrated_risk` |
| No verdict language | `assert_language_safe`, run on every graph before return |
| No LLM in the loop | `build_proofgraph` calls no model, no network and no LLM. It only re-presents verified numbers as a structure that can be argued with |

`provenance_statement` ships in every response:

> Every node in this graph is derived from a named dataset column or a named
> pipeline metric. No relationship, counterparty or value has been inferred or
> generated.

---

## 7. Why the champion had to be explainable

This is where §17 collided with the model tournament, and §17 won.

`tabpfn_top_60` scored **0.9110 ± 0.0044** OOF against the champion's
**0.7690 ± 0.0266** — a large, verified, leakage-free margin (see
`UPGRADE_GAP_ANALYSIS.md` §3.1.1). It was **not promoted**. TabPFN has no
faithful per-feature attribution path, so the only way to build a ProofGraph for
it is occlusion: one forward pass per feature. At a measured 438 s per
`predict_proba` call, that is **≈ 7.4 hours per case**.

Borrowing the champion's SHAP values to explain TabPFN's score was considered and
rejected: rank correlation between the two models' scores is Spearman 0.222, so
the resulting graph would be a plausible-looking explanation of a decision that
was made for other reasons. That is fabricated evidence with extra steps, and
rule 1 of this module forbids it.

The trade was made explicitly: **a served alert that cannot be proved is worth
less than a slightly lower-scoring one that can.**

---

## 8. Frontend contract

The graph renders on white, with black text and light-grey borders. Prosecution
and defence sit in two columns of equal visual weight — the defence column is not
collapsed, greyed, or placed below the fold, because a defence a reviewer has to
click to see is a defence that does not get read.

Node `type` drives layout; `source` is displayed verbatim next to every claim, so
the column that produced any assertion on screen can be looked up in the feature
registry without leaving the page.

---

## 9. Tests

| Test | Guards |
|---|---|
| `tests/unit/test_proofgraph.py` | node/edge construction, traceability failure on a sourceless node, forbidden-term rejection, disagreement statuses, courtroom verdict boundaries |
| `tests/unit/test_evidence_packet.py` | the packet's `source` fields resolve to real columns or metrics |
| `tests/integration/test_proofgraph_route.py` | `/v1/proofgraph/{case_id}` returns both sides populated and the provenance statement |
| release gate `no_verdict_language` | scans served payloads for the forbidden vocabulary |

---

## 10. Summary for a judge

Three claims, each mechanically enforced rather than asserted:

1. **Every alert proves why it was raised** — signed SHAP over firewall-admitted
   columns, each node naming its exact source column, with the legitimate-cohort
   median and percentile beside the account's own value.
2. **Every alert shows why it might be wrong** — a defence column of equal
   standing, fed by both negative SHAP and six measured structural doubts, plus a
   real non-escalated twin account to compare against.
3. **Nothing in it was generated** — traceability and language guardrails run
   before serialisation, the courtroom is deterministic Python, and the scoring
   path works with Ollama stopped.
