# Transaction Graph Adapter Report

Master prompt §18, addendum UPDATE 8.

Implementation: `src/muleguard/graph/adapter.py` (407 lines) ·
Routes: `POST /v1/graph/edges`, `GET /v1/graph/status`,
`GET /v1/graph/account/{account}`, `GET /v1/graph/neighbourhood/{account}` ·
Frontend: Graph Lab.

**Current state: `UNAVAILABLE`. That is the correct state and the default one.**

---

## 1. The temptation, and the refusal

The competition data is one row per account with 3,924 aggregate columns. There
is no sender, no receiver, and no transaction. A graph therefore cannot be built
from it.

The single most tempting mistake in this problem is to build one anyway — to
treat "accounts with similar UPI profiles" as an edge, run a community detection
algorithm, and produce a network diagram that looks like intelligence. It demos
beautifully. It encodes nothing except correlation between aggregate features,
and every structure visible in it is an artefact of the clustering rather than
evidence about an account.

Addendum UPDATE 8 forbids it in exactly those words:

> Do NOT derive fake sender/receiver relationships from F-columns.

This module is the shape of that refusal. **Edges come from an uploaded edge
list, or they do not exist.**

When no edge file has been supplied, every function reports:

```json
{
  "status": "UNAVAILABLE",
  "reason": "no transaction edge file has been supplied. The competition dataset
             is one aggregate row per account and contains no sender, receiver or
             transaction, so no graph can be derived from it.",
  "what_we_refuse_to_do": "MuleGuard does not fabricate edges from feature
             similarity between F-columns. A graph built that way encodes
             correlation between aggregates, not money movement, and every
             structure visible in it would be an artefact of the clustering
             rather than evidence about an account.",
  "how_to_enable": "POST an edge file to /v1/graph/edges with columns
             account_from, account_to, amount, timestamp."
}
```

The UI renders that message rather than an empty canvas or a placeholder
network. A judge who opens Graph Lab is told, in the product, why it is empty.

---

## 2. The isolation guarantee

> **These metrics never enter the mule model.**

The champion was fitted without any graph feature and is scored without one. A
case's calibrated risk is **identical** whether or not an edge file was ever
uploaded.

This is not a limitation, it is the design. If graph features fed the score,
there would be two different models depending on what a judge happened to bring
to the demo, and no way to compare a scored case across sessions. Graph findings
are **corroborating evidence displayed beside a score computed independently**.

Consequence worth stating plainly: uploading an edge file cannot improve our
hidden-validation score, and cannot damage it either.

---

## 3. The contract

Four columns:

```
account_from, account_to, amount, timestamp
```

An edge file is something a judge exports from whatever system they have, not a
format we get to dictate, so aliases are accepted:

| Canonical | Accepted aliases |
|---|---|
| `account_from` | `from`, `sender`, `source`, `src`, `payer`, `from_account`, `debit_account`, `originator` |
| `account_to` | `to`, `receiver`, `target`, `dst`, `payee`, `to_account`, `credit_account`, `beneficiary` |
| `amount` | `amt`, `value`, `txn_amount`, `transaction_amount` |
| `timestamp` | `time`, `datetime`, `txn_time`, `transaction_time`, `date`, `value_date` |

### Cleaning, with every drop counted

`load_edges` reports what it removed rather than silently improving the file:

| Reported | Why it matters |
|---|---|
| `n_self_loops_dropped` | a self-loop is an internal transfer between a customer's own products, not a counterparty relationship. Letting it through inflates fan-in on exactly the accounts that legitimately move money between their own accounts |
| `n_non_positive_amounts_dropped` | reversals and zero-value records are not money movement |
| `n_unparseable_timestamps` | those edges are **kept for structure** and **excluded from timing metrics** — a chain needs ordering, a fan-in count does not |
| `n_edges_supplied` vs `n_edges_used` | the gap is visible, not absorbed |

Timestamps are parsed best-effort and the failure is counted, not hidden. If no
usable edges remain, `EdgeSchemaError` is raised rather than an empty graph being
presented as a finding.

---

## 4. Per-account metrics

| Metric | Definition |
|---|---|
| `fan_in_counterparties` | distinct accounts paying in |
| `fan_out_counterparties` | distinct accounts paid out to |
| `value_in` / `value_out` / `net_value` | totals by value |
| `passthrough_ratio` | `min(value_in, value_out) / value_in` |
| `median_dwell_hours` | median hours between money arriving and the next outgoing transfer |
| `fan_in_fan_out_ratio` | the shape of the funnel |

Pass-through is measured **by value, not by count**, because ten small deposits
forwarded as one large transfer is the pattern, not the exception. A count-based
ratio would read that as 10:1 and miss it entirely.

---

## 5. Patterns, with thresholds fixed in advance

`GRAPH_THRESHOLDS` is defined in the module **before any edge file was seen**, so
a pattern cannot be defined into existence after looking at the data:

```python
fan_in_min = 10                 # counterparties paying in
fan_out_min = 10                # counterparties paid out to
passthrough_ratio_min = 0.90    # forwarded / received, by value
passthrough_window_hours = 48   # how quickly it must leave to count
rapid_hops_max_hours = 24       # hop-to-hop delay within a chain
chain_depth_min = 3             # hops before a path is a layering chain
```

| Pattern | Meaning |
|---|---|
| `FAN_IN_COLLECTION` | many distinct parties pay into this one account |
| `FAN_OUT_DISPERSAL` | this account disperses funds to many distinct parties |
| `RAPID_PASSTHROUGH` | almost everything received leaves again quickly; the account holds little of what flows through it |
| `LAYERING_CHAIN` | funds move through a chain of accounts with little delay at each hop |
| `CIRCULAR_FLOW` | value returns to an account it previously left |

**Every pattern returns the values that triggered it** — the counterparty count,
the observed ratio, the dwell hours, the actual path, and the threshold it was
compared against. A card that says "fan-in detected" without saying from how many
counterparties is not evidence, it is an assertion.

These are deliberately the classic mule-network structures, because the whole
point of an edge file is to see what the aggregates cannot: an account that
receives from forty parties and forwards to one, within hours, is a mule
signature no per-account column captures.

---

## 6. Bounded by construction

An uploaded file is untrusted input and the API must survive it.

| Guard | Bound | Reason |
|---|---|---|
| `longest_rapid_chain` | `max_depth = 8` | an unbounded DFS on a dense file is an easy way to hang the API; a chain deeper than eight hops adds nothing an analyst will read |
| `longest_rapid_chain` | `seen_states` on `(node, depth)` | prevents exponential re-exploration |
| `find_cycle` | `max_len = 6`, BFS | returns the *shortest* cycle, which is the readable one |
| `neighbourhood` | `max_nodes = 60`, and **the truncation is reported** | a 5,000-node hairball is not a visualisation |

Adjacency is plain dicts rather than a graph library: the only operations needed
are neighbour lookups and a bounded walk, and adding a dependency for that would
be a dependency a judge has to install offline.

---

## 7. What we deliberately did not build (UPDATE 7)

The addendum names a set of techniques not to copy. None of them are here:

| Not built | Why |
|---|---|
| mock GraphSAGE / GNN embeddings | there is no graph to embed, and a GNN over fabricated edges is fabrication with a neural network on top |
| persistent homology on the transaction graph | no edge file, and no reviewer could act on the output |
| community detection over feature similarity | this is precisely the forbidden move |
| synthetic trajectory generation | invented data presented as evidence |
| automatic fund freezing on a graph pattern | no automatic freezing anywhere in this system; high-impact actions require human approval |

---

## 8. Tests

| Test | Guards |
|---|---|
| `tests/unit/test_graph_adapter.py` | column aliasing, self-loop and non-positive-amount removal, fan-in/fan-out, pass-through by value, dwell computation, chain depth bound, cycle detection, neighbourhood truncation |
| `tests/integration/test_graph_routes.py` | `UNAVAILABLE` is the default response and carries the refusal text; upload → status → account → neighbourhood round-trip |
| release gate | asserts no graph feature appears in the champion's feature list |

---

## 9. Summary

The adapter is complete, tested, and **switched off**, because the data required
to switch it on does not exist in this competition's extract.

That is the deliverable. §18 marks the graph adapter optional; UPDATE 8 makes
faking it disqualifying. The honest implementation of an optional feature whose
inputs are absent is a working module that reports `UNAVAILABLE` and explains
what it would need — not a demo built on invented relationships.

If the organiser supplies an edge file, every row marked `NOT_AVAILABLE` in
`PATTERN_AVAILABILITY_MATRIX.md` §3 (#18 mule networks, #19 layering chains,
#17 smurfing rings) becomes live in one upload, with no model change and no
retraining.
