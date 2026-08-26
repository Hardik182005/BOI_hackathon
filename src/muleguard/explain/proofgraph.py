"""Trinetra Dual-Evidence ProofGraph (master prompt section 17).

    Every alert must prove why it was raised - and show why it might be wrong.

A one-sided explanation is an advocate, not an analyst. Standard SHAP output
answers "what pushed this score up"; it never answers "what should make me
doubt this", so a reviewer reads a ranked list of incriminating numbers and has
nothing to argue back with. Reviewers who cannot argue back stop reviewing.

The ProofGraph therefore always assembles two sides from the *same* verified
model output:

* **prosecution** - features whose SHAP contribution raised the score,
* **defence** - features whose SHAP contribution lowered it, plus the
  structural doubts the pipeline already measured: model disagreement, an
  ambiguous conformal set, out-of-distribution input, an unremarkable anomaly
  percentile, missing inputs, and a hard-negative verifier that declines to
  confirm.

Rules this module enforces mechanically, not by convention:

1. **No fabricated evidence.** Every node carries a ``source`` naming the exact
   column or the exact pipeline metric it came from, and
   :func:`assert_evidence_traceable` refuses to emit a graph containing a node
   that cannot name its origin. An LLM may narrate this structure; it may never
   add to it (addendum UPDATE 10).
2. **No transaction relationships are invented.** Nodes describe one account's
   own aggregate features. Counterparty edges appear only when a real edge file
   was uploaded, and that lives in the graph adapter, not here (UPDATE 8).
3. **Disagreement is uncertainty, never risk.** Where the model families
   disagree the graph says so on the defence side. It does not raise the score
   (UPDATE 6).
4. **No verdict language.** ``assert_language_safe`` rejects accusatory or
   absolute terms anywhere in the serialised graph.
5. **No quarantined column, on either side.** A feature the Feature
   Availability Firewall has quarantined is not admissible evidence, and being
   traceable does not make it admissible - a post-resolution column like
   ``MIN_RESOLVE_DAYS`` traces perfectly to a real dataset field and is still a
   fact about the investigation rather than about the account.
   :func:`assert_evidence_admissible` runs the firewall over every token the
   graph names, so the quarantine list and the evidence boundary refuse the
   same set by construction rather than by two lists agreeing.
"""
from __future__ import annotations

import datetime as dt
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import numpy as np

from muleguard.features import firewall
from muleguard.logging import get_logger
from muleguard.usp.control_attribution import (
    RELATION as CONTROL_RELATION, control_attribution, proofgraph_node)

log = get_logger("explain.proofgraph")

# --------------------------------------------------------------------------
# vocabulary
# --------------------------------------------------------------------------

NODE_ACCOUNT = "ACCOUNT"
NODE_EVIDENCE_FOR = "EVIDENCE_FOR"
NODE_EVIDENCE_AGAINST = "EVIDENCE_AGAINST"
NODE_PATTERN = "PATTERN"
NODE_MODEL_VOTE = "MODEL_VOTE"
NODE_UNCERTAINTY = "UNCERTAINTY"
NODE_COUNTERFACTUAL = "COUNTERFACTUAL"
NODE_TWIN = "COUNTERFACTUAL_TWIN"
NODE_DECISION = "DECISION"
NODE_CONTROL = "CONTROL_ATTRIBUTION"

# Terms the system must never emit. A behavioural model ranks accounts for
# human review; it does not establish criminality and cannot certify innocence.
FORBIDDEN_TERMS = (
    "guilty", "criminal", "fraudster", "permanently_safe", "permanently safe",
    "certified_clean", "certified clean", "auto_freeze", "auto freeze",
    "confirmed mule", "proven mule",
)

# Verdicts the courtroom is allowed to return. All of them describe an action
# for a person to take, none of them describe what the account holder is.
VERDICT_REVIEW = "REVIEW_RECOMMENDED"
VERDICT_ENHANCED = "ENHANCED_REVIEW_RECOMMENDED"
VERDICT_MONITOR = "MONITOR_ONLY"
VERDICT_INSUFFICIENT = "INSUFFICIENT_EVIDENCE_TO_ESCALATE"
VERDICT_NO_ACTION = "NO_ACTION_INDICATED"


class UntraceableEvidence(RuntimeError):
    """Raised when a node cannot name the column or metric it came from."""


class UnsafeLanguage(RuntimeError):
    """Raised when a forbidden term reaches the serialised graph."""


class InadmissibleEvidence(RuntimeError):
    """Raised when a quarantined column reaches either side of the graph."""


# --------------------------------------------------------------------------
# graph primitives
# --------------------------------------------------------------------------


@dataclass
class Node:
    """One assertion in the graph, with its provenance attached.

    ``source`` is mandatory and is the whole point of the structure: it is
    either a dataset column name (``F1702``, ``MG_PASSTHROUGH_L7D``) or the
    name of a pipeline metric (``model_agreement``, ``conformal_status``).
    A node whose source cannot be resolved never reaches the frontend.
    """

    id: str
    type: str
    label: str
    source: str
    detail: str = ""
    weight: float = 0.0
    value: Any = None
    reference: Any = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["weight"] = round(float(self.weight), 6)
        return d


@dataclass
class Edge:
    source: str
    target: str
    relation: str
    weight: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"source": self.source, "target": self.target,
                "relation": self.relation, "weight": round(float(self.weight), 6)}


def _fmt(v: Any, nd: int = 3) -> str:
    if v is None:
        return "not available"
    if isinstance(v, float):
        if math.isnan(v):
            return "not available"
        return f"{v:,.{nd}f}"
    return str(v)


# --------------------------------------------------------------------------
# counterfactual twin
# --------------------------------------------------------------------------


@dataclass
class TwinIndex:
    """Real legitimate development rows, used to answer "who looks like this?".

    The twin is deliberately a *real* account from the development cohort and
    never a synthesised point. "Here is an account with a similar profile that
    was not escalated, and here are the three numbers where they differ" is a
    statement a reviewer can check. A generated pseudo-account is not.
    """

    X: np.ndarray               # legitimate dev rows, model matrix order
    feature_names: list[str]
    row_labels: list[str]       # opaque references, never customer identifiers
    scale: np.ndarray           # per-feature IQR-ish scale for distance

    @classmethod
    def from_matrix(cls, X_dev: np.ndarray, y_dev: np.ndarray,
                    feature_names: Sequence[str],
                    row_labels: Sequence[str] | None = None,
                    max_rows: int = 4000, seed: int = 42) -> "TwinIndex":
        legit = np.asarray(X_dev)[np.asarray(y_dev) == 0]
        labels = ([str(x) for x in row_labels] if row_labels is not None
                  else [f"DEV-{i}" for i in range(len(np.asarray(y_dev)))])
        labels = [labels[i] for i, ok in enumerate(np.asarray(y_dev) == 0) if ok]
        if len(legit) > max_rows:  # cap the index so scoring stays interactive
            rng = np.random.default_rng(seed)
            pick = rng.choice(len(legit), size=max_rows, replace=False)
            legit, labels = legit[pick], [labels[i] for i in pick]
        with np.errstate(invalid="ignore"):
            q75 = np.nanpercentile(legit, 75, axis=0)
            q25 = np.nanpercentile(legit, 25, axis=0)
        scale = np.where(np.isfinite(q75 - q25) & (q75 - q25 > 0), q75 - q25, 1.0)
        return cls(X=legit, feature_names=list(feature_names),
                   row_labels=labels, scale=scale)

    def nearest(self, x_row: np.ndarray, focus_idx: Sequence[int],
                ) -> tuple[int, float] | None:
        """Nearest legitimate row, measured only on the deciding features.

        Distance over all 3,900 columns is dominated by noise; the comparison
        that means something to a reviewer is "similar on the features that
        actually drove this score".
        """
        idx = [j for j in focus_idx if 0 <= j < self.X.shape[1]]
        if not idx or len(self.X) == 0:
            return None
        A = self.X[:, idx]
        b = np.asarray(x_row)[idx]
        s = self.scale[idx]
        d = (A - b) / np.where(s == 0, 1.0, s)
        d = np.where(np.isfinite(d), d, 0.0)
        # count a missing-vs-present mismatch as one scale unit rather than zero
        miss = (np.isnan(A) != np.isnan(b)).astype(float)
        dist = np.sqrt((d ** 2).sum(axis=1) + miss.sum(axis=1))
        j = int(np.argmin(dist))
        return j, float(dist[j])


# --------------------------------------------------------------------------
# model disagreement (addendum UPDATE 6)
# --------------------------------------------------------------------------


def disagreement_profile(raw_scores: dict[str, float]) -> dict[str, Any]:
    """Per-family probabilities plus the four dispersion statistics.

    Addendum UPDATE 6 makes disagreement first-class *and* forbids treating it
    as incriminating. High dispersion means the families do not agree on what
    this account is - which is a reason to look harder, and never a reason to
    score it higher. ``interpretation`` says so in the payload itself so the
    property survives into the UI and the evidence packet.
    """
    names = sorted(raw_scores)
    vals = np.array([float(raw_scores[n]) for n in names], dtype=float)
    if len(vals) == 0:
        return {"families": {}, "status": "NO_MODELS", "interpretation": ""}
    ranks = np.argsort(np.argsort(vals))
    spread = float(vals.max() - vals.min())
    status = ("MODEL_CONSENSUS" if spread <= 0.15
              else "PARTIAL_AGREEMENT" if spread <= 0.35
              else "MODEL_DISAGREEMENT")
    return {
        "families": {n: round(float(v), 6) for n, v in zip(names, vals)},
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "rank_std": float(ranks.std()),
        "max_minus_min": spread,
        "status": status,
        "interpretation": (
            "Disagreement between model families is an uncertainty signal. It "
            "widens the review band; it does not raise the risk score."
        ),
    }


# --------------------------------------------------------------------------
# building the graph
# --------------------------------------------------------------------------


def _registry_lookup(feature: str, registry: dict[str, Any] | None) -> dict[str, Any]:
    """Resolve a column to its registry record, meta-features included.

    The persisted registry describes the 3,924 raw ``F####`` columns. The
    ``MG_*`` meta-features are attached in memory by the frame builder and are
    therefore absent from the file, so they are resolved from their own
    description table - otherwise the most interpretable features in the model
    would be the only ones rendering without an explanation.
    """
    if feature.startswith("MG_"):
        from muleguard.features.meta_features import META_DESCRIPTIONS
        desc = META_DESCRIPTIONS.get(feature)
        if desc:
            return {"variable_name": feature, "description": desc,
                    "availability_class": "BEHAVIORAL",
                    "feature_family": "DERIVED_META", "window": "NONE"}
    if not registry:
        return {}
    feats = registry.get("features", registry)
    rec = feats.get(feature) if isinstance(feats, dict) else None
    return rec or {}


def _evidence_nodes(reasons: list[dict[str, Any]], registry: dict[str, Any] | None,
                    ) -> tuple[list[Node], list[Node]]:
    """Split verified SHAP reasons into prosecution and defence nodes."""
    for_nodes: list[Node] = []
    against_nodes: list[Node] = []
    for i, r in enumerate(reasons):
        feat = r["feature"]
        rec = _registry_lookup(feat, registry)
        name = (r.get("verified_semantic_name") or rec.get("variable_name") or feat)
        desc = rec.get("description") or ""
        pct = r.get("legitimate_percentile")
        med = r.get("legitimate_cohort_median")
        shap = float(r.get("shap_contribution", 0.0))
        raises = r.get("direction") == "INCREASES_RISK"
        comparison = (
            f"value {_fmt(r.get('value'))} vs legitimate-cohort median "
            f"{_fmt(med)}"
            + (f" ({pct:.0f}th percentile of the legitimate cohort)"
               if isinstance(pct, (int, float)) else "")
        )
        node = Node(
            id=f"{'for' if raises else 'against'}:{feat}",
            type=NODE_EVIDENCE_FOR if raises else NODE_EVIDENCE_AGAINST,
            label=str(name),
            source=feat,
            detail=(f"{desc}. {comparison}." if desc else f"{comparison}."),
            weight=abs(shap),
            value=r.get("value"),
            reference=med,
            extra={
                "column": feat,
                "variable_name": rec.get("variable_name"),
                "description": desc,
                "availability_class": rec.get("availability_class"),
                "window": rec.get("window"),
                "feature_family": rec.get("feature_family"),
                "legitimate_percentile": pct,
                "shap_contribution": shap,
                "rank": i + 1,
            },
        )
        (for_nodes if raises else against_nodes).append(node)
    return for_nodes, against_nodes


def _structural_doubt_nodes(score: dict[str, Any]) -> list[Node]:
    """Doubts the pipeline already measured, promoted to first-class evidence.

    These numbers exist in every scoring response today but are rendered as
    small print. On the defence side of a ProofGraph they carry the weight they
    deserve: a MEDIUM-confidence alert on out-of-distribution input with an
    ambiguous conformal set is a materially weaker case than the risk score
    alone suggests.
    """
    out: list[Node] = []

    agree = score.get("model_agreement")
    if isinstance(agree, (int, float)) and agree < 0.85:
        out.append(Node(
            id="doubt:model_agreement", type=NODE_UNCERTAINTY,
            label="Model families disagree",
            source="model_agreement",
            detail=(f"Agreement {agree:.3f}: the boosting families do not "
                    "converge on this account. Treated as uncertainty, not as "
                    "additional risk."),
            weight=float(1.0 - agree), value=float(agree),
        ))

    conf = score.get("conformal_status")
    if conf and conf not in ("SINGLETON_HIGH", "SINGLETON_LOW"):
        out.append(Node(
            id="doubt:conformal", type=NODE_UNCERTAINTY,
            label="Conformal prediction set is not decisive",
            source="conformal_status",
            detail=(f"Status {conf}: at the configured confidence level the "
                    "prediction set does not exclude the legitimate label."),
            weight=0.5, value=conf,
        ))

    ood = score.get("ood_status")
    if ood and ood != "IN_DISTRIBUTION":
        out.append(Node(
            id="doubt:ood", type=NODE_UNCERTAINTY,
            label="Input differs from the training distribution",
            source="ood_status",
            detail=(f"Status {ood}. Scores on inputs unlike anything in "
                    "training are less reliable and the alert should be read "
                    "with that in mind."),
            weight=0.6, value=ood,
        ))

    anom = score.get("anomaly_percentile")
    if isinstance(anom, (int, float)) and anom < 80:
        out.append(Node(
            id="doubt:anomaly", type=NODE_EVIDENCE_AGAINST,
            label="Unremarkable against the unsupervised anomaly detector",
            source="anomaly_percentile",
            detail=(f"Isolation-forest percentile {anom:.0f}: the account does "
                    "not stand out to a detector that never saw the labels."),
            weight=float((80 - anom) / 100.0), value=float(anom),
        ))

    if score.get("verifier_confirms_risk") is False:
        vp = score.get("verifier_probability")
        out.append(Node(
            id="doubt:verifier", type=NODE_EVIDENCE_AGAINST,
            label="Hard-negative verifier does not confirm the alert",
            source="verifier_confirms_risk",
            detail=(f"The verifier, trained to separate true positives from "
                    f"look-alike legitimate accounts, returns "
                    f"{_fmt(vp)} and declines to confirm."),
            weight=0.7, value=False,
        ))
    return out


def _missingness_node(row_values: dict[str, Any], evidence_cols: Sequence[str]) -> Node | None:
    """Flag when the deciding features are themselves largely absent."""
    if not evidence_cols:
        return None
    missing = [c for c in evidence_cols
               if row_values.get(c) is None
               or (isinstance(row_values.get(c), float) and math.isnan(row_values[c]))]
    if len(missing) < max(2, len(evidence_cols) // 3):
        return None
    return Node(
        id="doubt:missing_inputs", type=NODE_UNCERTAINTY,
        label="Several deciding inputs are missing",
        source="input_missingness",
        detail=(f"{len(missing)} of {len(evidence_cols)} features behind this "
                f"score are absent for this account ({', '.join(missing[:4])}"
                f"{'...' if len(missing) > 4 else ''}). The model handles "
                "missing values natively, but the case rests on less data than "
                "it appears to."),
        weight=len(missing) / len(evidence_cols),
        value=len(missing),
    )


def build_proofgraph(
    *,
    account_reference: str,
    score: dict[str, Any],
    reasons: list[dict[str, Any]],
    registry: dict[str, Any] | None = None,
    row_values: dict[str, Any] | None = None,
    counterfactuals: list[dict[str, Any]] | None = None,
    twin: dict[str, Any] | None = None,
    patterns: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Assemble the dual-evidence graph for one scored account.

    Every argument is an output of the deterministic pipeline. Nothing here
    calls a model, a network, or an LLM: this function only re-presents
    verified numbers as a structure that can be argued with.
    """
    nodes: list[Node] = []
    edges: list[Edge] = []

    risk = float(score.get("calibrated_risk", 0.0))
    tier = score.get("risk_tier", "UNKNOWN")

    root = Node(
        id="account", type=NODE_ACCOUNT,
        label=f"Account {account_reference}",
        source="account_reference",
        detail=(f"Calibrated risk {risk:.3f}, review tier {tier}. This is a "
                "behavioural ranking for human review, not a finding about the "
                "account holder."),
        weight=risk, value=account_reference,
    )
    nodes.append(root)

    for_nodes, against_nodes = _evidence_nodes(reasons or [], registry)
    for n in for_nodes:
        nodes.append(n)
        edges.append(Edge("account", n.id, "RAISED_BY", n.weight))
    for n in against_nodes:
        nodes.append(n)
        edges.append(Edge("account", n.id, "MITIGATED_BY", n.weight))

    # Structural doubts carry two node types: some are genuine defence evidence
    # (the verifier declining to confirm), others are pure uncertainty (an
    # ambiguous conformal set). Both must reach the courtroom - keeping only
    # the UNCERTAINTY ones silently dropped the strongest defence points.
    structural = _structural_doubt_nodes(score)
    if row_values:
        mn = _missingness_node(row_values, [r["feature"] for r in (reasons or [])])
        if mn:
            structural.append(mn)
    for n in structural:
        nodes.append(n)
        edges.append(Edge("account", n.id, "QUALIFIED_BY", n.weight))

    # model votes (UPDATE 6)
    dis = disagreement_profile(score.get("raw_scores") or {})
    for fam, p in (dis.get("families") or {}).items():
        vid = f"vote:{fam}"
        nodes.append(Node(
            id=vid, type=NODE_MODEL_VOTE, label=f"{fam} probability",
            source=f"raw_scores.{fam}",
            detail=f"{fam} independently scores this account at {p:.4f}.",
            weight=float(p), value=float(p),
        ))
        edges.append(Edge("account", vid, "SCORED_BY", float(p)))

    # named behavioural patterns, if the caller matched any
    for p in patterns or []:
        pid = f"pattern:{p.get('id') or p.get('name')}"
        nodes.append(Node(
            id=pid, type=NODE_PATTERN, label=str(p.get("name", pid)),
            source=str(p.get("source", "pattern_card")),
            detail=str(p.get("detail", "")),
            weight=float(p.get("confidence", 0.0)),
            extra={"supporting_features": p.get("supporting_features", [])},
        ))
        edges.append(Edge("account", pid, "MATCHES_PATTERN",
                          float(p.get("confidence", 0.0))))
        for f in p.get("supporting_features", []):
            if any(n.id == f"for:{f}" for n in nodes):
                edges.append(Edge(f"for:{f}", pid, "SUPPORTS", 0.0))

    # counterfactual sensitivity (existing, kept verbatim in meaning)
    for i, c in enumerate(counterfactuals or []):
        cid = f"counterfactual:{c['feature']}"
        nodes.append(Node(
            id=cid, type=NODE_COUNTERFACTUAL,
            label=f"If {c['feature']} were typical",
            source=c["feature"],
            detail=(f"Moving {c['feature']} from {_fmt(c.get('from_value'))} to "
                    f"the legitimate-cohort median {_fmt(c.get('to_value'))} "
                    f"changes the model score from {_fmt(c.get('score_before'))} "
                    f"to {_fmt(c.get('score_after'))}"
                    + (" and crosses the review threshold."
                       if c.get("crosses_threshold") else ".")
                    + " Model sensitivity, not a causal claim."),
            weight=abs(float(c.get("score_before", 0)) - float(c.get("score_after", 0))),
            extra={"crosses_threshold": bool(c.get("crosses_threshold")), "rank": i + 1},
        ))
        edges.append(Edge("account", cid, "WOULD_CHANGE_IF", 0.0))

    if twin:
        nodes.append(Node(
            id="twin", type=NODE_TWIN,
            label=f"Closest non-escalated account: {twin.get('reference')}",
            source="counterfactual_twin",
            detail=(f"A real development-set account that was not escalated and "
                    f"matches on the deciding features (distance "
                    f"{_fmt(twin.get('distance'))}). The differences below are "
                    "where this case departs from it."),
            weight=float(twin.get("distance", 0.0)),
            extra={"differences": twin.get("differences", [])},
        ))
        edges.append(Edge("account", "twin", "COMPARED_WITH", 0.0))

    struct_against = [n for n in structural if n.type == NODE_EVIDENCE_AGAINST]
    struct_doubt = [n for n in structural if n.type == NODE_UNCERTAINTY]
    court = model_courtroom(for_nodes, against_nodes + struct_against,
                            struct_doubt, score, dis)

    nodes.append(Node(
        id="decision", type=NODE_DECISION, label=court["verdict"],
        source="action.policy",
        detail=court["verdict_rationale"],
        weight=risk, value=tier,
    ))
    edges.append(Edge("account", "decision", "RESULTS_IN", risk))

    # Account-control ambiguity (section 22). The edge runs decision ->
    # limitation by REQUIRES_HUMAN_VERIFICATION, not account -> limitation by
    # RAISED_BY: this node is not evidence that raised the score, it is a
    # condition on acting on the decision. It carries weight 0.0 and is excluded
    # from the courtroom tallies, so it cannot move a probability or a tier.
    control_card = control_attribution(
        risk_probability=risk, risk_tier=str(tier))
    control_node = proofgraph_node(control_card)
    nodes.append(Node(
        id=control_node["id"], type=control_node["type"],
        label=control_node["label"], source=control_node["source"],
        detail=control_node["detail"], weight=0.0,
        value=control_node["value"], extra=control_node["extra"]))
    edges.append(Edge("decision", control_node["id"],
                      CONTROL_RELATION, 0.0))

    graph = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "account_reference": account_reference,
        "model_version": score.get("model_version"),
        "calibrated_risk": risk,
        "risk_tier": tier,
        "nodes": [n.to_dict() for n in nodes],
        "edges": [e.to_dict() for e in edges],
        "disagreement": dis,
        "courtroom": court,
        "evidence_counts": {
            "prosecution": len(for_nodes),
            "defence": len(against_nodes) + len(struct_against),
            "uncertainty": len(struct_doubt),
        },
        "control_attribution": control_card,
        "provenance_statement": (
            "Every node in this graph is derived from a named dataset column or "
            "a named pipeline metric. No relationship, counterparty or value "
            "has been inferred or generated."
        ),
    }
    assert_evidence_traceable(graph)
    # Admissibility is asserted here, at the point of construction, and not only
    # at the point of serving. A graph is refused before it can be stored,
    # cached, exported or handed to a report writer - the 2026-08-26 incident
    # was a graph that had been built once and rendered for weeks afterwards, so
    # a check that only runs on the way out is a check that runs too late.
    assert_evidence_admissible(graph, registry)
    assert_language_safe(graph)
    return graph


# --------------------------------------------------------------------------
# Model Courtroom (addendum UPDATE 10)
# --------------------------------------------------------------------------


def model_courtroom(for_nodes: list[Node], against_nodes: list[Node],
                    doubt_nodes: list[Node], score: dict[str, Any],
                    disagreement: dict[str, Any]) -> dict[str, Any]:
    """PROSECUTION / DEFENCE / VERDICT over the graph's own evidence.

    Addendum UPDATE 10 is explicit that no LLM-created evidence is allowed
    here, so this function is pure and deterministic: it reads the nodes that
    were already built from verified pipeline output and arranges them into two
    arguments. A local model may later rephrase these strings for an analyst;
    it cannot add a bullet, because the bullets are computed here.

    The verdict is a recommendation about *work to be done*, never a
    characterisation of the customer.
    """
    risk = float(score.get("calibrated_risk", 0.0))
    tier = str(score.get("risk_tier", "UNKNOWN"))

    prosecution = [
        {"point": n.label, "because": n.detail, "source": n.source,
         "weight": round(n.weight, 6)}
        for n in sorted(for_nodes, key=lambda n: -n.weight)[:6]
    ]
    defence = [
        {"point": n.label, "because": n.detail, "source": n.source,
         "weight": round(n.weight, 6)}
        for n in sorted(against_nodes + doubt_nodes, key=lambda n: -n.weight)[:6]
    ]

    p_strength = sum(n.weight for n in for_nodes)
    d_strength = sum(n.weight for n in against_nodes) + sum(n.weight for n in doubt_nodes)
    balance = p_strength / (p_strength + d_strength) if (p_strength + d_strength) else 0.0

    contested = (disagreement.get("status") == "MODEL_DISAGREEMENT"
                 or bool(doubt_nodes))
    if tier in ("MONITOR", "NO_ACTION") or risk < 0.2:
        verdict = VERDICT_MONITOR if risk >= 0.05 else VERDICT_NO_ACTION
    elif not prosecution:
        verdict = VERDICT_INSUFFICIENT
    elif balance >= 0.65 and not contested:
        verdict = VERDICT_ENHANCED
    elif balance >= 0.5:
        verdict = VERDICT_REVIEW
    else:
        verdict = VERDICT_INSUFFICIENT

    rationale = (
        f"Prosecution weight {p_strength:.3f} against defence and uncertainty "
        f"weight {d_strength:.3f} (balance {balance:.2f}). "
        + ("Model families disagree on this account, so the case is treated as "
           "contested and the recommendation is deliberately conservative. "
           if disagreement.get("status") == "MODEL_DISAGREEMENT" else "")
        + f"Recommendation: {verdict.replace('_', ' ').lower()}. "
        "This is guidance for a human reviewer and carries no finding about "
        "the account holder."
    )

    return {
        "prosecution": prosecution,
        "defence": defence,
        "evidence_balance": round(balance, 4),
        "prosecution_weight": round(p_strength, 6),
        "defence_weight": round(d_strength, 6),
        "contested": bool(contested),
        "verdict": verdict,
        "verdict_rationale": rationale,
        "evidence_policy": (
            "Both columns are built only from verified model output. No "
            "language model contributed evidence to either side."
        ),
    }


# --------------------------------------------------------------------------
# guardrails
# --------------------------------------------------------------------------


def assert_evidence_traceable(graph: dict[str, Any]) -> None:
    """Every node must name a source, and every edge must join real nodes."""
    ids = set()
    for n in graph["nodes"]:
        if not n.get("source"):
            raise UntraceableEvidence(
                f"node {n.get('id')!r} has no source - refusing to emit")
        ids.add(n["id"])
    for e in graph["edges"]:
        if e["source"] not in ids or e["target"] not in ids:
            raise UntraceableEvidence(
                f"edge {e['source']} -> {e['target']} references a missing node")


def evidence_feature_tokens(graph: dict[str, Any]) -> list[str]:
    """Every dataset column this graph names, from every slot that can name one.

    A node's ``source`` is its primary claim of origin, but it is not the only
    place a raw column token lands. The SHAP column is stashed on
    ``extra["column"]``, a counterfactual twin lists one per row of
    ``extra["differences"]``, and a counterfactual node names the feature it
    would move. A quarantine check that reads only ``source`` is a check with
    three back doors, so collect all four.

    Tokens that are pipeline metrics rather than dataset columns
    (``model_agreement``, ``conformal_status``, ``MG_PASSTHROUGH_7D`` ...) are
    returned too. They are safe to return: the firewall resolves an unknown
    token to ``UNKNOWN_REVIEW``, which is not a forbidden class, so they pass
    without needing a hand-maintained allowlist here - and a hand-maintained
    allowlist is exactly the thing that would rot into a back door.
    """
    tokens: list[str] = []
    for n in graph.get("nodes") or []:
        if n.get("source"):
            tokens.append(str(n["source"]))
        extra = n.get("extra") or {}
        if not isinstance(extra, dict):
            continue
        if extra.get("column"):
            tokens.append(str(extra["column"]))
        if extra.get("feature"):
            tokens.append(str(extra["feature"]))
        for d in extra.get("differences") or []:
            if isinstance(d, dict) and d.get("feature"):
                tokens.append(str(d["feature"]))
    seen, out = set(), []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def assert_evidence_admissible(graph: dict[str, Any],
                               registry: dict[str, Any] | None = None) -> None:
    """No quarantined column may stand as prosecution or defence evidence.

    Delegates to :func:`muleguard.features.firewall.assert_clean` rather than
    re-stating the ban, so this boundary can never drift out of step with the
    quarantine manifest the release gate enforces.
    """
    try:
        firewall.assert_clean(evidence_feature_tokens(graph),
                              context="proofgraph evidence", registry=registry)
    except firewall.LeakageViolation as exc:
        raise InadmissibleEvidence(str(exc)) from exc


def assert_language_safe(payload: Any) -> None:
    """Reject accusatory or absolute terms anywhere in the payload."""
    def walk(v: Any) -> None:
        if isinstance(v, str):
            low = v.lower()
            for term in FORBIDDEN_TERMS:
                if term in low:
                    raise UnsafeLanguage(f"forbidden term {term!r} in output")
        elif isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, (list, tuple)):
            for x in v:
                walk(x)

    walk(payload)


def twin_differences(x_row: np.ndarray, twin_row: np.ndarray,
                     feature_names: Sequence[str], focus_idx: Sequence[int],
                     registry: dict[str, Any] | None = None,
                     top_k: int = 5) -> list[dict[str, Any]]:
    """Where the case departs from its closest non-escalated neighbour."""
    diffs = []
    for j in focus_idx:
        a, b = float(x_row[j]), float(twin_row[j])
        if math.isnan(a) and math.isnan(b):
            continue
        rec = _registry_lookup(feature_names[j], registry)
        diffs.append({
            "feature": feature_names[j],
            "variable_name": rec.get("variable_name"),
            "description": rec.get("description"),
            "this_account": None if math.isnan(a) else a,
            "twin_account": None if math.isnan(b) else b,
            "absolute_gap": abs((0.0 if math.isnan(a) else a)
                                - (0.0 if math.isnan(b) else b)),
        })
    diffs.sort(key=lambda d: -d["absolute_gap"])
    return diffs[:top_k]
