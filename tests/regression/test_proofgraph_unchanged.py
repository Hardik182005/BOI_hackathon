"""Section 30: the ProofGraph the judges already saw must still be that graph.

The upgrade is allowed to *add* two things to the graph - a COHORT_CONTEXT node
and a CONTROL_ATTRIBUTION node - and is allowed to change nothing else. Section
30 lists what has to survive verbatim: the existing node types, the evidence
counts, the score, the courtroom and the counterfactuals.

These tests build the graph twice: once as the code ships it, and once from the
same inputs with the additions removed. The second form is what the graph was
before this upgrade existed, so comparing the two is the regression - it fails
if an addition leaked into a number that was already being shown.
"""
from __future__ import annotations

import copy

import pytest

from muleguard.explain import proofgraph as pg

#: Everything the graph was allowed to contain before this upgrade.
PRE_CHANGE_NODE_TYPES = {
    pg.NODE_ACCOUNT, pg.NODE_EVIDENCE_FOR, pg.NODE_EVIDENCE_AGAINST,
    pg.NODE_PATTERN, pg.NODE_MODEL_VOTE, pg.NODE_UNCERTAINTY,
    pg.NODE_COUNTERFACTUAL, pg.NODE_TWIN, pg.NODE_DECISION,
}

#: The only two additions section 30 permits.
PERMITTED_ADDITIONS = {"CONTROL_ATTRIBUTION", "COHORT_CONTEXT"}

SCORE = {
    "calibrated_risk": 0.8137412345678901,
    "risk_tier": "URGENT_REVIEW",
    "model_version": "v-test",
    "raw_scores": {"lightgbm": 0.79, "xgboost": 0.83, "catboost": 0.81},
    "conformal_status": "AMBIGUOUS",
    "ood_status": "IN_DISTRIBUTION",
    "model_agreement": 0.94,
}

# The exact shape the SHAP reason builder emits - direction sentinel and
# ``shap_contribution`` key included, because a fixture that guesses those would
# silently build the wrong node types and test nothing.
REASONS = [
    {"feature": "F0001", "direction": "INCREASES_RISK", "shap_contribution": 0.31,
     "value": 12.0, "legitimate_cohort_median": 3.0, "legitimate_percentile": 99.0},
    {"feature": "F0002", "direction": "INCREASES_RISK", "shap_contribution": 0.18,
     "value": 4.0, "legitimate_cohort_median": 1.0, "legitimate_percentile": 96.0},
    {"feature": "F0003", "direction": "DECREASES_RISK", "shap_contribution": -0.12,
     "value": 1.0, "legitimate_cohort_median": 1.0, "legitimate_percentile": 50.0},
]

COUNTERFACTUALS = [
    {"feature": "F0001", "from_value": 12.0, "to_value": 3.0,
     "score_before": 0.81, "score_after": 0.42, "crosses_threshold": True},
]


def _build(**kw):
    return pg.build_proofgraph(
        account_reference="RV-TEST-0001", score=copy.deepcopy(SCORE),
        reasons=copy.deepcopy(REASONS),
        counterfactuals=copy.deepcopy(COUNTERFACTUALS), **kw)


@pytest.fixture(scope="module")
def graph():
    return _build()


@pytest.fixture(scope="module")
def core(graph):
    """The graph with the permitted additions stripped - i.e. the old graph."""
    g = copy.deepcopy(graph)
    added = {n["id"] for n in g["nodes"] if n["type"] in PERMITTED_ADDITIONS}
    g["nodes"] = [n for n in g["nodes"] if n["id"] not in added]
    g["edges"] = [e for e in g["edges"]
                  if e["source"] not in added and e["target"] not in added]
    return g


# --------------------------------------------------------------------------
# what was added, and nothing else
# --------------------------------------------------------------------------


def test_only_the_two_permitted_node_types_are_new(graph):
    types = {n["type"] for n in graph["nodes"]}
    unexpected = types - PRE_CHANGE_NODE_TYPES - PERMITTED_ADDITIONS
    assert unexpected == set(), f"unexpected node types: {sorted(unexpected)}"


def test_every_pre_change_node_type_that_the_inputs_imply_is_still_there(graph):
    types = {n["type"] for n in graph["nodes"]}
    for expected in (pg.NODE_ACCOUNT, pg.NODE_EVIDENCE_FOR,
                     pg.NODE_EVIDENCE_AGAINST, pg.NODE_MODEL_VOTE,
                     pg.NODE_COUNTERFACTUAL, pg.NODE_DECISION):
        assert expected in types, f"{expected} disappeared from the graph"


def test_the_added_node_is_marked_as_post_model_context(graph):
    """Section 30: additions must be clearly marked, not blended in.

    Weight zero and ``modifies_risk`` false are the machine-readable version of
    that marking; the label is the human-readable one.
    """
    control = [n for n in graph["nodes"] if n["type"] == "CONTROL_ATTRIBUTION"]
    assert len(control) == 1
    node = control[0]
    assert node["weight"] == 0.0
    assert node["extra"]["modifies_risk"] is False
    assert "not established" in node["label"].lower()


def test_the_addition_reaches_the_graph_only_through_the_decision(graph):
    """Section 22, checked as topology rather than as intent."""
    control_ids = {n["id"] for n in graph["nodes"]
                   if n["type"] in PERMITTED_ADDITIONS}
    touching = [e for e in graph["edges"]
                if e["source"] in control_ids or e["target"] in control_ids]
    assert touching, "the addition is orphaned"
    for e in touching:
        assert e["target"] in control_ids, (
            f"an added node is a *source* of {e['relation']} - it must only "
            f"ever be pointed at, never point back into the case")
        assert e["source"] == "decision", (
            f"added node reached from {e['source']!r}, must be 'decision'")
        assert e["relation"] == pg.CONTROL_RELATION
        assert e["relation"] != "RAISED_BY"
        assert e["weight"] == 0.0


# --------------------------------------------------------------------------
# the numbers that were already published
# --------------------------------------------------------------------------


def test_the_score_and_tier_are_passed_through_untouched(graph):
    assert graph["calibrated_risk"] == SCORE["calibrated_risk"]
    assert graph["risk_tier"] == SCORE["risk_tier"]
    root = next(n for n in graph["nodes"] if n["type"] == pg.NODE_ACCOUNT)
    assert root["weight"] == pytest.approx(SCORE["calibrated_risk"], abs=1e-6)


def test_evidence_counts_ignore_the_additions(graph, core):
    """The counts must describe the old graph, not the new one."""
    counted = (graph["evidence_counts"]["prosecution"]
               + graph["evidence_counts"]["defence"]
               + graph["evidence_counts"]["uncertainty"])
    core_evidence = [n for n in core["nodes"] if n["type"] in (
        pg.NODE_EVIDENCE_FOR, pg.NODE_EVIDENCE_AGAINST, pg.NODE_UNCERTAINTY)]
    assert counted == len(core_evidence)


def test_the_courtroom_never_saw_the_addition(graph):
    """No added node may appear as a prosecution or defence point."""
    added_labels = {n["label"] for n in graph["nodes"]
                    if n["type"] in PERMITTED_ADDITIONS}
    for side in ("prosecution", "defence"):
        for point in graph["courtroom"][side]:
            assert point["point"] not in added_labels
            assert "control_attribution" not in point["source"]


def test_the_verdict_is_the_one_the_evidence_alone_produces(graph, core):
    """Rebuild the courtroom from the core nodes and require the same verdict.

    This is the sharpest form of the section-30 claim: if the guardrail had
    contributed anything to the recommendation, a courtroom computed without it
    would land somewhere else.
    """
    by_id = {n["id"]: n for n in core["nodes"]}
    def _nodes(kind):
        return [pg.Node(id=n["id"], type=n["type"], label=n["label"],
                        source=n["source"], detail=n["detail"],
                        weight=n["weight"], value=n.get("value"))
                for n in by_id.values() if n["type"] == kind]

    rebuilt = pg.model_courtroom(
        _nodes(pg.NODE_EVIDENCE_FOR), _nodes(pg.NODE_EVIDENCE_AGAINST),
        _nodes(pg.NODE_UNCERTAINTY), SCORE,
        pg.disagreement_profile(SCORE["raw_scores"]))
    assert rebuilt["verdict"] == graph["courtroom"]["verdict"]
    assert rebuilt["prosecution_weight"] == graph["courtroom"]["prosecution_weight"]
    assert rebuilt["defence_weight"] == graph["courtroom"]["defence_weight"]
    assert rebuilt["evidence_balance"] == graph["courtroom"]["evidence_balance"]


def test_the_counterfactuals_are_unchanged(graph):
    cf = [n for n in graph["nodes"] if n["type"] == pg.NODE_COUNTERFACTUAL]
    assert len(cf) == len(COUNTERFACTUALS)
    node = cf[0]
    assert node["source"] == "F0001"
    assert node["extra"]["crosses_threshold"] is True
    assert node["weight"] == pytest.approx(abs(0.81 - 0.42), abs=1e-9)


def test_the_model_votes_are_unchanged(graph):
    votes = {n["source"]: n["value"] for n in graph["nodes"]
             if n["type"] == pg.NODE_MODEL_VOTE}
    assert votes == {f"raw_scores.{k}": v for k, v in SCORE["raw_scores"].items()}


def test_the_graph_is_deterministic_apart_from_its_timestamp():
    a, b = _build(), _build()
    for g in (a, b):
        g.pop("generated_utc")
    assert a == b


def test_every_node_is_still_traceable_and_the_language_guard_still_passes(graph):
    """Both invariants the graph shipped with, re-asserted after the addition."""
    pg.assert_evidence_traceable(graph)
    pg.assert_language_safe(graph)


def test_the_addition_did_not_change_the_graph_for_a_low_risk_account():
    """A MONITOR account must still get a MONITOR courtroom.

    Section 23 in the graph: nothing added by this upgrade may nudge a quiet
    account towards review.
    """
    low = dict(SCORE, calibrated_risk=0.04, risk_tier="MONITOR",
               raw_scores={"lightgbm": 0.05, "xgboost": 0.03, "catboost": 0.04})
    g = pg.build_proofgraph(account_reference="RV-TEST-0002", score=low,
                            reasons=copy.deepcopy(REASONS))
    assert g["risk_tier"] == "MONITOR"
    assert g["calibrated_risk"] == 0.04
    assert g["courtroom"]["verdict"] in (
        pg.VERDICT_MONITOR, pg.VERDICT_NO_ACTION, pg.VERDICT_INSUFFICIENT)
    control = next(n for n in g["nodes"] if n["type"] == "CONTROL_ATTRIBUTION")
    assert control["weight"] == 0.0
    assert g["control_attribution"]["automatic_actions_permitted"] == []
