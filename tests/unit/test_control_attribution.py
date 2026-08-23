"""The Account-Control Ambiguity Guardrail must stay a guardrail.

There are two ways a limitation card goes wrong. It can quietly acquire an
opinion - inferring from a high percentile that someone is a witting mule, a
criminal, a victim - which is section 20's prohibition. Or it can acquire
*influence*: a node wired to the score instead of to the decision, a status
that nudges a tier, an action taken automatically because the card looked
confident. Sections 22 and 23 exist for the second failure, and it is the more
dangerous one, because a card that changes an outcome does not look like a
card any more.
"""
from __future__ import annotations

import pytest

from muleguard.usp import control_attribution as ca

TIERS = ["CRITICAL_REVIEW", "URGENT_REVIEW", "STANDARD_REVIEW",
         "OOD_REVIEW", "MONITOR"]


@pytest.fixture
def card():
    return ca.control_attribution(risk_probability=0.97, risk_tier="CRITICAL_REVIEW")


# --------------------------------------------------------------------------
# section 20 - the three questions stay separate
# --------------------------------------------------------------------------


def test_the_card_answers_three_questions_not_one(card):
    assert card["behavioural_mule_risk"]["concept"] == ca.BEHAVIOURAL_RISK
    assert card["account_control_evidence"]["concept"] == ca.CONTROL_EVIDENCE
    assert card["intent_attribution"]["concept"] == ca.INTENT


def test_control_evidence_is_not_available_and_intent_is_unknown(card):
    assert card["account_control_evidence"]["status"] == ca.STATUS_NOT_AVAILABLE
    assert card["intent_attribution"]["status"] == ca.STATUS_UNKNOWN


@pytest.mark.parametrize("tier", TIERS)
@pytest.mark.parametrize("risk", [0.0, 0.5, 0.999999])
def test_no_score_however_extreme_produces_control_or_intent_evidence(tier, risk):
    """The whole point. A 0.999999 is still a statement about behaviour."""
    c = ca.control_attribution(risk_probability=risk, risk_tier=tier)
    assert c["account_control_evidence"]["status"] == ca.STATUS_NOT_AVAILABLE
    assert c["intent_attribution"]["status"] == ca.STATUS_UNKNOWN


def _all_text(node) -> list[str]:
    if isinstance(node, str):
        return [node]
    if isinstance(node, dict):
        out = []
        for k, v in node.items():
            out.append(str(k))
            out.extend(_all_text(v))
        return out
    if isinstance(node, (list, tuple)):
        return [t for v in node for t in _all_text(v)]
    return []


@pytest.mark.parametrize("tier", TIERS)
@pytest.mark.parametrize("term", ca.NEVER_INFERRED)
def test_no_person_category_is_ever_named_in_the_output(tier, term):
    """Section 20 forbids inferring these. Naming them in the payload would be
    the same mistake in quotation marks, so the card does not mention them."""
    c = ca.control_attribution(risk_probability=0.99, risk_tier=tier)
    for text in _all_text(c):
        assert term not in text.lower(), f"{term!r} appears in {text!r}"


def test_the_limitation_is_stated_not_implied(card):
    text = card["limitation_statement"].lower()
    assert "does not establish who controlled the account" in text
    assert "knowingly participated" in text


# --------------------------------------------------------------------------
# section 21 - the checklist
# --------------------------------------------------------------------------


def test_checklist_is_fixed_deterministic_and_unranked(card):
    ids = [i["id"] for i in card["verification_checklist"]]
    assert ids == [i["id"] for i in ca.VERIFICATION_CHECKLIST]
    again = ca.control_attribution(risk_probability=0.97,
                                   risk_tier="CRITICAL_REVIEW")
    assert card == again
    for item in card["verification_checklist"]:
        assert "score" not in item and "rank" not in item and "weight" not in item


@pytest.mark.parametrize("tier", TIERS)
def test_checklist_does_not_change_with_the_score(tier):
    """What a human should go and find out does not depend on how alarming the
    model found the account. A checklist that grew with the score would be a
    second, unvalidated risk signal."""
    low = ca.control_attribution(risk_probability=0.01, risk_tier="MONITOR")
    other = ca.control_attribution(risk_probability=0.99, risk_tier=tier)
    assert low["verification_checklist"] == other["verification_checklist"]


def test_every_check_says_the_project_does_not_hold_it(card):
    for item in card["verification_checklist"]:
        assert item["status"] == "NOT_IN_THIS_DATASET"
        assert item["checked"] is False
        assert item["why"], "a check with no stated purpose is a to-do, not a control"
    assert "does not hold these data sources" in card["checklist_note"]


def test_a_connected_source_is_reported_as_partial_never_as_resolved():
    """A deployment with a real device feed answers one of the three questions
    partially. It does not thereby establish intent."""
    c = ca.control_attribution(risk_probability=0.9, risk_tier="URGENT_REVIEW",
                               sources_available={"device_login_history": True})
    assert c["account_control_evidence"]["status"] == "PARTIAL_SEE_CHECKLIST"
    assert c["intent_attribution"]["status"] == ca.STATUS_UNKNOWN
    got = {i["id"]: i["status"] for i in c["verification_checklist"]}
    assert got["device_login_history"] == "AVAILABLE"
    assert got["sim_change"] == "NOT_IN_THIS_DATASET"


# --------------------------------------------------------------------------
# section 22 / 23 - influence, or the absence of it
# --------------------------------------------------------------------------


def test_the_card_reports_the_score_it_was_given_and_changes_nothing(card):
    assert card["behavioural_mule_risk"]["risk_probability"] == 0.97
    assert card["behavioural_mule_risk"]["risk_tier"] == "CRITICAL_REVIEW"
    assert card["affects_model_output"] is False


@pytest.mark.parametrize("tier,band", [
    ("CRITICAL_REVIEW", "HIGH"), ("URGENT_REVIEW", "HIGH"),
    ("STANDARD_REVIEW", "MODERATE"), ("OOD_REVIEW", "NOT_COMPARABLE"),
    ("MONITOR", "LOW"), ("", "LOW"),
])
def test_the_band_restates_the_tier_rather_than_recomputing_it(tier, band):
    """A band derived from the probability would be a second threshold set -
    one nobody validated. It is read off the tier the system actually issued."""
    c = ca.control_attribution(risk_probability=0.999, risk_tier=tier)
    assert c["behavioural_mule_risk"]["band"] == band


def test_no_action_is_ever_permitted_automatically(card):
    assert card["automatic_actions_permitted"] == []


@pytest.mark.parametrize("action", ca.NEVER_AUTOMATIC)
@pytest.mark.parametrize("tier", TIERS)
def test_the_forbidden_actions_are_never_offered(action, tier):
    c = ca.control_attribution(risk_probability=0.999, risk_tier=tier)
    assert action not in c["automatic_actions_permitted"]


def test_the_proofgraph_node_is_traceable_and_weightless(card):
    node = ca.proofgraph_node(card)
    assert node["type"] == "CONTROL_ATTRIBUTION"
    assert node["source"], "an untraceable node cannot be audited"
    assert node["weight"] == 0.0
    assert node["extra"]["modifies_risk"] is False
    assert node["value"] == ca.STATUS_NOT_AVAILABLE


def test_the_relation_to_the_decision_is_verification_not_evidence():
    """Section 22, precisely. RAISED_BY would make a limitation look like a
    reason for the conclusion rather than a condition on acting on it."""
    assert ca.RELATION == "REQUIRES_HUMAN_VERIFICATION"
    assert ca.RELATION != "RAISED_BY"


@pytest.mark.parametrize("term", ca.NEVER_INFERRED)
def test_the_proofgraph_node_names_no_person_category(card, term):
    for text in _all_text(ca.proofgraph_node(card)):
        assert term not in text.lower()


def test_the_node_carries_no_probability_of_its_own(card):
    node = ca.proofgraph_node(card)
    assert not ({"risk_probability", "calibrated_risk", "risk_tier",
                 "score", "probability"} & set(node))
