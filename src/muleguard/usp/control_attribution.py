"""Account-Control Ambiguity Guardrail (sections 20-23).

A behavioural classifier trained on account-level aggregates can rank how
unusual an account's activity is. It cannot tell you who was sitting at the
keyboard, or whether the person whose name is on the account knew what was
happening. Those are three different questions, and the honest system keeps
them apart:

    BEHAVIOURAL MULE RISK      - the frozen classifier answers this
    ACCOUNT-CONTROL EVIDENCE   - device, SIM, credential and KYC history would
    INTENT ATTRIBUTION         - an interview and an investigator would

The dataset behind this project contains the first kind of evidence and none of
the second or third. So this module reports ``NOT_AVAILABLE`` and ``UNKNOWN``
and declines to guess, because the alternative - reading intent off a
transaction pattern - is how an unwitting account holder becomes a suspect on
the strength of a percentile.

**This is a guardrail, not a model.** Nothing here touches ``risk_probability``,
``risk_tier``, ``model_votes``, the calibrator or a threshold. Section 22 is
specific about the wiring for a reason: the node attaches to the *decision* by
``REQUIRES_HUMAN_VERIFICATION``, never to the score by ``RAISED_BY``. An edge
pointing the other way would make a limitation look like evidence.

The checklist is what to go and find out, phrased as recommended enrichment.
None of those data sources are in this project, and the card says so rather
than implying a lookup that would silently return nothing.
"""
from __future__ import annotations

from typing import Any

#: The three questions, kept separate on purpose.
BEHAVIOURAL_RISK = "BEHAVIOURAL_MULE_RISK"
CONTROL_EVIDENCE = "ACCOUNT_CONTROL_EVIDENCE"
INTENT = "INTENT_ATTRIBUTION"

STATUS_NOT_AVAILABLE = "NOT_AVAILABLE"
STATUS_UNKNOWN = "UNKNOWN"

#: Section 22. The only relation by which a limitation may reach the decision.
RELATION = "REQUIRES_HUMAN_VERIFICATION"

#: Section 20. Categories that describe a *person*, which aggregate behaviour
#: cannot establish. Held here so the test suite can assert none of them is ever
#: emitted; deliberately not written into any payload, since naming them in the
#: output would be the same mistake in quotation marks.
NEVER_INFERRED = (
    "witting mule", "unwitting mule", "coerced mule",
    "criminal", "victim", "handler", "money launderer", "accomplice",
)

#: Section 23. Actions no automated path may take.
NEVER_AUTOMATIC = ("FREEZE", "FILE_STR", "DECLARE_MULE",
                   "DECLARE_CRIMINAL", "CERTIFY_CLEAN")

LIMITATION_STATEMENT = (
    "The supplied account-level feature dataset supports behavioural-risk "
    "assessment but does not establish who controlled the account or whether "
    "the account holder knowingly participated."
)

#: Section 21. Deterministic, fixed order, no scoring, no ranking. These are
#: things for a human to obtain - the project holds none of them.
VERIFICATION_CHECKLIST = (
    {"id": "device_login_history",
     "label": "Recent device / login ownership history",
     "why": "Shows whether the account was operated from the customer's own device."},
    {"id": "sim_change",
     "label": "SIM / mobile-number change",
     "why": "A recent change can indicate loss of control of the second factor."},
    {"id": "credential_reset",
     "label": "Credential / password-reset history",
     "why": "Resets close to the activity window bear on who held access."},
    {"id": "kyc_contact_change",
     "label": "KYC / contact-detail changes",
     "why": "Redirected contact details separate the holder from the operator."},
    {"id": "beneficiary_relationships",
     "label": "Beneficiary / counterparty relationships",
     "why": "Establishes whether counterparties are known to the customer."},
    {"id": "customer_interview",
     "label": "Customer confirmation / interview",
     "why": "The only source that speaks to awareness and intent."},
    {"id": "transaction_trail",
     "label": "Raw transaction trail",
     "why": "Aggregates cannot show the individual movements behind them."},
)


def _risk_band(risk: float, tier: str) -> str:
    """A word for the risk level, taken from the tier the policy already set.

    Deliberately derived rather than re-thresholded: inventing a second set of
    cut-offs here would create a number that could disagree with the decision
    the system actually made.
    """
    tier = str(tier or "").upper()
    if tier in ("CRITICAL_REVIEW", "URGENT_REVIEW"):
        return "HIGH"
    if tier == "STANDARD_REVIEW":
        return "MODERATE"
    if tier == "OOD_REVIEW":
        return "NOT_COMPARABLE"
    return "LOW"


def control_attribution(*, risk_probability: float, risk_tier: str,
                        sources_available: dict[str, bool] | None = None
                        ) -> dict[str, Any]:
    """The three-part statement for one account.

    ``sources_available`` is the hook for a deployment that genuinely has
    device or KYC-history feeds: pass ``{"device_login_history": True}`` and the
    checklist marks that item available. Nothing is assumed available by
    default, because assuming would produce a card claiming evidence exists
    when the lookup behind it would return nothing.
    """
    available = sources_available or {}
    checklist = [
        {**item,
         "status": "AVAILABLE" if available.get(item["id"]) else "NOT_IN_THIS_DATASET",
         "checked": False}
        for item in VERIFICATION_CHECKLIST
    ]
    any_control_source = any(available.get(i["id"]) for i in VERIFICATION_CHECKLIST)

    return {
        "behavioural_mule_risk": {
            "concept": BEHAVIOURAL_RISK,
            "status": "ASSESSED",
            "band": _risk_band(risk_probability, risk_tier),
            "risk_probability": float(risk_probability),
            "risk_tier": str(risk_tier),
            "source": "frozen champion classifier",
            "means": ("How unusual this account's aggregate behaviour is "
                      "relative to the training population."),
        },
        "account_control_evidence": {
            "concept": CONTROL_EVIDENCE,
            "status": (STATUS_NOT_AVAILABLE if not any_control_source
                       else "PARTIAL_SEE_CHECKLIST"),
            "source": "no device, SIM, credential or KYC-history feed is connected",
            "means": ("Who operated the account. Not derivable from aggregate "
                      "transaction features."),
        },
        "intent_attribution": {
            "concept": INTENT,
            "status": STATUS_UNKNOWN,
            "source": "no customer contact or investigation record is connected",
            "means": ("Whether the account holder was aware of the activity. "
                      "Requires a person to establish, not a model."),
        },
        "limitation_statement": LIMITATION_STATEMENT,
        "verification_checklist": checklist,
        "checklist_note": ("Recommended enrichment checks. This project does "
                           "not hold these data sources; the list describes "
                           "what to obtain before any high-impact action."),
        "automatic_actions_permitted": [],
        "affects_model_output": False,
    }


def proofgraph_node(card: dict[str, Any]) -> dict[str, Any]:
    """The CONTROL_ATTRIBUTION node, in the shape ProofGraph expects.

    Attached to the decision by :data:`RELATION`. Section 22 forbids reaching
    the score by ``RAISED_BY`` - a limitation is not evidence for a conclusion,
    it is a condition on acting upon one.
    """
    return {
        "id": "control_attribution",
        "type": "CONTROL_ATTRIBUTION",
        "label": "Account control and intent are not established",
        "source": "control_attribution.guardrail",
        "detail": LIMITATION_STATEMENT,
        "weight": 0.0,
        "value": STATUS_NOT_AVAILABLE,
        "extra": {
            "account_control_evidence": card["account_control_evidence"]["status"],
            "intent_attribution": card["intent_attribution"]["status"],
            "verification_checklist": [i["label"] for i in card["verification_checklist"]],
            "modifies_risk": False,
        },
    }
