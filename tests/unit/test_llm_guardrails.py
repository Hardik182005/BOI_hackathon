"""LLM validator must reject every class of hallucination; fallback always works."""
import json

import pytest

from muleguard.llm.deterministic_fallback import deterministic_narrative
from muleguard.llm.schemas import NarratorInput, ReasonFact
from muleguard.llm.validator import validate_llm_output


@pytest.fixture
def ctx() -> NarratorInput:
    return NarratorInput(
        account_reference="ACC-1029",
        calibrated_risk=0.92,
        risk_tier="CRITICAL_REVIEW",
        model_agreement=0.88,
        conformal_status="HIGH_RISK_SET",
        ood_status="IN_DISTRIBUTION",
        top_reasons=[
            ReasonFact(feature="F1702", value=10.4, legitimate_percentile=99.2,
                       direction="INCREASES_RISK", shap_contribution=0.21),
        ],
    )


def _valid_payload(ctx) -> dict:
    return {
        "summary": "Account ACC-1029 requires highest-priority human review. "
                   "F1702 is high relative to the legitimate cohort.",
        "risk_tier": "CRITICAL_REVIEW",
        "verified_risk_score": 0.92,
        "reason_codes": ["F1702:+0.210"],
        "recommended_checks": ["VERIFY_KYC", "ANALYST_REVIEW"],
        "limitations": [
            "Behavioural risk is not proof of criminal intent",
            "Final action requires human review",
        ],
    }


def test_valid_output_accepted(ctx):
    out, reasons = validate_llm_output(json.dumps(_valid_payload(ctx)), ctx)
    assert out is not None and reasons == []


def test_invalid_json_rejected(ctx):
    out, reasons = validate_llm_output("not json {", ctx)
    assert out is None and "invalid JSON" in reasons


def test_risk_score_change_rejected(ctx):
    p = _valid_payload(ctx)
    p["verified_risk_score"] = 0.55
    out, reasons = validate_llm_output(json.dumps(p), ctx)
    assert out is None and any("risk score altered" in r for r in reasons)


def test_tier_change_rejected(ctx):
    p = _valid_payload(ctx)
    p["risk_tier"] = "MONITOR"
    out, reasons = validate_llm_output(json.dumps(p), ctx)
    assert out is None and any("tier altered" in r for r in reasons)


def test_unknown_feature_rejected(ctx):
    p = _valid_payload(ctx)
    p["summary"] += " Also F999 shows rapid pass-through."
    out, reasons = validate_llm_output(json.dumps(p), ctx)
    assert out is None and any("F999" in r for r in reasons)


def test_invented_amount_rejected(ctx):
    p = _valid_payload(ctx)
    p["summary"] += " The account moved Rs. 4,50,000 last week."
    out, reasons = validate_llm_output(json.dumps(p), ctx)
    assert out is None and any("currency amount" in r for r in reasons)


def test_guilt_assertion_rejected(ctx):
    p = _valid_payload(ctx)
    p["summary"] = "This account holder is guilty of laundering."
    out, reasons = validate_llm_output(json.dumps(p), ctx)
    assert out is None and any("guilt" in r for r in reasons)


def test_action_outside_allowlist_rejected(ctx):
    p = _valid_payload(ctx)
    p["recommended_checks"] = ["FREEZE_ACCOUNT_NOW"]
    out, reasons = validate_llm_output(json.dumps(p), ctx)
    assert out is None and any("allowlist" in r for r in reasons)


def test_missing_limitations_rejected(ctx):
    p = _valid_payload(ctx)
    p["limitations"] = ["none"]
    out, reasons = validate_llm_output(json.dumps(p), ctx)
    assert out is None and any("limitation" in r for r in reasons)


def test_schema_violation_rejected(ctx):
    p = _valid_payload(ctx)
    p["verified_risk_score"] = 42.0  # out of [0,1]
    out, reasons = validate_llm_output(json.dumps(p), ctx)
    assert out is None and any("schema" in r for r in reasons)


def test_criminal_intent_phrase_is_allowed_in_limitations(ctx):
    # the phrase 'criminal intent' inside the required limitation must NOT trip
    out, reasons = validate_llm_output(json.dumps(_valid_payload(ctx)), ctx)
    assert out is not None


def test_deterministic_fallback_always_valid(ctx):
    out = deterministic_narrative(ctx)
    assert out.verified_risk_score == ctx.calibrated_risk
    assert out.risk_tier == ctx.risk_tier
    assert any("intent" in l.lower() for l in out.limitations)
    assert any("human review" in l.lower() for l in out.limitations)
    # fallback survives its own validator
    ok, reasons = validate_llm_output(out.model_dump_json(), ctx)
    assert ok is not None and reasons == []


def test_fallback_for_monitor_tier_never_certifies_safe(ctx):
    ctx2 = ctx.model_copy(update={"risk_tier": "MONITOR", "calibrated_risk": 0.02,
                                  "conformal_status": "LOW_RISK_SET"})
    out = deterministic_narrative(ctx2)
    assert "not currently flagged" in out.summary.lower()
    assert "safe" not in out.summary.lower()
