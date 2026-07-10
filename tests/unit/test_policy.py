"""Policy engine routing invariants."""
import pytest

from muleguard.action.policy import PolicyThresholds, decide, TIERS

THR = PolicyThresholds(
    critical_risk=0.85, urgent_risk=0.6, standard_risk=0.3,
    anomaly_escalation_pct=99.0,
)


def _decide(**over):
    base = dict(
        calibrated_risk=0.1, conformal_set="LOW_RISK_SET",
        ood_status="IN_DISTRIBUTION", anomaly_percentile=10.0,
        model_agreement=0.9, verifier_flag=None, thresholds=THR,
    )
    base.update(over)
    return decide(**base)


def test_ood_always_routes_to_ood_review():
    r = _decide(ood_status="OUT_OF_DISTRIBUTION", calibrated_risk=0.99)
    assert r["risk_tier"] == "OOD_REVIEW"


def test_high_risk_critical():
    r = _decide(calibrated_risk=0.95, conformal_set="HIGH_RISK_SET")
    assert r["risk_tier"] == "CRITICAL_REVIEW"
    assert r["decision"] == "HUMAN_REVIEW_REQUIRED"


def test_low_risk_monitor_is_not_certified_safe():
    r = _decide(calibrated_risk=0.05)
    assert r["risk_tier"] == "MONITOR"
    assert r["decision"] == "NOT_CURRENTLY_FLAGGED"
    assert any("not currently flagged" in l.lower() or "never" in l.lower()
               for l in r["limitations"])


def test_conformal_uncertain_escalates_monitor():
    r = _decide(calibrated_risk=0.05, conformal_set="UNCERTAIN_SET")
    assert r["risk_tier"] == "STANDARD_REVIEW"


def test_anomaly_challenger_escalates_monitor():
    r = _decide(calibrated_risk=0.05, anomaly_percentile=99.5)
    assert r["risk_tier"] == "STANDARD_REVIEW"


def test_lookalike_protection_caps_critical_to_urgent():
    r = _decide(calibrated_risk=0.95, verifier_flag=False, conformal_set="UNCERTAIN_SET")
    assert r["risk_tier"] == "URGENT_REVIEW"
    assert any("look-alike" in s for s in r["reasons"])


def test_lookalike_protection_does_not_cap_when_conformal_confident():
    r = _decide(calibrated_risk=0.95, verifier_flag=False, conformal_set="HIGH_RISK_SET")
    assert r["risk_tier"] == "CRITICAL_REVIEW"


def test_no_auto_action_ever():
    for risk in (0.01, 0.5, 0.99):
        r = _decide(calibrated_risk=risk)
        assert r["auto_action"] is None


def test_all_outputs_are_known_tiers():
    for risk in (0.0, 0.3, 0.6, 0.85, 1.0):
        for cs in ("HIGH_RISK_SET", "LOW_RISK_SET", "UNCERTAIN_SET"):
            r = _decide(calibrated_risk=risk, conformal_set=cs)
            assert r["risk_tier"] in TIERS


def test_never_uses_forbidden_language():
    r = _decide(calibrated_risk=0.99)
    text = str(r).lower()
    assert "guilty" not in text and "criminal " not in text.replace("criminal intent", "")
    assert "permanently_safe" not in text and "certified_clean" not in text
