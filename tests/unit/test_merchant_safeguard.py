"""The merchant safeguard must never be able to clear an account.

MODE A is the shipped behaviour and must leave the score untouched. MODE B
exists for reviewers who want the score itself dampened, and the tests here
pin the two properties that keep it from becoming a way for a fraud system to
talk itself out of a true positive: it is bounded, and it is always logged.
"""
from __future__ import annotations

import pytest

from muleguard.models.merchant import (
    DAMPENING_FLOOR,
    MODE_A,
    MODE_B,
    MerchantVerdict,
    apply_merchant_safeguard,
)


def _verdict(band: str) -> MerchantVerdict:
    return MerchantVerdict(
        legitimacy=0.95, band=band,
        mule_probability_from_business_evidence=0.02, components={})


STRONG = "STRONG_BUSINESS_EVIDENCE"


def test_mode_a_never_changes_the_score():
    r = apply_merchant_safeguard(calibrated_risk=0.8,
                                 merchant=_verdict(STRONG), mode=MODE_A)
    assert r["merchant_safeguard_applied"] is False
    assert r["before_score"] == r["after_policy_score"] == 0.8
    assert r["delta"] == 0.0


def test_mode_a_is_the_default():
    r = apply_merchant_safeguard(calibrated_risk=0.8, merchant=_verdict(STRONG))
    assert r["mode"] == MODE_A


def test_mode_b_dampens_only_on_strong_evidence():
    r = apply_merchant_safeguard(calibrated_risk=0.8, merchant=_verdict(STRONG),
                                 mode=MODE_B)
    assert r["merchant_safeguard_applied"] is True
    assert r["after_policy_score"] < r["before_score"]


@pytest.mark.parametrize("band", ["WEAK_BUSINESS_EVIDENCE", "NO_EVIDENCE", None])
def test_mode_b_does_nothing_without_strong_evidence(band):
    merchant = _verdict(band) if band else None
    r = apply_merchant_safeguard(calibrated_risk=0.8, merchant=merchant,
                                 mode=MODE_B)
    assert r["merchant_safeguard_applied"] is False
    assert r["after_policy_score"] == 0.8


def test_dampening_is_bounded_by_the_floor():
    """Even an absurd factor cannot drive the score toward zero."""
    r = apply_merchant_safeguard(calibrated_risk=1.0, merchant=_verdict(STRONG),
                                 mode=MODE_B, dampening_factor=0.01)
    assert r["after_policy_score"] >= DAMPENING_FLOOR
    assert r["dampening_factor"] == DAMPENING_FLOOR


def test_dampening_can_never_reach_zero():
    r = apply_merchant_safeguard(calibrated_risk=0.9, merchant=_verdict(STRONG),
                                 mode=MODE_B, dampening_factor=0.0)
    assert r["after_policy_score"] > 0.0


def test_the_record_always_carries_the_required_audit_fields():
    """A dampening that is not logged did not happen."""
    for mode in (MODE_A, MODE_B):
        r = apply_merchant_safeguard(calibrated_risk=0.8,
                                     merchant=_verdict(STRONG), mode=mode)
        for field in ("merchant_safeguard_applied", "before_score",
                      "after_policy_score", "policy_version", "mode", "reason"):
            assert field in r, f"{mode} record is missing {field}"


def test_an_unapplied_safeguard_is_still_recorded():
    """Silence must not be mistaken for evidence that nothing was considered."""
    r = apply_merchant_safeguard(calibrated_risk=0.8, merchant=None, mode=MODE_B)
    assert r["merchant_safeguard_applied"] is False
    assert "no dampening applied" in r["reason"]


def test_unknown_mode_is_refused():
    with pytest.raises(ValueError, match="unknown merchant safeguard mode"):
        apply_merchant_safeguard(calibrated_risk=0.8, merchant=None, mode="MODE_C")


def test_invalid_floor_is_refused():
    with pytest.raises(ValueError, match="floor"):
        apply_merchant_safeguard(calibrated_risk=0.8, merchant=_verdict(STRONG),
                                 mode=MODE_B, floor=0.0)


def test_guarantees_state_the_queue_is_never_cleared():
    r = apply_merchant_safeguard(calibrated_risk=0.8, merchant=_verdict(STRONG),
                                 mode=MODE_B)
    assert any("review queue" in g for g in r["guarantees"])
