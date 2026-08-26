"""evidence_guard: the boundary between a stored score and evidence a reviewer sees.

``current_model_version`` is monkeypatched throughout so these tests exercise
the *logic* that decides CURRENT vs RETIRED and clean vs quarantined, without
depending on which bundle happens to be on disk or paying to load it. The
bundle-dependent wiring (``assert_servable_as_current_evidence`` raising a 409,
the redaction actually reaching a route) is covered end to end in
``tests/integration/test_evidence_guard_api.py``.
"""
from __future__ import annotations

import pytest

from muleguard.api import evidence_guard as eg


@pytest.fixture(autouse=True)
def _pin_current_version(monkeypatch):
    monkeypatch.setattr(eg, "current_model_version", lambda: "2.0.0")


def test_provenance_current_for_the_live_bundle_version():
    prov = eg.provenance({"model_version": "2.0.0"})
    assert prov["status"] == eg.PROVENANCE_CURRENT
    assert prov["stored_model_version"] == "2.0.0"
    assert prov["current_model_version"] == "2.0.0"


def test_provenance_retired_for_any_other_version():
    """The retired CatBoost bundle from the incident: model_version 1.0.0."""
    prov = eg.provenance({"model_version": "1.0.0"})
    assert prov["status"] == eg.PROVENANCE_RETIRED
    assert prov["stored_model_version"] == "1.0.0"
    assert prov["current_model_version"] == "2.0.0"


def test_provenance_retired_when_the_payload_has_no_stamp_at_all():
    """A payload missing model_version is not CURRENT by default - it must
    prove it, and an absent stamp cannot."""
    prov = eg.provenance({})
    assert prov["status"] == eg.PROVENANCE_RETIRED
    assert prov["stored_model_version"] is None


def test_quarantined_reason_columns_flags_the_incident_columns():
    """F3898 (MIN_RESOLVE_DAYS) and F3914 (FALSE_POSITIVE) are exactly the two
    columns a reviewer saw presented as prosecution evidence."""
    reasons = [{"feature": "F3898"}, {"feature": "F3914"}, {"feature": "F1813"}]
    bad = eg.quarantined_reason_columns(reasons)
    assert set(bad) == {"F3898", "F3914"}


def test_quarantined_reason_columns_passes_a_clean_list():
    reasons = [{"feature": "F1813"}, {"feature": "F3799"}]
    assert eg.quarantined_reason_columns(reasons) == []


def test_quarantined_reason_columns_tolerates_missing_or_absent_reasons():
    # A row with no "feature" key and a payload with no reasons at all are
    # both real shapes (a score stored without explanations); neither should
    # raise, and neither names a quarantined column.
    assert eg.quarantined_reason_columns([{"value": 1.0}]) == []
    assert eg.quarantined_reason_columns(None) == []
    assert eg.quarantined_reason_columns([]) == []


def test_redact_retired_evidence_drops_keys_rather_than_emptying_them():
    """The whole point of the fix: a consumer that finds no ``top_reasons`` key
    has to ask why (and gets the answer in ``evidence_withheld``); a consumer
    that finds ``top_reasons: []`` would wrongly conclude the model had
    nothing to say."""
    score = {
        "model_version": "1.0.0",
        "calibrated_risk": 0.97,
        "risk_tier": "CRITICAL_REVIEW",
        "top_reasons": [{"feature": "F3898"}, {"feature": "F3914"}],
        "counterfactual_twin": {"reference": "LEGIT-1"},
    }
    status = eg.evidence_status("CASE-RETIRED", score)
    assert status["admissible_as_current_evidence"] is False
    assert status["reason"] == eg.RETIRED_ERROR

    redacted = eg.redact_retired_evidence("CASE-RETIRED", score, status)

    assert "top_reasons" not in redacted
    assert "counterfactual_twin" not in redacted
    # The score, tier and uncertainty fields survive as history - they are
    # what the retired model reported, and the case file is entitled to show
    # them, labelled.
    assert redacted["calibrated_risk"] == 0.97
    assert redacted["risk_tier"] == "CRITICAL_REVIEW"

    withheld = redacted["evidence_withheld"]
    assert withheld["reason"] == eg.RETIRED_ERROR
    assert withheld["withheld_keys"] == ["top_reasons", "counterfactual_twin"]
    assert withheld["audit_record_available_at"] == "/v1/proofgraph/CASE-RETIRED/provenance"


def test_redact_retired_evidence_flags_a_current_payload_that_names_a_quarantined_column():
    """Independent of provenance: a CURRENT payload should never name a
    quarantined column, but if a future bug produced one, it must still be
    withheld rather than shown."""
    score = {"model_version": "2.0.0", "top_reasons": [{"feature": "F3898"}]}
    status = eg.evidence_status("CASE-BUG", score)
    assert status["admissible_as_current_evidence"] is False
    assert status["reason"] == eg.INADMISSIBLE_ERROR
    redacted = eg.redact_retired_evidence("CASE-BUG", score, status)
    assert "top_reasons" not in redacted
    assert redacted["evidence_withheld"]["reason"] == eg.INADMISSIBLE_ERROR


def test_redact_retired_evidence_is_a_passthrough_when_admissible():
    """Nothing is withheld from evidence that is already admissible - the
    function must not touch a clean, current payload."""
    score = {"model_version": "2.0.0", "top_reasons": [{"feature": "F1813"}]}
    status = eg.evidence_status("CASE-CLEAN", score)
    assert status["admissible_as_current_evidence"] is True
    assert eg.redact_retired_evidence("CASE-CLEAN", score, status) == score


def test_evidence_status_reports_no_stored_payload_without_raising():
    status = eg.evidence_status("CASE-EMPTY", None)
    assert status["admissible_as_current_evidence"] is False
    assert status["reason"] == "NO_STORED_PAYLOAD"
    assert status["provenance"] is None
