"""Evidence-provenance gate, exercised through the real API and a real DB row.

The incident: 77 rows in ``artifacts/muleguard.db`` were written by a retired
CatBoost model (model_version 1.0.0, frozen 2026-07-10, before the Feature
Availability Firewall existed) and named ``F3898 MIN_RESOLVE_DAYS`` /
``F3914 FALSE_POSITIVE`` in ``top_reasons``. Three routes read
``scores.payload_json`` and rendered it without ever asking which model wrote
it, so a reviewer opening a case saw a retired explanation as though it were a
current finding.

These tests build one such row directly with ``db.record_score`` - the same
function the scoring pipeline uses, just called with a retired
``model_version`` - rather than trying to reload a CatBoost bundle that is no
longer shipped. That is the honest reproduction: the bug was never about what
produced the row, it was about what a route did with a row whose
``model_version`` did not match the champion's.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from muleguard import settings

BUNDLE = settings.MODELS_DIR / "final_bundle.joblib"
pytestmark = pytest.mark.skipif(not BUNDLE.exists(), reason="final bundle not built yet")

#: The 13 columns the Feature Availability Firewall quarantines. No serialised
#: response - case detail, proofgraph error body, report - may contain any of
#: these tokens once a payload is recognised as RETIRED or INADMISSIBLE.
QUARANTINED = {"F3924", "F3912", "F3913", "F3914", "F3915", "F3898", "F3899",
              "F2230", "__UNNAMED__0", "F3916", "F3917", "F3918", "F3892"}


@pytest.fixture(scope="module")
def client():
    tmp = tempfile.mkdtemp()
    os.environ["MULEGUARD_DB"] = str(Path(tmp) / "evidence_guard_api.db")
    import importlib

    from muleguard.api import database
    importlib.reload(database)
    from muleguard.api import main as api_main
    importlib.reload(api_main)
    from fastapi.testclient import TestClient

    with TestClient(api_main.app) as c:
        yield c


@pytest.fixture(scope="module")
def sample_features():
    from muleguard.data import ingest

    df = ingest.load_dataset()
    row = df.head(1).to_dicts()[0]
    row.pop(settings.TARGET_COLUMN, None)
    return {k: (str(v) if hasattr(v, "isoformat") else v) for k, v in row.items()}


def _retired_result() -> dict:
    """A payload shaped exactly like the incident: the two named columns, plus
    a counterfactual twin, stamped with the retired model's version."""
    return {
        "model_version": "1.0.0",
        "raw_scores": {"catboost_tuned_top60": 0.97},
        "ensemble_score": 0.97,
        "calibrated_risk": 0.97,
        "model_agreement": 1.0,
        "conformal_status": "HIGH_RISK_SET",
        "verifier_confirms_risk": None,
        "verifier_probability": None,
        "anomaly_percentile": 99.0,
        "ood_status": "IN_DISTRIBUTION",
        "ood_detail": {},
        "risk_tier": "CRITICAL_REVIEW",
        "decision": {},
        "reasons": [],
        "policy_version": "1",
        "auto_action": None,
        "limitations": ["not proof of criminal intent", "human review required"],
        "top_reasons": [
            {"feature": "F3898", "verified_semantic_name": "MIN_RESOLVE_DAYS",
             "value": 1.0, "legitimate_cohort_median": 3.0,
             "legitimate_percentile": 42.7, "shap_contribution": 1.87,
             "direction": "INCREASES_RISK"},
            {"feature": "F3914", "verified_semantic_name": "FALSE_POSITIVE",
             "value": 0.0, "legitimate_cohort_median": 1.0,
             "legitimate_percentile": 5.0, "shap_contribution": 0.9,
             "direction": "INCREASES_RISK"},
        ],
        "counterfactual_twin": {"reference": "LEGIT-0001", "distance": 0.4,
                                "differences": []},
    }


@pytest.fixture
def retired_case(client) -> str:
    """A case backed by a retired-model payload, inserted the way the
    pipeline itself would have written it on 2026-07-10."""
    from muleguard.api import database as db

    _, _, case_id = db.record_score("DEMO-RETIRED-EVIDENCE", _retired_result())
    assert case_id is not None  # CRITICAL_REVIEW always opens a case
    return case_id


def test_case_detail_is_reachable_but_redacts_retired_evidence(client, retired_case):
    r = client.get(f"/v1/cases/{retired_case}")
    assert r.status_code == 200, r.text
    body = r.json()

    status = body["evidence_status"]
    assert status["admissible_as_current_evidence"] is False
    assert status["reason"] == "RETIRED_EVIDENCE"
    assert "top_reasons" not in body["score"]
    assert "counterfactual_twin" not in body["score"]
    assert "evidence_withheld" in body["score"]

    # Naming *which* quarantined columns were withheld is deliberate audit
    # metadata: evidence_status and score.evidence_withheld both carry
    # quarantined_features_used, the same disclosure
    # /proofgraph/{id}/provenance makes on purpose (see evidence_guard's
    # module docstring). What must never reach the wire again is the withheld
    # evidence *itself* - the SHAP contribution, cohort percentile and
    # semantic name a reason row carries. Those keys exist nowhere except
    # inside a top_reasons/counterfactual_twin object, so their total absence
    # proves the evidence was dropped, not merely its column codes hidden.
    text = json.dumps(body)
    for evidentiary_key in ("shap_contribution", "legitimate_cohort_median",
                           "legitimate_percentile", "verified_semantic_name"):
        assert evidentiary_key not in text, (
            f"{evidentiary_key!r} leaked - withheld evidence reached the "
            "response, not just the name of the column it came from")


def test_proofgraph_refuses_a_retired_case(client, retired_case):
    r = client.get(f"/v1/proofgraph/{retired_case}")
    assert r.status_code == 409, r.text
    assert r.json()["detail"]["error"] == "RETIRED_EVIDENCE"


def test_courtroom_also_refuses_a_retired_case(client, retired_case):
    """The courtroom view calls the same _build() as /proofgraph; it must not
    be a second route someone forgot to gate."""
    r = client.get(f"/v1/courtroom/{retired_case}")
    assert r.status_code == 409, r.text
    assert r.json()["detail"]["error"] == "RETIRED_EVIDENCE"


def test_provenance_endpoint_serves_the_labelled_audit_record(client, retired_case):
    """The one place a retired payload's quarantined columns are allowed to
    appear: named explicitly, as an audit record, never as a graph."""
    r = client.get(f"/v1/proofgraph/{retired_case}/provenance")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["provenance"]["status"] == "RETIRED"
    assert body["admissible_as_current_evidence"] is False
    named = {q["feature"] for q in body["quarantined_features_used"]}
    assert named == {"F3898", "F3914"}
    assert named <= QUARANTINED  # both incident columns are on the firewall's own list


def test_report_generation_refuses_a_retired_case_without_reaching_the_llm(
        client, retired_case, monkeypatch):
    from muleguard.api import main as api_main

    def _must_not_be_called():
        raise AssertionError(
            "the narrator must never be reached for evidence that was "
            "refused before the narrative-building step")

    monkeypatch.setattr(api_main, "get_narrator", _must_not_be_called)

    r = client.post(f"/v1/reports/{retired_case}/generate")
    assert r.status_code == 409, r.text
    assert r.json()["detail"]["error"] == "RETIRED_EVIDENCE"


def test_report_generation_refuses_even_with_llm_disabled(client, retired_case):
    """use_llm=false still goes through case_detail()'s gate; the refusal
    must not be a side effect of the narrator being reachable."""
    r = client.post(f"/v1/reports/{retired_case}/generate?use_llm=false")
    assert r.status_code == 409, r.text
    assert r.json()["detail"]["error"] == "RETIRED_EVIDENCE"


def test_current_model_case_still_serves_full_admissible_evidence(client, sample_features):
    """Regression guard: the fix must not have made a healthy, current-model
    case unreachable or stripped its evidence."""
    r = client.post("/v1/score", json={"account_reference": "ACC-EVIDENCE-REGRESSION",
                                       "features": sample_features})
    assert r.status_code == 200, r.text
    case_id = r.json().get("case_id")
    if case_id is None:
        pytest.skip("sample row scored MONITOR; no case opened for it")

    detail = client.get(f"/v1/cases/{case_id}").json()
    status = detail["evidence_status"]
    assert status["admissible_as_current_evidence"] is True
    assert status["reason"] is None
    assert "top_reasons" in detail["score"]
    assert "evidence_withheld" not in detail["score"]

    # And the routes that refuse retired evidence must serve current evidence
    # normally - same gate, opposite outcome.
    g = client.get(f"/v1/proofgraph/{case_id}")
    assert g.status_code == 200, g.text
    assert g.json()["provenance"]["status"] == "CURRENT"
