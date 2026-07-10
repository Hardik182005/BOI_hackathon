"""API integration tests over the real bundle (skipped until it exists).

Uses FastAPI TestClient with a temp database per test session.
"""
import json
import os
import tempfile
from pathlib import Path

import pytest

from muleguard import settings

BUNDLE = settings.MODELS_DIR / "final_bundle.joblib"
pytestmark = pytest.mark.skipif(not BUNDLE.exists(), reason="final bundle not built yet")


@pytest.fixture(scope="module")
def client():
    tmp = tempfile.mkdtemp()
    os.environ["MULEGUARD_DB"] = str(Path(tmp) / "test.db")
    # database module reads env at import; re-import fresh
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


def test_health_live(client):
    r = client.get("/health/live")
    assert r.status_code == 200


def test_health_ready_reports_ollama_optional(client):
    r = client.get("/health/ready")
    assert r.status_code == 200
    assert r.json()["ollama_required"] is False


def test_score_roundtrip_and_determinism(client, sample_features):
    req = {"account_reference": "ACC-TEST-001", "features": sample_features}
    r1 = client.post("/v1/score", json=req)
    assert r1.status_code == 200, r1.text
    r2 = client.post("/v1/score", json=req)
    assert r1.json()["calibrated_risk"] == r2.json()["calibrated_risk"]
    assert r1.json()["risk_tier"] == r2.json()["risk_tier"]
    body = r1.json()
    assert 0.0 <= body["calibrated_risk"] <= 1.0
    assert body["limitations"]


def test_score_rejects_target_in_features(client, sample_features):
    bad = dict(sample_features)
    bad[settings.TARGET_COLUMN] = 1
    r = client.post("/v1/score", json={"account_reference": "ACC-X", "features": bad})
    assert r.status_code == 422


def test_score_missing_feature_schema_error(client, sample_features):
    from muleguard.models.scoring import load_bundle

    b = load_bundle()
    bad = dict(sample_features)
    for f in b["feature_list_selected"][:3]:
        bad.pop(f, None)
    r = client.post("/v1/score", json={"account_reference": "ACC-X", "features": bad})
    assert r.status_code == 422
    assert "SCHEMA_ERROR" in r.json()["detail"]


def test_batch_score(client, sample_features):
    reqs = [{"account_reference": f"ACC-B{i}", "features": sample_features} for i in range(3)]
    r = client.post("/v1/score/batch", json={"accounts": reqs})
    assert r.status_code == 200
    assert r.json()["n_scored"] == 3


def test_case_workflow_and_audit(client, sample_features):
    # force a case by scoring; any tier != MONITOR creates one, else create via decision on 404
    r = client.post("/v1/score", json={"account_reference": "ACC-CASE-1",
                                       "features": sample_features})
    case_id = r.json().get("case_id")
    if case_id is None:
        pytest.skip("sample row scored MONITOR; case workflow covered in e2e demo")
    d = client.post(f"/v1/cases/{case_id}/decision",
                    json={"actor": "analyst.k", "action": "ASSIGN",
                          "reason": "taking ownership for review"})
    assert d.status_code == 200
    f = client.post(f"/v1/cases/{case_id}/feedback",
                    json={"actor": "analyst.k", "verdict": "INCONCLUSIVE"})
    assert f.status_code == 200
    detail = client.get(f"/v1/cases/{case_id}").json()
    assert detail["actions"] and detail["feedback"]


def test_freeze_recommendation_requires_approver(client, sample_features):
    r = client.post("/v1/score", json={"account_reference": "ACC-CASE-2",
                                       "features": sample_features})
    case_id = r.json().get("case_id")
    if case_id is None:
        pytest.skip("sample row scored MONITOR")
    d = client.post(f"/v1/cases/{case_id}/decision",
                    json={"actor": "analyst.k", "action": "RECOMMEND_FREEZE",
                          "reason": "high risk, verified drivers"})
    assert d.status_code == 422  # no approved_by -> rejected


def test_report_generation_without_llm(client, sample_features):
    r = client.post("/v1/score", json={"account_reference": "ACC-CASE-3",
                                       "features": sample_features})
    case_id = r.json().get("case_id")
    if case_id is None:
        pytest.skip("sample row scored MONITOR")
    rep = client.post(f"/v1/reports/{case_id}/generate?use_llm=false")
    assert rep.status_code == 200
    body = rep.json()
    assert body["narrative"]["source"] == "deterministic"
    assert body["narrative"]["narrative"]["verified_risk_score"] == body["risk_and_uncertainty"]["calibrated_risk"]


def test_metrics_summary_served_from_artifacts(client):
    r = client.get("/v1/metrics/summary")
    assert r.status_code == 200
    assert "oof" in r.json()


def test_audit_log_append_only(client):
    import sqlite3

    from muleguard.api import database as db
    with pytest.raises(sqlite3.DatabaseError):
        with db.connect() as c:
            c.execute("UPDATE audit_events SET actor='hacker' WHERE id=1")
