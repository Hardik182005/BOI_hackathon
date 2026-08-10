"""ProofGraph API: a real scored case must yield traceable, two-sided evidence."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from muleguard import settings

BUNDLE = settings.MODELS_DIR / "final_bundle.joblib"
pytestmark = pytest.mark.skipif(not BUNDLE.exists(),
                                reason="final bundle not built yet")


@pytest.fixture(scope="module")
def client():
    tmp = tempfile.mkdtemp()
    os.environ["MULEGUARD_DB"] = str(Path(tmp) / "proofgraph_api.db")
    import importlib

    from muleguard.api import database
    importlib.reload(database)
    from muleguard.api import main as api_main
    importlib.reload(api_main)
    from fastapi.testclient import TestClient

    with TestClient(api_main.app) as c:
        yield c


@pytest.fixture(scope="module")
def case_id(client) -> str:
    """Score the highest-risk row we can find so the graph has something to say."""
    from muleguard.data import ingest

    df = ingest.load_dataset().head(60)
    best, best_risk = None, -1.0
    for row in df.to_dicts():
        row.pop(settings.TARGET_COLUMN, None)
        feats = {k: (str(v) if hasattr(v, "isoformat") else v)
                 for k, v in row.items()}
        r = client.post("/v1/score", json={"account_reference": "ACC-PG-TEST",
                                           "features": feats})
        assert r.status_code == 200, r.text
        body = r.json()
        if body["calibrated_risk"] > best_risk:
            best, best_risk = body["case_id"], body["calibrated_risk"]
    return best


def test_graph_is_traceable_and_two_sided(client, case_id):
    r = client.get(f"/v1/proofgraph/{case_id}")
    assert r.status_code == 200, r.text
    g = r.json()

    assert g["case_id"] == case_id
    assert g["nodes"] and g["edges"]
    for n in g["nodes"]:
        assert n["source"], f"node {n['id']} arrived without provenance"
    ids = {n["id"] for n in g["nodes"]}
    for e in g["edges"]:
        assert e["source"] in ids and e["target"] in ids
    assert g["evidence_counts"]["prosecution"] >= 1
    assert g["courtroom"]["verdict"]


def test_graph_never_uses_forbidden_language(client, case_id):
    body = str(client.get(f"/v1/proofgraph/{case_id}").json()).lower()
    for term in ("guilty", "criminal", "certified clean", "auto_freeze",
                 "permanently safe", "confirmed mule"):
        assert term not in body


def test_graph_invents_no_counterparty_edges(client, case_id):
    """UPDATE 8: DataSet.xlsx has no sender/receiver, so neither may the graph."""
    g = client.get(f"/v1/proofgraph/{case_id}").json()
    relations = {e["relation"] for e in g["edges"]}
    assert not (relations & {"SENT_TO", "RECEIVED_FROM", "TRANSFERRED_TO",
                             "COUNTERPARTY"})


def test_courtroom_endpoint_matches_the_graph(client, case_id):
    g = client.get(f"/v1/proofgraph/{case_id}").json()
    c = client.get(f"/v1/courtroom/{case_id}").json()
    assert c["courtroom"]["verdict"] == g["courtroom"]["verdict"]
    assert c["calibrated_risk"] == g["calibrated_risk"]
    assert "No language model contributed a fact" in c["note"]


def test_disagreement_is_reported_without_moving_the_score(client, case_id):
    """UPDATE 6: the graph reports spread; the score is whatever scoring said."""
    detail = client.get(f"/v1/cases/{case_id}").json()
    g = client.get(f"/v1/proofgraph/{case_id}").json()
    assert g["calibrated_risk"] == pytest.approx(
        detail["score"]["calibrated_risk"])
    assert g["disagreement"]["status"] in ("MODEL_CONSENSUS", "PARTIAL_AGREEMENT",
                                           "MODEL_DISAGREEMENT")


def test_unknown_case_is_a_404(client):
    assert client.get("/v1/proofgraph/NO-SUCH-CASE").status_code == 404
    assert client.get("/v1/courtroom/NO-SUCH-CASE").status_code == 404
