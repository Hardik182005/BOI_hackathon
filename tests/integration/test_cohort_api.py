"""Cohort Radar over HTTP: retrieval that a judge can drive, and cannot subvert.

The API is where the two USP guarantees meet a hostile input. A caller can send
a target column, a quarantined field, a resolution flag, a label - and the
response must be identical to the one they would have got without it. A caller
can also ask about an account nobody has scored, and the answer must be a
refusal rather than somebody else's row.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from muleguard import settings

BUNDLE = settings.MODELS_DIR / "final_bundle.joblib"
TRANSFORM = settings.MODELS_DIR / "cohort_radar_transform.joblib"
pytestmark = pytest.mark.skipif(
    not (BUNDLE.exists() and TRANSFORM.exists()),
    reason="champion bundle or cohort transform not built")


@pytest.fixture(scope="module")
def client():
    tmp = tempfile.mkdtemp()
    os.environ["MULEGUARD_DB"] = str(Path(tmp) / "cohort_api.db")
    import importlib

    from muleguard.api import database
    importlib.reload(database)
    from muleguard.api import main as api_main
    importlib.reload(api_main)
    from fastapi.testclient import TestClient

    with TestClient(api_main.app) as c:
        yield c


@pytest.fixture(scope="module")
def scored(client):
    """Score real rows until one opens a case, then keep that case."""
    from muleguard.data import ingest

    for row in ingest.load_dataset().head(80).to_dicts():
        # The scoring path rejects the target outright; the cohort path merely
        # ignores it. Both are correct, and the difference is why the tests
        # below send F3924 to /v1/cohort/search and not to /v1/score.
        row.pop(settings.TARGET_COLUMN, None)
        payload = {k: (str(v) if hasattr(v, "isoformat") else v)
                   for k, v in row.items()}
        r = client.post("/v1/score", json={"account_reference": "ACC-COHORT-1",
                                           "features": payload})
        assert r.status_code == 200, r.text
        body = r.json()
        if body.get("case_id"):
            return {"case_id": body["case_id"], "features": payload, "score": body}
    pytest.skip("no row in the sample opened a case")


def _refs(body) -> list[str]:
    return [n["account_reference"] for n in body["neighbors"]]


# --------------------------------------------------------------------------
# the manifest
# --------------------------------------------------------------------------


def test_manifest_is_served_and_declares_its_own_irrelevance_to_the_score(client):
    body = client.get("/v1/cohort/manifest").json()
    assert body["affects_model_score"] is False
    assert body["edge_relationship"] == "BEHAVIORALLY_SIMILAR_TO"
    assert body["quarantine"]["target_in_fingerprint"] is False
    assert body["disclaimer"]


# --------------------------------------------------------------------------
# cohort for a scored case
# --------------------------------------------------------------------------


def test_a_scored_case_gets_a_cohort_carrying_its_own_unchanged_risk(client, scored):
    r = client.get(f"/v1/cases/{scored['case_id']}/cohort?k=5")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["case_id"] == scored["case_id"]
    assert len(body["neighbors"]) == 5
    # The risk is read from the case, never recomputed by the radar.
    assert body["risk_probability"] == pytest.approx(
        scored["score"]["calibrated_risk"], abs=0.0)
    assert body["risk_tier"] == scored["score"]["risk_tier"]
    assert "unchanged by this lookup" in body["risk_source"]


def test_the_cohort_response_always_carries_the_disclaimer(client, scored):
    body = client.get(f"/v1/cases/{scored['case_id']}/cohort").json()
    assert "does not establish" in body["disclaimer"]
    assert "prioritisation signal" in body["action_policy"]


def test_k_is_capped_rather_than_honoured(client, scored):
    body = client.get(f"/v1/cases/{scored['case_id']}/cohort?k=9999").json()
    assert body["k"] == 25
    assert len(body["neighbors"]) <= 25


def test_an_unknown_case_is_a_404_not_a_guess(client):
    assert client.get("/v1/cases/CASE-DOES-NOT-EXIST/cohort").status_code == 404


def test_every_neighbour_is_a_reference_frame_row_never_a_customer_identifier(
        client, scored):
    body = client.get(f"/v1/cases/{scored['case_id']}/cohort?k=10").json()
    for ref in _refs(body):
        assert ref.startswith("RV-"), "the dataset carries no account numbers"


def test_the_query_account_is_never_returned_as_its_own_neighbour(client, scored):
    body = client.get(f"/v1/cases/{scored['case_id']}/cohort?k=10").json()
    assert body["query_account"] not in _refs(body)


# --------------------------------------------------------------------------
# search: what a caller may and may not influence
# --------------------------------------------------------------------------


@pytest.mark.parametrize("col,value", [
    ("F3924", 1), ("F3912", 999.0), ("F3918", -999.0),
    ("F2230", 12345.0), ("F3898", 1.0), ("F3899", 1.0),
])
def test_a_caller_cannot_move_retrieval_with_a_forbidden_column(client, scored,
                                                                col, value):
    base = client.post("/v1/cohort/search",
                       json={"features": scored["features"], "k": 10}).json()
    tampered = dict(scored["features"])
    tampered[col] = value
    after = client.post("/v1/cohort/search",
                        json={"features": tampered, "k": 10}).json()
    assert _refs(base) == _refs(after)
    assert [n["behavioral_similarity"] for n in base["neighbors"]] == \
           [n["behavioral_similarity"] for n in after["neighbors"]]


def test_a_label_column_supplied_by_a_judge_changes_nothing(client, scored):
    """Section 19. A judge upload arrives with the answer key attached; the
    radar must be provably unable to read it."""
    labelled = dict(scored["features"])
    labelled.update({"F3924": 1, "label": 1, "target": 1, "is_mule": "yes"})
    a = client.post("/v1/cohort/search", json={"features": labelled, "k": 10}).json()
    b = client.post("/v1/cohort/search",
                    json={"features": scored["features"], "k": 10}).json()
    assert _refs(a) == _refs(b)


def test_the_same_query_twice_returns_the_same_neighbours(client, scored):
    a = client.post("/v1/cohort/search",
                    json={"features": scored["features"], "k": 10}).json()
    b = client.post("/v1/cohort/search",
                    json={"features": scored["features"], "k": 10}).json()
    assert a["neighbors"] == b["neighbors"]
    assert a["mutual_edges"] == b["mutual_edges"]


def test_a_query_with_nothing_in_it_is_refused_not_answered(client):
    body = client.post("/v1/cohort/search",
                       json={"features": {"F1": None}, "k": 5}).json()
    assert body["cohort_summary"]["insufficient_data"] is True
    assert body["neighbors"] == []
    assert body["data_sufficiency"]["status"] == "INSUFFICIENT_DATA"


def test_search_needs_something_to_search_for(client):
    assert client.post("/v1/cohort/search", json={"k": 5}).status_code == 422


def test_an_account_nobody_scored_is_a_refusal_not_a_neighbouring_row(client):
    r = client.post("/v1/cohort/search",
                    json={"account_reference": "ACC-NEVER-SEEN", "k": 5})
    assert r.status_code == 404
    assert r.json()["detail"]["error"] == "ACCOUNT_NOT_IN_REFERENCE_FRAME"


def test_a_reference_frame_row_can_be_looked_up_directly(client):
    from muleguard.usp import cohort_radar

    row = int(cohort_radar.reference_row_index()[0])
    body = client.post("/v1/cohort/search",
                       json={"row_index": row, "k": 5}).json()
    assert body["query_account"] == f"RV-{row}"
    assert len(body["neighbors"]) == 5


def test_a_locked_test_row_cannot_be_reached_through_the_api(client):
    """The held-out set is held out from this layer too. A reference that
    resolves into it would make the locked test a thing the product consults."""
    import numpy as np

    from muleguard.data import split as split_mod

    locked = int(np.flatnonzero(np.asarray(split_mod.load_locked_test_mask()))[0])
    r = client.post("/v1/cohort/search",
                    json={"account_reference": f"RV-{locked}", "k": 5})
    assert r.status_code == 404


# --------------------------------------------------------------------------
# sections 22-23 over the wire
# --------------------------------------------------------------------------


def test_the_case_detail_carries_the_control_attribution_card(client, scored):
    body = client.get(f"/v1/cases/{scored['case_id']}").json()
    card = body["control_attribution"]
    assert card["account_control_evidence"]["status"] == "NOT_AVAILABLE"
    assert card["intent_attribution"]["status"] == "UNKNOWN"
    assert card["automatic_actions_permitted"] == []
    assert card["affects_model_output"] is False


def test_the_proofgraph_wires_the_limitation_to_the_decision_not_the_score(
        client, scored):
    """Section 22, checked on the wire. RAISED_BY would present a limitation as
    a reason for the conclusion."""
    graph = client.get(f"/v1/proofgraph/{scored['case_id']}").json()
    control = [n for n in graph["nodes"] if n["type"] == "CONTROL_ATTRIBUTION"]
    assert len(control) == 1
    edges = [e for e in graph["edges"] if e["target"] == control[0]["id"]]
    assert len(edges) == 1
    assert edges[0]["source"] == "decision"
    assert edges[0]["relation"] == "REQUIRES_HUMAN_VERIFICATION"
    assert control[0]["weight"] == 0.0


def test_the_limitation_node_is_not_counted_as_evidence(client, scored):
    graph = client.get(f"/v1/proofgraph/{scored['case_id']}").json()
    counts = graph["evidence_counts"]
    assert "CONTROL_ATTRIBUTION" not in counts


def test_looking_at_a_cohort_does_not_change_the_case(client, scored):
    """Section 23: cohort membership escalates nobody, including the subject."""
    before = client.get(f"/v1/cases/{scored['case_id']}").json()["case"]
    client.get(f"/v1/cases/{scored['case_id']}/cohort?k=25")
    after = client.get(f"/v1/cases/{scored['case_id']}").json()["case"]
    assert before["calibrated_risk"] == after["calibrated_risk"]
    assert before["risk_tier"] == after["risk_tier"]
    assert before["status"] == after["status"]


def test_a_neighbour_is_not_escalated_by_appearing_in_a_cohort(client, scored):
    """The neighbours are reported with the tier their own case would carry;
    appearing in someone else's panel does not open a case for them."""
    body = client.get(f"/v1/cases/{scored['case_id']}/cohort?k=10").json()
    with __import__("muleguard.api.database", fromlist=["x"]).connect() as c:
        rows = c.execute("SELECT account_reference FROM cases").fetchall()
    opened = {r["account_reference"] for r in rows}
    for ref in _refs(body):
        assert ref not in opened
