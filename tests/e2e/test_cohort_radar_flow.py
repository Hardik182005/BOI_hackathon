"""Section 35: the four-step judge demo, driven end to end over HTTP.

The demo is a claim about the product, so it is tested as one. Each step below
is a step of the script, in order, through the real API:

    1. open a high-risk account
    2. show WHY FLAGGED and WHY THIS MIGHT BE A FALSE POSITIVE
    3. FIND BEHAVIOURAL COHORT - 5 to 10 unusually similar accounts
    4. show Control Attribution

Two things are checked at every step. The first is that the step works. The
second is that the classifier's answer for the account is the same number
before the demo, in the middle of it, and after it - because the whole point of
the sequence is that steps 3 and 4 add context without touching the score.
"""
from __future__ import annotations

import importlib
import os

import pytest
from fastapi.testclient import TestClient

from muleguard import settings

BUNDLE = settings.MODELS_DIR / "final_bundle.joblib"
TRANSFORM = settings.MODELS_DIR / "cohort_radar_transform.joblib"
pytestmark = pytest.mark.skipif(not BUNDLE.exists(), reason="final bundle not built")


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """A live app on a throwaway database.

    A temporary DB rather than the developer's own: the demo opens cases, and a
    test that leaves cases behind in the working database changes what the next
    person sees on the dashboard.
    """
    os.environ["MULEGUARD_DB"] = str(tmp_path_factory.mktemp("demo") / "demo.db")
    from muleguard.api import database
    importlib.reload(database)
    from muleguard.api import main as api_main
    importlib.reload(api_main)
    with TestClient(api_main.app) as c:
        yield c


@pytest.fixture(scope="module")
def demo_rows():
    """Development rows only. The locked test is not demo material."""
    import polars as pl
    from muleguard.data import ingest, split as split_mod

    df = ingest.load_dataset()
    mask = split_mod.load_locked_test_mask()
    return df.filter(~pl.Series(mask)).head(120)


def _payload(row: dict) -> dict:
    row = dict(row)
    row.pop(settings.TARGET_COLUMN, None)
    return {k: (str(v) if hasattr(v, "isoformat") else v) for k, v in row.items()}


@pytest.fixture(scope="module")
def demo_case(client, demo_rows):
    """Step 1: a scored, high-risk account with an open case.

    The *riskiest* reviewable account, not the first one found. The demo script
    says "open a high-risk account", and step 2 asks for the prosecution
    column - a borderline case can legitimately have every reason pointing the
    other way, which would make step 2 fail on the data rather than on the code.
    """
    opened = []
    for i, row in enumerate(demo_rows.to_dicts()):
        r = client.post("/v1/score", json={
            "account_reference": f"DEMO-{i:04d}", "features": _payload(row)})
        if r.status_code != 200:
            continue
        body = r.json()
        if body.get("case_id") and body["risk_tier"] != "MONITOR":
            opened.append(body)
    if not opened:
        pytest.skip("no reviewable case opened from the demo rows")
    return max(opened, key=lambda b: float(b["calibrated_risk"]))


# --------------------------------------------------------------------------
# step 1 - open a high-risk account
# --------------------------------------------------------------------------


def test_step_1_the_account_opens_with_a_risk_and_a_tier(client, demo_case):
    detail = client.get(f"/v1/cases/{demo_case['case_id']}")
    assert detail.status_code == 200
    case = detail.json()["case"]
    assert case["calibrated_risk"] == pytest.approx(
        demo_case["calibrated_risk"], abs=1e-12)
    assert case["risk_tier"] == demo_case["risk_tier"] != "MONITOR"


# --------------------------------------------------------------------------
# step 2 - why flagged, and why this might be a false positive
# --------------------------------------------------------------------------


def test_step_2_both_columns_of_the_argument_are_available(client, demo_case):
    """WHY FLAGGED and WHY THIS MIGHT BE A FALSE POSITIVE.

    The second column is the one that matters here: a demo that only shows the
    prosecution is the product this upgrade is meant not to be.
    """
    r = client.get(f"/v1/courtroom/{demo_case['case_id']}")
    assert r.status_code == 200
    court = r.json()["courtroom"]
    assert court["prosecution"], "nothing to show under WHY FLAGGED"
    assert "defence" in court, "no false-positive column at all"
    assert court["verdict_rationale"]
    for point in court["prosecution"] + court["defence"]:
        assert point["source"], "an argument with no traceable source"


# --------------------------------------------------------------------------
# step 3 - find behavioural cohort
# --------------------------------------------------------------------------


needs_radar = pytest.mark.skipif(
    not TRANSFORM.exists(),
    reason="cohort transform not built - python -m muleguard.cli.build_cohort_radar")


@needs_radar
def test_step_3_the_cohort_button_returns_five_to_ten_similar_accounts(
        client, demo_case):
    r = client.get(f"/v1/cases/{demo_case['case_id']}/cohort", params={"k": 10})
    assert r.status_code == 200, r.text
    body = r.json()
    assert 5 <= len(body["neighbors"]) <= 10, "the demo asks for 5-10 accounts"
    assert body["case_id"] == demo_case["case_id"]
    for n in body["neighbors"]:
        assert n["account_reference"].startswith("RV-")
        assert n["main_shared_features"], "a neighbour with no stated reason"


@needs_radar
def test_step_3_the_narration_the_presenter_gives_is_the_one_the_api_returns(
        client, demo_case):
    """The demo script's sentence is a promise. This checks the API keeps it.

    "We do not fabricate criminal links" has to be true of the payload, not
    just of the slide - so the edge label, the disclaimer and the language
    guard are all asserted on the response an audience would be shown.
    """
    from muleguard.usp import cohort_radar as cr

    body = client.get(f"/v1/cases/{demo_case['case_id']}/cohort").json()
    assert body["disclaimer"] == cr.DISCLAIMER
    for e in body["mutual_edges"]:
        assert e["relationship"] == cr.EDGE_LABEL == "BEHAVIORALLY_SIMILAR_TO"
    cr.assert_language_safe(body)


@needs_radar
def test_step_3_does_not_change_the_account_it_was_opened_from(client, demo_case):
    """The heart of it. Before, during, after - the same number.

    Read back from the case record rather than from the cohort response, so a
    write to the database would be caught even if the response were consistent
    with itself.
    """
    before = client.get(f"/v1/cases/{demo_case['case_id']}").json()["case"]
    cohort = client.get(f"/v1/cases/{demo_case['case_id']}/cohort",
                        params={"k": 10}).json()
    after = client.get(f"/v1/cases/{demo_case['case_id']}").json()["case"]

    assert after["calibrated_risk"] == before["calibrated_risk"]
    assert after["risk_tier"] == before["risk_tier"]
    assert after["status"] == before["status"]
    # and the panel reports the case's own figure, not one of its own
    assert cohort["risk_probability"] == pytest.approx(
        before["calibrated_risk"], abs=1e-12)
    assert cohort["risk_tier"] == before["risk_tier"]
    assert cohort["risk_source"].startswith("frozen champion classifier")


@needs_radar
def test_step_3_leaves_every_neighbour_exactly_where_it_was(client, demo_case):
    """Section 23: appearing in a cohort must not open a case for anyone."""
    before = {c["case_id"] for c in client.get("/v1/cases").json()["cases"]}
    client.get(f"/v1/cases/{demo_case['case_id']}/cohort", params={"k": 25})
    after = {c["case_id"] for c in client.get("/v1/cases").json()["cases"]}
    assert after == before, "a cohort lookup created or removed a case"


# --------------------------------------------------------------------------
# step 4 - control attribution
# --------------------------------------------------------------------------


def test_step_4_the_limitation_is_stated_on_the_case(client, demo_case):
    """The three concepts, kept apart, in the response the analyst reads."""
    card = client.get(f"/v1/cases/{demo_case['case_id']}").json()["control_attribution"]
    assert card["behavioural_mule_risk"]["status"] == "ASSESSED"
    assert card["account_control_evidence"]["status"] == "NOT_AVAILABLE"
    assert card["intent_attribution"]["status"] == "UNKNOWN"
    assert card["automatic_actions_permitted"] == []
    assert card["affects_model_output"] is False
    assert card["verification_checklist"], "no enrichment checklist to show"


def test_step_4_the_limitation_hangs_off_the_decision_in_the_graph(
        client, demo_case):
    """Section 22 as the demo renders it: below the decision, dashed, weight 0."""
    graph = client.get(f"/v1/proofgraph/{demo_case['case_id']}").json()
    control = [n for n in graph["nodes"] if n["type"] == "CONTROL_ATTRIBUTION"]
    assert len(control) == 1
    edges = [e for e in graph["edges"] if e["target"] == control[0]["id"]]
    assert len(edges) == 1
    assert edges[0]["source"] == "decision"
    assert edges[0]["relation"] == "REQUIRES_HUMAN_VERIFICATION"
    assert control[0]["weight"] == 0.0


# --------------------------------------------------------------------------
# the whole sequence, as one claim
# --------------------------------------------------------------------------


@needs_radar
def test_the_full_demo_leaves_the_score_bit_identical(client, demo_case):
    """Run all four steps in order and re-score the same account afterwards.

    The individual steps above each check their own invariant. This one checks
    the sequence, because the failure this upgrade risks is cumulative: a
    lookup that caches, a card that writes, a graph that recomputes.
    """
    case_id = demo_case["case_id"]
    client.get(f"/v1/cases/{case_id}")
    client.get(f"/v1/courtroom/{case_id}")
    client.get(f"/v1/cases/{case_id}/cohort", params={"k": 10})
    client.get(f"/v1/proofgraph/{case_id}")

    case = client.get(f"/v1/cases/{case_id}").json()["case"]
    assert case["calibrated_risk"] == demo_case["calibrated_risk"]
    assert case["risk_tier"] == demo_case["risk_tier"]


@needs_radar
def test_the_manifest_backs_the_demo_up_if_a_judge_asks(client):
    """Step 3's claim is checkable on the spot, which is the point of saying it."""
    m = client.get("/v1/cohort/manifest").json()
    assert m["affects_model_score"] is False
    assert m["edge_relationship"] == "BEHAVIORALLY_SIMILAR_TO"
    assert m["quarantine"]["target_in_fingerprint"] is False
    assert m["fingerprint"]["features"], "a manifest that names no features"
    assert m["similarity_formula"]
