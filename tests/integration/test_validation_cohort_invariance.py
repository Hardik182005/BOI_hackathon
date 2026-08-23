"""Sections 28 and 29: a judge's labels must be inert.

An organiser uploads a file with the answer key still in it. Two things have to
be true, and neither is true by accident. The frozen champion must produce the
same scores it would have produced from the same file with the target removed -
otherwise the label reached the model. And the Cohort Radar must return the same
neighbours in the same order - otherwise the label reached retrieval, which is
the subtler failure, because a cohort conditioned on labels looks like a very
good cohort right up until the labels are not there.

The test is the spec's own: the same file, present versus removed, compared
end to end.
"""
from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path

import polars as pl
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
    os.environ["MULEGUARD_DB"] = str(Path(tmp) / "validation_cohort.db")
    import importlib

    from muleguard.api import database
    importlib.reload(database)
    from muleguard.api import main as api_main
    importlib.reload(api_main)
    from fastapi.testclient import TestClient

    with TestClient(api_main.app) as c:
        yield c


@pytest.fixture(scope="module")
def uploads() -> dict:
    """The same 40 rows twice: labels in, labels out."""
    from muleguard.data import ingest

    df = ingest.load_dataset().head(40)
    stripped = df.drop(settings.TARGET_COLUMN)

    def csv(frame: pl.DataFrame) -> bytes:
        buf = io.BytesIO()
        frame.write_csv(buf)
        return buf.getvalue()

    return {"labelled": csv(df), "unlabelled": csv(stripped), "frame": df}


def _run(client, payload: bytes, name: str):
    r = client.post("/v1/validation/run",
                    files={"file": (name, payload, "text/csv")})
    assert r.status_code == 200, r.text
    return r.json()


# --------------------------------------------------------------------------
# section 28 - the mandatory present-vs-removed test
# --------------------------------------------------------------------------


def test_the_same_file_with_and_without_the_target_scores_identically(client, uploads):
    """The frozen champion half of the guarantee."""
    a = _run(client, uploads["labelled"], "labelled.csv")
    b = _run(client, uploads["unlabelled"], "unlabelled.csv")

    pa = client.get(f"/v1/validation/seals/{a['seal_id']}/predictions").text
    pb = client.get(f"/v1/validation/seals/{b['seal_id']}/predictions").text
    assert pa == pb, "the presence of a label changed a prediction"

    dist_a = [s for s in a["steps"] if s["step"] == 3][0]["distributions"]
    dist_b = [s for s in b["steps"] if s["step"] == 3][0]["distributions"]
    assert dist_a == dist_b


def test_the_upload_is_recognised_as_labelled_only_when_it_is(client, uploads):
    a = _run(client, uploads["labelled"], "labelled.csv")
    b = _run(client, uploads["unlabelled"], "unlabelled.csv")
    assert a["labels_available_for_reveal"] is True
    assert a["target_column_detected"] == settings.TARGET_COLUMN
    assert b["labels_available_for_reveal"] is False


def test_the_cohort_ranking_is_identical_with_and_without_the_target(client, uploads):
    """The retrieval half. This is the test section 28 calls mandatory."""
    from muleguard.features.frame import attach_meta

    frame = uploads["frame"]
    stripped = frame.drop(settings.TARGET_COLUMN)
    rows_with = attach_meta(frame).to_dicts()
    rows_without = attach_meta(stripped).to_dicts()

    for i in range(0, len(rows_with), 7):     # a spread of rows, not just the first
        with_label = {k: (str(v) if hasattr(v, "isoformat") else v)
                      for k, v in rows_with[i].items()}
        without = {k: (str(v) if hasattr(v, "isoformat") else v)
                   for k, v in rows_without[i].items()}
        assert settings.TARGET_COLUMN in with_label
        assert settings.TARGET_COLUMN not in without

        a = client.post("/v1/cohort/search",
                        json={"features": with_label, "k": 10}).json()
        b = client.post("/v1/cohort/search",
                        json={"features": without, "k": 10}).json()
        assert a["neighbors"] == b["neighbors"], f"row {i}: the label moved retrieval"
        assert a["mutual_edges"] == b["mutual_edges"]


# --------------------------------------------------------------------------
# section 29 - the sealed protocol still holds
# --------------------------------------------------------------------------


def test_a_sealed_run_still_discloses_no_metric(client, uploads):
    """The USP additions must not have opened a side channel. A cohort panel
    that reported neighbour *accuracy* would disclose labels by another route."""
    body = _run(client, uploads["labelled"], "labelled.csv")
    assert body["ui_state"] == "PREDICTIONS_SEALED"
    flat = str(body).lower()
    for forbidden in ("pr_auc", "roc_auc", "recall_at_budget",
                      "precision_at_budget", "hit@5", "hit_at_5"):
        assert forbidden not in flat


def test_the_cohort_response_carries_no_label_derived_field(client, uploads):
    """What the radar may read: scores, safe feature rows, pattern cards. The
    absence of a label field is checked rather than assumed."""
    from muleguard.features.frame import attach_meta

    row = {k: (str(v) if hasattr(v, "isoformat") else v)
           for k, v in attach_meta(uploads["frame"]).to_dicts()[0].items()}
    body = client.post("/v1/cohort/search", json={"features": row, "k": 5}).json()

    banned = {settings.TARGET_COLUMN, "label", "is_mule", "y_true", "actual",
              "ground_truth", "outcome"}
    def walk(node):
        if isinstance(node, dict):
            for k, v in node.items():
                assert k not in banned, f"{k} reached a cohort response"
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
    walk(body)


def test_the_radar_cannot_be_asked_for_a_label(client, uploads):
    """The request model has no field for one, so a caller holding the answer
    key has nowhere to put it."""
    from muleguard.api.routes_cohort import CohortSearchRequest

    assert settings.TARGET_COLUMN not in CohortSearchRequest.model_fields
    for name in ("label", "labels", "y", "target", "is_mule"):
        assert name not in CohortSearchRequest.model_fields


def test_reveal_still_works_after_the_usp_additions(client, uploads):
    """Section 29 preserves the existing protocol; the reveal is the half that
    would break first if the seal had changed shape."""
    seal_id = _run(client, uploads["labelled"], "labelled.csv")["seal_id"]
    r = client.post(f"/v1/validation/{seal_id}/reveal",
                    files={"file": ("labels.csv", uploads["labelled"], "text/csv")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["verification"]["verified"] is True
    assert body["seal"]["state"] == "METRICS_REVEALED"
