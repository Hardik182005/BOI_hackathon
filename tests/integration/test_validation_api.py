"""Validation Lab API: the three-step order and the seal, end to end.

These run against the frozen bundle, so they are skipped until one is built.
The point of the suite is the *protocol*, not the numbers: a run must return a
seal and no metric, a reveal must refuse tampered predictions, and step 3 must
be unreachable when step 1 failed.
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
pytestmark = pytest.mark.skipif(not BUNDLE.exists(),
                                reason="final bundle not built yet")


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    tmp = tempfile.mkdtemp()
    os.environ["MULEGUARD_DB"] = str(Path(tmp) / "validation_api.db")
    import importlib

    from muleguard.api import database
    importlib.reload(database)
    from muleguard.api import main as api_main
    importlib.reload(api_main)
    from fastapi.testclient import TestClient

    with TestClient(api_main.app) as c:
        yield c


@pytest.fixture(scope="module")
def labelled_csv() -> tuple[bytes, pl.DataFrame]:
    """A small slice of the real dataset, labels included.

    Labels are left in the file deliberately: the endpoint is responsible for
    withholding them, and a test that pre-strips them would not exercise that.
    """
    from muleguard.data import ingest

    df = ingest.load_dataset().head(40)
    buf = io.BytesIO()
    df.write_csv(buf)
    return buf.getvalue(), df


def _post(client, payload: bytes, name="validation_slice.csv"):
    return client.post("/v1/validation/run",
                       files={"file": (name, payload, "text/csv")})


def test_run_returns_a_seal_and_no_metric(client, labelled_csv):
    payload, _ = labelled_csv
    r = _post(client, payload)
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["overall"] in ("PASS", "WARN")
    assert [s["step"] for s in body["steps"]] == [1, 2, 3]
    assert body["ui_state"] == "PREDICTIONS_SEALED"
    assert body["seal_id"]
    assert body["labels_available_for_reveal"] is True
    assert body["target_column_detected"] == settings.TARGET_COLUMN

    # nothing in the sealed response may leak a label-derived number
    flat = str(body).lower()
    for forbidden in ("pr_auc", "roc_auc", "recall_at_budget", "precision_at_budget"):
        assert forbidden not in flat, f"{forbidden} was disclosed before the reveal"


def test_reveal_reports_metrics_only_after_the_seal_verifies(client, labelled_csv):
    payload, df = labelled_csv
    seal_id = _post(client, payload).json()["seal_id"]

    r = client.post(f"/v1/validation/{seal_id}/reveal",
                    files={"file": ("labels.csv", payload, "text/csv")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["verification"]["verified"] is True
    assert body["metrics"]["status"] in ("OK", "SINGLE_CLASS_ONLY")
    assert body["seal"]["state"] == "METRICS_REVEALED"
    assert "before any label was read" in body["integrity_statement"]


def test_reveal_rejects_a_row_count_mismatch(client, labelled_csv):
    payload, df = labelled_csv
    seal_id = _post(client, payload).json()["seal_id"]

    short = io.BytesIO()
    df.head(10).write_csv(short)
    r = client.post(f"/v1/validation/{seal_id}/reveal",
                    files={"file": ("labels.csv", short.getvalue(), "text/csv")})
    assert r.status_code == 422
    assert "one to one" in r.json()["detail"]


def test_reveal_refuses_predictions_edited_after_sealing(client, labelled_csv):
    """The seal is only worth something if breaking it is fatal."""
    from muleguard.validation import sealed

    payload, _ = labelled_csv
    seal_id = _post(client, payload).json()["seal_id"]
    rec = sealed.load_seal(seal_id)
    preds = pl.read_csv(rec.prediction_path)
    preds.with_columns(pl.col("risk_score").reverse()).write_csv(rec.prediction_path)

    r = client.post(f"/v1/validation/{seal_id}/reveal",
                    files={"file": ("labels.csv", payload, "text/csv")})
    assert r.status_code == 409
    assert "SEAL_BROKEN" in r.json()["detail"]


def test_a_file_missing_required_features_stops_at_step_1(client):
    bad = pl.DataFrame({"F1": [1.0, 2.0], "F2": [3.0, 4.0]})
    buf = io.BytesIO()
    bad.write_csv(buf)
    r = _post(client, buf.getvalue(), name="too_few_columns.csv")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["overall"] == "FAIL"
    assert body["stopped_at_step"] == 1
    assert "seal_id" not in body          # nothing was sealed
    assert len(body["steps"]) == 1        # the shield never ran


def test_seals_are_listable_and_verifiable(client, labelled_csv):
    payload, _ = labelled_csv
    seal_id = _post(client, payload).json()["seal_id"]

    listing = client.get("/v1/validation/seals")
    assert listing.status_code == 200
    assert any(s["seal_id"] == seal_id for s in listing.json()["seals"])

    one = client.get(f"/v1/validation/seals/{seal_id}")
    assert one.status_code == 200
    assert one.json()["verification"]["verified"] is True
    assert len(one.json()["seal"]["prediction_sha256"]) == 64


def test_unknown_seal_is_a_404(client):
    assert client.get("/v1/validation/seals/seal-does-not-exist").status_code == 404
