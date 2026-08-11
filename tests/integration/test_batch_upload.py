"""Batch CSV/XLSX upload safety cases (QA spec section 14)."""
import io
import json
import os
import tempfile
from pathlib import Path

import polars as pl
import pytest

from muleguard import settings

BUNDLE = settings.MODELS_DIR / "final_bundle.joblib"
pytestmark = pytest.mark.skipif(not BUNDLE.exists(), reason="final bundle not built")


@pytest.fixture(scope="module")
def client():
    tmp = tempfile.mkdtemp()
    os.environ["MULEGUARD_DB"] = str(Path(tmp) / "upload_test.db")
    import importlib

    from muleguard.api import database
    importlib.reload(database)
    from muleguard.api import main as api_main
    importlib.reload(api_main)
    from fastapi.testclient import TestClient

    with TestClient(api_main.app) as c:
        yield c


@pytest.fixture(scope="module")
def sample_frame() -> pl.DataFrame:
    from muleguard.data import ingest

    return ingest.load_dataset().head(6)


def _csv_bytes(df: pl.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.write_csv(buf)
    return buf.getvalue()


def _post(client, content: bytes, name="batch.csv"):
    return client.post("/v1/score/file", files={"file": (name, content, "text/csv")})


def test_valid_small_file_scores_and_downloads_csv(client, sample_frame):
    df = sample_frame.drop(settings.TARGET_COLUMN)
    r = _post(client, _csv_bytes(df))
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("text/csv")
    assert int(r.headers["x-rows-scored"]) == 6
    body = r.text
    assert "calibrated_risk" in body and "model_version" in body
    # hidden threshold values never exposed
    assert "critical_risk" not in body


def test_target_column_present_is_ignored_safely(client, sample_frame):
    r = _post(client, _csv_bytes(sample_frame))  # includes F3924
    assert r.status_code == 200
    dropped = json.loads(r.headers["x-dropped-columns"])
    assert settings.TARGET_COLUMN in dropped


def test_missing_selected_feature_schema_error(client, sample_frame):
    from muleguard.models.scoring import load_bundle

    b = load_bundle()
    df = sample_frame.drop([settings.TARGET_COLUMN, b["feature_list_selected"][0]])
    r = _post(client, _csv_bytes(df))
    assert r.status_code == 422
    assert "SCHEMA_ERROR" in r.json()["detail"]


def test_duplicate_columns_rejected(client, sample_frame):
    df = sample_frame.drop(settings.TARGET_COLUMN)
    csv = _csv_bytes(df).decode()
    header, rest = csv.split("\n", 1)
    first_col = header.split(",")[0]
    dup = f"{first_col}," + header + "\n" + rest.replace("\n", ",0\n", rest.count("\n"))
    # simpler: duplicate a column by concatenating header twice
    cols = header.split(",")
    dup_csv = ",".join(cols + [cols[1]]) + "\n"
    for line in rest.strip().split("\n"):
        dup_csv += line + "," + line.split(",")[1] + "\n"
    r = _post(client, dup_csv.encode())
    assert r.status_code == 422


def test_corrupted_file_rejected_without_crash(client):
    r = _post(client, b"\x00\x01\x02 not a csv \xff", name="broken.xlsx")
    assert r.status_code == 422
    r2 = client.get("/health/live")
    assert r2.status_code == 200  # app alive


def test_wrong_extension_rejected(client):
    r = _post(client, b"a,b\n1,2\n", name="data.exe")
    assert r.status_code == 422


def test_oversized_upload_rejected(client, monkeypatch):
    from muleguard.api import routes_upload

    monkeypatch.setattr(routes_upload, "MAX_UPLOAD_BYTES", 100)
    r = _post(client, b"x" * 200)
    assert r.status_code == 413


def test_formula_injection_sanitised_in_output(client, sample_frame):
    df = sample_frame.drop(settings.TARGET_COLUMN)
    r = _post(client, _csv_bytes(df))
    for line in r.text.splitlines():
        for cell in line.split(","):
            assert not cell.startswith(("=", "+", "@")), cell


def test_extra_unknown_columns_ignored(client, sample_frame):
    df = sample_frame.drop(settings.TARGET_COLUMN).with_columns(
        pl.lit("hello").alias("TOTALLY_UNKNOWN_COL")
    )
    r = _post(client, _csv_bytes(df))
    assert r.status_code == 200


def test_reordered_columns_score_identically(client, sample_frame):
    df = sample_frame.drop(settings.TARGET_COLUMN)
    r1 = _post(client, _csv_bytes(df))
    r2 = _post(client, _csv_bytes(df.select(list(reversed(df.columns)))))
    assert r1.status_code == r2.status_code == 200
    risks1 = [l.split(",")[1] for l in r1.text.strip().splitlines()[1:]]
    risks2 = [l.split(",")[1] for l in r2.text.strip().splitlines()[1:]]
    assert risks1 == risks2


# --- schema inference on wide, ragged exports --------------------------------

def test_late_non_numeric_value_does_not_reject_the_file():
    """A column that turns non-numeric deep in the file must not 422 the upload.

    Polars sniffs a fixed number of rows to pick dtypes. On a 3,924-column
    export a column that is numeric for the first 200 rows and carries one
    "N/A" at row 900 raised ComputeError and the whole upload was refused - the
    organiser's file rejected over a single cell. This is the regression test
    for that: the bad cell becomes a null, which the imputer already handles.
    """
    from muleguard.api.routes_upload import _read_csv_tolerantly

    rows = [f"{i},1.5" for i in range(400)]
    rows[350] = "N/A,1.5"          # past the 200-row sniff window
    payload = ("F1,F2\n" + "\n".join(rows) + "\n").encode()

    df = _read_csv_tolerantly(payload)
    assert df.height == 400
    assert df.width == 2


def test_tolerant_reader_still_types_a_clean_file_numerically():
    """The fallbacks must not cost normal files their dtypes.

    Reading everything as text would "work" for every input and quietly turn a
    well-formed export into strings, so the first attempt has to be the one
    that wins here.
    """
    from muleguard.api.routes_upload import _read_csv_tolerantly

    payload = b"F1,F2\n1,2.5\n3,4.5\n"
    df = _read_csv_tolerantly(payload)
    assert df["F1"].dtype.is_integer()
    assert df["F2"].dtype.is_float()


def test_na_spellings_become_null_not_text():
    from muleguard.api.routes_upload import _read_csv_tolerantly

    payload = b"F1,F2\n1,NA\n2,N/A\n3,null\n4,\n"
    df = _read_csv_tolerantly(payload)
    assert df["F2"].null_count() == 4
