"""CSV/XLSX batch-upload scoring endpoint.

POST /v1/score/file  (multipart form: file=<csv|xlsx>)
  - size cap (default 60 MB) and MIME/extension validation
  - safe parsing (polars CSV / fastexcel XLSX); corrupted files -> 422
  - the target column F3924, index columns and unknown extras are DROPPED
    safely (recorded in the response header block, never used)
  - missing selected features -> 422 SCHEMA_ERROR (no silent fill)
  - duplicate column names -> 422
  - per-row scoring with the frozen bundle; progress recorded in the DB
  - output: JSON summary + downloadable CSV (formula-injection sanitised,
    includes model_version; hidden threshold values are NOT exposed)
"""
from __future__ import annotations

import io
import json
import uuid

import numpy as np
import polars as pl
from fastapi import APIRouter, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from muleguard import settings
from muleguard.api import database as db
from muleguard.explain.evidence_packet import csv_safe
from muleguard.logging import get_logger
from muleguard.models.scoring import SchemaError, load_bundle, score_rows

log = get_logger("api.upload")

router = APIRouter()

MAX_UPLOAD_BYTES = 60 * 1024 * 1024
ALLOWED_SUFFIXES = {".csv", ".xlsx"}
CHUNK_ROWS = 500


def _parse_upload(filename: str, payload: bytes) -> pl.DataFrame:
    suffix = ("." + filename.rsplit(".", 1)[-1].lower()) if "." in filename else ""
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(422, f"unsupported file type {suffix or '(none)'}; use .csv or .xlsx")
    try:
        if suffix == ".csv":
            header = payload.split(b"\n", 1)[0].decode("utf-8", "replace").strip("\r")
            names = [h.strip().strip('"') for h in header.split(",")]
            if len(names) != len(set(names)):
                raise HTTPException(422, "duplicate column names in upload")
            df = pl.read_csv(io.BytesIO(payload), infer_schema_length=200,
                             null_values=["NA", ""])
        else:
            import fastexcel

            df = fastexcel.read_excel(payload).load_sheet(0).to_polars()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(422, f"file could not be parsed safely: {type(e).__name__}")
    if df.height == 0:
        raise HTTPException(422, "file contains no data rows")
    if len(df.columns) != len(set(df.columns)):
        raise HTTPException(422, "duplicate column names in upload")
    return df


@router.post("/v1/score/file")
async def score_file(file: UploadFile):
    payload = await file.read()
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"file exceeds {MAX_UPLOAD_BYTES // (1024*1024)} MB limit")
    df = _parse_upload(file.filename or "", payload)

    bundle = load_bundle()
    dropped = [c for c in df.columns
               if c == settings.TARGET_COLUMN or c.startswith("__UNNAMED__")
               or c.strip().lower() in {"", "index", "id"}]
    if dropped:
        df = df.drop(dropped)

    batch_id = f"BATCH-{uuid.uuid4().hex[:10].upper()}"
    n = df.height
    rows_out: list[dict] = []
    try:
        for start in range(0, n, CHUNK_ROWS):
            chunk = df.slice(start, CHUNK_ROWS)
            results = score_rows(chunk, bundle=bundle, with_explanations=False)
            for i, r in enumerate(results):
                rows_out.append({
                    "row_number": start + i + 1,
                    "calibrated_risk": round(r["calibrated_risk"], 6),
                    "risk_tier": r["risk_tier"],
                    "review_status": r["decision"],
                    "model_agreement": round(r["model_agreement"], 4),
                    "conformal_status": r["conformal_status"],
                    "ood_status": r["ood_status"],
                    "model_version": r["model_version"],
                })
            db.audit("BATCH_PROGRESS", "system", correlation_id=batch_id,
                     detail={"scored": min(start + CHUNK_ROWS, n), "total": n})
    except SchemaError as e:
        raise HTTPException(422, f"SCHEMA_ERROR: {e}")

    out = pl.DataFrame(rows_out)
    tier_counts = {k: int(v) for k, v in
                   out["risk_tier"].value_counts().iter_rows()}
    db.audit("BATCH_SCORED", "system", correlation_id=batch_id,
             after={"rows": n, "tiers": tier_counts, "dropped_columns": dropped},
             model_version=bundle["version"])

    csv_buf = io.StringIO()
    out.with_columns([
        pl.col(c).map_elements(csv_safe, return_dtype=pl.Utf8).alias(c)
        for c in out.columns if out.schema[c] == pl.Utf8
    ]).write_csv(csv_buf)

    return StreamingResponse(
        io.BytesIO(csv_buf.getvalue().encode("utf-8")),
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="{batch_id}_scores.csv"',
            "X-Batch-Id": batch_id,
            "X-Rows-Scored": str(n),
            "X-Dropped-Columns": json.dumps(dropped),
            "X-Tier-Counts": json.dumps(tier_counts),
            "X-Model-Version": bundle["version"],
        },
    )
