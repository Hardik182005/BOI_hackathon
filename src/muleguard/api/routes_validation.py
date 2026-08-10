"""Validation Lab endpoints - the sealed hidden-validation workflow.

    POST /v1/validation/run           upload a file, run the three steps, seal
    POST /v1/validation/{id}/reveal   verify the seal, then report metrics
    GET  /v1/validation/seals         list every sealed run
    GET  /v1/validation/seals/{id}    one manifest, with live hash verification

The split between "run" and "reveal" is the product, not an implementation
detail. ``run`` withholds the target column, scores, writes the predictions and
hashes them; the response contains no metric at all, only a seal. ``reveal``
re-hashes the file on disk and refuses to score if it changed. A judge can
therefore establish, from the artifacts alone, that the reported numbers came
from predictions made before any label was read.

Uploaded rows are never used to retrain, recalibrate or re-threshold the model.
"""
from __future__ import annotations

import uuid

import numpy as np
import polars as pl
from fastapi import APIRouter, HTTPException, UploadFile

from muleguard import settings
from muleguard.api import database as db
from muleguard.api.routes_upload import MAX_UPLOAD_BYTES, _parse_upload
from muleguard.features.frame import attach_meta, meta_feature_names
from muleguard.logging import get_logger
from muleguard.models.scoring import SchemaError, load_bundle, score_rows
from muleguard.validation import lab, sealed

log = get_logger("api.validation")

router = APIRouter(prefix="/v1/validation", tags=["validation"])

SCORE_CHUNK = 500


_SHIELD_REF: tuple[np.ndarray, list[str], dict[str, list]] | None = None


def _training_matrix(bundle: dict) -> tuple[np.ndarray, list[str], dict[str, list]]:
    """The development matrix the shield compares an upload against.

    Restricted to the intersection of the bundle's features and the columns the
    firewall currently admits. The intersection is not defensive padding: a
    bundle frozen before a re-quarantine can legitimately name features that are
    no longer admissible, and comparing distributions on a post-resolution
    column would tell us about case outcomes rather than about covariate shift.
    Cached because the frame costs seconds to assemble and never changes within
    a process. Locked-test rows are excluded via the development split.
    """
    global _SHIELD_REF
    if _SHIELD_REF is not None:
        return _SHIELD_REF

    ref = bundle.get("shield_reference_matrix")
    if ref is not None:
        _SHIELD_REF = (np.asarray(ref), list(bundle["feature_list_selected"]),
                       bundle.get("cat_maps") or {})
        return _SHIELD_REF

    from muleguard.features.frame import build_model_frame
    from muleguard.models.harness import dev_split

    mf = build_model_frame()
    wanted = set(bundle["feature_list_selected"])
    keep = [i for i, n in enumerate(mf.feature_names) if n in wanted]
    dropped = sorted(wanted - set(mf.feature_names))
    if dropped:
        log.warning(
            "shield reference drops %d bundle feature(s) the firewall no longer "
            "admits (%s...); the shield compares the admissible intersection",
            len(dropped), ", ".join(dropped[:6]))
    if not keep:  # a wholly quarantined bundle - compare on everything admitted
        keep = list(range(len(mf.feature_names)))
    dev = dev_split()
    _SHIELD_REF = (mf.X[np.ix_(dev.row_index, keep)],
                   [mf.feature_names[i] for i in keep], mf.cat_maps)
    return _SHIELD_REF


def _upload_matrix(df: pl.DataFrame, names: list[str],
                   cat_maps: dict[str, list]) -> np.ndarray:
    """Encode the upload exactly as the reference matrix was encoded.

    Two things would otherwise wreck the shield. The MG_* block is derived
    rather than uploaded, so it has to be recomputed; and categorical columns
    are ordinal codes in the reference matrix, so casting the upload's raw
    strings to float would yield a column of NaN. Either mistake makes every
    upload look shifted for reasons that have nothing to do with the data, and
    the adversarial classifier would separate train from upload perfectly on an
    artifact of encoding.
    """
    from muleguard.features.preprocessing import encode_dataframe

    df = attach_meta(df)  # a no-op when the caller already derived the block
    present = [c for c in names if c in df.columns]
    X = np.full((df.height, len(names)), np.nan, dtype=float)
    if present:
        Xp, _, _ = encode_dataframe(df, present, cat_maps=cat_maps)
        pos = {n: i for i, n in enumerate(names)}
        X[:, [pos[c] for c in present]] = Xp
    return X


@router.post("/run")
async def run_validation(file: UploadFile) -> dict:
    """Step 1 -> 2 -> 3. Returns a seal, never a metric."""
    payload = await file.read()
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"file exceeds {MAX_UPLOAD_BYTES // (1024*1024)} MB limit")
    df = _parse_upload(file.filename or "", payload)

    try:
        bundle = load_bundle()
    except FileNotFoundError:
        raise HTTPException(503, "no model bundle available")

    required = list(bundle["feature_list_selected"])
    run_id = f"VAL-{uuid.uuid4().hex[:10].upper()}"

    # The target is removed before anything is scored. `y_held` stays in this
    # function's local scope and is deliberately never returned, logged or
    # persisted here - the reveal endpoint asks for labels again.
    scoring_df, y_held, target_name = sealed.withhold_target(df)

    # Derive the MG_* block before step 1 rather than after it. The champion
    # selects three meta-features, and those are row-wise functions of raw
    # columns that no uploader would ever supply - so checking schema integrity
    # against the raw file would fail every legitimate upload. Deriving first
    # means step 1 audits exactly the frame that gets scored. The derivation
    # happens after the target was withheld, so it cannot see a label.
    scoring_df = attach_meta(scoring_df)

    s1 = lab.step_1_schema_integrity(scoring_df, required,
                                     target_column=settings.TARGET_COLUMN,
                                     derived_features=meta_feature_names())
    if s1["verdict"] == lab.STEP_FAIL:
        db.audit("VALIDATION_SCHEMA_FAIL", "system", correlation_id=run_id,
                 detail={"missing": s1["n_missing_required"]})
        return {"run_id": run_id, "overall": lab.STEP_FAIL, "steps": [s1],
                "stopped_at_step": 1, "summary": s1["status_detail"],
                "protocol": lab.PROTOCOL_STATEMENT}

    train_X, names, cat_maps = _training_matrix(bundle)
    upload_X = _upload_matrix(scoring_df, names, cat_maps)
    s2 = lab.step_2_hidden_validation_shield(train_X, upload_X, names, s1)

    scores: list[float] = []
    try:
        for start in range(0, scoring_df.height, SCORE_CHUNK):
            chunk = scoring_df.slice(start, SCORE_CHUNK)
            for r in score_rows(chunk, bundle=bundle, with_explanations=False):
                scores.append(float(r["calibrated_risk"]))
    except SchemaError as e:
        raise HTTPException(422, f"SCHEMA_ERROR: {e}")

    s3 = lab.step_3_predictions(
        scoring_df, scores, model_version=bundle["version"],
        schema=s1, shield=s2,
        reference=[f"ROW-{i + 1}" for i in range(scoring_df.height)],
    )

    compat = s2["compatibility"]
    db.audit("VALIDATION_SEALED", "system", correlation_id=run_id,
             after={"seal_id": s3["seal"]["seal_id"],
                    "rows": scoring_df.height,
                    "compatibility": compat["score"]},
             model_version=bundle["version"])
    log.info("validation run %s sealed as %s (%d rows, compatibility %.1f)",
             run_id, s3["seal"]["seal_id"], scoring_df.height, compat["score"])

    return {
        "run_id": run_id,
        "overall": lab.STEP_WARN if any(s["verdict"] == lab.STEP_WARN
                                        for s in (s1, s2)) else lab.STEP_PASS,
        "steps": [s1, s2, s3],
        "seal_id": s3["seal"]["seal_id"],
        "ui_state": sealed.STATE_SEALED,
        "labels_available_for_reveal": y_held is not None,
        "target_column_detected": target_name,
        "compatibility_score": compat["score"],
        "compatibility_band": compat["band"],
        "protocol": lab.PROTOCOL_STATEMENT,
        "next_action": (
            "Reveal Validation Metrics" if y_held is not None else
            "No label column was present, so there is nothing to reveal; "
            "download the sealed predictions instead"),
    }


@router.post("/{seal_id}/reveal")
async def reveal(seal_id: str, file: UploadFile | None = None,
                 label_column: str | None = None) -> dict:
    """Verify the seal, then score it against labels supplied now.

    Labels are supplied at reveal time rather than carried over from the run,
    so the two phases stay physically separate: nothing in the scoring path
    ever had access to them.
    """
    try:
        rec = sealed.load_seal(seal_id)
    except FileNotFoundError:
        raise HTTPException(404, f"no sealed run {seal_id}")
    except sealed.BadSealId as e:
        raise HTTPException(422, str(e))

    if file is None:
        raise HTTPException(422, "a labelled file is required to reveal metrics")
    payload = await file.read()
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "label file too large")
    df = _parse_upload(file.filename or "", payload)

    col = label_column or settings.TARGET_COLUMN
    if col not in df.columns:
        raise HTTPException(422, f"label column {col!r} not found in the file")
    if df.height != rec.n_rows:
        raise HTTPException(
            422, f"label file has {df.height} rows but {rec.n_rows} were sealed; "
                 "the rows must correspond one to one, in the original order")

    y = df[col].cast(pl.Float64, strict=False).to_numpy()
    try:
        out = sealed.reveal_metrics(seal_id, y)
    except sealed.SealBroken as e:
        db.audit("VALIDATION_SEAL_BROKEN", "system", detail={"seal_id": seal_id})
        raise HTTPException(409, f"SEAL_BROKEN: {e}")

    db.audit("VALIDATION_REVEALED", "system",
             after={"seal_id": seal_id, "status": out["metrics"].get("status")})
    return out


@router.get("/seals")
def list_seals() -> dict:
    seals = sealed.list_seals()
    return {"count": len(seals), "seals": seals}


@router.get("/seals/{seal_id}")
def get_seal(seal_id: str) -> dict:
    try:
        rec = sealed.load_seal(seal_id)
    except FileNotFoundError:
        raise HTTPException(404, f"no sealed run {seal_id}")
    except sealed.BadSealId as e:
        raise HTTPException(422, str(e))
    ok, why = sealed.verify_seal(rec)
    return {"seal": rec.to_dict(),
            "verification": {"verified": ok, "detail": why}}
