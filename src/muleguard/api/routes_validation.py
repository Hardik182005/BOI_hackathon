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
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl
from fastapi import APIRouter, HTTPException, UploadFile
from fastapi.responses import FileResponse

from muleguard import settings
from muleguard.api import database as db
from muleguard.api.routes_upload import MAX_UPLOAD_BYTES, _parse_upload
from muleguard.features.frame import attach_meta, meta_feature_names
from muleguard.logging import get_logger
from muleguard.models.scoring import SchemaError, load_bundle, score_rows
from muleguard.validation import lab, sealed
from muleguard.action.submission import (
    ExportValidationError,
    SubmissionFormat,
    write_exports,
)
from muleguard.validation.column_mapping import apply_column_mapping

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

    # Headers may arrive as human-readable variable names rather than F-numbers.
    # Resolve them first, so a file whose data is correct is not failed for
    # spelling its columns differently. Only exact normalised matches rename;
    # the full plan is returned so the analyst sees every rename.
    df, column_plan = apply_column_mapping(df)
    if column_plan["n_renamed"]:
        db.audit("VALIDATION_COLUMNS_RESOLVED", "system", correlation_id=run_id,
                 detail={"renamed": column_plan["n_renamed"],
                         "ambiguous": len(column_plan["ambiguous"])})

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
                "column_resolution": column_plan,
                "protocol": lab.PROTOCOL_STATEMENT}

    train_X, names, cat_maps = _training_matrix(bundle)
    upload_X = _upload_matrix(scoring_df, names, cat_maps)
    s2 = lab.step_2_hidden_validation_shield(train_X, upload_X, names, s1)

    scores: list[float] = []
    tiers: Counter[str] = Counter()
    oods: Counter[str] = Counter()
    try:
        for start in range(0, scoring_df.height, SCORE_CHUNK):
            chunk = scoring_df.slice(start, SCORE_CHUNK)
            for r in score_rows(chunk, bundle=bundle, with_explanations=False):
                scores.append(float(r["calibrated_risk"]))
                tiers[str(r["risk_tier"])] += 1
                oods[str(r["ood_status"])] += 1
    except SchemaError as e:
        raise HTTPException(422, f"SCHEMA_ERROR: {e}")

    # Seal against the frame *as uploaded*, not the stripped copy. The seal has
    # to fingerprint the file the judge actually handed over, and it has to be
    # able to say whether a label was present at all - passing the stripped
    # frame made every seal report target_withheld=false, which reads as "no
    # label was withheld" when the truth was the opposite. The scores were
    # still produced from `scoring_df`; only the manifest changes.
    s3 = lab.step_3_predictions(
        df, scores, model_version=bundle["version"],
        schema=s1, shield=s2,
        reference=[f"ROW-{i + 1}" for i in range(scoring_df.height)],
    )

    # Inference diagnostics that are legitimate with no label in the building:
    # where the scores landed and how many rows the OOD lens considers unlike
    # anything the model was fitted on. They are derived from the predictions
    # that were just sealed, so they cost nothing extra and they are the only
    # honest answer to "how did my file score?" when no target was supplied.
    n = max(len(scores), 1)
    s3["distributions"] = {
        "risk": {
            "min": min(scores) if scores else None,
            "median": float(np.median(scores)) if scores else None,
            "p90": float(np.quantile(scores, 0.9)) if scores else None,
            "max": max(scores) if scores else None,
            "mean": float(np.mean(scores)) if scores else None,
        },
        "review_tier": [{"tier": t, "n": c, "share": round(c / n, 4)}
                        for t, c in tiers.most_common()],
        "ood": {"rate": round(oods.get("OUT_OF_DISTRIBUTION", 0) / n, 4),
                "counts": dict(oods)},
        "note": "computed from the sealed predictions; no label was read",
    }

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
        "column_resolution": column_plan,
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

    # Reject a label column that is not a label rather than coercing it. A
    # non-numeric column (YES/NO, TRUE/FALSE, a free-text verdict) casts to all
    # nulls, and every metric downstream is then computed on zero rows and
    # reported as "one class only" - which reads as a property of the judge's
    # data when it is really a parsing failure on our side.
    raw = df[col]
    labels = raw.cast(pl.Float64, strict=False)
    n_unparseable = int((labels.is_null() & raw.is_not_null()).sum())
    if n_unparseable:
        raise HTTPException(
            422, f"label column {col!r} has {n_unparseable} value(s) that are not "
                 "numeric; the target must be 0/1")
    seen = {v for v in labels.drop_nulls().unique().to_list()}
    if not seen <= {0.0, 1.0}:
        raise HTTPException(
            422, f"label column {col!r} contains values outside {{0, 1}} "
                 f"({sorted(seen)[:5]}); the target must be binary")
    y = labels.to_numpy()
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


@router.get("/seals/{seal_id}/predictions")
def download_predictions(seal_id: str) -> FileResponse:
    """The sealed prediction file itself, byte for byte.

    A judge who cannot take the predictions away cannot check them, so the
    manifest on its own is not enough: the workflow ends at "prediction
    download". The file is re-hashed before it is served and a mismatch is
    refused rather than quietly served, so what leaves this endpoint is either
    the artifact the seal names or nothing at all. It is streamed unmodified -
    sanitising it here would change the bytes the sealed hash covers.
    """
    try:
        rec = sealed.load_seal(seal_id)
    except FileNotFoundError:
        raise HTTPException(404, f"no sealed run {seal_id}")
    except sealed.BadSealId as e:
        raise HTTPException(422, str(e))
    ok, why = sealed.verify_seal(rec)
    if not ok:
        db.audit("VALIDATION_SEAL_BROKEN", "system", detail={"seal_id": seal_id})
        raise HTTPException(409, f"SEAL_BROKEN: {why}")
    path = Path(rec.prediction_path)
    if not path.exists():
        raise HTTPException(404, "the sealed prediction file is no longer on disk")
    db.audit("VALIDATION_PREDICTIONS_DOWNLOADED", "system",
             detail={"seal_id": seal_id, "sha256": rec.prediction_sha256})
    return FileResponse(path, media_type="text/csv",
                        filename=f"{seal_id}_predictions.csv")


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


# --------------------------------------------------------------------------
# Competition export
# --------------------------------------------------------------------------


@router.get("/seals/{seal_id}/submission")
def download_submission(seal_id: str, format: str = "minimal") -> FileResponse:
    """Reshape a sealed run into the organiser's submission file.

    Built from the sealed prediction CSV rather than by rescoring, so the file
    handed to the organiser is provably the same set of numbers that were
    hashed before any label was read. ``format`` is ``minimal`` (the two-column
    competition file) or ``analyst``.
    """
    if format not in {"minimal", "analyst"}:
        raise HTTPException(422, "format must be 'minimal' or 'analyst'")
    try:
        rec = sealed.load_seal(seal_id)
    except FileNotFoundError:
        raise HTTPException(404, f"no sealed run {seal_id}")
    except sealed.BadSealId as e:
        raise HTTPException(422, str(e))

    ok, why = sealed.verify_seal(rec)
    if not ok:
        db.audit("VALIDATION_SEAL_BROKEN", "system", detail={"seal_id": seal_id})
        raise HTTPException(409, f"SEAL_BROKEN: {why}")

    path = Path(rec.prediction_path)
    if not path.exists():
        raise HTTPException(404, "the sealed prediction file is no longer on disk")

    preds = pl.read_csv(path)
    fmt = SubmissionFormat()
    scores = [float(s) for s in preds[rec.score_column].to_list()]
    out_dir = settings.ARTIFACTS_DIR / "submissions" / seal_id
    try:
        manifest = write_exports(
            out_dir, scores=scores,
            records=(None if format == "minimal" else
                     [{"calibrated_risk": s} for s in scores]),
            row_ids=list(range(1, len(scores) + 1)), id_source="positional",
            fmt=fmt, policy_threshold=_policy_standard_threshold(),
            model_version=rec.model_version)
    except ExportValidationError as e:
        raise HTTPException(422, f"EXPORT_INVALID: {e}")

    db.audit("SUBMISSION_EXPORTED", "system",
             detail={"seal_id": seal_id, "format": format,
                     "sha256": manifest["minimal_sha256"]})
    target = Path(manifest["minimal_path"] if format == "minimal"
                  else manifest["analyst_path"])
    return FileResponse(target, media_type="text/csv",
                        filename=f"{seal_id}_{format}.csv")


def _policy_standard_threshold() -> float | None:
    """The standard-review cutoff of the model that actually produced the scores.

    Read from the loaded bundle, not from
    ``artifacts/model_registry/policy_snapshot.json``. That file was written at
    the 2026-07-10 freeze and still holds the **retired** CatBoost bundle's
    cutoffs (standard_risk 0.017246), while the champion's is 0.013183 - so
    every binary submission exported through the Validation Lab was being cut
    at a threshold derived from a different model's calibrated distribution
    than the one that produced the probabilities being cut. That is not a
    threshold change: the production thresholds are and remain the bundle's,
    and this makes the export use them instead of a superseded model's.

    Returns None if the bundle has no policy block, which leaves
    ``SubmissionFormat.binary_threshold`` on its declared default rather than
    silently substituting a number from somewhere else.
    """
    try:
        pol = load_bundle().get("policy_thresholds") or {}
        val = pol.get("standard_risk")
        return None if val is None else float(val)
    except Exception:  # noqa: BLE001 - export must not break on a bundle read
        return None
