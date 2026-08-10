"""Validation Lab: the fixed three-step order (addendum UPDATE 11).

    Step 1  Schema Integrity
    Step 2  Hidden Validation Shield
    Step 3  Predictions (sealed)

The order is the point. A team that scores first and inspects the file
afterwards has already had the opportunity to tune against it; a team that
checks schema and distribution *before* predicting, then seals the predictions,
has not. Each step returns a verdict and the pipeline refuses to advance past a
step that fails hard, so the sequence cannot be skipped by calling the
functions in a different order.

The Validation Compatibility Score summarises "can this model be trusted on
this file?" as one auditable number with its components broken out. It is a
statement about *inputs*, computed with no access to labels, and it never
changes a prediction - a low score is a warning printed next to the results,
not a silent adjustment to them (addendum UPDATE 3).
"""
from __future__ import annotations

import datetime as dt
from typing import Any, Sequence

import numpy as np
import polars as pl

from muleguard.logging import get_logger
from muleguard.validation import sealed

log = get_logger("validation.lab")

STEP_PASS = "PASS"
STEP_WARN = "WARN"
STEP_FAIL = "FAIL"

# Component weights of the Validation Compatibility Score, fixed before any
# file was ever scored so the number cannot be reverse-engineered to flatter a
# particular upload.
COMPAT_WEIGHTS = {
    "schema_completeness": 40,
    "distribution_compatibility": 30,
    "missingness_consistency": 15,
    "value_range_coverage": 15,
}
COMPAT_BANDS = (
    (85, "HIGH_COMPATIBILITY",
     "the file looks like the data the model was fitted on; read the metrics at face value"),
    (70, "ACCEPTABLE",
     "minor differences from training data; results are usable, absolute probabilities slightly less reliable"),
    (50, "DEGRADED",
     "material differences; prefer the ranking over the probabilities and check the shifted features"),
    (0, "INCOMPATIBLE",
     "the file differs from training data enough that scores should be treated as indicative only"),
)


class StepOrderError(RuntimeError):
    """Raised when a lab step is run before the step it depends on."""


# --------------------------------------------------------------------------
# Step 1 - schema integrity
# --------------------------------------------------------------------------


def step_1_schema_integrity(upload: pl.DataFrame,
                            required_features: Sequence[str],
                            *, target_column: str | None = None,
                            derived_features: Sequence[str] = ()) -> dict[str, Any]:
    """Does the file contain what the model needs, in a usable form?

    Runs before anything else because every later number is meaningless if the
    answer is no. Missing *required* features is a hard failure: silently
    zero-filling them would produce confident scores for accounts the model has
    no information about, which is the single most dangerous failure mode in
    this system.

    ``derived_features`` names required columns this system computes itself -
    the MG_* meta-feature block, which is a row-wise function of raw columns and
    is therefore never uploaded. ``upload`` must already carry them. They are
    counted as present because the model genuinely has them, but they are
    reported separately so nobody reads this step as a claim that the file
    contained 120 columns when it supplied 117. A derived feature whose own
    source columns are absent lands in ``all_null_required`` like any other
    empty column, so the missingness is still surfaced rather than hidden.
    """
    cols = set(upload.columns)
    required = list(required_features)
    derived = [c for c in derived_features if c in set(required)]
    missing = [c for c in required if c not in cols]
    present = [c for c in required if c in cols]
    extra = sorted(cols - set(required) - set(derived_features) - {target_column or ""})

    non_numeric: list[str] = []
    all_null: list[str] = []
    for c in present:
        s = upload[c]
        casted = s.cast(pl.Float64, strict=False)
        if casted.null_count() == upload.height and upload.height:
            all_null.append(c)
        elif s.dtype == pl.Utf8 and casted.null_count() > 0.5 * upload.height:
            non_numeric.append(c)

    dup_rows = upload.height - upload.unique().height
    completeness = len(present) / max(len(required), 1)

    if missing:
        verdict, status = STEP_FAIL, (
            f"{len(missing)} of {len(required)} required features are absent; "
            "scoring is refused because absent inputs cannot be distinguished "
            "from genuinely zero behaviour")
    elif all_null or dup_rows:
        verdict, status = STEP_WARN, (
            f"{len(all_null)} required features are entirely empty and "
            f"{dup_rows} duplicate rows are present; scoring proceeds and both "
            "facts are carried into the compatibility score")
    else:
        verdict, status = STEP_PASS, (
            "every required feature is present and parseable"
            + (f" ({len(derived)} of them derived from the uploaded raw columns "
               "rather than supplied directly)" if derived else ""))

    return {
        "step": 1,
        "name": "Schema Integrity",
        "verdict": verdict,
        "status_detail": status,
        "n_rows": upload.height,
        "n_columns": upload.width,
        "n_required": len(required),
        "n_present": len(present),
        "n_supplied_by_file": len(present) - len(derived),
        "derived_required": derived,
        "n_derived_required": len(derived),
        "derivation_note": (
            "MG_* meta-features are computed here from the file's own raw "
            "columns; nothing about that derivation consults training data, "
            "labels or other rows." if derived else None),
        "schema_completeness": round(completeness, 4),
        "missing_required": missing[:50],
        "n_missing_required": len(missing),
        "unexpected_columns": extra[:50],
        "n_unexpected_columns": len(extra),
        "all_null_required": all_null[:25],
        "unparseable_numeric": non_numeric[:25],
        "duplicate_rows": int(dup_rows),
        "target_column_present": bool(target_column and target_column in cols),
        "guarantee": "no label was read in this step",
    }


# --------------------------------------------------------------------------
# Step 2 - Hidden Validation Shield
# --------------------------------------------------------------------------


def _missingness_consistency(train_X: np.ndarray, upload_X: np.ndarray) -> float:
    tr = np.mean(np.isnan(train_X), axis=0)
    up = np.mean(np.isnan(upload_X), axis=0)
    return float(np.clip(1.0 - np.mean(np.abs(tr - up)), 0.0, 1.0))


def _value_range_coverage(train_X: np.ndarray, upload_X: np.ndarray) -> float:
    """Share of features whose uploaded median sits inside the training p1-p99.

    A cheap, readable check that catches unit changes and rescaled columns,
    which adversarial AUC alone can miss when the shift is monotone.
    """
    with np.errstate(invalid="ignore"):
        lo = np.nanpercentile(train_X, 1, axis=0)
        hi = np.nanpercentile(train_X, 99, axis=0)
        med = np.nanmedian(upload_X, axis=0)
    ok = np.isfinite(med) & np.isfinite(lo) & np.isfinite(hi) & (med >= lo) & (med <= hi)
    usable = np.isfinite(med) & np.isfinite(lo) & np.isfinite(hi)
    return float(ok.sum() / usable.sum()) if usable.sum() else 1.0


def _auc_to_compatibility(auc: float) -> float:
    """Map adversarial AUC to a 0-1 compatibility component.

    0.50 (indistinguishable) scores 1.0 and 1.00 (trivially separable) scores
    0.0, linearly. Deliberately linear rather than tuned: a threshold curve
    fitted to make our own uploads look good would defeat the purpose.
    """
    return float(np.clip((1.0 - auc) / 0.5, 0.0, 1.0))


def compatibility_score(schema: dict[str, Any], shield: dict[str, Any],
                        missingness: float, range_coverage: float) -> dict[str, Any]:
    """Combine the four components into one 0-100 score with its workings shown."""
    auc = shield.get("adversarial_auc")
    dist = _auc_to_compatibility(float(auc)) if auc is not None else 0.75
    parts = {
        "schema_completeness": float(schema.get("schema_completeness", 0.0)),
        "distribution_compatibility": dist,
        "missingness_consistency": float(missingness),
        "value_range_coverage": float(range_coverage),
    }
    contributions = {k: round(v * COMPAT_WEIGHTS[k], 2) for k, v in parts.items()}
    total = round(sum(contributions.values()), 2)
    band, meaning = next((b, m) for cut, b, m in COMPAT_BANDS if total >= cut)
    return {
        "score": total,
        "band": band,
        "meaning": meaning,
        "components": {k: round(v, 4) for k, v in parts.items()},
        "weights": dict(COMPAT_WEIGHTS),
        "contributions": contributions,
        "note": (
            "computed from inputs only, with no access to labels. A low score "
            "is reported alongside the predictions; it never modifies them."
        ),
    }


def step_2_hidden_validation_shield(
    train_X: np.ndarray, upload_X: np.ndarray,
    feature_names: Sequence[str], schema: dict[str, Any],
) -> dict[str, Any]:
    """Adversarial train-vs-upload check plus the compatibility score."""
    if schema.get("verdict") == STEP_FAIL:
        raise StepOrderError(
            "step 2 refused: schema integrity failed, so a distribution "
            "comparison would be measuring the missing columns, not the data")
    from muleguard.models.shield import adversarial_validation

    shield = adversarial_validation(train_X, upload_X, list(feature_names))
    miss = _missingness_consistency(train_X, upload_X)
    rng = _value_range_coverage(train_X, upload_X)
    compat = compatibility_score(schema, shield, miss, rng)

    band = shield.get("shift_band")
    verdict = (STEP_PASS if band in ("COMPATIBLE", None)
               else STEP_WARN if band in ("MILD_SHIFT", "MODERATE_SHIFT")
               else STEP_WARN)  # even SEVERE_SHIFT warns rather than blocks
    return {
        "step": 2,
        "name": "Hidden Validation Shield",
        "verdict": verdict,
        "status_detail": shield.get("verdict", shield.get("note", "")),
        "adversarial": shield,
        "compatibility": compat,
        "missingness_consistency": round(miss, 4),
        "value_range_coverage": round(rng, 4),
        "guarantees": [
            "the mule model is not retrained or recalibrated on uploaded rows",
            "validation labels are not inspected in this step",
            "no prediction is changed as a result of the adversarial AUC",
        ],
    }


# --------------------------------------------------------------------------
# Step 3 - sealed predictions
# --------------------------------------------------------------------------


def step_3_predictions(
    upload: pl.DataFrame,
    scores: Sequence[float],
    *,
    model_version: str,
    schema: dict[str, Any],
    shield: dict[str, Any],
    reference: Sequence[str] | None = None,
    extra_columns: dict[str, Sequence[Any]] | None = None,
) -> dict[str, Any]:
    """Seal the predictions. Labels remain unread after this returns."""
    if schema.get("verdict") == STEP_FAIL:
        raise StepOrderError("step 3 refused: schema integrity failed")
    if shield.get("step") != 2:
        raise StepOrderError("step 3 refused: the shield step has not been run")

    rec = sealed.seal_predictions(
        upload, scores, model_version=model_version,
        reference=reference, extra_columns=extra_columns,
    )
    return {
        "step": 3,
        "name": "Predictions (sealed)",
        "verdict": STEP_PASS,
        "status_detail": (
            "predictions written and hashed; metrics can now be revealed "
            "without any possibility of the predictions having changed"),
        "seal": rec.to_dict(),
        "ui_state": sealed.STATE_SEALED,
        "next_action": "Reveal Validation Metrics",
    }


def run_lab(
    upload: pl.DataFrame,
    *,
    required_features: Sequence[str],
    train_X: np.ndarray,
    upload_X: np.ndarray,
    feature_names: Sequence[str],
    scores: Sequence[float],
    model_version: str,
    target_column: str | None = None,
    reference: Sequence[str] | None = None,
    extra_columns: dict[str, Sequence[Any]] | None = None,
) -> dict[str, Any]:
    """Run all three steps in order and return the combined report.

    Stops and reports at the first hard failure rather than pressing on, so a
    file with the wrong schema produces a clear diagnosis instead of a page of
    plausible-looking numbers.
    """
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    s1 = step_1_schema_integrity(upload, required_features,
                                 target_column=target_column)
    steps = [s1]
    if s1["verdict"] == STEP_FAIL:
        return {
            "generated_utc": started,
            "steps": steps,
            "overall": STEP_FAIL,
            "stopped_at_step": 1,
            "summary": s1["status_detail"],
            "protocol": PROTOCOL_STATEMENT,
        }

    s2 = step_2_hidden_validation_shield(train_X, upload_X, feature_names, s1)
    steps.append(s2)
    s3 = step_3_predictions(upload, scores, model_version=model_version,
                            schema=s1, shield=s2, reference=reference,
                            extra_columns=extra_columns)
    steps.append(s3)

    overall = (STEP_WARN if any(s["verdict"] == STEP_WARN for s in steps)
               else STEP_PASS)
    compat = s2["compatibility"]
    return {
        "generated_utc": started,
        "steps": steps,
        "overall": overall,
        "stopped_at_step": None,
        "compatibility_score": compat["score"],
        "compatibility_band": compat["band"],
        "seal_id": s3["seal"]["seal_id"],
        "ui_state": sealed.STATE_SEALED,
        "summary": (
            f"Schema {s1['verdict']}, shift band "
            f"{s2['adversarial'].get('shift_band', 'n/a')}, compatibility "
            f"{compat['score']}/100 ({compat['band']}). "
            f"{len(scores)} predictions sealed under "
            f"{s3['seal']['prediction_sha256'][:12]}."
        ),
        "protocol": PROTOCOL_STATEMENT,
    }


PROTOCOL_STATEMENT = (
    "Validation runs in a fixed order: schema integrity, then a train-vs-upload "
    "distribution shield, then predictions - which are written and hashed "
    "before any label is read. The model is never retrained, recalibrated or "
    "threshold-tuned on an uploaded file, and no prediction is altered because "
    "of what the shield reports."
)
