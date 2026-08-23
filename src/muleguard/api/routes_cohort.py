"""Trinetra Mule-Farm Cohort Radar endpoints (section 16).

    GET  /v1/cohort/manifest              what the frozen transform is, exactly
    GET  /v1/cases/{case_id}/cohort       cohort for an account already scored
    POST /v1/cohort/search                cohort for a case, account or raw row

These read the classifier's output and never write to it. There is no endpoint
here that changes a risk probability, a tier or a decision, because a retrieval
layer that could do so would no longer be a retrieval layer - it would be part
of the model, and every published accuracy figure would need re-earning.

Availability is honest. If the frozen transform has not been built, these
return 503 with the command that builds it rather than fitting one on demand:
a transform quietly fitted inside a web request would be fitted on whatever
data that process happened to hold, which is exactly the reproducibility hole
section 8 exists to close.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from muleguard.api import database as db
from muleguard.logging import get_logger
from muleguard.usp import cohort_radar

log = get_logger("api.cohort")

router = APIRouter(tags=["cohort"])

MAX_K = 25
BUILD_HINT = "python -m muleguard.cli.build_cohort_radar"


class CohortSearchRequest(BaseModel):
    """A cohort query. Every field is optional except that one must identify a row.

    No target label is accepted or required - section 16. A caller holding
    labels cannot pass them in, so a cohort can never be conditioned on one.
    """

    case_id: str | None = None
    account_reference: str | None = None
    row_index: int | None = None
    features: dict[str, Any] | None = None
    k: int = Field(default=10, ge=1, le=MAX_K)
    include_explanations: bool = True


def _unavailable(exc: Exception) -> HTTPException:
    return HTTPException(503, {
        "error": "COHORT_RADAR_UNAVAILABLE",
        "detail": str(exc),
        "build_with": BUILD_HINT,
    })


def _guarded(payload: dict[str, Any]) -> dict[str, Any]:
    """Last checkpoint before an answer leaves the process.

    The language guard runs on the rendered response rather than on the code
    that assembled it. Checking intentions catches nothing; this reads the words
    an analyst would actually be shown.
    """
    cohort_radar.assert_language_safe(payload)
    return payload


@router.get("/v1/cohort/manifest")
def cohort_manifest() -> dict[str, Any]:
    """What the radar is, in enough detail to reproduce or refute it."""
    try:
        return _guarded(cohort_radar.manifest())
    except cohort_radar.CohortRadarUnavailable as exc:
        raise _unavailable(exc)


@router.get("/v1/cases/{case_id}/cohort")
def cohort_for_case(case_id: str, k: int = 10,
                    include_explanations: bool = True) -> dict[str, Any]:
    """Behaviourally similar accounts for an account already scored.

    The fingerprint comes from what was submitted with the score, not from a
    re-scoring. Re-deriving it would risk answering about a row the case was
    never about.
    """
    k = max(1, min(int(k), MAX_K))
    with db.connect() as c:
        case = c.execute("SELECT * FROM cases WHERE case_id=?", (case_id,)).fetchone()
    if not case:
        raise HTTPException(404, "case not found")

    features = db.fingerprint_for_case(case_id)
    if not features:
        # Cases scored before fingerprints were stored are not an error and are
        # not silently answered with somebody else's row.
        raise HTTPException(409, {
            "error": "FINGERPRINT_NOT_STORED",
            "detail": ("this case was scored before cohort fingerprints were "
                       "retained, so its feature values are not available. "
                       "Re-score the account to enable cohort lookup."),
            "case_id": case_id,
        })
    try:
        result = cohort_radar.cohort_for_features(
            features, k=k, query_reference=case["account_reference"],
            with_explanations=include_explanations)
    except cohort_radar.CohortRadarUnavailable as exc:
        raise _unavailable(exc)
    result["case_id"] = case_id
    result["risk_probability"] = float(case["calibrated_risk"])
    result["risk_tier"] = case["risk_tier"]
    result["risk_source"] = "frozen champion classifier, unchanged by this lookup"
    return _guarded(result)


@router.post("/v1/cohort/search")
def cohort_search(req: CohortSearchRequest) -> dict[str, Any]:
    """Cohort for a case, a stored account, a reference row or a raw payload.

    Accepts a Validation Lab row directly, which is the point: a judge can
    upload a file, score it and ask what its rows resemble without any of it
    having been seen when the transform was frozen.
    """
    k = max(1, min(int(req.k), MAX_K))
    try:
        if req.case_id:
            return cohort_for_case(req.case_id, k=k,
                                   include_explanations=req.include_explanations)
        if req.features:
            return _guarded(cohort_radar.cohort_for_features(
                req.features, k=k,
                query_reference=req.account_reference or "SUBMITTED_ROW",
                with_explanations=req.include_explanations))
        if req.row_index is not None:
            return _guarded(cohort_radar.cohort_for_row(
                req.row_index, k=k,
                with_explanations=req.include_explanations))
        if req.account_reference:
            row = cohort_radar.row_index_for_reference(req.account_reference)
            if row is None:
                raise HTTPException(404, {
                    "error": "ACCOUNT_NOT_IN_REFERENCE_FRAME",
                    "detail": (f"{req.account_reference!r} is not a reference-frame "
                               "account and no features were supplied. Send "
                               "`features`, or a `case_id` that has been scored."),
                })
            return _guarded(cohort_radar.cohort_for_row(
                row, k=k, with_explanations=req.include_explanations))
    except cohort_radar.CohortRadarUnavailable as exc:
        raise _unavailable(exc)
    raise HTTPException(422, "supply one of case_id, features, row_index or "
                             "account_reference")
