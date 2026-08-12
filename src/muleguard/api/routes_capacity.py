"""Analyst Capacity Optimizer endpoints (upgrade 3).

    GET  /v1/capacity/curve   the precomputed capacity curve, whole
    POST /v1/capacity/plan    answer one capacity question

The curve is not computed here. It is read from
``artifacts/metrics/capacity_curve.json``, which
``muleguard.cli.capacity_curve`` writes from the stored development
out-of-fold predictions. A request therefore never touches a model, never
touches the dataset, and cannot produce a number that disagrees with the
artifact a judge can open on disk.

The plan endpoint takes exactly one of two questions - a review capacity, or a
tolerance for false alarms - and returns what the measured curve says about it
plus a *recommended* threshold. The recommendation is advisory: it is returned
with ``applied: false``, it names the frozen policy band it would sit in, and
no route in this module writes a threshold anywhere. Changing the served policy
is a human decision made through the case workflow, not an API side effect.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, model_validator

from muleguard import settings
from muleguard.logging import get_logger
from muleguard.models import capacity
from muleguard.utils import load_json

log = get_logger("api.capacity")

router = APIRouter(tags=["capacity"])

CURVE_PATH = settings.METRICS_DIR / "capacity_curve.json"

_CURVE: dict[str, Any] | None = None
_CURVE_MTIME: float | None = None


def _curve() -> dict[str, Any]:
    """Load the artifact, re-reading it if it has been regenerated.

    Cached because it is a few hundred kilobytes and a dashboard polls; keyed
    on mtime so that regenerating the curve during a demo does not require a
    backend restart to take effect.
    """
    global _CURVE, _CURVE_MTIME
    if not CURVE_PATH.exists():
        raise HTTPException(
            503, "no capacity curve artifact yet - run "
                 "`python -m muleguard.cli.capacity_curve` to build it from the "
                 "stored development out-of-fold predictions")
    mtime = CURVE_PATH.stat().st_mtime
    if _CURVE is None or _CURVE_MTIME != mtime:
        _CURVE = load_json(CURVE_PATH)
        _CURVE_MTIME = mtime
        log.info("capacity curve loaded: %d points, champion=%s",
                 len(_CURVE["points"]),
                 _CURVE["provenance"]["champion_model"])
    return _CURVE


class CapacityRequest(BaseModel):
    """Exactly one of the two questions, never both."""

    review_capacity: int | None = Field(
        default=None, ge=1,
        description="accounts the analyst team can review per day")
    max_fp_per_1000: float | None = Field(
        default=None, ge=0,
        description="maximum acceptable false alarms per 1,000 accounts")
    fp_basis: str = Field(
        default="legitimate", pattern="^(legitimate|screened)$",
        description="denominator for max_fp_per_1000: per 1,000 legitimate "
                    "accounts (1000 x FPR, the repository convention) or per "
                    "1,000 accounts screened")

    @model_validator(mode="after")
    def _exactly_one(self) -> "CapacityRequest":
        given = [self.review_capacity is not None, self.max_fp_per_1000 is not None]
        if sum(given) != 1:
            raise ValueError(
                "give exactly one of review_capacity or max_fp_per_1000; the "
                "panel answers one question at a time so that the answer is "
                "unambiguous about which constraint is binding")
        return self


@router.get("/v1/capacity/curve")
def capacity_curve() -> dict[str, Any]:
    """The whole measured curve, its provenance and its caveats."""
    return _curve()


@router.post("/v1/capacity/plan")
def capacity_plan(req: CapacityRequest) -> dict[str, Any]:
    curve = _curve()
    try:
        if req.review_capacity is not None:
            plan = capacity.answer_for_budget(curve, req.review_capacity)
        else:
            plan = capacity.answer_for_fp_per_1000(
                curve, float(req.max_fp_per_1000), basis=req.fp_basis)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    plan["evaluation"] = curve["evaluation"]
    plan["frozen_policy_position"] = curve["frozen_policy_position"]
    plan["limitations"] = [
        "Measured on development out-of-fold predictions only; the locked test "
        "split was not opened to produce this answer.",
        "Recall at a review budget is bounded by the "
        f"{curve['evaluation']['n_positives']} confirmed mules present in that "
        "split, and it is not projected onto a larger book.",
        "The recommended threshold is a proposal for an authorised human to "
        "approve. The frozen policy thresholds are unchanged by this request.",
    ]
    return plan
