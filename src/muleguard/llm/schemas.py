"""Pydantic schemas for the guarded LLM narrator.

The LLM receives ONLY verified structured facts (NarratorInput) and must
return NarratorOutput. Any deviation is rejected by the validator and the
deterministic fallback is used instead.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

RiskTier = Literal["CRITICAL_REVIEW", "URGENT_REVIEW", "STANDARD_REVIEW", "OOD_REVIEW", "MONITOR"]

REQUIRED_LIMITATIONS = [
    "Behavioural risk is not proof of criminal intent",
    "Final action requires human review",
]

ALLOWED_CHECKS = [
    "VERIFY_KYC",
    "CHECK_DEVICE_HISTORY",
    "CHECK_REGISTRY",
    "ANALYST_REVIEW",
    "CONTACT_CUSTOMER_SUPPORT_PROCESS",
    "REVIEW_RECENT_ACTIVITY",
]


class ReasonFact(BaseModel):
    feature: str
    value: float | str | None
    legitimate_percentile: float | None = Field(default=None, ge=0.0, le=100.0)
    direction: Literal["INCREASES_RISK", "DECREASES_RISK"]
    shap_contribution: float
    verified_semantic_name: str | None = None


class NarratorInput(BaseModel):
    account_reference: str
    calibrated_risk: float = Field(ge=0.0, le=1.0)
    risk_tier: RiskTier
    model_agreement: float = Field(ge=0.0, le=1.0)
    conformal_status: Literal["HIGH_RISK_SET", "LOW_RISK_SET", "UNCERTAIN_SET"]
    ood_status: Literal["IN_DISTRIBUTION", "OUT_OF_DISTRIBUTION"]
    top_reasons: list[ReasonFact]
    allowed_checks: list[str] = Field(default_factory=lambda: list(ALLOWED_CHECKS))
    limitations: list[str] = Field(default_factory=lambda: list(REQUIRED_LIMITATIONS))


class NarratorOutput(BaseModel):
    summary: str = Field(min_length=1, max_length=1200)
    risk_tier: RiskTier
    verified_risk_score: float = Field(ge=0.0, le=1.0)
    reason_codes: list[str] = Field(max_length=10)
    recommended_checks: list[str] = Field(max_length=8)
    limitations: list[str] = Field(min_length=1, max_length=8)

    @field_validator("summary")
    @classmethod
    def summary_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("summary is blank")
        return v
