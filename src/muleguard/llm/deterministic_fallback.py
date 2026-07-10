"""Deterministic narrative generator - always available, no LLM required.

Produces the same NarratorOutput schema from templates over verified facts.
This is the guaranteed path: scoring and case reports work with Ollama
stopped, and it is the automatic fallback when the validator rejects LLM text.
"""
from __future__ import annotations

from muleguard.llm.schemas import REQUIRED_LIMITATIONS, NarratorInput, NarratorOutput

_TIER_PHRASES = {
    "CRITICAL_REVIEW": "requires highest-priority human review",
    "URGENT_REVIEW": "requires prioritised human review",
    "STANDARD_REVIEW": "has been placed in the routine review queue",
    "OOD_REVIEW": "produced data unlike the model's training distribution, so the "
                  "model score is not trusted and a human review is required",
    "MONITOR": "is not currently flagged; monitoring continues",
}


def deterministic_narrative(ctx: NarratorInput) -> NarratorOutput:
    top = sorted(ctx.top_reasons, key=lambda r: -abs(r.shap_contribution))[:5]
    frags = []
    for r in top:
        name = r.verified_semantic_name or r.feature
        if r.legitimate_percentile is not None and r.direction == "INCREASES_RISK":
            frags.append(
                f"{name} is at the {r.legitimate_percentile:.1f}th percentile of the "
                f"legitimate cohort and increases the model score"
            )
        elif r.direction == "INCREASES_RISK":
            frags.append(f"{name} increases the model score")
        else:
            frags.append(f"{name} decreases the model score")

    summary = (
        f"Account {ctx.account_reference} {_TIER_PHRASES[ctx.risk_tier]}. "
        f"Calibrated behavioural risk is {ctx.calibrated_risk:.2f} with model "
        f"agreement {ctx.model_agreement:.2f}; conformal status {ctx.conformal_status}, "
        f"input status {ctx.ood_status}. "
    )
    if frags:
        summary += "Main technical drivers: " + "; ".join(frags[:3]) + "."

    reason_codes = [
        f"{(r.verified_semantic_name or r.feature)}:"
        f"{'+' if r.direction == 'INCREASES_RISK' else '-'}"
        f"{abs(r.shap_contribution):.3f}"
        for r in top
    ]
    checks = list(ctx.allowed_checks[:4]) or ["ANALYST_REVIEW"]

    return NarratorOutput(
        summary=summary,
        risk_tier=ctx.risk_tier,
        verified_risk_score=ctx.calibrated_risk,
        reason_codes=reason_codes,
        recommended_checks=checks,
        limitations=list(dict.fromkeys([*ctx.limitations, *REQUIRED_LIMITATIONS]))[:8],
    )
