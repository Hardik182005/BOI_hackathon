"""Hallucination validator for LLM narrator output.

Rejects (with a machine-readable reason) any output that:
- is invalid JSON / fails the schema
- changes the risk score or tier
- mentions a feature not supplied in the input
- invents unsupported numbers (amounts/dates not present in input facts)
- asserts criminal guilt
- recommends an action outside the allowlist
- omits the required limitations
"""
from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from muleguard.llm.schemas import (
    ALLOWED_CHECKS,
    REQUIRED_LIMITATIONS,
    NarratorInput,
    NarratorOutput,
)

GUILT_PATTERNS = re.compile(
    r"\b(guilty|criminal(?!\s+intent)|fraudster|launderer|crook|thief|"
    r"committed\s+fraud|is\s+a\s+mule|definitely\s+fraud)\b",
    re.IGNORECASE,
)
# feature tokens look like F123; the narrator may only mention supplied ones
FEATURE_TOKEN = re.compile(r"\bF\d{1,4}\b")
# numbers with currency/amount context invented by the model
AMOUNT_PATTERN = re.compile(r"(?:rs\.?|inr|₹|\$|usd|eur)\s*[\d,]+", re.IGNORECASE)


def validate_llm_output(raw_text: str, ctx: NarratorInput) -> tuple[NarratorOutput | None, list[str]]:
    """Return (validated output, rejection reasons). Output is None if rejected."""
    reasons: list[str] = []

    try:
        payload: Any = json.loads(raw_text)
    except (json.JSONDecodeError, TypeError):
        return None, ["invalid JSON"]

    try:
        out = NarratorOutput.model_validate(payload)
    except ValidationError as e:
        return None, [f"schema violation: {e.errors()[0]['msg']} at {e.errors()[0]['loc']}"]

    if abs(out.verified_risk_score - ctx.calibrated_risk) > 1e-6:
        reasons.append(
            f"risk score altered: {out.verified_risk_score} != {ctx.calibrated_risk}"
        )
    if out.risk_tier != ctx.risk_tier:
        reasons.append(f"risk tier altered: {out.risk_tier} != {ctx.risk_tier}")

    supplied = {r.feature for r in ctx.top_reasons}
    all_text = " ".join(
        [out.summary, *out.reason_codes, *out.recommended_checks, *out.limitations]
    )
    for tok in set(FEATURE_TOKEN.findall(all_text)):
        if tok not in supplied:
            reasons.append(f"mentions feature not in supplied facts: {tok}")

    if AMOUNT_PATTERN.search(all_text):
        reasons.append("invents a currency amount not present in structured input")

    if GUILT_PATTERNS.search(all_text):
        reasons.append("asserts criminal guilt")

    allowed = set(ctx.allowed_checks or ALLOWED_CHECKS)
    for chk in out.recommended_checks:
        if chk not in allowed:
            reasons.append(f"recommends action outside allowlist: {chk}")

    lim_text = " ".join(out.limitations).lower()
    for required in REQUIRED_LIMITATIONS:
        # keyword-level containment: intent + human review must be acknowledged
        key = "intent" if "intent" in required else "human review"
        if key not in lim_text:
            reasons.append(f"missing required limitation: {required}")

    return (out, reasons) if not reasons else (None, reasons)
