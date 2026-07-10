"""Deterministic decision-policy engine.

Pure function of structured inputs -> review tier + routing. No model object,
no LLM, no network. Thresholds come from a frozen policy snapshot produced on
dev OOF data. The engine can only recommend REVIEW tiers; punitive actions
require an authorised human via the case workflow (api.routes_cases).

Tier semantics (never "guilty", never "certified safe"):
  CRITICAL_REVIEW - highest-priority human review
  URGENT_REVIEW   - prioritised human review
  STANDARD_REVIEW - routine review queue
  OOD_REVIEW      - input unlike training data; model output not trusted
  MONITOR         - not currently flagged; monitoring continues
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

TIERS = ("CRITICAL_REVIEW", "URGENT_REVIEW", "STANDARD_REVIEW", "OOD_REVIEW", "MONITOR")


@dataclass(frozen=True)
class PolicyThresholds:
    """Frozen at build time from dev OOF distributions."""

    critical_risk: float          # calibrated risk >= this -> CRITICAL band
    urgent_risk: float            # calibrated risk >= this -> URGENT band
    standard_risk: float          # calibrated risk >= this -> STANDARD band
    anomaly_escalation_pct: float # anomaly percentile that escalates MONITOR
    policy_version: str = "1.0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "critical_risk": self.critical_risk,
            "urgent_risk": self.urgent_risk,
            "standard_risk": self.standard_risk,
            "anomaly_escalation_pct": self.anomaly_escalation_pct,
            "policy_version": self.policy_version,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PolicyThresholds":
        return cls(
            critical_risk=float(d["critical_risk"]),
            urgent_risk=float(d["urgent_risk"]),
            standard_risk=float(d["standard_risk"]),
            anomaly_escalation_pct=float(d["anomaly_escalation_pct"]),
            policy_version=str(d.get("policy_version", "1.0")),
        )


def decide(
    *,
    calibrated_risk: float,
    conformal_set: str,
    ood_status: str,
    anomaly_percentile: float | None,
    model_agreement: float,
    verifier_flag: bool | None,
    thresholds: PolicyThresholds,
) -> dict[str, Any]:
    """Route one account. Returns tier + machine-readable reasons.

    Precedence:
      1. OOD -> OOD_REVIEW (model output not trusted for this input)
      2. risk bands (calibrated risk from the frozen scorer)
      3. safety escalations: conformal UNCERTAIN or verifier disagreement
         lift MONITOR/no-band cases into STANDARD_REVIEW; anomaly challenger
         lifts MONITOR into STANDARD_REVIEW
      4. Lens-2 protection: verifier says look-alike AND conformal not
         confidently high -> cap CRITICAL to URGENT (protects false positives;
         never suppresses review entirely)
    """
    reasons: list[str] = []
    if ood_status == "OUT_OF_DISTRIBUTION":
        return _result("OOD_REVIEW", ["input unlike training data; score not trusted"],
                       calibrated_risk, thresholds)

    if calibrated_risk >= thresholds.critical_risk:
        tier = "CRITICAL_REVIEW"
        reasons.append(f"calibrated risk {calibrated_risk:.3f} in critical band")
    elif calibrated_risk >= thresholds.urgent_risk:
        tier = "URGENT_REVIEW"
        reasons.append(f"calibrated risk {calibrated_risk:.3f} in urgent band")
    elif calibrated_risk >= thresholds.standard_risk:
        tier = "STANDARD_REVIEW"
        reasons.append(f"calibrated risk {calibrated_risk:.3f} in standard band")
    else:
        tier = "MONITOR"
        reasons.append(f"calibrated risk {calibrated_risk:.3f} below review bands")

    if tier == "MONITOR":
        if conformal_set == "UNCERTAIN_SET":
            tier = "STANDARD_REVIEW"
            reasons.append("conformal abstention: evidence insufficient for a confident pass")
        elif anomaly_percentile is not None and anomaly_percentile >= thresholds.anomaly_escalation_pct:
            tier = "STANDARD_REVIEW"
            reasons.append(
                f"anomaly challenger disagrees (percentile {anomaly_percentile:.1f}); second-opinion review"
            )

    if tier == "CRITICAL_REVIEW" and verifier_flag is False and conformal_set != "HIGH_RISK_SET":
        tier = "URGENT_REVIEW"
        reasons.append(
            "look-alike protection: hard-negative verifier disagrees and conformal "
            "evidence is not conclusive; routed to review, not fast-tracked"
        )

    if tier != "MONITOR" and model_agreement < 0.5:
        reasons.append(f"low model agreement ({model_agreement:.2f}); treat score cautiously")

    return _result(tier, reasons, calibrated_risk, thresholds)


def _result(tier: str, reasons: list[str], risk: float, thr: PolicyThresholds) -> dict[str, Any]:
    assert tier in TIERS
    return {
        "risk_tier": tier,
        "decision": "HUMAN_REVIEW_REQUIRED" if tier != "MONITOR" else "NOT_CURRENTLY_FLAGGED",
        "reasons": reasons,
        "policy_version": thr.policy_version,
        "auto_action": None,  # policy engine NEVER executes actions
        "limitations": [
            "Behavioural risk is not proof of criminal intent",
            "Final action requires an authorised human analyst",
            "Low risk means 'not currently flagged', never 'certified safe'",
        ],
    }
