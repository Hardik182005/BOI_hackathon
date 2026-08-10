"""Merchant Legitimacy Verifier (addendum UPDATE 9).

The naive false-positive fix is to notice that merchants trip mule rules and
multiply their score by 0.7. That is unacceptable here for three reasons: the
constant is invented, it silently rewrites a calibrated probability into
something that is no longer a probability, and it is unfalsifiable - nobody can
say afterwards whether it helped or hid a real mule.

This module replaces it with something learnable and auditable. A small model
is fitted on the merchant/business evidence view alone - POS acceptance, GST
activity, balance profile, standing instructions, loans, tenure - and asked the
same question the main model is asked: is this account a mule? Its output is
therefore not a hand-written "merchantness" score but a measured statement of
how much *exculpatory* weight the business evidence carries on its own.

What it may and may not do is fixed:

  * It NEVER modifies calibrated risk, the model score, or any threshold.
  * It NEVER suppresses review. The strongest thing it can do is lower the
    confidence attached to an automatic escalation, which routes a case to a
    human sooner rather than later.
  * Every adjustment it contributes is emitted with its inputs, so an auditor
    can reconstruct the decision from the record alone.

The verifier is fitted out-of-fold on the development split, exactly like every
other model here, and it never sees the locked test set.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any

import numpy as np

from muleguard.logging import get_logger

log = get_logger("models.merchant")

# Bands for the exculpatory evidence, fixed before the verifier was fitted.
# They are quantiles of the DEVELOPMENT distribution, so "strong" means
# "stronger business evidence than 80% of this book", not an absolute claim.
LEGITIMACY_BANDS = (
    (0.80, "STRONG_BUSINESS_EVIDENCE"),
    (0.50, "SOME_BUSINESS_EVIDENCE"),
    (0.00, "LITTLE_BUSINESS_EVIDENCE"),
)

CONFIDENCE_HIGH = "HIGH"
CONFIDENCE_MEDIUM = "MEDIUM"
CONFIDENCE_LOW = "LOW"

DISCLAIMER = (
    "Business evidence is exculpatory context, never a whitelist. A genuine "
    "merchant can also be a mule, and mule networks are known to use business "
    "accounts precisely because they attract less scrutiny. This verifier can "
    "only reduce the confidence attached to an automatic escalation; it cannot "
    "clear an account, cannot lower its risk score, and cannot remove it from "
    "a review queue."
)


@dataclass
class MerchantVerdict:
    """One account's business-evidence reading."""

    legitimacy: float          # 0-1, higher = stronger legitimate-business evidence
    band: str
    mule_probability_from_business_evidence: float
    components: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "legitimacy": round(float(self.legitimacy), 5),
            "band": self.band,
            "mule_probability_from_business_evidence": round(
                float(self.mule_probability_from_business_evidence), 6),
            "components": self.components,
            "disclaimer": DISCLAIMER,
        }


class MerchantLegitimacyVerifier:
    """Learned exculpatory-evidence model over the merchant/business view.

    ``legitimacy`` is one minus the percentile of the fitted model's mule
    probability within the development distribution. Percentile rather than raw
    probability because the bands must mean something stable across refits: a
    model that becomes better calibrated would otherwise silently move every
    account's band without anyone changing a threshold.
    """

    def __init__(self, feature_names: list[str]) -> None:
        self.feature_names = list(feature_names)
        self.model: Any = None
        self._reference: np.ndarray | None = None
        self.fitted_utc: str | None = None
        self.dev_pr_auc: float | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, *, seed: int = 42) -> "MerchantLegitimacyVerifier":
        from lightgbm import LGBMClassifier

        pos = max(int(y.sum()), 1)
        clf = LGBMClassifier(
            n_estimators=250, learning_rate=0.05, num_leaves=15,
            min_child_samples=15, subsample=0.9, subsample_freq=1,
            colsample_bytree=0.7, reg_lambda=1.0,
            scale_pos_weight=float((len(y) - pos) / pos),
            random_state=seed, n_jobs=2, verbose=-1,
        )
        clf.fit(X, y)
        self.model = clf
        self._reference = np.sort(clf.predict_proba(X)[:, 1])
        self.fitted_utc = dt.datetime.now(dt.timezone.utc).isoformat()
        log.info("merchant verifier fitted on %d rows x %d business features "
                 "(%d positives)", X.shape[0], X.shape[1], int(y.sum()))
        return self

    def _percentile(self, p: np.ndarray) -> np.ndarray:
        if self._reference is None or not len(self._reference):
            return np.full(len(p), 0.5)
        return np.searchsorted(self._reference, p, side="right") / len(self._reference)

    def verdicts(self, X: np.ndarray, contributions: list[dict] | None = None
                 ) -> list[MerchantVerdict]:
        if self.model is None:
            raise RuntimeError("verifier is not fitted")
        p = self.model.predict_proba(X)[:, 1]
        legit = 1.0 - self._percentile(p)
        out = []
        for i in range(len(p)):
            band = next(b for cut, b in LEGITIMACY_BANDS if legit[i] >= cut)
            comp = contributions[i] if contributions else {}
            out.append(MerchantVerdict(
                legitimacy=float(legit[i]), band=band,
                mule_probability_from_business_evidence=float(p[i]),
                components=comp,
            ))
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_names": self.feature_names,
            "n_features": len(self.feature_names),
            "fitted_utc": self.fitted_utc,
            "dev_pr_auc": self.dev_pr_auc,
            "bands": {b: cut for cut, b in LEGITIMACY_BANDS},
            "contract": DISCLAIMER,
        }


def escalation_confidence(
    *,
    risk_tier: str,
    calibrated_risk: float,
    conformal_set: str,
    model_agreement: float,
    merchant: MerchantVerdict | None,
) -> dict[str, Any]:
    """How confident are we in escalating this case automatically?

    Separate from the risk score on purpose. The score answers "how mule-like is
    this behaviour?"; this answers "how sure are we that a machine should push
    this up the queue without a person looking first?". Strong business evidence
    lowers the second without touching the first, which is exactly the
    distinction UPDATE 9 asks for and the one a blind multiplier destroys.
    """
    adjustments: list[dict[str, Any]] = []
    confidence = CONFIDENCE_HIGH

    if conformal_set == "UNCERTAIN_SET":
        confidence = CONFIDENCE_MEDIUM
        adjustments.append({
            "factor": "conformal_abstention",
            "observed": conformal_set,
            "effect": "confidence HIGH -> MEDIUM",
            "rationale": "the conformal predictor declined to commit to a class",
        })

    if model_agreement < 0.5:
        before = confidence
        confidence = CONFIDENCE_MEDIUM if confidence == CONFIDENCE_HIGH else CONFIDENCE_LOW
        adjustments.append({
            "factor": "model_disagreement",
            "observed": round(float(model_agreement), 4),
            "effect": f"confidence {before} -> {confidence}",
            "rationale": (
                "the base models disagree about this account. Disagreement is "
                "uncertainty, not evidence of guilt - it does not raise the risk "
                "score (addendum UPDATE 6)"),
        })

    if merchant is not None and merchant.band == "STRONG_BUSINESS_EVIDENCE":
        before = confidence
        confidence = CONFIDENCE_MEDIUM if confidence == CONFIDENCE_HIGH else CONFIDENCE_LOW
        adjustments.append({
            "factor": "merchant_legitimacy_verifier",
            "observed": {
                "band": merchant.band,
                "legitimacy": round(merchant.legitimacy, 4),
                "mule_probability_from_business_evidence": round(
                    merchant.mule_probability_from_business_evidence, 6),
            },
            "effect": f"confidence {before} -> {confidence}",
            "rationale": (
                "the business-evidence model, using merchant features alone, "
                "does not corroborate the alert. The case stays in the review "
                "queue at its measured risk; only the confidence in escalating "
                "it without human eyes is reduced"),
        })

    return {
        "auto_escalation_confidence": confidence,
        "risk_tier_unchanged": risk_tier,
        "calibrated_risk_unchanged": round(float(calibrated_risk), 6),
        "adjustments": adjustments,
        "guarantees": [
            "no adjustment modified the calibrated risk or any threshold",
            "no adjustment removed this account from a review queue",
            "every adjustment above lists the value that triggered it",
        ],
        "merchant": merchant.to_dict() if merchant is not None else None,
    }
