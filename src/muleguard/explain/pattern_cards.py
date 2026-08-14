"""Semantic Pattern Cards - named typologies matched from measured behaviour.

``build_proofgraph`` has always accepted a ``patterns=`` argument and drawn
PATTERN nodes for whatever it was given. Nothing produced them, so the layer
was inert. This module is the producer.

A card is *not* a model output and does not affect a score. It is a named
typology from ``docs/PATTERN_AVAILABILITY_MATRIX.md`` matched against the
account's own measured meta-features, carrying the values that triggered it.
Two rules keep this from turning into a second, unvalidated risk model:

* only typologies the matrix grades **DIRECT** are matched. A PROXY typology
  is a single alert flag; presenting it as a detected pattern would claim
  observation of relationships this dataset does not contain.
* a card states its evidence and its threshold. "Pass-through detected" with
  no number is an assertion; "debits are 1.4x credits over 7 days, threshold
  1.0" is a fact a reviewer can disagree with.

Confidence is how far past its threshold the evidence sits, capped at 1.0. It
is a display weight for the ProofGraph edge, not a probability.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

__all__ = ["PatternCard", "PATTERN_DEFINITIONS", "match_patterns"]


@dataclass(frozen=True)
class PatternDefinition:
    """One typology: which meta-feature carries it and where the line sits."""

    id: str
    name: str
    feature: str
    threshold: float
    direction: str  # "above" or "below"
    # Full scale span used to normalise confidence, so a value just over the
    # line does not read as certainty.
    span: float
    detail: str
    typology_row: int  # row in PATTERN_AVAILABILITY_MATRIX.md


# Only DIRECT-grade typologies (matrix rows 1-10). PROXY rows are deliberately
# absent - see the module docstring.
PATTERN_DEFINITIONS: tuple[PatternDefinition, ...] = (
    PatternDefinition(
        "passthrough_7d", "Rapid pass-through (7 days)",
        "MG_PASSTHROUGH_7D", 1.0, "above", 1.0,
        "Debits over the last 7 days match or exceed credits: money is leaving "
        "about as fast as it arrives.", 1),
    PatternDefinition(
        "passthrough_31d", "Sustained pass-through (31 days)",
        "MG_PASSTHROUGH_31D", 1.0, "above", 1.0,
        "The same balance of debits to credits held across 31 days, so this is "
        "not a single unusual week.", 1),
    PatternDefinition(
        "low_retention", "Funds not retained",
        "MG_RETENTION_RATIO", 0.10, "below", 0.10,
        "Average balance is small against the value credited: incoming funds "
        "are not being held.", 2),
    PatternDefinition(
        "dormant_burst", "Dormant reactivation / burst",
        "MG_BURST_7_31", 0.20, "above", 0.40,
        "A disproportionate share of the last 31 days' activity landed in the "
        "last 7, against a 0.226 steady-state baseline.", 3),
    PatternDefinition(
        "rail_fragmentation", "Layering across payment rails",
        "MG_RAIL_FRAGMENTATION", 4.0, "above", 6.0,
        "Material debit activity spread across several distinct rails in 7 "
        "days, which fragments a trail that one rail would keep together.", 4),
    PatternDefinition(
        "cashout_pressure", "Cash-out pressure",
        "MG_CASHOUT_PRESSURE", 0.50, "above", 0.50,
        "A large share of credited value left as cash or ATM withdrawal, which "
        "ends the electronic trail.", 5),
    PatternDefinition(
        "balance_drain", "Throughput beyond balance capacity",
        "MG_BALANCE_DRAIN", 3.0, "above", 7.0,
        "Seven-day debits are several times the average balance: the account "
        "is moving far more value than it holds.", 6),
    PatternDefinition(
        "new_account_activity", "New-account abuse",
        "MG_NEW_ACCOUNT_ACTIVITY", 0.60, "above", 0.40,
        "A young account is moving large electronic value for its tenure.", 7),
    PatternDefinition(
        "profile_mismatch", "Peer-group anomaly",
        "MG_PROFILE_MISMATCH", 2.0, "above", 4.0,
        "Balance behaviour deviates from the bank's own occupation and segment "
        "peer groups for this customer.", 8),
    PatternDefinition(
        "alert_convergence", "Multi-signal convergence",
        "MG_ALERT_CONVERGENCE", 3.0, "above", 5.0,
        "Several independent pre-decision alert types fired on this account at "
        "once.", 9),
    PatternDefinition(
        "odd_hour", "Odd-hour activity",
        "MG_ODD_HOUR_ALERT_RATIO", 0.50, "above", 0.50,
        "A majority of this account's alerts were raised outside normal "
        "banking hours.", 10),
)


@dataclass(frozen=True)
class PatternCard:
    id: str
    name: str
    confidence: float
    detail: str
    source: str
    supporting_features: list[str]
    observed_value: float
    threshold: float
    direction: str
    typology_row: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "confidence": round(self.confidence, 4),
            "detail": self.detail,
            "source": self.source,
            "supporting_features": list(self.supporting_features),
            "evidence": {
                "feature": self.supporting_features[0],
                "observed_value": round(self.observed_value, 6),
                "threshold": self.threshold,
                "direction": self.direction,
            },
            "typology_row": self.typology_row,
            "contract": (
                "a matched typology describes measured behaviour; it is not a "
                "model output and did not contribute to the risk score"),
        }


def _confidence(value: float, d: PatternDefinition) -> float:
    """How far past the line, normalised to the definition's span."""
    if d.direction == "above":
        excess = value - d.threshold
    else:
        excess = d.threshold - value
    if excess <= 0:
        return 0.0
    return min(1.0, excess / d.span) if d.span > 0 else 1.0


def match_patterns(row_values: dict[str, Any],
                   definitions: Sequence[PatternDefinition] = PATTERN_DEFINITIONS,
                   ) -> list[dict[str, Any]]:
    """Return the typology cards this account's own values support.

    ``row_values`` maps feature name to value - the raw row, meta-features
    included. Missing or non-numeric features simply do not match: an absent
    measurement is not evidence of a pattern, and inventing one from a default
    would be exactly the fabrication the availability matrix rules out.
    """
    cards: list[dict[str, Any]] = []
    for d in definitions:
        raw = row_values.get(d.feature)
        if raw is None:
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if value != value:  # NaN
            continue
        matched = value > d.threshold if d.direction == "above" else value < d.threshold
        if not matched:
            continue
        cards.append(PatternCard(
            id=d.id, name=d.name, confidence=_confidence(value, d),
            detail=d.detail, source="pattern_card:availability_matrix",
            supporting_features=[d.feature], observed_value=value,
            threshold=d.threshold, direction=d.direction,
            typology_row=d.typology_row,
        ).to_dict())

    cards.sort(key=lambda c: -c["confidence"])
    return cards
