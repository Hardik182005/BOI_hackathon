"""Mondrian (class-conditional) split-conformal prediction for abstention.

Implemented directly (MAPIE API compatibility not assumed; procedure documented):

Calibration (on OOF predictions only):
  - nonconformity for class 1: 1 - p(x);  for class 0: p(x)
  - per-class quantile q_c at miscoverage alpha with the standard
    (n+1) finite-sample correction

Prediction for a new score p:
  - class c is in the prediction set iff nonconformity_c(p) <= q_c
  - {1} only        -> HIGH_RISK_SET
  - {0} only        -> LOW_RISK_SET      (never certifies "safe"; monitoring continues)
  - {0,1} or {}     -> UNCERTAIN_SET     (route to human review)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from muleguard.logging import get_logger

log = get_logger("models.conformal")


@dataclass
class MondrianConformal:
    alpha: float = 0.1
    q_pos: float = 1.0  # quantile of (1 - p) among true positives
    q_neg: float = 1.0  # quantile of p among true negatives

    def fit(self, probs: np.ndarray, y: np.ndarray) -> "MondrianConformal":
        pos_scores = 1.0 - probs[y == 1]
        neg_scores = probs[y == 0]
        self.q_pos = _finite_sample_quantile(pos_scores, self.alpha)
        self.q_neg = _finite_sample_quantile(neg_scores, self.alpha)
        log.info(
            "conformal fit: alpha=%.2f q_pos=%.4f (n=%d) q_neg=%.4f (n=%d)",
            self.alpha, self.q_pos, len(pos_scores), self.q_neg, len(neg_scores),
        )
        return self

    def predict_set(self, probs: np.ndarray) -> list[str]:
        out = []
        for p in np.atleast_1d(probs):
            in_pos = (1.0 - p) <= self.q_pos
            in_neg = p <= self.q_neg
            if in_pos and not in_neg:
                out.append("HIGH_RISK_SET")
            elif in_neg and not in_pos:
                out.append("LOW_RISK_SET")
            else:
                out.append("UNCERTAIN_SET")
        return out

    def to_dict(self) -> dict:
        return {"alpha": self.alpha, "q_pos": self.q_pos, "q_neg": self.q_neg}

    @classmethod
    def from_dict(cls, d: dict) -> "MondrianConformal":
        m = cls(alpha=d["alpha"])
        m.q_pos, m.q_neg = d["q_pos"], d["q_neg"]
        return m


def _finite_sample_quantile(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    if n == 0:
        return 1.0  # no calibration data -> everything conformal (max caution)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = min(k, n)
    return float(np.sort(scores)[k - 1])


def empirical_coverage(conf: MondrianConformal, probs: np.ndarray, y: np.ndarray) -> dict:
    sets = conf.predict_set(probs)
    sets = np.array(sets)
    pos_covered = float(np.mean(np.isin(sets[y == 1], ["HIGH_RISK_SET", "UNCERTAIN_SET"]))) if (y == 1).any() else None
    neg_covered = float(np.mean(np.isin(sets[y == 0], ["LOW_RISK_SET", "UNCERTAIN_SET"]))) if (y == 0).any() else None
    return {
        "target_coverage": 1 - conf.alpha,
        "positive_coverage": pos_covered,
        "negative_coverage": neg_covered,
        "abstention_rate": float(np.mean(sets == "UNCERTAIN_SET")),
        "share_high": float(np.mean(sets == "HIGH_RISK_SET")),
        "share_low": float(np.mean(sets == "LOW_RISK_SET")),
    }
