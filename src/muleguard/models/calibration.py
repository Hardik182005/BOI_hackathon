"""Probability calibration selected on OOF data (never locked test).

Compares Platt (sigmoid) scaling vs isotonic regression by Brier score and
ECE using internal cross-fitting on the OOF predictions. With ~65 dev
positives, isotonic is at high overfit risk; the selector prefers Platt
unless isotonic wins BOTH Brier and ECE by a margin.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from muleguard.evaluation.metrics import expected_calibration_error
from muleguard.logging import get_logger

log = get_logger("models.calibration")


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


@dataclass
class PlattCalibrator:
    a: float = 1.0
    b: float = 0.0

    def fit(self, scores: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        lr = LogisticRegression(C=1e6, max_iter=2000)  # unregularised Platt
        lr.fit(_logit(scores).reshape(-1, 1), y)
        self.a = float(lr.coef_[0][0])
        self.b = float(lr.intercept_[0])
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        z = self.a * _logit(scores) + self.b
        return 1.0 / (1.0 + np.exp(-z))


@dataclass
class IsotonicCalibrator:
    iso: IsotonicRegression | None = None

    def fit(self, scores: np.ndarray, y: np.ndarray) -> "IsotonicCalibrator":
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.iso.fit(scores, y)
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        return self.iso.predict(scores)


def crossfit_calibrated(scores, y, method: Literal["platt", "isotonic"], seed=42, n_folds=5):
    """Cross-fitted calibrated probabilities (each point calibrated by a model
    that never saw it) - the honest way to compare calibrators on OOF data."""
    out = np.zeros_like(scores, dtype=float)
    skf = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
    for tr, va in skf.split(scores.reshape(-1, 1), y):
        cal = PlattCalibrator() if method == "platt" else IsotonicCalibrator()
        cal.fit(scores[tr], y[tr])
        out[va] = cal.predict(scores[va])
    return np.clip(out, 0.0, 1.0)


def select_calibrator(scores: np.ndarray, y: np.ndarray, seed: int = 42) -> dict:
    """Return comparison + winner. Isotonic must beat Platt on BOTH Brier and
    ECE by >2% relative to overcome the small-positives overfit prior."""
    from sklearn.metrics import brier_score_loss

    results = {}
    for method in ("platt", "isotonic"):
        p = crossfit_calibrated(scores, y, method, seed=seed)
        results[method] = {
            "brier": float(brier_score_loss(y, p)),
            "ece": float(expected_calibration_error(y, p)["ece"]),
        }
    iso, pla = results["isotonic"], results["platt"]
    iso_wins = iso["brier"] < pla["brier"] * 0.98 and iso["ece"] < pla["ece"] * 0.98
    winner = "isotonic" if iso_wins else "platt"
    return {
        "comparison": results,
        "winner": winner,
        "rule": "isotonic needs >=2% relative improvement on BOTH Brier and ECE "
                "(few positives -> prefer simpler calibrator)",
    }


def fit_final_calibrator(scores: np.ndarray, y: np.ndarray, method: str):
    cal = PlattCalibrator() if method == "platt" else IsotonicCalibrator()
    return cal.fit(scores, y)
