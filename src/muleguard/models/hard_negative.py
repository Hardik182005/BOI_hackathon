"""Lens 2 hard-negative verifier.

Mined from OOF predictions only: legitimate accounts the screener scores
highest (the false-positive battlefield). A second model is trained to
separate confirmed positives from these hard negatives plus a background
sample; at scoring time its verdict is a *protective* signal - disagreement
with the screener routes to review instead of fast-tracking punishment.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from muleguard import settings
from muleguard.logging import get_logger

log = get_logger("models.hard_negative")


def mine_hard_negatives(
    oof_scores: np.ndarray, y: np.ndarray, band_quantile: float = 0.98,
    background_ratio: int = 3, seed: int = 42,
) -> dict[str, np.ndarray]:
    """Indices of hard negatives (top-scored legit) + background sample."""
    rng = np.random.default_rng(seed)
    neg_idx = np.where(y == 0)[0]
    neg_scores = oof_scores[neg_idx]
    thr = np.quantile(neg_scores, band_quantile)
    hard = neg_idx[neg_scores >= thr]
    easy = neg_idx[neg_scores < thr]
    background = rng.choice(easy, size=min(len(easy), background_ratio * len(hard)), replace=False)
    return {
        "hard_negative_idx": hard,
        "background_idx": background,
        "positive_idx": np.where(y == 1)[0],
        "band_threshold": np.array([thr]),
    }


@dataclass
class HardNegativeVerifier:
    """LightGBM verifier: positives vs (hard negatives + background)."""

    seed: int = 42
    model: object | None = None
    decision_threshold: float = 0.5

    def fit(self, X: np.ndarray, y: np.ndarray, mined: dict[str, np.ndarray]) -> "HardNegativeVerifier":
        import lightgbm as lgb

        idx = np.concatenate([mined["positive_idx"], mined["hard_negative_idx"], mined["background_idx"]])
        Xv, yv = X[idx], y[idx]
        self.model = lgb.LGBMClassifier(
            objective="binary", n_estimators=400, learning_rate=0.05,
            num_leaves=15, min_child_samples=15,
            colsample_bytree=0.6, subsample=0.8, subsample_freq=1,
            scale_pos_weight=(yv == 0).sum() / max(yv.sum(), 1),
            random_state=self.seed, deterministic=True, force_row_wise=True,
            n_jobs=int(settings.load_config("train")["n_jobs"]), verbose=-1,
        )
        self.model.fit(Xv, yv)
        log.info(
            "verifier trained: %d pos vs %d hard-neg + %d background",
            len(mined["positive_idx"]), len(mined["hard_negative_idx"]), len(mined["background_idx"]),
        )
        return self

    def confirms_risk(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(bool verdicts, verifier probabilities). True = verifier also sees
        mule-like behaviour; False = looks like a known false-positive pattern."""
        p = self.model.predict_proba(X)[:, 1]
        return p >= self.decision_threshold, p
