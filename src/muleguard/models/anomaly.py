"""Lens 3 challengers: Isolation Forest anomaly signal + OOD detector.

Both are second opinions. They never override the supervised score; the
policy engine uses them to ESCALATE to review or to declare OOD_REVIEW.
Fitted on development data only.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import IsolationForest

from muleguard import settings
from muleguard.logging import get_logger

log = get_logger("models.anomaly")


@dataclass
class AnomalyChallenger:
    """Isolation Forest over the legitimate dev cohort, reduced feature view.

    Scores are converted to percentiles against the dev score distribution,
    so downstream logic reasons in [0, 100] regardless of library internals.
    """

    contamination: float = 0.01
    n_estimators: int = 300
    seed: int = 42
    forest: IsolationForest | None = None
    medians_: np.ndarray | None = None
    dev_scores_: np.ndarray | None = None

    def fit(self, X_dev: np.ndarray, y_dev: np.ndarray) -> "AnomalyChallenger":
        X_legit = X_dev[y_dev == 0]
        self.medians_ = np.nanmedian(X_legit, axis=0)
        self.medians_ = np.where(np.isnan(self.medians_), 0.0, self.medians_)
        Xi = self._impute(X_legit)
        self.forest = IsolationForest(
            n_estimators=self.n_estimators, contamination=self.contamination,
            random_state=self.seed, n_jobs=int(settings.load_config("train")["n_jobs"]),
        )
        self.forest.fit(Xi)
        self.dev_scores_ = -self.forest.score_samples(self._impute(X_dev))
        return self

    def _impute(self, X: np.ndarray) -> np.ndarray:
        return np.where(np.isnan(X), self.medians_, X)

    def anomaly_percentile(self, X: np.ndarray) -> np.ndarray:
        raw = -self.forest.score_samples(self._impute(X))
        return 100.0 * np.searchsorted(np.sort(self.dev_scores_), raw, side="right") / len(self.dev_scores_)


@dataclass
class OODDetector:
    """Multi-signal OOD detection on the reduced feature view (dev-fitted).

    Signals:
      - missingness-profile z-score (per-row missing count vs dev distribution)
      - selected-feature range violations (outside dev [min, max] widened 20%)
      - kNN distance in median-imputed, robust-scaled space vs dev quantile
    OOD if missingness z OR violation share OR kNN distance exceed thresholds.
    """

    missingness_z: float = 4.0
    violation_share: float = 0.20
    knn_quantile: float = 0.999
    k: int = 10
    stats_: dict = field(default_factory=dict)

    def fit(self, X_dev: np.ndarray) -> "OODDetector":
        miss = np.isnan(X_dev).mean(axis=1)
        self.stats_["miss_mu"] = float(miss.mean())
        self.stats_["miss_sd"] = float(miss.std() + 1e-12)

        lo = np.nanmin(X_dev, axis=0)
        hi = np.nanmax(X_dev, axis=0)
        span = np.where(hi - lo <= 0, 1.0, hi - lo)
        self.stats_["lo"] = lo - 0.2 * span
        self.stats_["hi"] = hi + 0.2 * span

        med = np.nanmedian(X_dev, axis=0)
        med = np.where(np.isnan(med), 0.0, med)
        q75, q25 = np.nanpercentile(X_dev, 75, axis=0), np.nanpercentile(X_dev, 25, axis=0)
        iqr = np.where((q75 - q25) <= 0, 1.0, q75 - q25)
        self.stats_["med"], self.stats_["iqr"] = med, iqr
        Z = self._scale(X_dev)
        self.stats_["ref"] = Z
        d = self._knn_dist(Z, exclude_self=True)
        self.stats_["knn_thr"] = float(np.quantile(d, self.knn_quantile))
        return self

    def _scale(self, X: np.ndarray) -> np.ndarray:
        Xi = np.where(np.isnan(X), self.stats_["med"], X)
        return (Xi - self.stats_["med"]) / self.stats_["iqr"]

    def _knn_dist(self, Z: np.ndarray, exclude_self: bool = False) -> np.ndarray:
        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(n_neighbors=self.k + (1 if exclude_self else 0))
        nn.fit(self.stats_["ref"])
        d, _ = nn.kneighbors(Z)
        return d[:, -1]

    def status(self, X: np.ndarray) -> tuple[list[str], list[dict]]:
        miss = np.isnan(X).mean(axis=1)
        z = (miss - self.stats_["miss_mu"]) / self.stats_["miss_sd"]
        with np.errstate(invalid="ignore"):
            viol = (
                ((X < self.stats_["lo"]) | (X > self.stats_["hi"])) & ~np.isnan(X)
            ).mean(axis=1)
        d = self._knn_dist(self._scale(X))
        out_status, out_detail = [], []
        for i in range(len(X)):
            checks = {
                "missingness_z": float(z[i]),
                "range_violation_share": float(viol[i]),
                "knn_distance": float(d[i]),
                "knn_threshold": self.stats_["knn_thr"],
            }
            is_ood = (
                z[i] > self.missingness_z
                or viol[i] > self.violation_share
                or d[i] > self.stats_["knn_thr"]
            )
            out_status.append("OUT_OF_DISTRIBUTION" if is_ood else "IN_DISTRIBUTION")
            out_detail.append(checks)
        return out_status, out_detail
