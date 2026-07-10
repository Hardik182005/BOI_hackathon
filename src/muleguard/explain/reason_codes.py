"""SHAP-based reason codes with cohort context.

For anonymised features the language is strictly numerical-comparative:
"F1702 is high relative to the legitimate cohort and increases the model
score." No invented business semantics. The interpretable registry maps only
VERIFIED semantic names (from the raw categorical columns).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from muleguard.logging import get_logger

log = get_logger("explain.reason_codes")

# Only columns whose meaning is verifiable from the raw data itself
# (decoded categorical values / parsed dates). Everything else stays Fxxxx.
VERIFIED_SEMANTIC_NAMES: dict[str, str] = {
    "F3886": "account/product type",
    "F3888": "account opening date",
    "F3889": "tenure/vintage bucket",
    "F3890": "region/locality type",
    "F3891": "occupation",
    "F3892": "gender",
    "F3893": "customer segment",
}


class CohortReference:
    """Per-feature quantiles of the legitimate dev cohort (for percentiles)."""

    def __init__(self, X_dev: np.ndarray, y_dev: np.ndarray, feature_names: list[str]):
        self.feature_names = feature_names
        legit = X_dev[y_dev == 0]
        # store sorted non-missing values per feature for percentile lookup
        self._sorted = [np.sort(legit[:, j][~np.isnan(legit[:, j])]) for j in range(legit.shape[1])]
        self.medians = np.array([
            float(np.median(s)) if len(s) else float("nan") for s in self._sorted
        ])

    def percentile(self, j: int, value: float) -> float | None:
        s = self._sorted[j]
        if len(s) == 0 or np.isnan(value):
            return None
        return float(100.0 * np.searchsorted(s, value, side="right") / len(s))


def shap_reason_codes(
    model: Any,
    X_rows: np.ndarray,
    feature_names: list[str],
    cohort: CohortReference,
    top_k: int = 5,
) -> list[list[dict[str, Any]]]:
    """Per-row top-k signed SHAP contributions with cohort percentiles.

    Uses LightGBM's native TreeSHAP (pred_contrib) - exact for tree models.
    """
    contrib = model.predict(X_rows, pred_contrib=True) if hasattr(model, "predict") else None
    if contrib is None or contrib.shape[1] != len(feature_names) + 1:
        raise RuntimeError("pred_contrib unavailable or shape mismatch")
    shap_vals = contrib[:, :-1]
    base = contrib[:, -1]

    out: list[list[dict[str, Any]]] = []
    for i in range(X_rows.shape[0]):
        order = np.argsort(-np.abs(shap_vals[i]), kind="stable")[:top_k]
        rows = []
        for j in order:
            val = X_rows[i, j]
            pct = cohort.percentile(j, float(val)) if not np.isnan(val) else None
            rows.append({
                "feature": feature_names[j],
                "verified_semantic_name": VERIFIED_SEMANTIC_NAMES.get(feature_names[j]),
                "value": None if np.isnan(val) else float(val),
                "legitimate_cohort_median": None if np.isnan(cohort.medians[j]) else float(cohort.medians[j]),
                "legitimate_percentile": pct,
                "shap_contribution": float(shap_vals[i, j]),
                "direction": "INCREASES_RISK" if shap_vals[i, j] > 0 else "DECREASES_RISK",
            })
        out.append(rows)
    return out


def counterfactual_sensitivity(
    model: Any,
    x_row: np.ndarray,
    feature_names: list[str],
    reason_rows: list[dict[str, Any]],
    cohort: CohortReference,
    threshold: float,
    max_features: int = 3,
) -> list[dict[str, Any]]:
    """Model-sensitivity examples: move top INCREASES_RISK features to the
    legitimate cohort median, rescore, report the score change.

    Explicitly labelled 'model sensitivity example' - no causal claim.
    Only mutable behavioural (numeric, non-registry) features are moved.
    """
    name_idx = {f: j for j, f in enumerate(feature_names)}
    results = []
    for r in reason_rows:
        if len(results) >= max_features or r["direction"] != "INCREASES_RISK":
            continue
        if r["verified_semantic_name"] is not None:
            continue  # demographic/registry-like fields are never proposed for change
        j = name_idx[r["feature"]]
        med = cohort.medians[j]
        if np.isnan(med):
            continue
        x_mod = x_row.copy()
        x_mod[j] = med
        p_orig = float(model.predict(x_row.reshape(1, -1))[0])
        p_mod = float(model.predict(x_mod.reshape(1, -1))[0])
        results.append({
            "kind": "model sensitivity example (not a causal statement)",
            "feature": r["feature"],
            "from_value": r["value"],
            "to_value": float(med),
            "score_before": p_orig,
            "score_after": p_mod,
            "crosses_threshold": bool(p_orig >= threshold > p_mod),
        })
    return results
