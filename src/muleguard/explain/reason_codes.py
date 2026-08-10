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


def tree_shap(model: Any, family: str, X_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact TreeSHAP contributions from the model that produced the score.

    Dispatching on family matters more than it looks. Explaining one model
    while a different one decides the case would put fabricated evidence in
    front of an analyst: the reasons would be internally consistent and simply
    not be the reasons the alert was raised. Each library computes exact
    TreeSHAP in raw-margin space, so the returned contributions mean the same
    thing whichever family won the tournament.

    Returns ``(contributions, base_value)`` with shapes ``(n, n_features)``
    and ``(n,)``.
    """
    if family == "lightgbm":
        contrib = model.booster_.predict(X_rows, pred_contrib=True)
    elif family == "xgboost":
        import xgboost as xgb

        booster = model.get_booster()
        dm = xgb.DMatrix(X_rows, feature_names=booster.feature_names)
        contrib = booster.predict(dm, pred_contribs=True)
    elif family == "catboost":
        from catboost import Pool

        contrib = model.get_feature_importance(Pool(X_rows), type="ShapValues")
    else:
        raise RuntimeError(f"no TreeSHAP path for model family {family!r}")

    contrib = np.asarray(contrib)
    if contrib.ndim != 2 or contrib.shape[1] != X_rows.shape[1] + 1:
        raise RuntimeError(
            f"{family} returned SHAP of shape {contrib.shape}; expected "
            f"(n, {X_rows.shape[1] + 1})")
    return contrib[:, :-1], contrib[:, -1]


def _proba(model: Any, X: np.ndarray) -> np.ndarray:
    """Positive-class probability, uniform across the three families."""
    return np.asarray(model.predict_proba(X))[:, 1]


def shap_reason_codes(
    model: Any,
    X_rows: np.ndarray,
    feature_names: list[str],
    cohort: CohortReference,
    top_k: int = 5,
    family: str = "lightgbm",
) -> list[list[dict[str, Any]]]:
    """Per-row top-k signed SHAP contributions with cohort percentiles.

    ``model`` is the fitted estimator of ``family`` - the one whose score the
    alert is based on - not a bare booster.
    """
    shap_vals, base = tree_shap(model, family, X_rows)
    if shap_vals.shape[1] != len(feature_names):
        raise RuntimeError("SHAP width does not match the kept feature list")

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
    family: str = "lightgbm",
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
        p_orig = float(_proba(model, x_row.reshape(1, -1))[0])
        p_mod = float(_proba(model, x_mod.reshape(1, -1))[0])
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
