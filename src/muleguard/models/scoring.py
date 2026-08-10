"""Deterministic scoring service over the frozen bundle.

Given raw feature rows (dict or DataFrame with original column names), produce
the complete Trinetra output: base scores, calibrated risk, agreement,
conformal set, verifier verdict, anomaly percentile, OOD status, policy tier
and SHAP reason codes. Pure function of (bundle, input) - no randomness, no
network, no LLM.
"""
from __future__ import annotations

from typing import Any

import joblib
import numpy as np
import polars as pl

from muleguard import settings
from muleguard.action.policy import PolicyThresholds, decide
from muleguard.explain.reason_codes import counterfactual_sensitivity, shap_reason_codes
from muleguard.logging import get_logger
from muleguard.models.calibration import PlattCalibrator  # noqa: F401 (unpickle)
from muleguard.models.conformal import MondrianConformal

log = get_logger("models.scoring")

_BUNDLE_CACHE: dict[str, Any] | None = None


def load_bundle(path=None) -> dict[str, Any]:
    global _BUNDLE_CACHE
    if _BUNDLE_CACHE is None or path is not None:
        p = path or settings.MODELS_DIR / "final_bundle.joblib"
        _BUNDLE_CACHE = joblib.load(p)
        log.info("bundle loaded: winner=%s sha-fp=%s",
                 _BUNDLE_CACHE["winner_oof_name"],
                 _BUNDLE_CACHE["data_fingerprint_sha256"][:12])
    return _BUNDLE_CACHE


class SchemaError(ValueError):
    """Raised when required selected features are missing from a request."""


_DATE_FORMATS = ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%m-%d-%Y", "%d-%m-%Y"]


def _canonicalize_request(rows: pl.DataFrame, selected: list[str],
                          schema: dict[str, str]) -> pl.DataFrame:
    """Coerce raw request values to the training schema per column.

    numeric: parse floats ("NA"/blank -> null); date: parse known formats;
    categorical: keep utf8 (trained cat_maps applied downstream). Values that
    cannot be coerced become null (missing), which the OOD/missingness
    signals see - never a silent wrong number.
    """
    exprs = []
    for c in selected:
        kind = schema.get(c, "numeric")
        col = pl.col(c)
        if kind == "numeric":
            e = (
                col.cast(pl.Utf8, strict=False).str.strip_chars()
                .replace({"NA": None, "": None})
                .cast(pl.Float64, strict=False)
            )
        elif kind == "date":
            base = col.cast(pl.Utf8, strict=False).str.strip_chars()
            e = None
            for fmt in _DATE_FORMATS:
                p = base.str.strptime(pl.Date, fmt, strict=False)
                e = p if e is None else e.fill_null(p)
        else:  # categorical
            e = (
                col.cast(pl.Utf8, strict=False).str.strip_chars()
                .replace({"NA": None, "": None})
            )
        exprs.append(e.alias(c))
    return rows.select(exprs)


def _matrix_from_rows(rows: pl.DataFrame, bundle: dict[str, Any]) -> np.ndarray:
    """Build the kept-feature matrix from raw-named input columns.

    MG_* meta-features are derived here when the bundle needs them: they are
    row-wise functions of raw columns, so a caller supplies raw data and the
    block is recomputed rather than uploaded. Nothing about that derivation
    consults the training set.

    Missing SELECTED features raise SchemaError (never silent zero-fill);
    unknown extra columns are ignored; categorical codes come from the
    TRAINED mappings frozen in the bundle (unseen category -> -1).
    """
    from muleguard.features.preprocessing import encode_dataframe

    selected = bundle["feature_list_selected"]
    needs_meta = [c for c in bundle.get("meta_features_required", [])
                  if c not in rows.columns]
    if needs_meta:
        from muleguard.features.frame import attach_meta

        rows = attach_meta(rows)

    missing = [c for c in selected if c not in rows.columns]
    if missing:
        raise SchemaError(
            f"required features missing from request: {missing[:8]}"
            + ("..." if len(missing) > 8 else "")
        )
    canon = _canonicalize_request(rows, selected, bundle.get("feature_schema", {}))
    X, _, _ = encode_dataframe(canon, selected, cat_maps=bundle.get("cat_maps"))
    return bundle["preprocessor"].transform(X)


def score_rows(rows: pl.DataFrame, bundle: dict[str, Any] | None = None,
               with_explanations: bool = True) -> list[dict[str, Any]]:
    b = bundle or load_bundle()
    Xp = _matrix_from_rows(rows, b)
    thresholds = PolicyThresholds.from_dict(b["policy_thresholds"])
    conformal = MondrianConformal.from_dict(b["conformal"])

    base_scores = {m: b["models"][m].predict_proba(Xp)[:, 1] for m in b["models"]}
    S = np.column_stack(list(base_scores.values()))
    family = b.get("winner_family", "lightgbm")
    if family == "ensemble" and b.get("stacker") is not None:
        order = list(b["models"].keys())  # lightgbm, xgboost, catboost
        Sb = np.column_stack([base_scores[m] for m in order])
        Z = np.log(np.clip(Sb, 1e-7, 1 - 1e-7) / (1 - np.clip(Sb, 1e-7, 1 - 1e-7)))
        raw = b["stacker"].predict_proba(Z)[:, 1]
    else:
        raw = base_scores.get(family, base_scores["lightgbm"])
    calibrated = np.clip(b["calibrator"].predict(raw), 0.0, 1.0)
    agreement = 1.0 - (S.max(axis=1) - S.min(axis=1))
    conf_sets = conformal.predict_set(calibrated)
    verifier_flags, verifier_probs = b["verifier"].confirms_risk(Xp)
    anom_pct = b["anomaly"].anomaly_percentile(Xp)
    ood_status, ood_detail = b["ood"].status(Xp)
    # rank percentile vs dev OOF calibrated distribution is approximated by
    # the anomaly dev reference; we use calibrated risk directly for ranks
    # Explain the model that produced `raw`, not a convenient stand-in. When an
    # ensemble wins there is no single tree to attribute the stacked score to,
    # so we explain the strongest base model and say so in the record rather
    # than presenting its reasons as the ensemble's.
    explain_family = family if family in b["models"] else "lightgbm"
    reasons = None
    if with_explanations:
        reasons = shap_reason_codes(
            b["models"][explain_family], Xp, b["feature_list_kept"], b["cohort"],
            family=explain_family,
        )

    out = []
    for i in range(len(Xp)):
        pol = decide(
            calibrated_risk=float(calibrated[i]),
            conformal_set=conf_sets[i],
            ood_status=ood_status[i],
            anomaly_percentile=float(anom_pct[i]),
            model_agreement=float(agreement[i]),
            verifier_flag=bool(verifier_flags[i]),
            thresholds=thresholds,
        )
        rec: dict[str, Any] = {
            "model_version": b["version"],
            "raw_scores": {m: float(base_scores[m][i]) for m in base_scores},
            "ensemble_score": float(S[i].mean()),
            "calibrated_risk": float(calibrated[i]),
            "model_agreement": float(agreement[i]),
            "conformal_status": conf_sets[i],
            "verifier_confirms_risk": bool(verifier_flags[i]),
            "verifier_probability": float(verifier_probs[i]),
            "anomaly_percentile": float(anom_pct[i]),
            "ood_status": ood_status[i],
            "ood_detail": ood_detail[i],
            "scoring_model": family,
            "explanation_model": explain_family,
            "explanation_is_of_the_scoring_model": bool(explain_family == family),
            **pol,
        }
        if reasons is not None:
            rec["top_reasons"] = reasons[i]
        out.append(rec)
    return out


def counterfactual_for_row(row_matrix: np.ndarray, reason_rows: list[dict],
                           bundle: dict[str, Any] | None = None) -> list[dict]:
    b = bundle or load_bundle()
    thresholds = PolicyThresholds.from_dict(b["policy_thresholds"])
    booster = b["models"]["lightgbm"].booster_
    return counterfactual_sensitivity(
        booster, row_matrix, b["feature_list_kept"], reason_rows,
        b["cohort"], threshold=thresholds.standard_risk,
    )
