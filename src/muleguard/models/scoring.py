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
from muleguard.models.merchant import (
    DAMPENING_FLOOR, MODE_A, MerchantVerdict, apply_merchant_safeguard,
)

log = get_logger("models.scoring")

_BUNDLE_CACHE: dict[str, Any] | None = None


class BundleUnreadable(RuntimeError):
    """Raised when the model file exists but cannot be deserialised."""


def _read_bundle(p) -> dict[str, Any]:
    """Deserialise one bundle, turning a mangled file into a clear failure.

    A truncated or overwritten joblib surfaces from the pickle machinery as
    ``KeyError: 110`` or a bare ``ValueError`` about array bytes, which tells an
    operator nothing about which file is wrong. The model artifact is the one
    thing whose corruption must be unmistakable.
    """
    try:
        b = joblib.load(p)
    except FileNotFoundError:
        raise
    except Exception as exc:  # noqa: BLE001 - any deserialisation failure
        raise BundleUnreadable(
            f"model bundle at {p} could not be read ({type(exc).__name__}: {exc}); "
            "the file is corrupt or was written by an incompatible version"
        ) from exc
    if not isinstance(b, dict) or "feature_list_kept" not in b:
        raise BundleUnreadable(
            f"model bundle at {p} loaded but is not a MuleGuard bundle")
    return b


def load_bundle(path=None) -> dict[str, Any]:
    """The frozen bundle, cached for the life of the process.

    An explicit ``path`` is a diagnostic call - reading some other bundle to
    compare it - and deliberately does not replace the cached production model.
    """
    global _BUNDLE_CACHE
    if path is not None:
        return _read_bundle(path)
    if _BUNDLE_CACHE is None:
        p = settings.MODELS_DIR / "final_bundle.joblib"
        _BUNDLE_CACHE = _read_bundle(p)
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


RISK_REFERENCE_PATH = settings.REGISTRY_DIR / "risk_percentile_reference.json"

_RISK_REFERENCE: dict[str, Any] | None | bool = False  # False = not yet looked for


def _risk_reference() -> dict[str, Any] | None:
    """The frozen dev calibrated-risk grid used to rank a new score.

    Absent reference means the field is reported as None. A percentile with no
    stated reference distribution is a number that looks meaningful and is not.
    """
    global _RISK_REFERENCE
    if _RISK_REFERENCE is False:
        try:
            with open(RISK_REFERENCE_PATH, "r", encoding="utf-8") as fh:
                import json

                ref = json.load(fh)
            _RISK_REFERENCE = ref if ref.get("sorted_scores") else None
        except (FileNotFoundError, ValueError):
            _RISK_REFERENCE = None
    return _RISK_REFERENCE or None


def _risk_percentiles(calibrated: np.ndarray) -> np.ndarray | None:
    ref = _risk_reference()
    if ref is None:
        return None
    grid = np.asarray(ref["sorted_scores"], dtype=float)
    if grid.size == 0:
        return None
    # Fraction of the dev distribution at or below this score, as 0-100.
    return np.searchsorted(grid, calibrated, side="right") / grid.size * 100.0


MERCHANT_VERIFIER_PATH = settings.MODELS_DIR / "merchant_verifier.joblib"

_MERCHANT: dict[str, Any] | None | bool = False  # False = not yet looked for


def _merchant_artifact() -> dict[str, Any] | None:
    """The frozen Merchant Legitimacy Verifier, loaded once.

    Absent artifact means no row carries business evidence, and the safeguard
    record says so. Standing in a band derived from the champion's own features
    would make the exculpatory evidence a restatement of the accusation, which
    is the one thing the verifier exists not to be.
    """
    global _MERCHANT
    if _MERCHANT is False:
        try:
            _MERCHANT = joblib.load(MERCHANT_VERIFIER_PATH)
        except (FileNotFoundError, OSError) as exc:
            log.info("merchant verifier not available (%s); scored rows will "
                     "record the safeguard with no business evidence", exc)
            _MERCHANT = None
    return _MERCHANT or None


def _merchant_verdicts(rows: pl.DataFrame, n: int) -> list[MerchantVerdict | None]:
    """One business-evidence reading per row, or None where none can be read.

    The verifier reads the profile/merchant view - over a thousand columns the
    champion does not select - so a request carrying only the selected features
    contains no business evidence at all. That is reported as absent rather
    than as weak: "no evidence was supplied" and "the evidence is
    unconvincing" are different statements, and only the first is true here.
    """
    art = _merchant_artifact()
    if art is None:
        return [None] * n
    names = art["feature_names"]
    if any(c not in rows.columns for c in names):
        return [None] * n
    from muleguard.features.preprocessing import encode_dataframe

    try:
        # The merchant view is numeric throughout the canonical dataset - it
        # holds no categorical or date column - so the numeric branch of the
        # request canonicaliser is the whole schema for it.
        canon = _canonicalize_request(rows, names, {})
        X, _, _ = encode_dataframe(canon, names)
        return art["verifier"].verdicts(art["preprocessor"].transform(X))
    except Exception as exc:  # noqa: BLE001 - context must never take down a score
        log.warning("merchant verdicts unavailable: %s", exc)
        return [None] * n


def _merchant_safeguard_policy() -> dict[str, Any]:
    """The safeguard knobs, from configs/thresholds.yaml.

    Read from configuration rather than taken as an argument because which mode
    ran is a policy fact about the bank, not about the code path that happened
    to score the row.
    """
    cfg = settings.load_config("thresholds").get("merchant_safeguard") or {}
    return {
        "mode": str(cfg.get("mode", MODE_A)),
        "dampening_factor": float(cfg.get("dampening_factor", 0.85)),
        "floor": float(cfg.get("floor", DAMPENING_FLOOR)),
    }


def score_rows(rows: pl.DataFrame, bundle: dict[str, Any] | None = None,
               with_explanations: bool = True,
               with_counterfactual: bool = True) -> list[dict[str, Any]]:
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
    # Committee spread. `S` holds one column per base model, so the row-wise
    # mean and standard deviation say how tightly the committee agrees on this
    # account. `model_agreement` compresses the same thing to one number;
    # `prediction_std` is the raw dispersion an analyst can reason about.
    pred_mean = S.mean(axis=1)
    pred_std = S.std(axis=1)
    # Rank against the frozen dev calibrated distribution. The reference is
    # built by muleguard.cli.build_risk_reference from DEV rows only; when it
    # has not been built the field is None rather than a guessed rank.
    risk_pct = _risk_percentiles(calibrated)
    # Explain the model that produced `raw`, not a convenient stand-in. When an
    # ensemble wins there is no single tree to attribute the stacked score to,
    # so we explain the strongest base model and say so in the record rather
    # than presenting its reasons as the ensemble's.
    explain_family = family if family in b["models"] else "lightgbm"
    # Business evidence, read from the merchant view alone. The safeguard runs
    # on every row - including the rows with no verdict - so the record always
    # states which mode was in force rather than leaving a reader to infer from
    # a missing key that nothing was considered.
    merchant_policy = _merchant_safeguard_policy()
    merchant = _merchant_verdicts(rows, len(Xp))
    reasons = None
    if with_explanations:
        reasons = shap_reason_codes(
            b["models"][explain_family], Xp, b["feature_list_kept"], b["cohort"],
            family=explain_family,
        )

    def route(i: int, risk: float) -> dict[str, Any]:
        return decide(
            calibrated_risk=risk,
            conformal_set=conf_sets[i],
            ood_status=ood_status[i],
            anomaly_percentile=float(anom_pct[i]),
            model_agreement=float(agreement[i]),
            verifier_flag=bool(verifier_flags[i]),
            thresholds=thresholds,
        )

    out = []
    for i in range(len(Xp)):
        safeguard = apply_merchant_safeguard(
            calibrated_risk=float(calibrated[i]),
            merchant=merchant[i],
            mode=merchant_policy["mode"],
            policy_version=thresholds.policy_version,
            dampening_factor=merchant_policy["dampening_factor"],
            floor=merchant_policy["floor"],
        )
        # Only a safeguard that fired may move a tier. The record's
        # after_policy_score is rounded to six places for the audit log, and
        # routing MODE A on that echo instead of on the score itself could flip
        # a row sitting within 1e-6 of a threshold - a dampening nobody asked
        # for, produced by a rounding.
        policy_risk = (float(safeguard["after_policy_score"])
                       if safeguard["merchant_safeguard_applied"]
                       else float(calibrated[i]))
        pol = route(i, policy_risk)
        if safeguard["merchant_safeguard_applied"]:
            measured = route(i, float(calibrated[i]))
            if pol["risk_tier"] == "MONITOR" and measured["risk_tier"] != "MONITOR":
                # The safeguard's own guarantee - printed on this very record -
                # is that no mode removes an account from a review queue. A
                # dampening that would end this account's oversight is capped
                # here: MODE B may lower a case's priority, never its right to
                # be looked at. The dampened score stays in the record, so the
                # cap is as auditable as the dampening was.
                pol = measured
                pol["reasons"].append(
                    f"{safeguard['reason']}. The dampened score fell below every "
                    "review band, so the tier the measured score earned is kept")
            else:
                # The tier was decided on the dampened score, and every reason
                # string quotes that score. Without this line the routing reads
                # as though the measured risk produced the tier.
                pol["reasons"].append(safeguard["reason"])
        rec: dict[str, Any] = {
            "model_version": b["version"],
            "raw_scores": {m: float(base_scores[m][i]) for m in base_scores},
            "ensemble_score": float(S[i].mean()),
            "calibrated_risk": float(calibrated[i]),
            "risk_percentile": (None if risk_pct is None
                                else float(risk_pct[i])),
            "model_agreement": float(agreement[i]),
            "prediction_mean": float(pred_mean[i]),
            "prediction_std": float(pred_std[i]),
            "conformal_status": conf_sets[i],
            "verifier_confirms_risk": bool(verifier_flags[i]),
            "verifier_probability": float(verifier_probs[i]),
            "anomaly_percentile": float(anom_pct[i]),
            "ood_status": ood_status[i],
            "ood_detail": ood_detail[i],
            "scoring_model": family,
            "explanation_model": explain_family,
            "explanation_is_of_the_scoring_model": bool(explain_family == family),
            "merchant_safeguard": safeguard,
            **pol,
        }
        if reasons is not None:
            rec["top_reasons"] = reasons[i]
            # Counterfactual twin: the smallest set of behavioural changes that
            # moves this row across the decision boundary. Attached only where
            # reasons exist, since it is computed from them, and only for rows
            # the policy actually flags - there is nothing to counterfactual
            # about an account that was not flagged.
            if with_counterfactual and pol.get("decision") != "NO_ACTION":
                try:
                    rec["counterfactual_twin"] = counterfactual_sensitivity(
                        b["models"][explain_family].booster_
                        if explain_family == "lightgbm"
                        else b["models"][explain_family],
                        Xp[i], b["feature_list_kept"], reasons[i], b["cohort"],
                        threshold=thresholds.standard_risk, family=explain_family,
                    )
                except Exception as exc:  # noqa: BLE001
                    # An explanation extra must never take down a score.
                    log.warning("counterfactual twin unavailable: %s", exc)
                    rec["counterfactual_twin"] = []
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
