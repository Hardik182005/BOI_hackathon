"""Leakage firewall: per-feature target-relationship audit and quarantine.

Every feature gets:
- pairwise-complete point-biserial correlation with the target
- single-feature CROSS-VALIDATED PR-AUC (direction chosen on train folds only,
  evaluated on held-out folds - never same-data scoring)
- quantile-binned mutual information with the target (missing = own bin)
- perfect/near-perfect separation flag
- exact label-reconstruction check
- identifier-likeness check

These audit statistics are computed over the full file for SAFETY (they are
used only to EXCLUDE features). Predictive selection is done in-fold elsewhere.
"""
from __future__ import annotations

import datetime as dt
from typing import Any

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from muleguard import settings
from muleguard.logging import get_logger

log = get_logger("data.leakage")

# Columns that are never features, regardless of statistics.
ALWAYS_QUARANTINED_REASONS = {
    settings.TARGET_COLUMN: "target variable (F3924)",
    "F3912": "suspected target leak flagged in submission; re-verified by this audit",
}


def numeric_feature_matrix(df: pl.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Float32 matrix of all numeric columns except the target."""
    cols = [
        c for c in df.columns
        if c != settings.TARGET_COLUMN and df.schema[c].is_numeric()
    ]
    X = df.select([pl.col(c).cast(pl.Float32) for c in cols]).to_numpy()
    return X, cols


def pointbiserial_with_target(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Vectorised pairwise-complete Pearson r between each column and y."""
    mask = ~np.isnan(X)
    yb = y.astype(np.float64)[:, None]
    n = mask.sum(axis=0).astype(np.float64)
    Xz = np.where(mask, X, 0.0).astype(np.float64)
    sum_x = Xz.sum(axis=0)
    sum_x2 = (Xz * Xz).sum(axis=0)
    sum_y = (yb * mask).sum(axis=0)
    sum_y2 = sum_y  # y is 0/1 so y^2 == y
    sum_xy = (Xz * yb).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = sum_xy - sum_x * sum_y / n
        var_x = sum_x2 - sum_x**2 / n
        var_y = sum_y2 - sum_y**2 / n
        r = cov / np.sqrt(var_x * var_y)
    r[~np.isfinite(r)] = 0.0
    r[n < 10] = 0.0
    return r


def single_feature_cv_ap(
    X: np.ndarray, y: np.ndarray, n_folds: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Cross-validated single-feature PR-AUC and ROC-AUC per column.

    Per fold: impute missing with TRAIN median, choose score direction by
    TRAIN AP, evaluate on the VALIDATION fold. Returns fold-mean AP and AUC.
    """
    n_cols = X.shape[1]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    ap_sum = np.zeros(n_cols)
    auc_sum = np.zeros(n_cols)
    for tr_idx, va_idx in skf.split(X[:, :1], y):
        ytr, yva = y[tr_idx], y[va_idx]
        for j in range(n_cols):
            xtr = X[tr_idx, j].astype(np.float64)
            xva = X[va_idx, j].astype(np.float64)
            med = np.nanmedian(xtr)
            if np.isnan(med):
                med = 0.0
            xtr = np.where(np.isnan(xtr), med, xtr)
            xva = np.where(np.isnan(xva), med, xva)
            if np.all(xtr == xtr[0]):
                ap_sum[j] += ytr.mean()  # no-skill
                auc_sum[j] += 0.5
                continue
            # direction on TRAIN only
            ap_pos = average_precision_score(ytr, xtr)
            ap_neg = average_precision_score(ytr, -xtr)
            sign = 1.0 if ap_pos >= ap_neg else -1.0
            ap_sum[j] += average_precision_score(yva, sign * xva)
            try:
                auc_sum[j] += roc_auc_score(yva, sign * xva)
            except ValueError:
                auc_sum[j] += 0.5
    return ap_sum / n_folds, auc_sum / n_folds


def binned_mutual_information(X: np.ndarray, y: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Plug-in MI (bits) between quantile-binned feature (+missing bin) and y."""
    n_rows, n_cols = X.shape
    mi = np.zeros(n_cols)
    y1 = y == 1
    p_y1 = y1.mean()
    h_y = -(p_y1 * np.log2(p_y1) + (1 - p_y1) * np.log2(1 - p_y1)) if 0 < p_y1 < 1 else 0.0
    for j in range(n_cols):
        x = X[:, j].astype(np.float64)
        miss = np.isnan(x)
        codes = np.full(n_rows, 0, dtype=np.int32)  # bin 0 = missing
        vals = x[~miss]
        if len(vals) > 0 and not np.all(vals == vals[0]):
            edges = np.unique(np.quantile(vals, np.linspace(0, 1, n_bins + 1)[1:-1]))
            codes[~miss] = np.searchsorted(edges, vals, side="right") + 1
        elif len(vals) > 0:
            codes[~miss] = 1
        # contingency
        n_codes = codes.max() + 1
        joint = np.zeros((n_codes, 2))
        np.add.at(joint, (codes, y1.astype(int)), 1.0)
        joint /= n_rows
        px = joint.sum(axis=1, keepdims=True)
        py = joint.sum(axis=0, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            terms = joint * np.log2(joint / (px * py))
        mi[j] = np.nansum(terms)
    return np.clip(mi, 0.0, h_y if h_y > 0 else None)


def exact_label_reconstruction(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fraction of non-missing rows where the (binarised) feature equals y.

    Detects exact copies and affine recodings of the target among two-valued
    columns. Returns match rate per column (0 where not two-valued).
    """
    n_cols = X.shape[1]
    rate = np.zeros(n_cols)
    for j in range(n_cols):
        x = X[:, j].astype(np.float64)
        miss = np.isnan(x)
        vals = np.unique(x[~miss])
        if len(vals) != 2:
            continue
        lo, hi = vals
        for mapping in ((lo, hi), (hi, lo)):  # try both polarity mappings
            yhat = np.where(x == mapping[1], 1, 0)
            m = (~miss)
            r = (yhat[m] == y[m]).mean() if m.any() else 0.0
            rate[j] = max(rate[j], r)
    return rate


def run_leakage_audit(df: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, Any]]:
    cfg = settings.load_config("data")["audit"]
    y = df[settings.TARGET_COLUMN].cast(pl.Int32).to_numpy()
    if not np.isin(np.unique(y[~np.isnan(y.astype(float))]), [0, 1]).all():
        raise RuntimeError("target contains values outside {0,1}")

    X, cols = numeric_feature_matrix(df)
    log.info("leakage audit over %d numeric features", len(cols))

    corr = pointbiserial_with_target(X, y)
    cv_ap, cv_auc = single_feature_cv_ap(
        X, y, n_folds=int(cfg["single_feature_cv_folds"]), seed=settings.GLOBAL_SEED
    )
    mi = binned_mutual_information(X, y)
    recon = exact_label_reconstruction(X, y)

    n_rows = df.height
    n_unique = np.array([df[c].drop_nulls().n_unique() for c in cols])
    n_nonnull = np.array([n_rows - df[c].null_count() for c in cols])
    int_valued = np.zeros(len(cols), dtype=bool)
    for j, c in enumerate(cols):
        v = df[c].drop_nulls().cast(pl.Float64).to_numpy()
        int_valued[j] = len(v) > 0 and bool(np.all(v == np.round(v)))
    id_like = int_valued & (n_unique / np.maximum(n_nonnull, 1) >= float(cfg["identifier_unique_ratio"])) & (n_unique > 1000)

    audit = pl.DataFrame({
        "feature": cols,
        "target_corr": corr,
        "single_feature_cv_pr_auc": cv_ap,
        "single_feature_cv_roc_auc": cv_auc,
        "mutual_information_bits": mi,
        "label_reconstruction_rate": recon,
        "identifier_like": id_like,
    })

    sus_corr = float(cfg["suspicious_abs_corr"])
    sus_ap = float(cfg["suspicious_single_feature_ap"])
    audit = audit.with_columns([
        (pl.col("target_corr").abs() >= sus_corr).alias("flag_high_corr"),
        (pl.col("single_feature_cv_pr_auc") >= sus_ap).alias("flag_high_single_ap"),
        (pl.col("label_reconstruction_rate") >= 0.99).alias("flag_label_copy"),
        ((pl.col("single_feature_cv_roc_auc") >= 0.999) | (pl.col("single_feature_cv_roc_auc") <= 0.001)).alias("flag_perfect_separation"),
        pl.col("identifier_like").alias("flag_identifier"),
    ])
    audit = audit.with_columns(
        (pl.col("flag_high_corr") | pl.col("flag_high_single_ap") | pl.col("flag_label_copy")
         | pl.col("flag_perfect_separation") | pl.col("flag_identifier")).alias("suspicious")
    )

    summary = {
        "audited_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "n_numeric_features_audited": len(cols),
        "thresholds": {"abs_corr": sus_corr, "single_feature_cv_ap": sus_ap},
        "n_flagged": int(audit["suspicious"].sum()),
        "flagged": audit.filter(pl.col("suspicious")).sort(
            "single_feature_cv_pr_auc", descending=True
        ).to_dicts(),
    }
    return audit, summary


def build_quarantine(
    audit: pl.DataFrame, df: pl.DataFrame, index_candidates: list[str]
) -> dict[str, Any]:
    """Assemble the quarantine list with reasons and dispositions."""
    entries: list[dict[str, str]] = []
    for col, reason in ALWAYS_QUARANTINED_REASONS.items():
        if col in df.columns:
            entries.append({"feature": col, "reason": reason, "disposition": "EXCLUDED_FROM_ALL_TRAINING"})
    for col in index_candidates:
        entries.append({"feature": col, "reason": "row-index / identifier column", "disposition": "EXCLUDED_FROM_ALL_TRAINING"})
    for row in audit.filter(pl.col("suspicious")).iter_rows(named=True):
        if row["feature"] in {e["feature"] for e in entries}:
            continue
        reasons = []
        if row["flag_label_copy"]:
            reasons.append(f"label reconstruction rate {row['label_reconstruction_rate']:.3f}")
        if row["flag_high_corr"]:
            reasons.append(f"|target corr| {abs(row['target_corr']):.3f}")
        if row["flag_high_single_ap"]:
            reasons.append(f"single-feature CV PR-AUC {row['single_feature_cv_pr_auc']:.3f}")
        if row["flag_perfect_separation"]:
            reasons.append("near-perfect separation")
        if row["flag_identifier"]:
            reasons.append("identifier-like cardinality")
        entries.append({
            "feature": row["feature"],
            "reason": "; ".join(reasons),
            "disposition": "EXCLUDED_FROM_ALL_TRAINING",
        })
    return {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "policy": "quarantined features are excluded from every model, ensemble, "
                  "selector, calibration and explanation; ablation runs that "
                  "include them are labelled REJECTED LEAKAGE evidence only",
        "quarantine": entries,
    }
