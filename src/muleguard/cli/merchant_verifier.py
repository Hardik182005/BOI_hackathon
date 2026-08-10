"""Fit and evaluate the Merchant Legitimacy Verifier (addendum UPDATE 9).

    python -m muleguard.cli.merchant_verifier

Fits the business-evidence model on the development split, measures it
out-of-fold so its usefulness is a number rather than an assertion, and freezes
it next to the main bundle. The evaluation is the point: a verifier that cannot
separate anything would be a decorative confidence knob, and this CLI is what
proves it is not.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
from typing import Any

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import joblib  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from muleguard import settings  # noqa: E402
from muleguard.features.frame import build_model_frame  # noqa: E402
from muleguard.logging import get_logger  # noqa: E402
from muleguard.models import harness  # noqa: E402
from muleguard.models.merchant import (  # noqa: E402
    LEGITIMACY_BANDS, MerchantLegitimacyVerifier,
)
from muleguard.utils import save_json, set_global_seed  # noqa: E402

log = get_logger("cli.merchant_verifier")

VIEW = "E_profile_merchant"
MODEL_PATH = settings.MODELS_DIR / "merchant_verifier.joblib"
REPORT_JSON = settings.METRICS_DIR / "merchant_verifier_v2.json"

# The merchant view as configured also carries TOTAL_ALL_RAILS - aggregate
# debit/credit volume across every payment rail. That is the primary mule
# signal, not evidence of a legitimate business, and leaving it in would make
# this a second copy of the main model wearing a different label. The whole
# value of exculpatory evidence is that it is INDEPENDENT of the accusation, so
# the family is excluded here and the exclusion is recorded in the artifact.
EXCLUDED_FAMILIES = ("TOTAL_ALL_RAILS",)


def _oof_scores(X: np.ndarray, y: np.ndarray, dev, seed: int) -> np.ndarray:
    """Out-of-fold probabilities over the development split.

    Fitted per fold on the same saved folds every other model uses, so the
    reported PR-AUC is comparable with the tournament's numbers rather than an
    in-sample figure that would flatter the verifier.
    """
    from muleguard.features.preprocessing import FoldPreprocessor
    from muleguard.models.merchant import MerchantLegitimacyVerifier as V

    oof = np.full(len(y), np.nan)
    ids = dev.fold_ids[0]
    for k in np.unique(ids):
        tr, va = ids != k, ids == k
        prep = FoldPreprocessor(mode="tree")
        Xtr = prep.fit_transform(X[tr], [str(i) for i in range(X.shape[1])])
        Xva = prep.transform(X[va])
        v = V([str(i) for i in range(X.shape[1])]).fit(Xtr, y[tr], seed=seed + int(k))
        oof[va] = v.model.predict_proba(Xva)[:, 1]
    return oof


def _bands_from(p: np.ndarray) -> list[str]:
    """Band every row from its percentile within the supplied score vector.

    Used on the out-of-fold scores so the band prevalence below is measured on
    predictions made by models that never saw the row. Banding the refitted
    model's in-sample scores would report how well the verifier memorised the
    development split, and would have made the strong-evidence band look purer
    than it is.
    """
    ref = np.sort(p)
    legit = 1.0 - np.searchsorted(ref, p, side="right") / len(ref)
    return [next(b for cut, b in LEGITIMACY_BANDS if lg >= cut) for lg in legit]


def _top_drivers(verifier, names: list[str], k: int = 15) -> list[dict[str, Any]]:
    """The features the fitted verifier leans on, with their family names."""
    from muleguard.features import dictionary as fd
    from muleguard.features.frame import augmented_registry

    reg = augmented_registry()
    gain = verifier.model.booster_.feature_importance("gain")
    order = np.argsort(-gain)[:k]
    return [{"feature": names[i],
             "family": fd.describe(names[i], reg)["feature_family"],
             "direction": fd.describe(names[i], reg).get("direction"),
             "gain": round(float(gain[i]), 1)} for i in order]


def _business_features(mf) -> list[str]:
    """The merchant view minus the families that are really the mule signal."""
    from muleguard.features import dictionary as fd
    from muleguard.features.frame import augmented_registry

    reg = augmented_registry()
    return [f for f in mf.feature_names
            if fd.describe(f, reg)["feature_family"] not in EXCLUDED_FAMILIES]


def run(seed: int = 42) -> dict[str, Any]:
    set_global_seed(seed)
    full = build_model_frame(view=VIEW)
    keep = _business_features(full)
    mf = full.subset(keep)
    dev = harness.dev_split(1)
    X, y = mf.X[dev.row_index], mf.y[dev.row_index]
    log.info("merchant view: %d features (%d excluded as %s), %d dev rows, "
             "%d positives", X.shape[1], len(full.feature_names) - len(keep),
             ", ".join(EXCLUDED_FAMILIES), X.shape[0], int(y.sum()))

    oof = _oof_scores(X, y, dev, seed)
    ap = float(average_precision_score(y, oof))
    auc = float(roc_auc_score(y, oof))
    prevalence = float(y.mean())

    from muleguard.features.preprocessing import FoldPreprocessor

    prep = FoldPreprocessor(mode="tree")
    Xp = prep.fit_transform(X, mf.feature_names)
    verifier = MerchantLegitimacyVerifier(list(mf.feature_names)).fit(Xp, y, seed=seed)
    verifier.dev_pr_auc = round(ap, 5)

    # Bands measured out-of-fold, not from the refitted model's own training
    # scores. This is the number that decides whether the verifier is wired in
    # at all, so it has to be one the verifier could not have cheated on.
    oof_bands = _bands_from(oof)
    bands = {b: 0 for _, b in LEGITIMACY_BANDS}
    strong_pos = strong_n = 0
    for band, label in zip(oof_bands, y):
        bands[band] += 1
        if band == "STRONG_BUSINESS_EVIDENCE":
            strong_n += 1
            strong_pos += int(label)

    joblib.dump({"verifier": verifier, "preprocessor": prep,
                 "feature_names": list(mf.feature_names), "view": VIEW},
                MODEL_PATH)

    # What the verifier actually reads, named rather than asserted. An auditor
    # who suspects this is the mule model in disguise can check the families
    # here against the exclusion above.
    drivers = _top_drivers(verifier, list(mf.feature_names))

    # The honest test of the whole idea: are mules rarer among accounts the
    # verifier calls strong business evidence than in the book as a whole?
    strong_prevalence = strong_pos / max(strong_n, 1)
    useful = strong_prevalence < prevalence

    payload = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "view": VIEW,
        "n_features": X.shape[1],
        "n_features_in_view_before_exclusion": len(full.feature_names),
        "excluded_families": list(EXCLUDED_FAMILIES),
        "exclusion_rationale": (
            "TOTAL_ALL_RAILS carries aggregate debit/credit volume across every "
            "payment rail, which is the primary mule signal rather than evidence "
            "of a legitimate business. Exculpatory evidence is only worth "
            "anything if it is independent of the accusation, so the family is "
            "dropped from this model even though it would raise its PR-AUC."),
        "n_dev_rows": int(X.shape[0]),
        "n_positives": int(y.sum()),
        "prevalence": round(prevalence, 6),
        "oof_pr_auc": round(ap, 5),
        "oof_roc_auc": round(auc, 5),
        "lift_over_prevalence": round(ap / prevalence, 2) if prevalence else None,
        "band_counts": bands,
        "bands_measured_on": "out_of_fold_predictions",
        "top_evidence_drivers": drivers,
        "strong_business_evidence": {
            "n_accounts": strong_n,
            "n_mules_among_them": strong_pos,
            "mule_prevalence": round(strong_prevalence, 6),
            "book_prevalence": round(prevalence, 6),
            "relative_prevalence": round(strong_prevalence / prevalence, 4)
            if prevalence else None,
        },
        "verdict": (
            "USEFUL_AS_EXCULPATORY_CONTEXT" if useful else
            "NOT_EXCULPATORY_DO_NOT_USE_TO_LOWER_CONFIDENCE"),
        "interpretation": (
            "Mules are rarer among accounts with strong business evidence than "
            "in the book overall, so the band carries genuine exculpatory "
            "information." if useful else
            "Mules are NOT rarer among accounts with strong business evidence. "
            "The band must not be used to lower escalation confidence, because "
            "doing so would deprioritise exactly the accounts mule networks "
            "prefer. Wire the verifier as reporting-only until this changes."),
        "what_this_model_may_do": [
            "lower the confidence attached to an automatic escalation",
            "appear as exculpatory evidence in the ProofGraph defence panel",
        ],
        "what_this_model_may_never_do": [
            "modify the calibrated risk, the model score or any threshold",
            "remove an account from a review queue",
            "act as a merchant whitelist",
            "apply a fixed multiplier to any score",
        ],
        "replaces": (
            "the competitor pattern of multiplying a merchant's risk score by a "
            "hand-picked constant such as 0.70, which is unmeasurable and "
            "silently turns a calibrated probability into a non-probability"),
        "model_path": str(MODEL_PATH.name),
    }
    save_json(payload, REPORT_JSON)
    log.info("merchant verifier: OOF PR-AUC %.5f (prevalence %.5f), %s",
             ap, prevalence, payload["verdict"])
    log.info("strong-evidence band: %d accounts, %d mules (prevalence %.5f vs %.5f)",
             strong_n, strong_pos, strong_prevalence, prevalence)
    return payload


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Fit the Merchant Legitimacy Verifier")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args(argv)
    run(seed=a.seed)


if __name__ == "__main__":
    main()
