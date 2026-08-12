"""Build the Missed-Mule Error Atlas from the development out-of-fold run.

Usage::

    python -m muleguard.cli.error_atlas
    python -m muleguard.cli.error_atlas --budget 100
    python -m muleguard.cli.error_atlas --no-shap        # skip the fold refits

For every labelled mule the served champion did NOT place inside the analyst
budget, this measures why - closest mule-labelled account and closest
legitimate account in feature space, all four model families' scores,
attributions, missingness, merchant/context evidence, anomaly percentile and
neighbour distance - and assigns one of six categories with the ordered rule
list in :mod:`muleguard.models.error_atlas`.

Read-only with respect to the ML core. It loads the frozen bundle, the stored
out-of-fold predictions and the feature matrix; it writes exactly two files,
``artifacts/metrics/error_atlas.json`` and ``docs/ERROR_ATLAS.md``, and hashes
its inputs before and after to prove it changed none of them. No score, no
threshold, no prediction, no feature set and no model is touched. Nothing here
consults the locked test split.

No language model is involved at any point. Every field in the artifact is a
measured quantity, a value copied from a configuration file, or a fixed string
written by hand in this repository.
"""
from __future__ import annotations

import argparse
import os
import time
from typing import Any

# Thread hygiene BEFORE the ML imports. This is not boilerplate: XGBoost's
# `hist` tree builder sums gradients in thread-order, so a different thread
# count reproduces the stored out-of-fold scores only to ~0.16 absolute. The
# tournament that produced oof_v2.parquet ran at 4, so the refits used here for
# attributions must too, otherwise the attributions would explain a model that
# never scored these accounts. Verified per fold at run time and recorded.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "4")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from muleguard import settings  # noqa: E402
from muleguard.logging import get_logger  # noqa: E402
from muleguard.models import error_atlas as atlas  # noqa: E402
from muleguard.models import harness  # noqa: E402
from muleguard.utils import (git_info, load_json, save_json,  # noqa: E402
                             set_global_seed, sha256_file)

log = get_logger("cli.error_atlas")

OUT_JSON = settings.METRICS_DIR / "error_atlas.json"
OUT_DOC = settings.DOCS_DIR / "ERROR_ATLAS.md"

#: Families the specification requires a score for, mapped to the model name in
#: oof_v2.parquet. Each family is taken at the width it was actually run at.
FAMILY_MODELS = {
    "catboost": "catboost_top_120",
    "lightgbm": "lightgbm_top_120",
    "xgboost": "xgboost_top_120",
    "tabpfn": "tabpfn_top_60",
}

READ_ONLY_INPUTS = {
    "oof_v2.parquet": settings.PREDICTIONS_DIR / "oof_v2.parquet",
    "final_bundle.joblib": settings.MODELS_DIR / "final_bundle.joblib",
    "merchant_verifier.joblib": settings.MODELS_DIR / "merchant_verifier.joblib",
    "selected_features_v2.json": settings.FEATURES_DIR / "selected_features_v2.json",
    "promotion_decision_v2.json": settings.METRICS_DIR / "promotion_decision_v2.json",
    "thresholds.yaml": settings.REPO_ROOT / "configs" / "thresholds.yaml",
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _digests() -> dict[str, str]:
    return {k: sha256_file(p) for k, p in READ_ONLY_INPUTS.items() if p.exists()}


def _champion() -> str:
    """The served champion, read from the promotion decision, never hardcoded."""
    return load_json(settings.METRICS_DIR / "promotion_decision_v2.json")["promoted"]


def _budget() -> tuple[int, str]:
    """The analyst budget, read from the deployed policy configuration."""
    th = settings.load_config("thresholds")
    k = int(th["tiers"]["urgent_review"]["daily_alert_capacity"])
    return k, "configs/thresholds.yaml tiers.urgent_review.daily_alert_capacity"


def _repeat_scores(oof: pl.DataFrame, model: str) -> np.ndarray:
    """(n_repeats, n_dev_rows) matrix of stored out-of-fold scores."""
    df = oof.filter(pl.col("model") == model).sort(["repeat", "row_index"])
    reps = sorted(df["repeat"].unique().to_list())
    return np.vstack([df.filter(pl.col("repeat") == r)["score"].to_numpy()
                      for r in reps])


def _conventions(S: np.ndarray, y: np.ndarray, k: int) -> dict[str, Any]:
    """The three ways a repeated CV run can be collapsed to one ranking.

    They disagree, so all three are reported. The Atlas classifies on
    score_average because that is the convention build_lenses_v2 uses to derive
    the thresholds the system actually serves - i.e. it is the deployed
    operating point, not the flattering one.
    """
    pos = np.where(y == 1)[0]
    out: dict[str, Any] = {}

    r_sc = atlas.dense_rank_desc(S.mean(axis=0))
    out["score_average"] = r_sc

    ranks = np.vstack([atlas.dense_rank_desc(S[i]) for i in range(S.shape[0])])
    out["rank_average"] = atlas.dense_rank_desc(-ranks.mean(axis=0))

    per_repeat_recall = float(np.mean([(ranks[i][pos] <= k).mean()
                                       for i in range(ranks.shape[0])]))
    summary = {
        "score_average": {
            "recall_at_budget": round(float((r_sc[pos] <= k).mean()), 6),
            "n_misses": int((r_sc[pos] > k).sum()),
        },
        "rank_average": {
            "recall_at_budget": round(float((out["rank_average"][pos] <= k).mean()), 6),
            "n_misses": int((out["rank_average"][pos] > k).sum()),
        },
        "per_repeat_mean": {
            "recall_at_budget": round(per_repeat_recall, 6),
            "n_misses_per_repeat": [int((ranks[i][pos] > k).sum())
                                    for i in range(ranks.shape[0])],
        },
    }
    return {"ranks": out, "per_repeat_ranks": ranks, "summary": summary}


# ---------------------------------------------------------------------------
# Attributions
# ---------------------------------------------------------------------------


def _fold_attributions(Xdev: np.ndarray, ydev: np.ndarray, names: list[str],
                       dev, target_rows: np.ndarray, family: str,
                       stored: np.ndarray) -> dict[str, Any]:
    """Exact TreeSHAP for the target rows, from protocol-identical fold refits.

    The tournament stored scores but not the per-fold models, so the model that
    produced a given out-of-fold score has to be rebuilt. It is rebuilt with the
    same preprocessor, the same fold assignment and the same deterministic fold
    seed, and then the refit's scores are compared against the stored ones over
    the whole validation fold. That check is reported per fold: if a fold does
    not reproduce, its attributions describe a protocol-identical twin rather
    than the exact scoring model, and the artifact says so instead of quietly
    presenting them as the same thing.
    """
    from muleguard.explain.reason_codes import tree_shap
    from muleguard.features.preprocessing import FoldPreprocessor
    from muleguard.models import core_models

    # The split object can offer more repeats than the tournament actually ran;
    # the stored score matrix is the authority on how many exist.
    n_rep = int(stored.shape[0])
    if len(dev.fold_ids) < n_rep:
        raise ValueError("development split offers %d repeats but oof_v2.parquet "
                         "stores %d" % (len(dev.fold_ids), n_rep))
    contrib = {int(r): np.zeros((n_rep, len(names)), dtype=float) for r in target_rows}
    seen = {int(r): np.zeros(n_rep, dtype=bool) for r in target_rows}
    checks: list[dict[str, Any]] = []
    name_pos = {n: j for j, n in enumerate(names)}

    for rep in range(n_rep):
        ids = dev.fold_ids[rep]
        for fold in sorted(int(v) for v in np.unique(ids)):
            va = ids == fold
            wanted = [int(r) for r in target_rows if va[r]]
            if not wanted:
                continue
            tr = ~va
            prep = FoldPreprocessor(mode="tree")
            Xtr = prep.fit_transform(Xdev[tr], names)
            Xva = prep.transform(Xdev[va])
            t0 = time.perf_counter()
            s, model = core_models.SCORERS[family](
                Xtr, ydev[tr], Xva, harness.fold_seed(rep, fold), return_model=True)
            diff = float(np.abs(s - stored[rep][va]).max())
            checks.append({
                "repeat": rep, "fold": fold,
                "n_validation_rows": int(va.sum()),
                "n_target_rows_in_fold": len(wanted),
                "max_abs_score_difference_vs_stored_oof": diff,
                "reproduces_stored_oof": bool(diff == 0.0),
                "n_features_kept_by_fold_preprocessor": int(len(prep.kept_features)),
                "seconds": round(time.perf_counter() - t0, 2),
            })
            local = np.where(va)[0]
            sel = [int(np.where(local == r)[0][0]) for r in wanted]
            c, _ = tree_shap(model, family, Xva[sel])
            for t, r in enumerate(wanted):
                for j, fname in enumerate(prep.kept_features):
                    contrib[r][rep, name_pos[fname]] = c[t, j]
                seen[r][rep] = True

    return {"contributions": contrib, "seen": seen, "fold_checks": checks}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(budget: int | None = None, with_shap: bool = True,
        seed: int = settings.GLOBAL_SEED) -> dict[str, Any]:
    t_start = time.perf_counter()
    set_global_seed(seed)
    digests_before = _digests()

    from muleguard.features import dictionary as fd
    from muleguard.features.frame import augmented_registry, build_model_frame

    champion = _champion()
    family = champion.split("_top_")[0]
    default_k, k_source = _budget()
    k = int(budget) if budget else default_k
    if budget:
        k_source = f"--budget command line override (default was {default_k} from {k_source})"

    bundle = joblib.load(settings.MODELS_DIR / "final_bundle.joblib")
    names = list(bundle["feature_list_selected"])
    qman = load_json(settings.FEATURES_DIR / "quarantined_features.json")
    quarantined = {e["feature"] for e in qman["quarantine"]}
    if quarantined != atlas.QUARANTINED_FEATURES:
        raise ValueError(
            "the quarantine manifest and the Atlas's frozen copy disagree; "
            "resolve the difference rather than trusting either: manifest-only "
            f"{sorted(quarantined - atlas.QUARANTINED_FEATURES)}, atlas-only "
            f"{sorted(atlas.QUARANTINED_FEATURES - quarantined)}")
    overlap = sorted(set(names) & quarantined)
    if overlap:
        raise ValueError(f"champion input set names quarantined columns: {overlap}")

    mf = build_model_frame(view=None).subset(names)
    dev = harness.dev_split()
    Xdev, ydev = mf.X[dev.row_index], mf.y[dev.row_index]
    Xp = bundle["preprocessor"].transform(Xdev)
    n_dev = len(ydev)

    oof = pl.read_parquet(settings.PREDICTIONS_DIR / "oof_v2.parquet")
    stored_rows = np.sort(oof["row_index"].unique().to_numpy())
    if not np.array_equal(stored_rows, np.sort(dev.row_index)):
        raise ValueError("oof_v2.parquet row_index set does not match the "
                         "development split; alignment cannot be assumed")

    S = {f: _repeat_scores(oof, m) for f, m in FAMILY_MODELS.items()
         if m in oof["model"].unique().to_list()}
    conv = _conventions(S[family], ydev, k)
    rank = conv["ranks"]["score_average"]
    fam_rank = {f: atlas.dense_rank_desc(S[f].mean(axis=0)) for f in S}
    fam_repeat_rank = {f: np.vstack([atlas.dense_rank_desc(S[f][i])
                                     for i in range(S[f].shape[0])]) for f in S}

    pos = np.where(ydev == 1)[0]
    missed = pos[rank[pos] > k]
    caught = pos[rank[pos] <= k]
    log.info("champion=%s budget=%d -> %d misses, %d caught, of %d labelled "
             "mules in %d development rows", champion, k, len(missed),
             len(caught), len(pos), n_dev)

    # --- lenses (reused, not reinvented) ---------------------------------
    ood = bundle["ood"]
    med, iqr = ood.stats_["med"], ood.stats_["iqr"]
    Z = atlas.robust_scale(Xp, med, iqr)
    ood_status_all, ood_detail_all = ood.status(Xp)
    anomaly_all = bundle["anomaly"].anomaly_percentile(Xp)
    hard_neg_flags, hard_neg_probs = bundle["verifier"].confirms_risk(Xp)
    knn_all = np.array([d["knn_distance"] for d in ood_detail_all])
    rvs_all = np.array([d["range_violation_share"] for d in ood_detail_all])
    miss_frac_all = np.isnan(Xp).mean(axis=1)
    lens_obs = _lens_observations(ood, ood_status_all, knn_all, rvs_all, ydev)

    geom_rows = np.concatenate([missed, caught])
    geom = {int(r): g for r, g in zip(
        geom_rows, atlas.neighbour_geometry(Z, ydev, geom_rows,
                                            atlas.DEFAULT_CONSTANTS.neighbourhood_k))}

    # --- merchant / context evidence -------------------------------------
    merchant = _merchant_evidence(missed)

    # --- feature families for the missingness profile --------------------
    reg = augmented_registry()
    fam_of = {n: fd.describe(n, reg)["feature_family"] for n in names}

    # --- attributions -----------------------------------------------------
    shap_block: dict[str, Any] = {
        "method": "NOT_COMPUTED",
        "reason": "disabled with --no-shap",
    }
    attr = None
    if with_shap:
        attr = _fold_attributions(Xdev, ydev, names, dev, missed, family, S[family])
        ok = all(c["reproduces_stored_oof"] for c in attr["fold_checks"])
        shap_block = {
            "method": "EXACT_TREESHAP",
            "implementation": "muleguard.explain.reason_codes.tree_shap "
                              "(shap.TreeExplainer, interventional off, exact)",
            "not_a_fallback": "Model-native feature importances were NOT used. "
                              "Per-row exact TreeSHAP was affordable at this "
                              "scale, so no substitution was made.",
            "computed_on": "the out-of-fold model for each repeat, rebuilt from "
                           "the same folds, preprocessor and deterministic fold "
                           "seed the tournament used",
            "all_folds_reproduce_stored_oof_exactly": ok,
            "attribution_fidelity": (
                "EXACT - every refit reproduced its stored out-of-fold scores to "
                "0.0 absolute, so these attributions come from the same model "
                "that produced the score being explained" if ok else
                "PROTOCOL_IDENTICAL_TWIN - at least one refit did not reproduce "
                "its stored out-of-fold scores exactly; see fold_checks. The "
                "attributions describe a model built by the identical protocol, "
                "not necessarily the exact scoring model."),
            "dropped_column_convention": (
                "The per-fold preprocessor removes constant and exactly "
                "duplicated columns, so a column absent from a fold's model gets "
                "contribution 0.0 for that repeat. A duplicated column's "
                "contribution is carried by its surviving twin."),
            "fold_checks": attr["fold_checks"],
        }

    # --- measurements ------------------------------------------------------
    def measure(r: int) -> atlas.MissMeasurements:
        det = ood_detail_all[int(r)]
        g = geom[int(r)]
        peers = sum(1 for f in S if f != family and fam_rank[f][r] <= k)
        below = sum(1 for f in S if fam_rank[f][r] > n_dev / 2)
        return atlas.MissMeasurements(
            champion_rank=int(rank[r]),
            budget_k=k,
            missing_fraction=float(miss_frac_all[r]),
            missingness_z=float((miss_frac_all[r] - ood.stats_["miss_mu"])
                                / ood.stats_["miss_sd"]),
            range_violation_share=float(det["range_violation_share"]),
            knn_distance=float(det["knn_distance"]),
            knn_threshold=float(det["knn_threshold"]),
            n_peer_families_within_budget=peers,
            n_mule_labels_in_neighbourhood=int(g["n_mule_labels_in_neighbourhood"]),
            distance_to_nearest_known_mule=float(g["distance_to_nearest_known_mule"]),
            distance_to_nearest_legitimate_account=float(
                g["distance_to_nearest_legitimate_account"]),
            n_families_below_dev_median=below,
            n_families_measured=len(S),
        )

    records: list[dict[str, Any]] = []
    counts = {c: 0 for c in atlas.CATEGORIES}
    for r in missed:
        r = int(r)
        m = measure(r)
        verdict = atlas.classify(m)
        counts[verdict["category"]] += 1
        g = geom[r]
        records.append({
            "dev_row_position": r,
            "dataset_row_index": int(dev.row_index[r]),
            "label": "MULE_LABELLED_IN_DEVELOPMENT_DATA",
            "why_this_is_a_miss": {
                "champion": champion,
                "champion_rank": int(rank[r]),
                "analyst_budget_k": k,
                "aggregation": "score_average over repeats (the served convention)",
                "n_repeats_outside_budget": int(
                    (fam_repeat_rank[family][:, r] > k).sum()),
                "n_repeats": int(S[family].shape[0]),
                "per_repeat_rank": [int(v) for v in fam_repeat_rank[family][:, r]],
            },
            "category": verdict["category"],
            "classification": verdict,
            "measurements": m.to_dict(),
            "family_scores": {
                f: {
                    "model": FAMILY_MODELS[f],
                    "score_repeat_average": round(float(S[f][:, r].mean()), 6),
                    "per_repeat_score": [round(float(v), 6) for v in S[f][:, r]],
                    "rank": int(fam_rank[f][r]),
                    "within_budget": bool(fam_rank[f][r] <= k),
                    "score_percentile_in_development": round(
                        atlas.percentile_of(S[f][:, r].mean(), S[f].mean(axis=0)), 3),
                } for f in sorted(S)
            },
            "closest_known_mule": {
                "relation": atlas.NEIGHBOUR_RELATION,
                "not_a_transaction_link": True,
                "meaning": atlas.NEIGHBOUR_DISCLAIMER,
                "dataset_row_index": int(dev.row_index[g["nearest_mule_labelled_row"]]),
                "dev_row_position": int(g["nearest_mule_labelled_row"]),
                "distance": round(float(g["distance_to_nearest_known_mule"]), 6),
                "that_account_champion_rank": int(rank[g["nearest_mule_labelled_row"]]),
                "that_account_was_surfaced_within_budget": bool(
                    rank[g["nearest_mule_labelled_row"]] <= k),
            },
            "closest_legitimate_account": {
                "relation": atlas.NEIGHBOUR_RELATION,
                "not_a_transaction_link": True,
                "meaning": atlas.NEIGHBOUR_DISCLAIMER,
                "dataset_row_index": int(dev.row_index[g["nearest_legitimate_row"]]),
                "dev_row_position": int(g["nearest_legitimate_row"]),
                "distance": round(float(g["distance_to_nearest_legitimate_account"]), 6),
                "closer_than_nearest_mule_labelled_account": bool(
                    g["distance_to_nearest_legitimate_account"]
                    < g["distance_to_nearest_known_mule"]),
            },
            "nearest_neighbour_distance": {
                "space": "champion input space (120 columns), median-imputed then "
                         "robust-scaled by the deployed OOD lens's medians and IQRs",
                "k": int(g["neighbourhood_k"]),
                "mean_distance_to_k_nearest": round(float(g["mean_neighbour_distance"]), 6),
                "n_mule_labels_among_k_nearest": int(g["n_mule_labels_in_neighbourhood"]),
                "ood_lens_knn_distance": round(float(knn_all[r]), 6),
                "ood_lens_knn_gate": round(float(ood_detail_all[r]["knn_threshold"]), 6),
                "ood_lens_knn_distance_percentile_in_development": round(
                    atlas.percentile_of(knn_all[r], knn_all), 3),
            },
            "missingness_pattern": atlas.missingness_profile(
                Xp[r], names, fam_of, ood.stats_["miss_mu"], ood.stats_["miss_sd"]),
            "merchant_context_evidence": merchant.get(r, {
                "status": "NOT_COMPUTED",
                "reason": "merchant verifier artifact unavailable"}),
            "anomaly_score": {
                "anomaly_percentile": round(float(anomaly_all[r]), 4),
                "detector": "IsolationForest challenger from the served bundle",
                "fitted_on": "development rows NOT carrying a mule label, so this "
                             "account was not part of the detector's fit",
                "reference_distribution": "all development rows",
                "interpretation": "100 = most anomalous relative to the "
                                  "development cohort. This is an unsupervised "
                                  "lens; it is not a risk score.",
            },
            "hard_negative_verifier": {
                "confirms_risk": bool(hard_neg_flags[r]),
                "probability": round(float(hard_neg_probs[r]), 6),
                "caveat": "fitted on all development rows including this one; "
                          "this reading is in-sample and is reported as context, "
                          "not as independent corroboration",
            },
            "top_shap_features": (
                atlas.top_attributions(attr["contributions"][r], names, Xp[r])
                if attr is not None and attr["seen"][r].all()
                else [{"status": "NOT_COMPUTED",
                       "reason": shap_block.get("reason",
                                                "no fold produced this row")}]),
        })

    control = [measure(int(r)) for r in caught]
    base_rates = atlas.rule_firing_rates(control)
    base_rates["control_group"] = (
        "the %d labelled mules the champion DID surface within the budget"
        % len(caught))
    base_rates["why"] = ("A rule that fires as often on caught mules as on "
                         "missed ones explains nothing. These rates are the "
                         "discriminating power of each rule, measured rather "
                         "than assumed.")

    payload = _payload(champion, family, k, k_source, n_dev, pos, missed, caught,
                       conv, S, fam_rank, records, counts, base_rates,
                       shap_block, lens_obs, quarantined, dev.row_index,
                       digests_before, time.perf_counter() - t_start)

    atlas.assert_no_quarantined_feature(payload, context="error_atlas.json")
    atlas.assert_read_only_contract(digests_before, _digests())
    save_json(payload, OUT_JSON)
    OUT_DOC.write_text(_markdown(payload), encoding="utf-8")
    log.info("wrote %s and %s", OUT_JSON, OUT_DOC)
    return payload


def _lens_observations(ood, statuses, knn, rvs, ydev) -> dict[str, Any]:
    """What the served OOD lens does on development data, measured.

    Written after a contradiction: the gate value and the maximum development
    distance were read as showing an inert gate, and the two numbers did not
    agree. Measuring instead of asserting shows the gate does fire - it is the
    q99.9 of the development distances, so a handful of rows sit above it by
    construction - and that the reason no miss is OOD is simply that no missed
    account was flagged.
    """
    st = np.asarray(statuses)
    flagged = np.where(st != "IN_DISTRIBUTION")[0]
    thr = float(ood.stats_["knn_thr"])
    return {
        "why_this_section_exists": (
            "The OOD category delegates to the served lens rather than inventing "
            "a second definition of out-of-distribution, so what that lens can "
            "and cannot do on this data determines what the category can mean."),
        "knn_gate": {
            "gate_value": round(thr, 4),
            "gate_definition": "quantile %s of the development k-NN distances, "
                               "k=%d (configs/thresholds.yaml ood.knn_quantile)"
                               % (settings.load_config("thresholds")["ood"]["knn_quantile"],
                                  getattr(ood, "k", atlas.DEFAULT_CONSTANTS.neighbourhood_k)),
            "max_development_knn_distance": round(float(knn.max()), 4),
            "median_development_knn_distance": round(float(np.median(knn)), 4),
            "n_development_rows_above_gate": int((knn > thr).sum()),
            "gate_fires_on_development_data": bool((knn > thr).any()),
        },
        "range_violation_limb": {
            "max_development_range_violation_share": round(float(rvs.max()), 6),
            "n_development_rows_above_threshold": int(
                (rvs > atlas.DEFAULT_CONSTANTS.range_violation_share).sum()),
            "observation": (
                "Zero, and necessarily so: the lens's accepted ranges were "
                "widened from these same development rows, so no development row "
                "can violate them. This limb is a production-drift detector and "
                "carries no information in this in-sample analysis. Reported so "
                "that a zero count is not mistaken for evidence."),
        },
        "development_rows_flagged": {
            "n_flagged": int(len(flagged)),
            "n_flagged_carrying_a_mule_label": int(ydev[flagged].sum()),
            "observation": (
                "The lens flags %d development rows and none of them carries a "
                "mule label. So the OOD_PATTERN count of zero among the misses "
                "is not an artefact of an incapable gate - the gate does fire, "
                "just never on these accounts."
                % len(flagged)),
        },
        "the_atlas_did_not_change_any_of_this": (
            "Adjusting a served threshold is outside what this instrument is "
            "permitted to do. These are observations for whoever owns the lens."),
    }


def _merchant_evidence(rows: np.ndarray) -> dict[int, dict[str, Any]]:
    """Merchant Legitimacy Verifier reading for each missed account."""
    path = settings.MODELS_DIR / "merchant_verifier.joblib"
    if not path.exists() or not len(rows):
        return {}
    from muleguard.features.frame import build_model_frame
    from muleguard.models.merchant import DISCLAIMER

    mv = joblib.load(path)
    mvf = build_model_frame(view=mv["view"]).subset(mv["feature_names"])
    dev = harness.dev_split()
    Xm = mv["preprocessor"].transform(mvf.X[dev.row_index])
    verdicts = mv["verifier"].verdicts(Xm[rows])
    out: dict[int, dict[str, Any]] = {}
    for r, v in zip(rows, verdicts):
        d = v.to_dict() if hasattr(v, "to_dict") else dict(v)
        out[int(r)] = {
            "source": "Merchant Legitimacy Verifier (%s view, %d business "
                      "features)" % (mv["view"], len(mv["feature_names"])),
            "verdict": d,
            "policy": DISCLAIMER,
            "caveat": "fitted on all development rows including this one; the "
                      "reading is in-sample and is context, not corroboration",
        }
    return out


def _payload(champion, family, k, k_source, n_dev, pos, missed, caught, conv, S,
             fam_rank, records, counts, base_rates, shap_block, lens_obs,
             quarantined, dev_row_index, digests, seconds) -> dict[str, Any]:
    from datetime import datetime, timezone

    cross = {}
    for f in S:
        if f == family:
            continue
        cross[f] = {
            "model": FAMILY_MODELS[f],
            "own_misses_at_budget": int((fam_rank[f][pos] > k).sum()),
            "catches_n_of_champion_misses": int(sum(1 for r in missed
                                                    if fam_rank[f][r] <= k)),
            "champion_misses_it_catches_dataset_row_index": [
                int(dev_row_index[r]) for r in missed if fam_rank[f][r] <= k],
            "has_n_misses_champion_catches": int(sum(1 for r in caught
                                                     if fam_rank[f][r] > k)),
        }
    missed_by_all = [int(dev_row_index[r]) for r in missed
                     if all(fam_rank[f][r] > k for f in S)]

    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "artifact": "Missed-Mule Error Atlas",
        "purpose": (
            "Diagnostic only. For every labelled mule the served champion did "
            "not place inside the analyst budget, measure why and assign one "
            "category from a fixed, ordered rule list."),
        "what_this_artifact_must_never_be_used_for": [
            "Creating a rule, feature, threshold or override for an individual "
            "account. Any pattern found here is a hypothesis and must beat the "
            "existing model through nested cross-validation before it changes "
            "anything.",
            "Any statement about an account beyond 'carries a mule label in the "
            "development data' and 'the model ranked it here'.",
            "Any claim of a relationship between two accounts. Neighbour fields "
            "are feature-space similarity only; this dataset has no edge table.",
        ],
        "read_only_contract": {
            "modifies_scores_thresholds_features_or_models": False,
            "inputs_hashed_before_and_after": True,
            "input_sha256": digests,
            "writes": ["artifacts/metrics/error_atlas.json", "docs/ERROR_ATLAS.md"],
        },
        "provenance": {
            "no_language_model_used": True,
            "evidence_policy": "Every field is a measured quantity, a value "
                               "copied from a configuration file, or a fixed "
                               "string written by hand in this repository. No "
                               "generated evidence.",
            "split": "development out-of-fold only; the locked test split was "
                     "not read and no locked-test number appears here",
            "git": git_info(settings.REPO_ROOT),
            "seed": settings.GLOBAL_SEED,
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "seconds": round(seconds, 1),
        },
        "scope": {
            "champion": champion,
            "champion_family": family,
            "analyst_budget_k": k,
            "budget_source": k_source,
            "development_rows": int(n_dev),
            "labelled_mules": int(len(pos)),
            "prevalence": round(float(len(pos) / n_dev), 6),
            "n_missed": int(len(missed)),
            "n_caught": int(len(caught)),
            "recall_at_budget": round(float(len(caught) / len(pos)), 6),
            "statistical_caution": (
                "%d misses. Every count below is a count over %d accounts. "
                "Nothing here is a rate that would survive a confidence "
                "interval, and no pattern seen in a handful of accounts should "
                "be treated as established. The categories describe these "
                "accounts; they do not estimate a population."
                % (len(missed), len(missed))),
        },
        "aggregation_sensitivity": {
            "note": "The three ways of collapsing repeated CV to one ranking "
                    "disagree. All three are reported rather than choosing the "
                    "most favourable.",
            "convention_used": "score_average",
            "why": "build_lenses_v2 averages raw scores across repeats to derive "
                   "the thresholds the system serves, so this is the deployed "
                   "operating point.",
            "conventions": conv["summary"],
        },
        "categories": {
            "definitions": atlas.rule_book(),
            "constants": atlas.DEFAULT_CONSTANTS.to_dict(),
            "constant_sources": {
                "missingness_z": "configs/thresholds.yaml ood.missingness_z_threshold",
                "range_violation_share": "configs/thresholds.yaml "
                                         "ood.range_violation_share_threshold",
                "missing_fraction": "definitional - half of the champion's inputs",
                "threshold_miss_capacity_multiple": "chosen: twice the published "
                                                    "analyst capacity. The only "
                                                    "free parameter in the rule "
                                                    "list.",
                "neighbourhood_k": "OODDetector.k from the served bundle",
            },
            "assignment": "First rule that fires wins. Every rule is evaluated "
                          "for every miss and all verdicts are recorded, so the "
                          "reader can see which explanations were available and "
                          "which the priority order selected.",
            "determinism": "Pure function of the measurement vector. No "
                           "randomness, no per-account branch, no state.",
            "counts": counts,
        },
        "rule_base_rates_on_caught_mules": base_rates,
        "cross_family_agreement": {
            "note": "Same folds, same labels, same budget. A family that catches "
                    "a champion miss shows the signal was recoverable from this "
                    "data.",
            "per_family": cross,
            "champion_misses_missed_by_every_measured_family_dataset_row_index":
                missed_by_all,
            "n_missed_by_every_measured_family": len(missed_by_all),
        },
        "lens_observations": lens_obs,
        "quarantine_guard": {
            "n_quarantined_columns": len(quarantined),
            "quarantined_columns_named_in_this_artifact": 0,
            "enforced_by": "muleguard.models.error_atlas."
                           "assert_no_quarantined_feature, which raises rather "
                           "than emits",
        },
        "attribution_method": shap_block,
        "misses": records,
        "hypotheses_for_nested_cv_testing": _hypotheses(counts, cross, missed,
                                                        base_rates, missed_by_all,
                                                        k, champion),
    }


# ---------------------------------------------------------------------------
# Hypotheses - derived from the measured counts, acted on by nobody
# ---------------------------------------------------------------------------


def _hypotheses(counts, cross, missed, base_rates, missed_by_all, k,
                champion) -> dict[str, Any]:
    """Patterns worth testing, emitted only when the measurement triggers them.

    Each entry is generated from a measured quantity, carries the measurement
    that raised it, and is marked untested. None of them has been implemented,
    wired, or allowed to influence any score in this repository - that is the
    whole point of the upgrade: the Atlas finds candidates, nested CV decides.
    """
    hyps: list[dict[str, Any]] = []

    best = max(cross.items(), key=lambda kv: kv[1]["catches_n_of_champion_misses"],
               default=None)
    if best and best[1]["catches_n_of_champion_misses"] >= 2:
        f, d = best
        hyps.append({
            "id": "H1_recall_oriented_ensemble",
            "trigger": "%s catches %d of the champion's %d misses at K=%d, and "
                       "has %d miss%s the champion catches."
                       % (d["model"], d["catches_n_of_champion_misses"],
                          len(missed), k, d["has_n_misses_champion_catches"],
                          "" if d["has_n_misses_champion_catches"] == 1 else "es"),
            "hypothesis": "A blend of %s with %s, selected on recall at the "
                          "analyst budget rather than on PR-AUC, surfaces more "
                          "labelled mules within the same queue than %s alone."
                          % (champion, d["model"], champion),
            "how_to_test": "Nested CV with recall@K as the selection objective, "
                           "blend weights chosen in the inner loop only, scored "
                           "in the outer loop. Must beat the champion on the "
                           "outer folds, not on these misses.",
            "why_it_might_fail": "challenger_review_v2 already found the rank "
                                 "blend does not beat the best single member on "
                                 "PR-AUC. Recall at a budget is a different "
                                 "objective, but the prior is not encouraging, "
                                 "and %d overlapping accounts is far too few to "
                                 "select a blend weight on."
                                 % d["catches_n_of_champion_misses"],
            "status": "UNTESTED - NOT IMPLEMENTED, NOT WIRED, NO EFFECT ON ANY "
                      "SCORE",
        })

    if counts.get("LOOKALIKE_MULE", 0) >= 2:
        r = base_rates["rates"].get("lookalike_mule", {})
        hyps.append({
            "id": "H2_neighbourhood_label_density_feature",
            "trigger": "%d misses sit in a neighbourhood with no mule-labelled "
                       "account in it and a legitimate account nearer than any "
                       "mule-labelled one; the same test fires on %s of %s "
                       "caught mules, so it separates but far from cleanly."
                       % (counts["LOOKALIKE_MULE"], r.get("n_fired"),
                          base_rates["n_control_rows"]),
            "hypothesis": "A feature counting mule labels among an account's k "
                          "nearest neighbours in the champion's input space adds "
                          "signal the tree models are not extracting.",
            "how_to_test": "Nested CV. The neighbour graph and the label counts "
                           "must be built inside each training fold only - "
                           "building them on all development rows leaks the "
                           "validation labels straight into the feature and will "
                           "produce a spectacular and entirely false gain.",
            "why_it_might_fail": "With this prevalence a fold-local neighbourhood "
                                 "contains almost no positives, so the feature "
                                 "may be near-constant zero and add nothing.",
            "status": "UNTESTED - NOT IMPLEMENTED, NOT WIRED, NO EFFECT ON ANY "
                      "SCORE",
        })

    if counts.get("MISSING_DATA", 0) == 0:
        hyps.append({
            "id": "H3_missingness_is_not_the_explanation",
            "trigger": "Zero misses met either missingness test.",
            "hypothesis": "A negative result, recorded so it is not "
                          "rediscovered: missing data does not explain these "
                          "misses. Missingness-structure work is worth doing on "
                          "its own merits but should not be justified by this "
                          "Atlas.",
            "how_to_test": "Nothing to test here; this entry exists to close off "
                           "a wrong lead.",
            "why_it_might_fail": "Not applicable.",
            "status": "OBSERVATION - NOT ACTED ON",
        })

    if counts.get("THRESHOLD_MISS", 0) >= 1:
        hyps.append({
            "id": "H4_operating_point_sensitivity",
            "trigger": "%d miss(es) were ranked within twice the budget - the "
                       "champion ranked them highly and the queue length "
                       "excluded them." % counts["THRESHOLD_MISS"],
            "hypothesis": "For these accounts the binding constraint is review "
                          "capacity, not model quality. Their recovery is a "
                          "capacity question and no model change would be "
                          "credited for it.",
            "how_to_test": "Not a modelling change. If capacity is ever revised, "
                           "the gain must be attributed to capacity and not "
                           "reported as a model improvement.",
            "why_it_might_fail": "Capacity is an operational constraint set "
                                 "outside this system.",
            "status": "OBSERVATION - NOT ACTED ON",
        })

    if missed_by_all:
        hyps.append({
            "id": "H5_irreducible_on_current_features",
            "trigger": "%d miss(es) were outside the budget for every measured "
                       "family." % len(missed_by_all),
            "hypothesis": "These accounts are not separable by any estimator on "
                          "the current feature set. If they are to be recovered "
                          "it will take new evidence, not a new estimator.",
            "how_to_test": "Nested CV on any genuinely new evidence source. "
                           "Re-tuning existing models against these accounts "
                           "would be fitting to %d rows and is exactly the "
                           "failure mode this Atlas is written to avoid."
                           % len(missed_by_all),
            "why_it_might_fail": "There may be no further evidence available in "
                                 "this dataset.",
            "status": "UNTESTED - NOT IMPLEMENTED, NOT WIRED, NO EFFECT ON ANY "
                      "SCORE",
        })

    return {
        "standing_rule": (
            "None of these has been acted on. Nothing in this file has changed a "
            "score, a threshold, a feature set or a model, and none of these "
            "hypotheses may do so until it has beaten the existing champion "
            "through nested cross-validation. Hand-writing a rule for any "
            "individual account listed in this artifact would be a failure of "
            "the exercise, not a fix."),
        "count": len(hyps),
        "hypotheses": hyps,
    }


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------


def _markdown(p: dict[str, Any]) -> str:
    s, cat = p["scope"], p["categories"]
    L: list[str] = []
    a = L.append
    a("# Missed-Mule Error Atlas\n")
    a("Generated %s from `artifacts/metrics/error_atlas.json`. Do not edit by "
      "hand - regenerate with `python -m muleguard.cli.error_atlas`.\n"
      % p["generated_utc"])
    a("## What this is\n")
    a("For every account carrying a mule label in the development data that the "
      "served champion did **not** place inside the analyst review budget, this "
      "measures why and assigns one category.\n")
    a("It is a diagnostic instrument, not a patch mechanism. It never modifies a "
      "score, a threshold, a prediction, a feature set or a model; it hashes its "
      "inputs before and after to prove it. No language model is involved - "
      "every field is a measured quantity, a configuration value, or a fixed "
      "string written by hand in this repository. Everything is development "
      "out-of-fold; the locked test split was not read.\n")
    a("**The rule that matters:** a pattern found here leaves this system as a "
      "hypothesis for nested cross-validation. It does not become a rule, a "
      "feature or an override. Writing a special case for any individual account "
      "listed below would be a failure of this exercise.\n")

    a("## Scope\n")
    a("| | |\n|---|---|")
    a("| Champion | `%s` |" % s["champion"])
    a("| Analyst budget K | %d (%s) |" % (s["analyst_budget_k"], s["budget_source"]))
    a("| Development rows | %d |" % s["development_rows"])
    a("| Labelled mules | %d (prevalence %.4f) |" % (s["labelled_mules"], s["prevalence"]))
    a("| Surfaced within budget | %d |" % s["n_caught"])
    a("| **Missed** | **%d** |" % s["n_missed"])
    a("| Recall at budget | %.4f |\n" % s["recall_at_budget"])
    a("> %s\n" % s["statistical_caution"])

    a("## Which number counts as a miss\n")
    a("Repeated cross-validation can be collapsed to a single ranking three "
      "ways, and they disagree. All three are reported; the Atlas uses "
      "`score_average` because `build_lenses_v2` averages raw scores to derive "
      "the thresholds the system serves, so that is the deployed operating "
      "point - not the most flattering one.\n")
    a("| Convention | Recall@K | Misses |\n|---|---|---|")
    for name, d in p["aggregation_sensitivity"]["conventions"].items():
        n = d.get("n_misses", d.get("n_misses_per_repeat"))
        a("| %s | %.4f | %s |" % (name, d["recall_at_budget"], n))
    a("")

    a("## Category definitions\n")
    a("Six categories, one per miss. Rules are evaluated in the order below and "
      "the **first rule that fires wins**; every rule is still evaluated and "
      "recorded, so the artifact shows which explanations were available and "
      "which the order selected. The order runs from *the model could not have "
      "known* to *the model had every chance and still did not*.\n")
    a("Constants: `%s`. Four of the six rules have no free parameter - their "
      "tests are definitional. `missingness_z` and `range_violation_share` are "
      "the served OOD lens's own published thresholds, reused rather than "
      "reinvented. `threshold_miss_capacity_multiple` is the one chosen number.\n"
      % cat["constants"])
    a("| # | Category | Exact test | Count |\n|---|---|---|---|")
    for d in cat["definitions"]:
        a("| %d | `%s` | `%s` | %d |" % (d["priority"], d["category"], d["test"],
                                         cat["counts"][d["category"]]))
    a("")
    for d in cat["definitions"]:
        a("**%s** - %s\n" % (d["category"], d["rationale"]))

    br = p["rule_base_rates_on_caught_mules"]
    a("## Do these rules actually discriminate?\n")
    a("A rule that fires just as often on the mules the champion *did* surface "
      "explains nothing. Firing rates over %s:\n" % br["control_group"])
    a("| Rule | Fires on caught mules |\n|---|---|")
    for r, d in br["rates"].items():
        a("| `%s` | %d / %d (%.3f) |" % (r, d["n_fired"], br["n_control_rows"],
                                         d["rate"] if d["rate"] is not None else float("nan")))
    a("")

    a("## Per-miss findings\n")
    a("Distances are in the champion's robust-scaled input space; `d(mule)` and "
      "`d(legit)` are to the nearest mule-labelled and nearest legitimate "
      "development account respectively, the account itself excluded.\n")
    a("| Row | Champion rank | Category | d(mule) | d(legit) | Mules in 10-NN | "
      "Peer families in budget | Missing frac | Anomaly pct |\n"
      "|---|---|---|---|---|---|---|---|---|")
    for m in p["misses"]:
        mm = m["measurements"]
        a("| %d | %d | `%s` | %.2f | %.2f | %d | %d | %.3f | %.1f |"
          % (m["dataset_row_index"], mm["champion_rank"], m["category"],
             mm["distance_to_nearest_known_mule"],
             mm["distance_to_nearest_legitimate_account"],
             mm["n_mule_labels_in_neighbourhood"],
             mm["n_peer_families_within_budget"], mm["missing_fraction"],
             m["anomaly_score"]["anomaly_percentile"]))
    a("")
    a("Full evidence per account - all four family scores, top attributions, "
      "missingness by feature family, merchant/context verdict, anomaly "
      "percentile, neighbour distances and every rule's verdict - is in "
      "`artifacts/metrics/error_atlas.json` under `misses`.\n")
    a("**Neighbour fields are feature-space proximity only.** This dataset has "
      "no edge table. \"Closest known mule\" means the nearest account in the "
      "champion's own 120-column input space after robust scaling. It is not a "
      "transfer, a shared counterparty or any relationship between the two "
      "accounts, and it must never be presented as one.\n")

    cf = p["cross_family_agreement"]
    a("## Cross-family agreement\n")
    a("Same folds, same labels, same budget.\n")
    a("| Family | Own misses | Catches n of champion's %d | Has misses champion "
      "catches |\n|---|---|---|---|" % s["n_missed"])
    for f, d in cf["per_family"].items():
        a("| `%s` | %d | %d | %d |" % (d["model"], d["own_misses_at_budget"],
                                       d["catches_n_of_champion_misses"],
                                       d["has_n_misses_champion_catches"]))
    a("")
    a("%d of the champion's misses were outside the budget for **every** "
      "measured family.\n" % cf["n_missed_by_every_measured_family"])

    lo = p["lens_observations"]
    g, rl, fl = lo["knn_gate"], lo["range_violation_limb"], lo["development_rows_flagged"]
    a("## Observations on the served OOD lens\n")
    a("`OOD_PATTERN` delegates to the lens the system already serves rather than "
      "inventing a second definition, so what that lens does on this data "
      "decides what the category can mean. Zero misses landed in it, and the "
      "reason matters.\n")
    a("| | |\n|---|---|")
    a("| k-NN gate | %.2f (%s) |" % (g["gate_value"], g["gate_definition"]))
    a("| Largest development k-NN distance | %.2f |" % g["max_development_knn_distance"])
    a("| Median development k-NN distance | %.2f |" % g["median_development_knn_distance"])
    a("| Development rows above the gate | %d |" % g["n_development_rows_above_gate"])
    a("| Rows the lens flags in total | %d |" % fl["n_flagged"])
    a("| ...of which carry a mule label | %d |\n" % fl["n_flagged_carrying_a_mule_label"])
    a("The gate **does** fire - it is a development quantile, so a few rows sit "
      "above it by construction. %s\n" % fl["observation"])
    a("The range-violation limb is a different matter: its maximum over all "
      "development rows is %.4f. %s\n"
      % (rl["max_development_range_violation_share"], rl["observation"]))
    a("**The Atlas changed none of this.** %s\n"
      % lo["the_atlas_did_not_change_any_of_this"])

    am = p["attribution_method"]
    a("## Attribution method\n")
    a("Method: **%s**. %s\n" % (am["method"], am.get("not_a_fallback", "")))
    if am["method"] != "NOT_COMPUTED":
        a("The tournament stored scores but not per-fold models, so each scoring "
          "model was rebuilt with the same folds, preprocessor and deterministic "
          "fold seed, then checked against its stored out-of-fold scores across "
          "the whole validation fold. Fidelity:\n")
        a("> %s\n" % am["attribution_fidelity"])
        a("Reproduction required `OMP_NUM_THREADS=4`. XGBoost's `hist` builder "
          "sums gradients in thread order, so at a different thread count the "
          "refits diverged from the stored scores by up to 0.16 absolute. That "
          "is a property of the estimator, not a bug found here, but it means "
          "attributions computed at the wrong thread count would explain a model "
          "that never scored these accounts.\n")

    hy = p["hypotheses_for_nested_cv_testing"]
    a("## Hypotheses for nested-CV testing\n")
    a("> **%s**\n" % hy["standing_rule"])
    for h in hy["hypotheses"]:
        a("### %s\n" % h["id"])
        a("- **Status:** `%s`" % h["status"])
        a("- **What triggered it:** %s" % h["trigger"])
        a("- **Hypothesis:** %s" % h["hypothesis"])
        a("- **How to test it:** %s" % h["how_to_test"])
        a("- **Why it might fail:** %s\n" % h["why_it_might_fail"])

    a("## Limits\n")
    a("- %d misses. These are descriptions of %d accounts, not estimates of a "
      "population. No confidence interval would survive this sample size.\n"
      % (s["n_missed"], s["n_missed"]))
    a("- The anomaly detector, the OOD lens, the hard-negative verifier and the "
      "merchant verifier in the served bundle were fitted on development data. "
      "Their readings on these accounts are in-sample context, not independent "
      "corroboration. The IsolationForest is the exception: it was fitted on "
      "rows not carrying a mule label, so these accounts were outside its fit.\n")
    a("- Quarantined columns cannot appear anywhere above: the writer raises "
      "rather than emits if one is present.\n")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the Missed-Mule Error Atlas")
    ap.add_argument("--budget", type=int, default=None,
                    help="analyst budget K (default: the served urgent-review "
                         "capacity from configs/thresholds.yaml)")
    ap.add_argument("--no-shap", action="store_true",
                    help="skip the per-fold refits used for exact TreeSHAP")
    ap.add_argument("--seed", type=int, default=settings.GLOBAL_SEED)
    args = ap.parse_args()
    p = run(budget=args.budget, with_shap=not args.no_shap, seed=args.seed)
    s = p["scope"]
    print("champion=%s  K=%d  misses=%d / %d labelled mules  recall=%.4f"
          % (s["champion"], s["analyst_budget_k"], s["n_missed"],
             s["labelled_mules"], s["recall_at_budget"]))
    for c, n in p["categories"]["counts"].items():
        if n:
            print("  %-20s %d" % (c, n))
    print("hypotheses raised (all untested, none acted on): %d"
          % p["hypotheses_for_nested_cv_testing"]["count"])
    print("wrote %s" % OUT_JSON)
    print("wrote %s" % OUT_DOC)


if __name__ == "__main__":
    main()
