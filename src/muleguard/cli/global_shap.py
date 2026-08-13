"""Global feature importance by exact TreeSHAP, for the §56 importance plot.

Run::

    .venv/Scripts/python.exe -m muleguard.cli.global_shap

The Error Atlas already computes exact TreeSHAP, but only for the rows it
investigates - eleven missed mules. A global importance ranking needs every
development row, and storing a full per-row attribution matrix for them is
neither necessary nor cheap. This pass accumulates ``sum |shap|`` and
``sum shap`` per feature as each fold is scored and keeps nothing else, so the
memory cost is two vectors the width of the champion's feature set.

Attribution is out-of-fold, like every other number in this project: each row is
attributed by the fold model that did **not** train on it. Each refit is checked
against the stored out-of-fold scores before its attributions are used, and the
check is reported per fold rather than assumed - if a fold does not reproduce,
the artifact says its attributions come from a protocol-identical twin.

Writes ``artifacts/metrics/global_shap_importance.json``.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import time
from typing import Any

# Thread hygiene before the ML imports, matching the tournament that produced
# the stored out-of-fold scores; a different thread count can change sums.
os.environ.setdefault("OMP_NUM_THREADS", "4")

import joblib
import numpy as np
import polars as pl

from muleguard import settings
from muleguard.logging import configure, get_logger
from muleguard.models import harness
from muleguard.utils import load_json, save_json, set_global_seed

log = get_logger("cli.global_shap")

OUT = settings.METRICS_DIR / "global_shap_importance.json"


def _champion() -> str:
    return load_json(settings.METRICS_DIR / "promotion_decision_v2.json")["promoted"]


def _repeat_scores(oof: pl.DataFrame, model: str) -> np.ndarray:
    sub = oof.filter(pl.col("model") == model).sort(["repeat", "row_index"])
    n_rep = sub["repeat"].n_unique()
    return sub["score"].to_numpy().reshape(n_rep, -1)


def run(seed: int = settings.GLOBAL_SEED, top_n: int = 30) -> dict[str, Any]:
    set_global_seed(seed)
    t0 = time.perf_counter()

    from muleguard.explain.reason_codes import tree_shap
    from muleguard.features.frame import build_model_frame
    from muleguard.features.preprocessing import FoldPreprocessor
    from muleguard.models import core_models

    champion = _champion()
    family = champion.split("_top_")[0]
    bundle = joblib.load(settings.MODELS_DIR / "final_bundle.joblib")
    names = list(bundle["feature_list_selected"])

    quarantined = {e["feature"] for e in
                   load_json(settings.FEATURES_DIR / "quarantined_features.json")["quarantine"]}
    overlap = sorted(set(names) & quarantined)
    if overlap:
        raise ValueError(f"champion input set names quarantined columns: {overlap}")

    mf = build_model_frame(view=None).subset(names)
    dev = harness.dev_split()
    Xdev, ydev = mf.X[dev.row_index], mf.y[dev.row_index]

    oof = pl.read_parquet(settings.PREDICTIONS_DIR / "oof_v2.parquet")
    stored = _repeat_scores(oof, champion)
    n_rep = int(stored.shape[0])
    if len(dev.fold_ids) < n_rep:
        raise ValueError(f"development split offers {len(dev.fold_ids)} repeats but "
                         f"the stored out-of-fold matrix has {n_rep}")

    name_pos = {n: j for j, n in enumerate(names)}
    abs_sum = np.zeros(len(names))
    signed_sum = np.zeros(len(names))
    rows_attributed = 0
    checks: list[dict[str, Any]] = []

    for rep in range(n_rep):
        ids = dev.fold_ids[rep]
        for fold in sorted(int(v) for v in np.unique(ids)):
            va = ids == fold
            tr = ~va
            prep = FoldPreprocessor(mode="tree")
            Xtr = prep.fit_transform(Xdev[tr], names)
            Xva = prep.transform(Xdev[va])
            t_fold = time.perf_counter()
            scores, model = core_models.SCORERS[family](
                Xtr, ydev[tr], Xva, harness.fold_seed(rep, fold), return_model=True)
            diff = float(np.abs(scores - stored[rep][va]).max())
            contrib, _ = tree_shap(model, family, Xva)
            # Columns are the fold preprocessor's kept features, which can be a
            # subset; map them back by name rather than by position.
            for j, fname in enumerate(prep.kept_features):
                abs_sum[name_pos[fname]] += np.abs(contrib[:, j]).sum()
                signed_sum[name_pos[fname]] += contrib[:, j].sum()
            rows_attributed += int(va.sum())
            checks.append({
                "repeat": rep, "fold": fold, "n_validation_rows": int(va.sum()),
                "max_abs_score_difference_vs_stored_oof": diff,
                "reproduces_stored_oof": bool(diff == 0.0),
                "n_features_kept_by_fold_preprocessor": int(len(prep.kept_features)),
                "seconds": round(time.perf_counter() - t_fold, 2),
            })
            log.info("%s r%d f%d  rows %4d  reproduces=%s  %.1fs", family, rep, fold,
                     int(va.sum()), diff == 0.0, checks[-1]["seconds"])

    mean_abs = abs_sum / max(rows_attributed, 1)
    mean_signed = signed_sum / max(rows_attributed, 1)
    order = np.argsort(mean_abs)[::-1]
    ranking = [{
        "rank": i + 1,
        "feature": names[j],
        "mean_abs_shap": float(mean_abs[j]),
        "mean_signed_shap": float(mean_signed[j]),
        "share_of_total_abs": float(mean_abs[j] / mean_abs.sum()) if mean_abs.sum() else 0.0,
    } for i, j in enumerate(order[:top_n])]

    all_reproduce = all(c["reproduces_stored_oof"] for c in checks)
    payload = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "champion": champion,
        "family": family,
        "n_features": len(names),
        "n_dev_rows": int(len(ydev)),
        "n_repeats": n_rep,
        "rows_attributed": rows_attributed,
        "attribution_method": {
            "method": "EXACT_TREESHAP",
            "implementation": "muleguard.explain.reason_codes.tree_shap",
            "aggregation": "mean |SHAP| over all out-of-fold attributions, "
                           "averaged over rows and repeats alike",
            "out_of_fold": "each row is attributed by a model that did not train "
                           "on it, so the ranking is not an in-sample artefact",
            "fidelity": ("EXACT - every fold refit reproduced its stored "
                         "out-of-fold scores to 0.0 absolute"
                         if all_reproduce else
                         "PROTOCOL_IDENTICAL_TWIN - at least one refit did not "
                         "reproduce the stored scores exactly; those folds' "
                         "attributions describe an identically-built model, not "
                         "the exact scoring model"),
        },
        "what_this_is_not": "A causal statement. SHAP explains this model's "
                            "output, not the mechanics of money muling.",
        "ranking": ranking,
        "fold_checks": checks,
        "total_seconds": round(time.perf_counter() - t0, 1),
    }
    save_json(payload, OUT)
    log.info("wrote %s - top feature %s (mean |SHAP| %.5f), %d folds, %.1fs",
             OUT, ranking[0]["feature"], ranking[0]["mean_abs_shap"],
             len(checks), payload["total_seconds"])
    return payload


def main(argv: list[str] | None = None) -> int:
    configure()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--top-n", type=int, default=30,
                    help="how many features to store in the ranking (default 30)")
    ap.add_argument("--seed", type=int, default=settings.GLOBAL_SEED)
    args = ap.parse_args(argv)
    run(seed=args.seed, top_n=args.top_n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
