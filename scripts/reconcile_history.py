"""Reconcile every historical headline metric against the current generation.

Final-validation prompt section 7 requires that each historical number either be
traced and explained, or be declared unreproducible. It also forbids silently
quoting the most flattering one. This script does the tracing mechanically so
the document that cites it cannot drift from the artifacts.

The two generations differ in *two* ways at once - the feature sets were rebuilt
behind the availability firewall, and the repeat count dropped from 5 to 3. A
raw v1-vs-v2 delta therefore confounds them. Because generation 1 stored
``pr_auc_per_repeat`` and the first three repeats use byte-identical fold
assignments, the two effects can be separated: re-averaging generation 1 over
its first three repeats isolates the feature-set effect on its own.

Writes ``artifacts/metrics/historical_reconciliation.csv``.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "artifacts" / "metrics"
FEATURES = ROOT / "artifacts" / "features"

#: generation-1 name -> generation-2 name. Only pairs that are genuinely the
#: same experiment appear here; anything whose configuration also changed is
#: listed in UNPAIRED with the reason.
PAIRS = {
    "dummy_prevalence": "dummy_prevalence",
    "lightgbm_tuned_full": "lightgbm_full_pool",
    "lightgbm_tuned_top60": "lightgbm_top_60",
    "lightgbm_tuned_top30": "lightgbm_top_30",
    "lightgbm_tuned_top15": "lightgbm_top_15",
    "xgboost_tuned_top60": "xgboost_top_60",
    "catboost_tuned_top60": "catboost_top_60",
    "tabpfn_top60": "tabpfn_top_60",
}

UNPAIRED = {
    "logistic_l2": "generation 2 replaced full-pool L2 with logistic_top30; "
                   "different feature pool AND different penalty, not comparable",
    "lightgbm_baseline": "no generation-2 counterpart; retained only as the "
                         "ACCEPTED arm of the F3912 leakage ablation",
    "REJECTED_leakage_lgbm_with_F3912": "deliberate leak demonstration; never a "
                                        "candidate, never comparable",
}


def _load(name: str) -> dict:
    return json.loads((METRICS / name).read_text(encoding="utf-8"))


def main() -> None:
    v1 = _load("oof_metrics.json")["models"]
    v2 = _load("tournament_v2.json")["models"]
    quarantine = {
        e["feature"] if isinstance(e, dict) else e
        for e in _load_features("quarantined_features.json")["quarantine"]
    }

    rows = []
    for old, new in PAIRS.items():
        a, b = v1[old], v2[new]
        per = a.get("pr_auc_per_repeat")
        # like-for-like: generation 1 re-averaged over its first three repeats,
        # whose fold assignments are identical to generation 2's three repeats.
        v1_3 = float(np.mean(per[:3])) if per else float("nan")
        rows.append({
            "gen1_model": old,
            "gen2_model": new,
            "gen1_pr_auc_5rep": round(float(a["pr_auc_mean"]), 5),
            "gen1_pr_auc_first3rep": round(v1_3, 5),
            "gen2_pr_auc_3rep": round(float(b["pr_auc_mean"]), 5),
            "delta_repeat_count": round(v1_3 - float(a["pr_auc_mean"]), 5),
            "delta_feature_set": round(float(b["pr_auc_mean"]) - v1_3, 5),
            "delta_total": round(float(b["pr_auc_mean"]) - float(a["pr_auc_mean"]), 5),
        })

    out = pl.DataFrame(rows).sort("delta_feature_set")
    out.write_csv(METRICS / "historical_reconciliation.csv")

    print("dev-split control")
    print("  identical dev rows across repeat counts:", _dev_rows_identical())
    print()
    print("quarantined features present in each generation's compact sets")
    for gen, fname, key in (("gen1", "selected_features.json", "compact_sets"),
                            ("gen2", "selected_features_v2.json", "pools")):
        d = _load_features(fname)
        sets = d[key] if gen == "gen1" else d[key]["ALL_ADMISSIBLE"]["compact_sets"]
        for k, v in sets.items():
            if isinstance(v, list) and v and isinstance(v[0], str):
                hit = sorted(set(v) & quarantine)
                print(f"  {gen} {k:14s} n={len(v):4d}  quarantined={hit or 'NONE'}")
    print()
    print(out.write_csv().rstrip())
    print()
    for k, why in UNPAIRED.items():
        print(f"UNPAIRED {k}: {why}")


def _load_features(name: str) -> dict:
    return json.loads((FEATURES / name).read_text(encoding="utf-8"))


def _dev_rows_identical() -> bool:
    from muleguard.models import harness
    h = lambda n: hashlib.sha256(
        np.ascontiguousarray(np.sort(harness.dev_split(n).row_index))).hexdigest()
    return h(5) == h(3)


if __name__ == "__main__":
    main()
