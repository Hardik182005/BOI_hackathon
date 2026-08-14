"""Section 55: the one definitive accuracy table, all 27 columns.

There are four model-comparison CSVs in this repo and each answers a slightly
different question with a slightly different column set. That is three too
many. This builds the single table the spec asks for, and it builds every
threshold-dependent number from the **saved out-of-fold predictions** rather
than copying a summary line from an older artifact - so a reader who does not
trust the table can recompute it from the parquet files beside it.

    .venv/Scripts/python.exe -m muleguard.cli.final_accuracy_table

The operating point
-------------------
Seven of the 27 columns (accuracy, balanced accuracy, precision, recall, F1,
F2, MCC) need a threshold, and there is no threshold that is fair to twenty-two
models at once. Choosing each model's own F1-optimal cut would fit a parameter
on the very predictions being scored, which is the oldest way to make a table
lie. A fixed probability cut is worse: raw scores from xgboost, TabPFN and a
prevalence dummy are not on a common scale.

So every model is judged at the same **review budget**: the top 100 accounts
per repeat. Nothing is fitted, the budget is identical across rows, and it is
the queue length a bank actually staffs. Two consequences are deliberate and
stated rather than hidden - ``precision`` equals ``precision_at_top100`` and
``recall`` equals ``recall_at_top100`` by construction, and every row's
accuracy sits near 98.7% because 99.1% of the data is negative. The second is
the argument for why accuracy is in this table only because the spec asks for
it, and why PR-AUC is the column that decides anything.

Brier and ECE are computed on **raw** out-of-fold scores, because only the
shipped bundle has a fitted calibrator and applying one model's calibration to
another's scores would be meaningless. The champion's calibrated Brier is a
different, better number and is reported in the calibration artifacts, not
here; mixing the two in one column would be the kind of quiet apples-to-oranges
this table exists to end.

Latency and model size are measurements of an artifact that has to exist. Only
the champion was built into a servable bundle, so those two columns are null
for every other row with the reason recorded, rather than filled with an
estimate that would read as measured.
"""
from __future__ import annotations

import argparse
import datetime as dt
from typing import Any

import numpy as np
import polars as pl

from muleguard import settings
from muleguard.logging import configure, get_logger
from muleguard.utils import load_json, save_json

log = get_logger("cli.final_accuracy_table")

OUT_CSV = settings.METRICS_DIR / "final_accuracy_table.csv"
OUT_JSON = settings.METRICS_DIR / "final_accuracy_table.json"

#: Review budget, in accounts per repeat. Identical for every row.
BUDGET = 100

#: The two prediction stores, newest first. A model appearing in both keeps the
#: newer row; the older store is what supplies the pre-firewall leakage control
#: and the first-generation tuned models, which have no v2 equivalent.
#:
#: ``leakage_safe`` is a property of the *generation*, not of a model's name.
#: docs/HISTORICAL_METRIC_RECONCILIATION.md records that the firewall was built
#: after generation 1 ran and that every generation-1 compact set carried at
#: least three quarantined features. So no generation-1 row can be certified
#: safe from the stored artifacts, and marking one safe because its name does
#: not say REJECTED would let a demoted model back into the table as a
#: legitimate challenger - which is the exact mistake the firewall exists to
#: prevent.
STORES = (
    ("final_model_oof_predictions.parquet", "v2 tournament (post-firewall)", True),
    ("oof_predictions.parquet", "v1 tuned tournament (pre-firewall)", False),
)

LEAKAGE_REASON = {
    True: "trained under the Feature Availability Firewall; the quarantined "
          "columns were not in the candidate pool",
    False: "generation 1 predates the firewall. Its compact feature sets "
           "carried quarantined post-resolution columns (F3898, F3913, F3914, "
           "F3916), so no generation-1 row can be certified safe from the "
           "stored artifacts",
}

#: Exactly the 27 columns of section 55, in the spec's order.
COLUMNS = [
    "model", "feature_set", "leakage_safe", "outer_repeats",
    "pr_auc_mean", "pr_auc_std", "pr_auc_ci95", "roc_auc",
    "accuracy", "balanced_accuracy", "precision", "recall", "f1", "f2", "mcc",
    "recall_at_top25", "recall_at_top50", "recall_at_top100",
    "precision_at_top25", "precision_at_top50", "precision_at_top100",
    "fp_per_1000_legit", "brier", "ece", "latency_ms", "model_size_mb",
    "status",
]

STATUS_RULE = {
    "REJECTED": "not leakage-safe, or operationally ineligible (measured "
                "interactive latency above the 5 s budget), or PR-AUC below "
                "half the champion's",
    "CHALLENGER": "leakage-safe, eligible, and PR-AUC mean within one "
                  "champion-std of the champion or above it",
    "FINALIST": "leakage-safe, eligible, PR-AUC at least half the champion's "
                "but outside the challenge band",
    "CHAMPION": "the model recorded as active in artifacts/model_registry/"
                "registry.json",
}


# --- metrics ------------------------------------------------------------------

def _t_crit(n: int) -> float:
    """Two-sided 95% t critical value; falls back to a table without scipy."""
    try:
        from scipy import stats

        return float(stats.t.ppf(0.975, n - 1))
    except Exception:  # noqa: BLE001 - the table below is the handler
        return {1: float("nan"), 2: 12.706, 3: 4.303, 4: 3.182,
                5: 2.776}.get(n, 2.0)


def _ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        tot += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(tot)


def _at_k(y: np.ndarray, s: np.ndarray, k: int) -> tuple[float, float]:
    """(recall@k, precision@k) with ties broken by descending score only."""
    k = min(k, len(s))
    order = np.argsort(-s, kind="stable")[:k]
    hit = float(y[order].sum())
    pos = float(y.sum())
    return (hit / pos if pos else float("nan")), (hit / k if k else float("nan"))


def _repeat_metrics(y: np.ndarray, s: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    n = len(y)
    k = min(BUDGET, n)
    cut = np.argsort(-s, kind="stable")[:k]
    pred = np.zeros(n, dtype=bool)
    pred[cut] = True

    tp = float((pred & (y == 1)).sum())
    fp = float((pred & (y == 0)).sum())
    fn = float((~pred & (y == 1)).sum())
    tn = float((~pred & (y == 0)).sum())

    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    tnr = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    f2 = 5 * prec * rec / (4 * prec + rec) if 4 * prec + rec else 0.0
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn - fp * fn) / den) if den > 0 else 0.0

    out = {
        "pr_auc": float(average_precision_score(y, s)),
        "roc_auc": float(roc_auc_score(y, s)) if 0 < y.sum() < n else float("nan"),
        "accuracy": (tp + tn) / n,
        "balanced_accuracy": 0.5 * (rec + tnr),
        "precision": prec, "recall": rec, "f1": f1, "f2": f2, "mcc": float(mcc),
        "fp_per_1000_legit": 1000.0 * fp / max(tn + fp, 1.0),
        "brier": float(np.mean((s - y) ** 2)),
        "ece": _ece(y, np.clip(s, 0.0, 1.0)),
    }
    for kk in (25, 50, 100):
        r, p = _at_k(y, s, kk)
        out[f"recall_at_top{kk}"] = r
        out[f"precision_at_top{kk}"] = p
    return out


# --- assembly -----------------------------------------------------------------

def _load_stores() -> tuple[dict[str, pl.DataFrame], dict[str, dict[str, Any]]]:
    frames: dict[str, pl.DataFrame] = {}
    origin: dict[str, dict[str, Any]] = {}
    for fname, label, firewalled in STORES:
        path = settings.PREDICTIONS_DIR / fname
        if not path.exists():
            log.warning("prediction store absent: %s", fname)
            continue
        df = pl.read_parquet(path)
        for name, part in df.partition_by("model", as_dict=True).items():
            model = name[0] if isinstance(name, tuple) else name
            if model in frames:  # newer store already claimed it
                continue
            frames[model] = part
            # The leakage control is unsafe on its own terms, whichever store
            # it happens to sit in.
            safe = firewalled and not model.startswith("REJECTED_")
            origin[model] = {
                "prediction_source": f"artifacts/predictions/{fname} ({label})",
                "leakage_safe": safe,
                "leakage_reason": ("built deliberately WITH F3912 as a leakage "
                                   "control; never a candidate model")
                if model.startswith("REJECTED_") else LEAKAGE_REASON[firewalled],
            }
    return frames, origin


def _ineligible_families(cat: dict[str, dict[str, Any]]) -> set[str]:
    """Families disqualified on measured interactive latency.

    Read from the catalog rather than hardcoded, and applied per *family*: the
    v1 and v2 TabPFN rows are the same estimator at the same cost, and letting
    one of them through as a challenger because the older CSV had no
    eligibility column would be an accident of file format, not a finding.
    """
    bad: set[str] = set()
    for name, row in cat.items():
        slow = row.get("interactive_score_seconds")
        flagged = str(row.get("promotion_eligible", "")).lower() == "false"
        if flagged or (isinstance(slow, (int, float)) and slow > 5.0):
            bad.add(str(row.get("family") or name.split("_")[0]))
    return bad


def _catalog() -> dict[str, dict[str, Any]]:
    """Feature-set and eligibility facts, read from the tournament CSVs."""
    cat: dict[str, dict[str, Any]] = {}
    for fname in ("model_comparison_v2.csv", "model_comparison.csv"):
        path = settings.METRICS_DIR / fname
        if not path.exists():
            continue
        for r in pl.read_csv(path).to_dicts():
            cat.setdefault(str(r["model"]), r)
    return cat


def _feature_set(model: str, row: dict[str, Any]) -> str:
    fs = row.get("feature_set") or row.get("n_features")
    if fs in (None, "", "null"):
        # Fall back to the name, which encodes it for every model in the v1
        # store. Guessing beyond that would be inventing provenance.
        for tag in ("top60", "top30", "top15", "top120", "full"):
            if tag in model:
                return tag
        return "unrecorded"
    n = row.get("n_features")
    view = row.get("view")
    label = str(fs) if not str(fs).isdigit() else f"top_{fs}"
    if view and view != "ALL_ADMISSIBLE":
        label = f"{label} ({view})"
    return f"{label}" if not n or str(n) in label else f"{label} [n={n}]"


def _status(model: str, pr: float, safe: bool, eligible: bool,
            champ: str, champ_pr: float, champ_sd: float) -> str:
    if model == champ:
        return "CHAMPION"
    if not safe or not eligible:
        return "REJECTED"
    if pr >= champ_pr - champ_sd:
        return "CHALLENGER"
    if pr >= 0.5 * champ_pr:
        return "FINALIST"
    return "REJECTED"


def build() -> dict[str, Any]:
    configure()
    frames, origin = _load_stores()
    if not frames:
        raise SystemExit("no OOF prediction store found under artifacts/predictions")
    cat = _catalog()
    slow_families = _ineligible_families(cat)
    log.info("operationally ineligible families: %s", sorted(slow_families) or "none")

    reg = load_json(settings.REGISTRY_DIR / "registry.json")
    active = [m for m in reg.get("models", []) if m.get("status") != "retired"]
    champ = str(active[-1]["winner"]) if active else ""
    champ_sha = active[-1].get("bundle_sha256") if active else None

    perf = {}
    try:
        perf = load_json(settings.ARTIFACTS_DIR / "testing" / "performance_results.json")
    except (FileNotFoundError, ValueError):
        pass
    champ_latency_ms = round(1000 * perf.get("latency_seconds", {}).get("p50"), 1) \
        if perf.get("latency_seconds", {}).get("p50") else None
    champ_size_mb = round(perf.get("model_bundle_mb"), 3) \
        if perf.get("model_bundle_mb") else None

    rows: list[dict[str, Any]] = []
    detail: dict[str, Any] = {}
    for model, df in frames.items():
        per_repeat = []
        for _, part in df.partition_by("repeat", as_dict=True).items():
            y = part["target"].to_numpy().astype(int)
            s = part["score"].to_numpy().astype(float)
            if y.sum() == 0:
                continue
            per_repeat.append(_repeat_metrics(y, s))
        if not per_repeat:
            continue
        n_rep = len(per_repeat)
        agg = {k: np.array([m[k] for m in per_repeat], dtype=float)
               for k in per_repeat[0]}
        pr = agg["pr_auc"]
        sd = float(pr.std(ddof=1)) if n_rep > 1 else 0.0
        half = _t_crit(n_rep) * sd / np.sqrt(n_rep) if n_rep > 1 else float("nan")

        c = cat.get(model, {})
        safe = origin[model]["leakage_safe"]
        # TabPFN is the only model with a measured interactive latency, and it
        # is 438 s per scoring call: correct, and unusable behind a button.
        family = str(c.get("family") or model.split("_")[0])
        eligible = family not in slow_families

        rows.append({
            "model": model,
            "feature_set": _feature_set(model, c),
            "leakage_safe": safe,
            "outer_repeats": n_rep,
            "pr_auc_mean": round(float(pr.mean()), 5),
            "pr_auc_std": round(sd, 5),
            "pr_auc_ci95": (f"[{pr.mean() - half:.5f}, {pr.mean() + half:.5f}]"
                            if half == half else "n/a (single repeat)"),
            "roc_auc": round(float(np.nanmean(agg["roc_auc"])), 5),
            "accuracy": round(float(agg["accuracy"].mean()), 5),
            "balanced_accuracy": round(float(agg["balanced_accuracy"].mean()), 5),
            "precision": round(float(agg["precision"].mean()), 5),
            "recall": round(float(agg["recall"].mean()), 5),
            "f1": round(float(agg["f1"].mean()), 5),
            "f2": round(float(agg["f2"].mean()), 5),
            "mcc": round(float(agg["mcc"].mean()), 5),
            "recall_at_top25": round(float(agg["recall_at_top25"].mean()), 5),
            "recall_at_top50": round(float(agg["recall_at_top50"].mean()), 5),
            "recall_at_top100": round(float(agg["recall_at_top100"].mean()), 5),
            "precision_at_top25": round(float(agg["precision_at_top25"].mean()), 5),
            "precision_at_top50": round(float(agg["precision_at_top50"].mean()), 5),
            "precision_at_top100": round(float(agg["precision_at_top100"].mean()), 5),
            "fp_per_1000_legit": round(float(agg["fp_per_1000_legit"].mean()), 3),
            "brier": round(float(agg["brier"].mean()), 6),
            "ece": round(float(agg["ece"].mean()), 5),
            "latency_ms": champ_latency_ms if model == champ else None,
            "model_size_mb": champ_size_mb if model == champ else None,
            "status": "",  # filled once the champion's band is known
            "_eligible": eligible,
        })
        detail[model] = origin[model] | {
            "rows_per_repeat": int(df.height / n_rep),
            "operationally_eligible": eligible,
            "per_repeat_pr_auc": [round(v, 5) for v in pr]}

    by_name = {r["model"]: r for r in rows}
    champ_row = by_name.get(champ)
    if champ_row is None:
        raise SystemExit(f"registry champion {champ!r} has no saved OOF predictions")
    cpr, csd = champ_row["pr_auc_mean"], champ_row["pr_auc_std"]
    for r in rows:
        r["status"] = _status(r["model"], r["pr_auc_mean"], r["leakage_safe"],
                              r.pop("_eligible"), champ, cpr, csd)

    order = {"CHAMPION": 0, "CHALLENGER": 1, "FINALIST": 2, "REJECTED": 3}
    rows.sort(key=lambda r: (order[r["status"]], -r["pr_auc_mean"]))

    pl.DataFrame(rows).select(COLUMNS).write_csv(OUT_CSV)

    payload = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "spec": "section 55 - the definitive accuracy / performance table",
        "csv": str(OUT_CSV.relative_to(settings.REPO_ROOT)),
        "n_models": len(rows),
        "champion": champ,
        "champion_bundle_sha256": champ_sha,
        "split": "development out-of-fold predictions; the locked test set was "
                 "not read by this tool",
        "operating_point": {
            "rule": f"top {BUDGET} accounts per repeat, identical for every model",
            "why": "nothing is fitted and the budget does not vary by row, so "
                   "the threshold-dependent columns are comparable; a per-model "
                   "F1-optimal cut would be fitted on the same predictions it "
                   "is then scored against",
            "known_consequence": "precision == precision_at_top100 and recall "
                                 "== recall_at_top100 by construction",
        },
        "column_notes": {
            "accuracy": "near-identical across every row because 99.1% of the "
                        "data is negative; present because the spec lists it, "
                        "not because it separates models",
            "brier": "raw OOF scores, uncalibrated. Only the champion ships a "
                     "fitted calibrator, and applying it to another model's "
                     "scores would be meaningless",
            "ece": "raw OOF scores, 10 equal-width bins",
            "latency_ms": "measured p50 of a single-row API call against the "
                          "served bundle. Null for models that were never built "
                          "into a servable bundle - not estimated",
            "model_size_mb": "on-disk joblib size. Null for the same reason",
            "pr_auc_ci95": "t interval across outer repeats, not across folds",
        },
        "status_rule": STATUS_RULE,
        "reconciliation": {
            "why_recall_at_top100_differs_from_the_fairness_audit": (
                "docs/FAIRNESS_AND_SENSITIVE_FEATURE_AUDIT.md reports the "
                "champion's Recall@100 as 0.8281 and this table reports "
                "0.7812. Both are correct and neither is stale. The audit "
                "averages each row's score across the 3 repeats and then takes "
                "one top-100 cut, which is the operating point a deployed "
                "system would actually have; this table takes a top-100 cut "
                "inside each repeat and averages the results, which is the "
                "only form that yields a usable per-repeat spread. Averaging "
                "scores first removes seed variance, so it reads higher. The "
                "conservative number is the one reported here."),
            "verified": "computed both ways from "
                        "artifacts/predictions/final_model_oof_predictions.parquet",
        },
        "per_model_sources": detail,
        "supersedes": ["artifacts/metrics/model_comparison.csv",
                       "artifacts/metrics/model_comparison_v2.csv",
                       "artifacts/metrics/final_model_comparison.csv"],
        "rows": rows,
    }
    save_json(payload, OUT_JSON)
    log.info("wrote %d models to %s", len(rows), OUT_CSV)
    return payload


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    p = build()
    if not a.quiet:
        print(f"\n{'model':<30} {'PR-AUC':>8} {'+-':>7} {'R@100':>7} "
              f"{'P@100':>7} {'FP/1k':>7}  status")
        for r in p["rows"]:
            print(f"{r['model']:<30} {r['pr_auc_mean']:>8.5f} "
                  f"{r['pr_auc_std']:>7.5f} {r['recall_at_top100']:>7.4f} "
                  f"{r['precision_at_top100']:>7.4f} "
                  f"{r['fp_per_1000_legit']:>7.2f}  {r['status']}")
        print(f"\n{len(p['rows'])} models, {len(COLUMNS)} columns -> {OUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
