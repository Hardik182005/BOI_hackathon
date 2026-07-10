"""Training CLI.

Subcommands:
  baselines  - dummy / logistic-L2 / LightGBM through saved folds (leakage-free)
               + the F3912 REJECTED-LEAKAGE ablation evidence run
  (tournament / advanced live in muleguard.cli.tournament)

Every run writes:
  artifacts/predictions/oof_predictions.parquet  (appended per model)
  artifacts/metrics/oof_metrics.json             (merged per model)
"""
from __future__ import annotations

import argparse
import datetime as dt
import time

import numpy as np
import polars as pl

from muleguard import settings
from muleguard.evaluation.metrics import full_metric_report
from muleguard.logging import get_logger
from muleguard.models import baselines as bl
from muleguard.utils import load_json, save_json, set_global_seed

log = get_logger("cli.train")

OOF_STORE = settings.PREDICTIONS_DIR / "oof_predictions.parquet"
OOF_METRICS = settings.METRICS_DIR / "oof_metrics.json"


def append_oof(preds: pl.DataFrame) -> None:
    if OOF_STORE.exists():
        existing = pl.read_parquet(OOF_STORE)
        model = preds["model"][0]
        existing = existing.filter(pl.col("model") != model)  # rerun replaces
        preds = pl.concat([existing, preds])
    preds.write_parquet(OOF_STORE)


def evaluate_oof(preds: pl.DataFrame, model_name: str) -> dict:
    """Per-repeat metrics + pooled mean/std across repeats."""
    cfg = settings.load_config("train")
    budgets = [int(b) for b in cfg["alert_budgets"]]
    fprs = [float(f) for f in cfg["fpr_targets"]]
    per_repeat = []
    for rep in sorted(preds["repeat"].unique().to_list()):
        sub = preds.filter(pl.col("repeat") == rep)
        rpt = full_metric_report(
            sub["target"].to_numpy(), sub["score"].to_numpy(),
            budgets=budgets, fpr_targets=fprs,
            n_boot=500, seed=settings.GLOBAL_SEED,
            split_label=f"OOF repeat {rep} (dev, natural prevalence)",
        )
        per_repeat.append(rpt)
    ap = [r["pr_auc"]["point"] for r in per_repeat]
    roc = [r["roc_auc"]["point"] for r in per_repeat]
    return {
        "model": model_name,
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "n_repeats": len(per_repeat),
        "pr_auc_mean": float(np.mean(ap)),
        "pr_auc_std": float(np.std(ap)),
        "pr_auc_per_repeat": ap,
        "roc_auc_mean": float(np.mean(roc)),
        "per_repeat": per_repeat,
    }


def merge_metrics(entry: dict, extra_key: str | None = None) -> None:
    merged = load_json(OOF_METRICS) if OOF_METRICS.exists() else {"models": {}}
    key = extra_key or entry["model"]
    merged["models"][key] = entry
    merged["updated_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    save_json(merged, OOF_METRICS)


def run_baselines(n_repeats: int | None, include_leakage_ablation: bool = True) -> None:
    set_global_seed(settings.GLOBAL_SEED)
    runs: list[tuple[str, dict]] = [
        ("dummy_prevalence", dict(scorer=bl.fit_score_dummy, mode="linear")),
        ("logistic_l2", dict(scorer=bl.fit_score_logistic, mode="linear")),
        ("lightgbm_baseline", dict(scorer=bl.fit_score_lightgbm, mode="tree")),
    ]
    summary_lines = []
    for name, kw in runs:
        t0 = time.perf_counter()
        log.info("training %s ...", name)
        preds = bl.run_oof(name, kw["scorer"], kw["mode"], n_repeats=n_repeats)
        append_oof(preds)
        entry = evaluate_oof(preds, name)
        entry["runtime_seconds"] = round(time.perf_counter() - t0, 1)
        merge_metrics(entry)
        summary_lines.append(
            f"{name}: PR-AUC {entry['pr_auc_mean']:.4f} +/- {entry['pr_auc_std']:.4f} "
            f"({entry['runtime_seconds']}s)"
        )
        log.info(summary_lines[-1])

    if include_leakage_ablation:
        t0 = time.perf_counter()
        log.info("REJECTED-LEAKAGE ablation: LightGBM WITH F3912 (evidence only)")
        preds = bl.run_oof(
            "REJECTED_leakage_lgbm_with_F3912", bl.fit_score_lightgbm, "tree",
            n_repeats=min(n_repeats or 2, 2), extra_allowed=["F3912"],
        )
        append_oof(preds)
        entry = evaluate_oof(preds, "REJECTED_leakage_lgbm_with_F3912")
        entry["runtime_seconds"] = round(time.perf_counter() - t0, 1)
        entry["status"] = "REJECTED LEAKAGE - evidence only, never a candidate model"
        merge_metrics(entry)

        clean = load_json(OOF_METRICS)["models"]["lightgbm_baseline"]
        ablation = {
            "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "without_f3912": {
                "model": "lightgbm_baseline",
                "pr_auc_mean": clean["pr_auc_mean"],
                "pr_auc_std": clean["pr_auc_std"],
                "status": "ACCEPTED (leakage-free)",
            },
            "with_f3912": {
                "model": "REJECTED_leakage_lgbm_with_F3912",
                "pr_auc_mean": entry["pr_auc_mean"],
                "pr_auc_std": entry["pr_auc_std"],
                "status": "REJECTED LEAKAGE - never used",
            },
            "note": "F3912 is quarantined; this comparison exists only as evidence "
                    "of why. The inflated number is not a model result.",
        }
        save_json(ablation, settings.METRICS_DIR / "with_vs_without_f3912.json")
        summary_lines.append(
            f"[REJECTED LEAKAGE] with F3912: PR-AUC {entry['pr_auc_mean']:.4f} "
            f"vs clean {clean['pr_auc_mean']:.4f}"
        )

    print("BASELINES DONE\n" + "\n".join(summary_lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["baselines"])
    ap.add_argument("--repeats", type=int, default=None,
                    help="limit CV repeats (default: all saved repeats)")
    ap.add_argument("--no-ablation", action="store_true")
    args = ap.parse_args()
    if args.command == "baselines":
        run_baselines(args.repeats, include_leakage_ablation=not args.no_ablation)
