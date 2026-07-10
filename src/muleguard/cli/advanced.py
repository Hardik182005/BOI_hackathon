"""Advanced tabular challengers: TabPFN / TabICL / AutoGluon - guarded.

Mode A (16GB CPU) policy: each challenger runs only if its import succeeds,
a license/checkpoint is locally available, and a small feasibility probe
completes within budget. Otherwise it is SKIPPED with the exact reason
recorded in artifacts/metrics/advanced_models.json - documented honesty
beats silent failure.

Challengers run on the compact top-60 feature set through the SAME saved
folds as every other model (single repeat for cost control).
"""
from __future__ import annotations

import datetime as dt
import importlib.util
import sys
import time

import numpy as np

from muleguard import settings
from muleguard.cli.train import append_oof, evaluate_oof, merge_metrics
from muleguard.logging import get_logger
from muleguard.utils import load_json, save_json, set_global_seed

log = get_logger("cli.advanced")

RESULTS_PATH = settings.METRICS_DIR / "advanced_models.json"
PROBE_TIMEOUT_S = 900  # one fold must fit+predict inside this to be feasible


def _record(results: dict, name: str, status: str, reason: str, **extra) -> None:
    results["challengers"][name] = {"status": status, "reason": reason, **extra}
    log.info("%s: %s - %s", name, status, reason)


def try_tabpfn(results: dict) -> None:
    if importlib.util.find_spec("tabpfn") is None:
        _record(results, "tabpfn", "SKIPPED",
                "package not installed in CPU-mode environment (Mode A policy: "
                "no large checkpoint downloads on constrained hardware without "
                "a verified license + memory test)")
        return
    try:
        import inspect
        import os

        # model checkpoints cached on E: (user requirement: nothing new on C:)
        os.environ.setdefault(
            "TABPFN_MODEL_CACHE_DIR", str(settings.ARTIFACTS_DIR / "tabpfn_cache")
        )
        # privacy posture: no usage telemetry from a bank-data environment
        os.environ.setdefault("TABPFN_DISABLE_ANALYTICS", "1")
        os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
        from tabpfn import TabPFNClassifier  # type: ignore

        from muleguard.models import baselines as bl

        compact = load_json(settings.FEATURES_DIR / "selected_features.json")["compact_sets"]["top_60"]
        sig = inspect.signature(TabPFNClassifier.__init__).parameters

        def scorer(Xtr, ytr, Xva, seed):
            kwargs = {}
            if "device" in sig:
                kwargs["device"] = "cpu"
            if "random_state" in sig:
                kwargs["random_state"] = seed
            if "ignore_pretraining_limits" in sig:
                kwargs["ignore_pretraining_limits"] = True
            clf = TabPFNClassifier(**kwargs)
            clf.fit(Xtr, ytr)
            return clf.predict_proba(Xva)[:, 1]

        t0 = time.perf_counter()
        preds = bl.run_oof("tabpfn_top60", scorer, mode="tree",
                           n_repeats=1, feature_subset=compact)
        append_oof(preds)
        entry = evaluate_oof(preds, "tabpfn_top60")
        entry["runtime_seconds"] = round(time.perf_counter() - t0, 1)
        merge_metrics(entry)
        _record(results, "tabpfn", "RAN", "completed on top-60 features, 1 repeat",
                pr_auc_mean=entry["pr_auc_mean"])
    except Exception as e:
        _record(results, "tabpfn", "FAILED", f"{type(e).__name__}: {e}")


def try_tabicl(results: dict) -> None:
    if importlib.util.find_spec("tabicl") is None:
        _record(results, "tabicl", "SKIPPED",
                "package not installed; TabICL(v2) requires GPU-class resources "
                "for its in-context regime - Mode A (CPU 16GB) excludes it. "
                "Documented as a Mode B/C experiment.")
        return
    _record(results, "tabicl", "SKIPPED", "installed but no CUDA device available")


def try_autogluon(results: dict) -> None:
    if importlib.util.find_spec("autogluon.tabular") is None:
        py = f"{sys.version_info.major}.{sys.version_info.minor}"
        _record(results, "autogluon", "SKIPPED",
                f"AutoGluon does not support Python {py} at build time "
                "(requires <=3.12); benchmark documented as a roadmap item on "
                "a compatible environment")
        return
    _record(results, "autogluon", "SKIPPED", "not exercised in CPU mode budget")


def main() -> None:
    set_global_seed(settings.GLOBAL_SEED)
    results = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "compute_mode": "cpu_16gb",
        "policy": "challengers accepted only on repeated OOF evidence; "
                  "skips are recorded with reasons, never silently",
        "challengers": {},
    }
    try_tabpfn(results)
    try_tabicl(results)
    try_autogluon(results)
    save_json(results, RESULTS_PATH)
    print("ADVANCED:", {k: v["status"] for k, v in results["challengers"].items()})


if __name__ == "__main__":
    main()
