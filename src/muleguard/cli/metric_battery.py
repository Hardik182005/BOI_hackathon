"""Compute the §24 - §27 metric battery for one stored set of predictions.

    python -m muleguard.cli.metric_battery \
        --source artifacts/predictions/oof_v2.parquet \
        --model xgboost_top_120 --protocol FLAT

Writes (or merges into) ``artifacts/metrics/metric_battery.json``. Nothing is
trained here: every number comes from prediction arrays already on disk, which
is what makes the command safe to run while a cross-validation job is using the
machine, and what makes it reproducible afterwards.

The predictions source is an argument rather than a constant for one specific
reason. At the time this was first run the full nested cross-validation was
still executing, so the only nested predictions on disk were from a preliminary,
under-powered run. The primary honest estimate for this project is the nested
one, so the command has to be re-runnable against the final nested store the
moment it lands - and every run stamps the source file's hash and modification
time into the artifact so a reader can tell which input produced which numbers.

Each run is stored under its own key (``PROTOCOL:model``) and later runs merge
rather than overwrite, so the flat, nested and reference-holdout views of the
same champion sit in one document and can be read against each other. ``--fresh``
starts the document over.

Sources understood
------------------
long store   ``row_index, repeat, model, target, score`` - one row per account
             per repeat per model. ``--model`` selects; the repeats become the
             rows of the score matrix.
wide store   ``row_index, target, raw_lightgbm, calibrated_risk, ...`` - the
             single-pass holdout format. One repeat, and the calibrated column
             is taken as shipped rather than refitted.

The locked test is treated as a reference split throughout: its metrics are
computed and labelled, and no threshold search is permitted against it.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from muleguard import settings
from muleguard.logging import get_logger
from muleguard.models import metric_battery as mbat
from muleguard.utils import load_json, save_json

log = get_logger("cli.metric_battery")

OUT = settings.METRICS_DIR / "metric_battery.json"
LENS_JSON = settings.METRICS_DIR / "lens_stack_oof_v2.json"
FOLDS = settings.SPLITS_DIR / "cv_folds.parquet"
BUNDLE = settings.MODELS_DIR / "final_bundle.joblib"
PERF_JSON = settings.ARTIFACTS_DIR / "testing" / "performance_results.json"
SEED_VAR = settings.METRICS_DIR / "seed_variance_v2.json"
STRESS = settings.METRICS_DIR / "stability_stress_v2.json"

#: Filename fragments that mean "this is the held-out split". Matching one of
#: them forces the reference protocol and disables the §27 threshold search, so
#: an operator cannot tune against the holdout by forgetting a flag.
HOLDOUT_MARKERS = ("holdout", "locked_test")

PROTOCOLS = ("FLAT", "NESTED", "NESTED_PRELIMINARY", "HOLDOUT_REFERENCE")


def _sha256(path: Path) -> str:
    """Content hash of a predictions store, streamed so a large parquet is fine."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def _source_provenance(path: Path) -> dict[str, Any]:
    """Everything needed to tell two runs of this command apart.

    A parquet file carries no ``generated_utc`` of its own, so the modification
    time and content hash stand in for it, and where the store has a companion
    metrics file that *does* record one it is quoted as well. Without this a
    reader cannot tell a preliminary nested run from the final one, and those
    two produce very different numbers.
    """
    stat = path.stat()
    companion = {
        "oof_v2.parquet": settings.METRICS_DIR / "tournament_v2.json",
        "nested_oof.parquet": settings.METRICS_DIR / "nested_cv.json",
    }.get(path.name)
    generated = None
    if companion is not None and companion.exists():
        try:
            generated = load_json(companion).get("generated_utc")
        except Exception:                                    # pragma: no cover
            generated = None
    return {
        "path": str(path),
        "sha256_prefix": _sha256(path)[:16],
        "size_bytes": int(stat.st_size),
        "modified_utc": dt.datetime.fromtimestamp(
            stat.st_mtime, dt.timezone.utc).isoformat(timespec="seconds"),
        "companion_metrics_file": str(companion) if companion else None,
        "companion_generated_utc": generated,
    }


def _load_long(df: pl.DataFrame, model: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Row index, labels and an ``(n_repeats, n)`` score matrix for one model."""
    sub = df.filter(pl.col("model") == model)
    if sub.is_empty():
        raise SystemExit(f"no rows stored for model {model!r}; available: "
                         f"{sorted(df['model'].unique().to_list())}")
    repeats = sorted(sub["repeat"].unique().to_list())
    rows, y, ri = [], None, None
    for r in repeats:
        s = sub.filter(pl.col("repeat") == r).sort("row_index")
        rows.append(s["score"].to_numpy())
        if y is None:
            y, ri = s["target"].to_numpy(), s["row_index"].to_numpy()
        elif not np.array_equal(ri, s["row_index"].to_numpy()):
            raise SystemExit(f"repeat {r} covers different accounts than repeat "
                             f"{repeats[0]}; the store is inconsistent")
    return ri, np.asarray(y).astype(int), np.vstack(rows)


def _load_wide(df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Row index, labels, a one-row score matrix and the shipped calibrated column."""
    s = df.sort("row_index")
    score_col = next((c for c in ("raw_lightgbm", "score", "raw_score")
                      if c in s.columns), None)
    if score_col is None:
        raise SystemExit(f"no raw score column found in {s.columns}")
    cal = s["calibrated_risk"].to_numpy() if "calibrated_risk" in s.columns else None
    return (s["row_index"].to_numpy(), s["target"].to_numpy().astype(int),
            s[score_col].to_numpy().reshape(1, -1), cal)


def _fold_ids(row_index: np.ndarray, n_repeats: int) -> np.ndarray | None:
    """Outer-fold labels per account per repeat, if the split file covers them.

    Joined on ``row_index`` rather than filtered by membership, so a mismatch
    shows up as missing rows instead of a silently reordered array.
    """
    if not FOLDS.exists():
        return None
    folds = pl.read_parquet(FOLDS)
    cols = [c for c in folds.columns if c.startswith("repeat_")]
    if len(cols) < n_repeats:
        return None
    want = pl.DataFrame({"row_index": np.asarray(row_index, dtype=np.int64)})
    joined = want.join(folds.with_columns(pl.col("row_index").cast(pl.Int64)),
                       on="row_index", how="left")
    if joined.height != len(row_index) or joined[cols[0]].null_count():
        log.warning("fold ids do not cover every scored account; fold-level "
                    "PR-AUC will be skipped")
        return None
    return np.vstack([joined[cols[r]].to_numpy() for r in range(n_repeats)])


def _calibrated(
    y: np.ndarray, S: np.ndarray, mode: str, shipped: np.ndarray | None
) -> tuple[np.ndarray | None, str]:
    """The probability vector the frozen thresholds are expressed on.

    ``crossfit`` is the default because ``policy_version 1.0`` was read off a
    cross-fitted calibration of exactly these repeat-averaged scores; using it
    means the thresholds land where they landed when they were frozen. ``bundle``
    applies the shipped calibrator instead, which is what the scoring API emits
    and therefore differs by a few accounts at each band. ``shipped`` takes a
    calibrated column that is already stored, which is the only correct option
    for the holdout file - recalibrating a held-out split against its own labels
    would be fitting on it.
    """
    if mode == "none":
        return None, "none"
    if mode == "shipped":
        if shipped is None:
            raise SystemExit("--calibration shipped: no calibrated_risk column")
        return np.asarray(shipped, dtype=float), "shipped_column"
    raw = mbat._repeat_mean(S)
    if mode == "bundle":
        if not BUNDLE.exists():
            raise SystemExit(f"--calibration bundle: {BUNDLE} not found")
        import joblib

        b = joblib.load(BUNDLE)
        cal = b.get("calibrator")
        if cal is None:
            raise SystemExit("bundle contains no calibrator")
        return (np.asarray(cal.predict(raw), dtype=float),
                f"frozen_bundle_{b.get('calibrator_method', 'unknown')}")
    from muleguard.models.calibration import crossfit_calibrated, select_calibrator

    winner = select_calibrator(raw, y, seed=settings.GLOBAL_SEED)["winner"]
    return (crossfit_calibrated(raw, y, winner, seed=settings.GLOBAL_SEED),
            f"crossfit_{winner}")


def _operational() -> dict[str, Any]:
    """§24's operational block, measured where it can be and cited where it cannot.

    Model size is a file stat and is therefore fresh. Latency is not: the
    recorded measurement was taken on an idle machine, and re-running it while a
    nested cross-validation is saturating every core would produce a number that
    describes the load, not the model. A stale measurement labelled stale is more
    useful than a fresh measurement that is wrong, so the recorded figures are
    quoted with their date and a note - including the fact that the bundle on
    disk today is not the same size as the bundle they were taken against.
    """
    out: dict[str, Any] = {"bundle_path": str(BUNDLE)}
    if BUNDLE.exists():
        st = BUNDLE.stat()
        out["bundle_bytes_now"] = int(st.st_size)
        out["bundle_mb_now"] = round(st.st_size / 1e6, 6)
        out["bundle_modified_utc"] = dt.datetime.fromtimestamp(
            st.st_mtime, dt.timezone.utc).isoformat(timespec="seconds")
    if PERF_JSON.exists():
        perf = load_json(PERF_JSON)
        out["latency_seconds_recorded"] = dict(perf.get("latency_seconds", {}))
        out["batch_rows_per_second_recorded"] = perf.get(
            "batch_rows_per_second_locked_test")
        out["recorded_model_bundle_mb"] = perf.get("model_bundle_mb")
        out["latency_source"] = str(PERF_JSON)
        out["latency_generated_utc"] = perf.get("generated_utc")
        recorded_mb = perf.get("model_bundle_mb")
        now_mb = out.get("bundle_mb_now")
        out["recorded_size_matches_bundle_on_disk"] = (
            None if recorded_mb is None or now_mb is None
            else abs(float(recorded_mb) - float(now_mb)) < 0.01)
        out["latency_measurement_note"] = (
            "Not re-measured for this battery. Every core on this machine is "
            "occupied by a running cross-validation job, so a fresh timing would "
            "measure the load rather than the model. The recorded figures were "
            "taken on 2026-07-10 against a bundle of a different size from the "
            "one on disk today, so they describe an earlier build of the same "
            "pipeline and are quoted, not claimed as current.")
    out["peak_ram_note"] = (
        "Peak resident memory was not instrumented in this run; adding a "
        "measurement here would mean inventing one.")
    return out


def _referenced_stability() -> dict[str, Any]:
    """Seed and positive-removal stability, quoted from the artifacts that hold them.

    §24 asks for both. Neither can be recomputed from stored predictions - one
    needs new fits under new seeds, the other needs refits with mules removed -
    and this battery does not train. So they are read from the artifacts that
    measured them, with their paths attached, and marked as quoted rather than
    recomputed.
    """
    out: dict[str, Any] = {}
    if SEED_VAR.exists():
        d = load_json(SEED_VAR)
        out["seed_variance"] = {
            "source": str(SEED_VAR),
            "recomputed_here": False,
            **{k: d[k] for k in
               ("n_seeds", "n_repeats", "pr_auc_mean", "pr_auc_std", "pr_auc_min",
                "pr_auc_max", "spread", "mean", "std", "min", "max")
               if k in d},
        }
    if STRESS.exists():
        d = load_json(STRESS)
        out["positive_removal"] = {"source": str(STRESS), "recomputed_here": False,
                                   **{k: v for k, v in d.items()
                                      if not isinstance(v, (list, dict))}}
        for key in ("positive_removal", "prediction_rank_stability",
                    "feature_rank_stability"):
            if isinstance(d.get(key), dict):
                out.setdefault("detail", {})[key] = d[key]
    return out


def _holdout_admissibility(observed_pr_auc: float | None,
                           observed_calibrated_pr_auc: float | None = None) -> dict[str, Any]:
    """What the stored holdout vector actually is, and what may be quoted from it.

    This block exists because of a trap that is easy to fall into and expensive
    to fall into. ``artifacts/predictions/holdout_predictions.parquet`` is the
    only per-row holdout vector in the repository, and it does **not** belong to
    the served champion - it is the retired pre-firewall stack. Its two figures
    are the highest numbers anywhere in the project, and both were ruled
    inadmissible: one belongs to a model whose feature pool predates the leakage
    firewall, and neither was produced by the model that ships.

    The champion's own holdout result was produced under a sealed protocol that
    revealed summary metrics and a prediction hash but not the predictions
    themselves, so it can be quoted and cannot be given an interval. Both facts
    are recorded here rather than left for a reader to discover.

    Provenance is established by matching against both stored columns, not one.
    The parquet carries a raw score and a shipped calibrated score, and the
    retired run's recorded figure was computed on the calibrated one; comparing
    only the raw column would report "matches nothing" and leave the ownership
    of the vector looking merely unknown rather than positively identified.
    """
    hm = load_json(settings.METRICS_DIR / "holdout_metrics.json") if (
        settings.METRICS_DIR / "holdout_metrics.json").exists() else {}
    champ = hm.get("current_champion", {})
    retired = hm.get("retired_run", {}).get("metrics", {})
    retired_ap = (retired.get("pr_auc") or {}).get("point")
    champ_ap = champ.get("pr_auc")

    def close(a: Any, b: Any) -> bool:
        return (a is not None and b is not None
                and abs(float(a) - float(b)) < 5e-3)

    matches_retired_raw = close(observed_pr_auc, retired_ap)
    matches_retired_cal = close(observed_calibrated_pr_auc, retired_ap)
    matches_retired = matches_retired_raw or matches_retired_cal
    matches_champion = (close(observed_pr_auc, champ_ap)
                        or close(observed_calibrated_pr_auc, champ_ap))
    return {
        "ruling": "HISTORICAL HOLDOUT - FOR REFERENCE ONLY",
        "ruling_source": str(settings.DOCS_DIR / "LOCKED_TEST_RULING.md"),
        "stored_vector_belongs_to": (
            "the served champion" if matches_champion
            else "the retired pre-firewall stack, not the served champion"),
        "stored_vector_pr_auc_observed": observed_pr_auc,
        "stored_calibrated_column_pr_auc_observed": observed_calibrated_pr_auc,
        "champion_holdout_pr_auc_recorded": champ_ap,
        "retired_run_pr_auc_recorded": retired_ap,
        "observed_matches_champion_figure": bool(matches_champion),
        "observed_matches_retired_run": bool(matches_retired),
        "matched_on_column": (
            "calibrated_risk" if matches_retired_cal
            else "raw_score" if matches_retired_raw else None),
        "identification": (
            "The stored calibrated column reproduces the retired run's recorded "
            "PR-AUC, which identifies the vector positively rather than leaving "
            "its ownership unknown." if matches_retired_cal else
            "Neither stored column reproduces a recorded figure, so the vector "
            "could not be positively identified; it is treated as inadmissible "
            "because it is demonstrably not the champion's."),
        "stored_vector_admissible_for_reporting": bool(matches_champion),
        "why_inadmissible": (
            "The stored per-row holdout vector was produced by the retired "
            "generation-1 scorer on a pre-firewall feature pool. It is the "
            "highest-scoring set of numbers in the project and it describes a "
            "model that was rejected, so quoting it as the system's holdout "
            "performance would be selecting the most flattering historical "
            "number - exactly what the working discipline forbids."),
        "reportable_champion_figure": {
            "model": champ.get("model"),
            "pr_auc": champ.get("pr_auc"),
            "roc_auc": champ.get("roc_auc"),
            "at_budget": champ.get("at_budget"),
            "source": champ.get("source"),
            "prediction_sha256": champ.get("prediction_sha256"),
            "interval_computable": False,
            "why_no_interval": (
                "The champion was scored under a sealed protocol that revealed "
                "summary metrics and a prediction hash only. No per-row score "
                "vector is stored, so no bootstrap interval can be computed for "
                "it without re-scoring the holdout - which would be a second "
                "touch of a split that is already ruled reference-only."),
            "recomputed_here": False,
        },
    }


def _protocol_for(source: Path, requested: str | None) -> tuple[str, bool]:
    """Resolve the protocol label and whether a threshold search is permitted."""
    name = source.name.lower()
    if any(m in name for m in HOLDOUT_MARKERS):
        if requested and requested != "HOLDOUT_REFERENCE":
            log.warning("source %s is a held-out split; overriding --protocol %s "
                        "with HOLDOUT_REFERENCE", source.name, requested)
        return "HOLDOUT_REFERENCE", False
    if requested:
        return requested, True
    return ("NESTED_PRELIMINARY" if "nested" in name else "FLAT"), True


def main() -> None:
    ap = argparse.ArgumentParser(
        description="compute the §24-§27 metric battery from stored predictions")
    ap.add_argument("--source", required=True, type=Path,
                    help="parquet store of predictions (long or wide format)")
    ap.add_argument("--model", default=None,
                    help="model name inside a long-format store; defaults to the "
                         "champion recorded in lens_stack_oof_v2.json")
    ap.add_argument("--protocol", default=None, choices=PROTOCOLS,
                    help="how these predictions were produced; a held-out source "
                         "always forces HOLDOUT_REFERENCE")
    ap.add_argument("--label", default=None, help="human description of the split")
    ap.add_argument("--calibration", default="crossfit",
                    choices=("crossfit", "bundle", "shipped", "none"))
    ap.add_argument("--n-boot", type=int, default=mbat.N_BOOT_DEFAULT)
    ap.add_argument("--alpha", type=float, default=mbat.ALPHA)
    ap.add_argument("--seed", type=int, default=settings.GLOBAL_SEED)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--fresh", action="store_true",
                    help="discard existing runs in the artifact instead of merging")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    source: Path = args.source if args.source.is_absolute() else (
        settings.REPO_ROOT / args.source)
    if not source.exists():
        raise SystemExit(f"no such predictions store: {source}")

    lens = load_json(LENS_JSON) if LENS_JSON.exists() else {}
    frozen = dict(lens.get("policy_thresholds", {}))
    protocol, allow_search = _protocol_for(source, args.protocol)

    df = pl.read_parquet(source)
    shipped_cal = None
    if "model" in df.columns:
        model = args.model or str(lens.get("winner", ""))
        if not model:
            raise SystemExit("--model is required for a long-format store")
        ri, y, S = _load_long(df, model)
    else:
        model = args.model or "final_bundle_lightgbm"
        ri, y, S, shipped_cal = _load_wide(df)
    n, n_pos = len(y), int(y.sum())
    log.info("source=%s model=%s protocol=%s rows=%d positives=%d repeats=%d",
             source.name, model, protocol, n, n_pos, S.shape[0])

    cal_mode = args.calibration
    if protocol == "HOLDOUT_REFERENCE" and cal_mode == "crossfit" and shipped_cal is not None:
        log.info("held-out split: using the stored calibrated_risk column rather "
                 "than refitting a calibrator against held-out labels")
        cal_mode = "shipped"
    calibrated, cal_desc = _calibrated(y, S, cal_mode, shipped_cal)

    fold_ids = _fold_ids(ri, S.shape[0]) if protocol != "HOLDOUT_REFERENCE" else None

    doc = mbat.build_battery(
        y=y, S=S, calibrated=calibrated, fold_ids=fold_ids,
        frozen_thresholds=frozen, n_boot=int(args.n_boot), seed=int(args.seed),
        alpha=float(args.alpha),
        split_label=args.label or f"{source.stem} ({protocol})",
        protocol=protocol, allow_threshold_search=allow_search,
    )
    doc["model"] = model
    if protocol == "HOLDOUT_REFERENCE":
        observed = (doc["ranking"]["pr_auc"]["mean"]
                    if doc["ranking"].get("defined") else None)
        observed_cal = (doc.get("aggregation_reconciliation") or {}).get(
            "ap_of_calibrated_mean_score")
        doc["holdout_admissibility"] = _holdout_admissibility(observed, observed_cal)
        if not doc["holdout_admissibility"]["stored_vector_admissible_for_reporting"]:
            log.warning("this holdout vector (PR-AUC %.5f) is NOT the served "
                        "champion's (recorded %.5f); it is the retired "
                        "pre-firewall stack and is marked inadmissible - do not "
                        "quote it as system performance", observed or 0.0,
                        doc["holdout_admissibility"]["champion_holdout_pr_auc_recorded"]
                        or 0.0)
    doc["operational"] = _operational()
    doc["stability"].update(_referenced_stability())
    doc["provenance"] = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "generator": "muleguard.cli.metric_battery",
        "predictions_source": _source_provenance(source),
        "calibration": cal_desc,
        "frozen_policy_source": str(LENS_JSON),
        "retraining_performed": False,
        "thresholds_modified": False,
        "command": (f"python -m muleguard.cli.metric_battery --source "
                    f"{args.source} --model {model} --protocol {protocol} "
                    f"--calibration {cal_mode} --n-boot {args.n_boot}"),
    }
    doc["interpretation"] = _interpretation(protocol)

    r = doc["ranking"]
    if r.get("defined"):
        iv = doc["intervals"]["stratified"]["pr_auc"]
        iv2 = doc["intervals"]["resample_accounts"]["pr_auc"]
        log.info("PR-AUC %.5f  per-repeat sd %.5f  stratified CI [%.5f, %.5f]  "
                 "account CI [%.5f, %.5f]",
                 r["pr_auc"]["mean"], r["pr_auc"]["std"],
                 iv["ci_low"], iv["ci_high"], iv2["ci_low"], iv2["ci_high"])
    for row in (doc.get("at_frozen_thresholds") or []):
        log.info("%-16s thr %.5f -> alerts %5d  recall %.4f  precision %.4f  "
                 "F1 %.4f  MCC %.4f", row["tier"], row["threshold"], row["alerts"],
                 row["recall"] or 0.0, row["precision"] or 0.0,
                 row["f1"] or 0.0, row["mcc"] or 0.0)

    if args.dry_run:
        log.info("dry run: %s NOT written", args.out)
        return

    key = f"{protocol}:{model}"
    existing: dict[str, Any] = {}
    if args.out.exists() and not args.fresh:
        try:
            existing = load_json(args.out)
        except Exception:                                    # pragma: no cover
            log.warning("could not read %s; starting a fresh document", args.out)
    runs = dict(existing.get("runs", {}))
    runs[key] = doc
    save_json({
        "schema_version": mbat.SCHEMA_VERSION,
        "written_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "primary_run_key": _primary_key(runs),
        "run_keys": sorted(runs),
        "reading_order": [
            "NESTED is the primary honest estimate for this project (§9).",
            "FLAT is a historical development figure whose feature selection "
            "pooled information across all development folds; it reads high.",
            "HOLDOUT_REFERENCE is reportable only as a labelled historical "
            "reference and was not used to select or tune anything.",
        ],
        "runs": runs,
    }, args.out)
    log.info("wrote %s (run key %s, %d run(s) in the document)",
             args.out, key, len(runs))


def _primary_key(runs: dict[str, Any]) -> str | None:
    """Which stored run a reader should quote as the headline.

    Nested beats flat beats reference, and a preliminary nested run does not
    outrank a flat one - an under-powered nested estimate is not more honest
    than a well-powered optimistic one, it is just differently wrong, and saying
    which is which is the point of storing both.
    """
    for prefix in ("NESTED:", "FLAT:", "NESTED_PRELIMINARY:", "HOLDOUT_REFERENCE:"):
        for k in sorted(runs):
            if k.startswith(prefix):
                return k
    return None


def _interpretation(protocol: str) -> dict[str, Any]:
    """The sentence that has to travel with the numbers."""
    base = {
        "FLAT": {
            "status": "HISTORICAL DEVELOPMENT ESTIMATE",
            "usable_for_selection": True,
            "reading": ("Feature selection pooled importance across every "
                        "development fold, so a row helped choose the columns "
                        "before it was scored as held out. The optimism this "
                        "introduces is small in expectation but not zero at 64 "
                        "positives; the nested figure is the one to quote."),
        },
        "NESTED": {
            "status": "PRIMARY HONEST ESTIMATE",
            "usable_for_selection": True,
            "reading": ("Selection, tuning and the feature-set size were all "
                        "decided inside the outer training partition. This is the "
                        "number §9 asks for."),
        },
        "NESTED_PRELIMINARY": {
            "status": "SUPERSEDED - UNDER-POWERED NESTED RUN",
            "usable_for_selection": False,
            "reading": ("A single repeat with a very small tuning budget. It is "
                        "stored so the final nested run can be compared against "
                        "it, and it must not be quoted as the nested result."),
        },
        "HOLDOUT_REFERENCE": {
            "status": "HISTORICAL HOLDOUT - FOR REFERENCE ONLY",
            "usable_for_selection": False,
            "reading": ("This split was touched before the current protocol was "
                        "adopted, so it is reported as a labelled historical "
                        "reference and never as a selection or tuning signal. No "
                        "threshold search is run against it."),
        },
    }
    return base.get(protocol, {"status": "UNLABELLED", "usable_for_selection": False,
                               "reading": "protocol not recognised"})


if __name__ == "__main__":
    main()
