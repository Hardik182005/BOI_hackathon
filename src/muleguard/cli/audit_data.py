"""End-to-end data audit CLI.

Steps (all idempotent, artifacts land in known folders):
  1. read-only raw copy + SHA-256
  2. one-time XLSX -> Parquet conversion (+schema)
  3. independent-engine validation (openpyxl streaming pass)
  4. dataset fingerprint JSON
  5. per-feature profile
  6. index-candidate detection
  7. leakage audit (corr / CV single-feature PR-AUC / MI / reconstruction)
  8. quarantine list -> configs/leakage_quarantine.yaml (+ artifacts JSON)
  9. DATA_AUDIT_REPORT.md and LEAKAGE_AUDIT.md (artifacts/reports + docs)

Usage: python -m muleguard.cli.audit_data [--skip-validate] [--force-convert]
"""
from __future__ import annotations

import argparse
import datetime as dt

import numpy as np
import polars as pl
import yaml

from muleguard import settings
from muleguard.data import ingest, leakage, profile as prof_mod
from muleguard.logging import get_logger
from muleguard.utils import save_json, timer

log = get_logger("cli.audit_data")


def detect_index_candidates(df: pl.DataFrame) -> list[str]:
    out: list[str] = []
    n = df.height
    for c in df.columns:
        if c == settings.TARGET_COLUMN:
            continue
        name_flag = (
            c.strip().lower() in {"", "index", "id", "unnamed: 0", "sr", "sno", "s.no"}
            or c.startswith("__UNNAMED__")
        )
        seq_flag = False
        s = df[c]
        if s.dtype.is_numeric() and s.null_count() == 0:
            arr = s.cast(pl.Float64).to_numpy()
            seq_flag = bool(
                np.array_equal(arr, np.arange(n, dtype=np.float64))
                or np.array_equal(arr, np.arange(1, n + 1, dtype=np.float64))
            )
        if name_flag or seq_flag:
            out.append(c)
    return out


def _write_markdown_reports(
    df: pl.DataFrame,
    prof: pl.DataFrame,
    prof_summary: dict,
    audit: pl.DataFrame,
    leak_summary: dict,
    quarantine: dict,
    fingerprint: dict,
    index_candidates: list[str],
) -> None:
    tgt = fingerprint["target_distribution"]
    prevalence = fingerprint["positive_prevalence"]

    def md_table(rows: list[dict], cols: list[str]) -> str:
        head = "| " + " | ".join(cols) + " |\n|" + "|".join("---" for _ in cols) + "|\n"
        body = "".join(
            "| " + " | ".join(str(r.get(c, "")) for c in cols) + " |\n" for r in rows
        )
        return head + body

    top_corr = (
        audit.with_columns(pl.col("target_corr").abs().alias("abs_corr"))
        .sort("abs_corr", descending=True).head(15)
        .select(["feature", "target_corr", "single_feature_cv_pr_auc",
                 "mutual_information_bits", "suspicious"])
        .with_columns([pl.col(c).round(4) for c in
                       ["target_corr", "single_feature_cv_pr_auc", "mutual_information_bits"]])
        .to_dicts()
    )
    cat_cols = [c for c in df.columns if not df.schema[c].is_numeric()]
    cat_rows = prof.filter(pl.col("feature").is_in(cat_cols)).select(
        ["feature", "dtype", "n_unique", "missing_rate", "top_values"]
    ).to_dicts()

    data_report = f"""# Data Audit Report

Generated {dt.datetime.now(dt.timezone.utc).isoformat()} by `muleguard.cli.audit_data`.
Dataset fingerprint: raw SHA-256 `{fingerprint['raw_file']['sha256'][:16]}…`, parquet SHA-256 `{fingerprint['parquet_file']['sha256'][:16]}…`.
All numbers below are **measured from the raw file by this pipeline** (split: full dataset — audit only, no modelling decision selects features from these numbers).

## Shape and target

| Item | Value |
|---|---|
| Rows (accounts) | {fingerprint['n_rows']:,} |
| Columns (total incl. target) | {fingerprint['n_cols']:,} |
| Target column | `{settings.TARGET_COLUMN}` |
| Positives (1 = suspicious/mule) | {tgt['positives_1']:,} |
| Negatives (0) | {tgt['negatives_0']:,} |
| Null targets | {tgt['null']:,} |
| Positive prevalence | {prevalence:.4%} (~1 : {round((tgt['negatives_0']) / max(tgt['positives_1'], 1))}) |
| Independent engine validation | {"PASSED" if fingerprint['independent_validation']['passed'] else "FAILED: " + "; ".join(fingerprint['independent_validation']['problems'])} |

## Data quality

| Item | Value |
|---|---|
| Missing cells (features) | {prof_summary['n_missing_cells']:,} ({prof_summary['missing_cell_rate']:.2%}) |
| Columns with any missing | {prof_summary['n_cols_with_missing']:,} |
| Constant columns | {prof_summary['n_constant']:,} |
| Quasi-constant columns (≥99% one value or missing) | {prof_summary['n_quasi_constant']:,} |
| Exact duplicate columns | {prof_summary['n_exact_duplicate_cols']:,} |
| Dtypes | {prof_summary['dtype_counts']} |
| Index-like columns detected | {index_candidates or 'none'} |

## Non-numeric / interpretable columns

{md_table(cat_rows, ["feature", "dtype", "n_unique", "missing_rate", "top_values"]) if cat_rows else "None detected."}

## Strongest single-feature signals (top 15 by |corr|)

Single-feature PR-AUC is **cross-validated** (direction chosen on train folds, scored on held-out folds).

{md_table(top_corr, ["feature", "target_corr", "single_feature_cv_pr_auc", "mutual_information_bits", "suspicious"])}

Full per-feature table: `artifacts/reports/data_profile.csv` and `artifacts/reports/leakage_audit_table.csv`.
"""

    flagged_rows = [
        {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()
         if k in ("feature", "target_corr", "single_feature_cv_pr_auc",
                  "label_reconstruction_rate", "flag_high_corr", "flag_high_single_ap",
                  "flag_label_copy", "flag_perfect_separation", "flag_identifier")}
        for r in leak_summary["flagged"]
    ]
    leak_report = f"""# Leakage Audit

Generated {dt.datetime.now(dt.timezone.utc).isoformat()} by `muleguard.cli.audit_data`.

## Method

For **every numeric feature**: pairwise-complete correlation with `{settings.TARGET_COLUMN}`,
**cross-validated** single-feature PR-AUC ({settings.load_config('data')['audit']['single_feature_cv_folds']}-fold, direction fixed on train folds),
quantile-binned mutual information (missing as own bin), exact label-reconstruction
check on two-valued columns, near-perfect separation and identifier-likeness checks.
Audit statistics are computed on the full file **for safety screening only** — they are
used to *exclude* features, never to select them (selection happens inside CV folds).

Flag thresholds: |corr| ≥ {leak_summary['thresholds']['abs_corr']}, single-feature CV PR-AUC ≥ {leak_summary['thresholds']['single_feature_cv_ap']}.

## Result: {leak_summary['n_flagged']} feature(s) flagged

{md_table(flagged_rows, ["feature", "target_corr", "single_feature_cv_pr_auc", "label_reconstruction_rate", "flag_high_corr", "flag_high_single_ap", "flag_label_copy", "flag_perfect_separation", "flag_identifier"]) if flagged_rows else "No feature crossed the flag thresholds."}

## Quarantine (mandatory + flagged)

{md_table(quarantine['quarantine'], ["feature", "reason", "disposition"])}

Policy: {quarantine['policy']}

The with/without-F3912 ablation (evidence that the quarantine matters) is produced by
the baseline phase: `artifacts/metrics/with_vs_without_f3912.json` and
`artifacts/plots/leakage_ablation.png` — the F3912 run is labelled **REJECTED LEAKAGE**.
"""

    for name, content in (("DATA_AUDIT_REPORT.md", data_report), ("LEAKAGE_AUDIT.md", leak_report)):
        for base in (settings.REPORTS_DIR, settings.DOCS_DIR):
            (base / name).write_text(content, encoding="utf-8")
    log.info("markdown reports written to artifacts/reports and docs")


def main(skip_validate: bool = False, force_convert: bool = False) -> None:
    settings.ensure_dirs()
    timings: dict[str, float] = {}

    with timer("raw_copy", timings):
        raw_info = ingest.ensure_raw_copy()
    with timer("convert", timings):
        ingest.convert_to_parquet(force=force_convert)
    df = ingest.load_dataset()

    if skip_validate:
        fp_path = settings.REPO_ROOT / settings.load_config("data")["interim"]["fingerprint_json"]
        if fp_path.exists():
            from muleguard.utils import load_json
            validation = load_json(fp_path)["independent_validation"]
        else:
            raise SystemExit("--skip-validate requires an existing fingerprint")
    else:
        with timer("independent_validation", timings):
            validation = ingest.validate_against_independent_engine()
        if not validation["passed"]:
            raise SystemExit(f"CONVERSION VALIDATION FAILED: {validation['problems']}")

    with timer("fingerprint", timings):
        fingerprint = ingest.build_fingerprint(raw_info, validation)

    with timer("profile", timings):
        prof = prof_mod.profile_dataset(df)
        prof_summary = prof_mod.summarize_profile(prof, df)
    prof.write_parquet(settings.REPORTS_DIR / "data_profile.parquet")
    prof.write_csv(settings.REPORTS_DIR / "data_profile.csv")
    save_json(prof_summary, settings.REPORTS_DIR / "data_profile_summary.json")

    index_candidates = detect_index_candidates(df)

    with timer("leakage_audit", timings):
        audit, leak_summary = leakage.run_leakage_audit(df)
    audit.write_parquet(settings.REPORTS_DIR / "leakage_audit_table.parquet")
    audit.write_csv(settings.REPORTS_DIR / "leakage_audit_table.csv")
    save_json(leak_summary, settings.REPORTS_DIR / "leakage_audit_summary.json")

    quarantine = leakage.build_quarantine(audit, df, index_candidates)
    with open(settings.CONFIG_DIR / "leakage_quarantine.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(quarantine, fh, sort_keys=False, allow_unicode=True)
    save_json(quarantine, settings.FEATURES_DIR / "quarantined_features.json")

    _write_markdown_reports(
        df, prof, prof_summary, audit, leak_summary, quarantine, fingerprint, index_candidates
    )
    save_json(timings, settings.REPORTS_DIR / "audit_timings.json")

    print(
        f"AUDIT OK rows={fingerprint['n_rows']} cols={fingerprint['n_cols']} "
        f"pos={fingerprint['target_distribution']['positives_1']} "
        f"neg={fingerprint['target_distribution']['negatives_0']} "
        f"prevalence={fingerprint['positive_prevalence']:.4%} "
        f"constant={prof_summary['n_constant']} quasi={prof_summary['n_quasi_constant']} "
        f"dup_cols={prof_summary['n_exact_duplicate_cols']} "
        f"flagged={leak_summary['n_flagged']} quarantined={len(quarantine['quarantine'])} "
        f"validation={'PASS' if fingerprint['independent_validation']['passed'] else 'FAIL'}"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-validate", action="store_true")
    ap.add_argument("--force-convert", action="store_true")
    args = ap.parse_args()
    main(skip_validate=args.skip_validate, force_convert=args.force_convert)
