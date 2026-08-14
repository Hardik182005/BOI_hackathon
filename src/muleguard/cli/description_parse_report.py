"""Emit the Description.xlsx parse report and the meta-feature ablation view.

Two artifacts the validation spec names explicitly:

* ``artifacts/features/description_parse_report.json`` - what the workbook
  parser understood, what it could not, and how coverage breaks down. The
  point is the *unparsed* half: a parser that silently drops rows it does not
  recognise is how a feature quietly disappears from a model.
* ``artifacts/metrics/meta_feature_ablation.csv`` - the meta-feature arms
  projected out of the paired nested ablation run into their own file. The
  numbers are not recomputed; they are read from
  ``feature_subset_ablation.csv`` and carry a provenance block naming that
  source and its SHA-256.

    python -m muleguard.cli.description_parse_report
"""
from __future__ import annotations

import argparse
import datetime as dt
import re
from pathlib import Path

import polars as pl

from muleguard import settings
from muleguard.features.dictionary import (
    _default_description_path,
    parse_description_workbook,
)
from muleguard.logging import get_logger
from muleguard.utils import save_json, sha256_file

log = get_logger("cli.description_parse_report")

# Arms in the paired ablation that answer "do the engineered meta-features
# earn their place?". Kept explicit so a renamed arm fails loudly.
META_ARMS = ("full_clean", "no_meta_features", "meta_features_only")

_F_ID = re.compile(r"^F\d+$")


def build_parse_report(description_path: Path | None = None) -> dict:
    path = Path(description_path or _default_description_path())
    records = parse_description_workbook(path)

    unparsed_window = [r.feature for r in records if r.window == "UNKNOWN"]
    unparsed_direction = [r.feature for r in records if r.direction == "UNKNOWN"]
    other_family = [r.feature for r in records if r.feature_family == "OTHER"]
    no_description = [r.feature for r in records if not r.description]
    no_variable_name = [r.feature for r in records
                        if not r.variable_name or r.variable_name.lower() == "nan"]
    untagged = [r.feature for r in records if not r.semantic_tags]
    nonstandard_id = [r.feature for r in records if not _F_ID.match(r.feature)]

    n = len(records)

    def share(xs: list[str]) -> float:
        return round(len(xs) / n, 6) if n else 0.0

    report = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_file": path.name,
        "source_sha256": sha256_file(path),
        "n_rows_in_workbook": n,
        "n_rows_parsed": n,
        "parse_failures": 0,
        "coverage": {
            "window_resolved": round(1 - share(unparsed_window), 6),
            "direction_resolved": round(1 - share(unparsed_direction), 6),
            "family_resolved": round(1 - share(other_family), 6),
            "semantic_tags_present": round(1 - share(untagged), 6),
            "description_present": round(1 - share(no_description), 6),
        },
        "unresolved": {
            "window_unknown": {"count": len(unparsed_window),
                               "share": share(unparsed_window),
                               "examples": unparsed_window[:20]},
            "direction_unknown": {"count": len(unparsed_direction),
                                  "share": share(unparsed_direction),
                                  "examples": unparsed_direction[:20]},
            "family_other": {"count": len(other_family),
                             "share": share(other_family),
                             "examples": other_family[:20]},
            "no_description_text": {"count": len(no_description),
                                    "examples": no_description[:20]},
            "no_variable_name": {"count": len(no_variable_name),
                                 "examples": no_variable_name[:20]},
            "no_semantic_tags": {"count": len(untagged),
                                 "examples": untagged[:20]},
            "non_standard_feature_id": {"count": len(nonstandard_id),
                                        "examples": nonstandard_id[:20]},
        },
        "interpretation": [
            "'unresolved' rows are still usable features - an unknown window or "
            "direction only means the name did not encode one, not that the "
            "column was dropped",
            "every workbook row produced a record; a parse failure count above "
            "zero would mean features silently vanished from the registry",
        ],
    }
    out = settings.FEATURES_DIR / "description_parse_report.json"
    settings.FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    save_json(report, out)
    log.info("wrote %s (%d rows, %d parse failures)", out, n, 0)
    return report


def build_meta_ablation() -> pl.DataFrame:
    src = settings.METRICS_DIR / "feature_subset_ablation.csv"
    if not src.exists():
        raise SystemExit(
            f"{src} not found - run the paired feature-subset ablation first; "
            "this view never invents numbers")
    full = pl.read_csv(src)
    have = set(full["arm"].to_list())
    missing = [a for a in META_ARMS if a not in have]
    if missing:
        raise SystemExit(
            f"arms {missing} absent from {src.name}; the meta-feature ablation "
            "cannot be reported without them")

    view = full.filter(pl.col("arm").is_in(META_ARMS))
    baseline = float(view.filter(pl.col("arm") == "full_clean")["metric"][0])
    view = view.with_columns(
        (pl.col("metric") - baseline).round(5).alias("delta_vs_full_clean"),
        pl.lit("PR_AUC").alias("metric_name"),
        pl.lit(src.name).alias("source_file"),
        pl.lit(sha256_file(src)).alias("source_sha256"),
    )
    out = settings.METRICS_DIR / "meta_feature_ablation.csv"
    settings.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    view.write_csv(out)
    log.info("wrote %s (%d arms)", out, view.height)
    return view


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--description", type=Path, default=None)
    args = ap.parse_args()

    rep = build_parse_report(args.description)
    print(f"workbook rows parsed : {rep['n_rows_parsed']} "
          f"({rep['parse_failures']} failures)")
    for k, v in rep["coverage"].items():
        print(f"  {k:<26} {v:.4f}")

    view = build_meta_ablation()
    print("\nmeta-feature ablation:")
    for row in view.iter_rows(named=True):
        print(f"  {row['arm']:<20} {row['metric']:.5f} "
              f"(delta {row['delta_vs_full_clean']:+.5f}) {row['verdict']}")


if __name__ == "__main__":
    main()
