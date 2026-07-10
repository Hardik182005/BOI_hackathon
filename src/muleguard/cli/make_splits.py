"""Create the immutable locked-test and CV fold files.

Refuses to overwrite existing split files unless --force is given (the locked
test must never drift once baselines have seen the dev set).
"""
from __future__ import annotations

import argparse

from muleguard import settings
from muleguard.data import ingest, split as split_mod
from muleguard.features.preprocessing import load_quarantine_list
from muleguard.logging import get_logger

log = get_logger("cli.make_splits")


def main(force: bool = False) -> None:
    cfg = settings.load_config("data")["splits"]
    locked_path = settings.REPO_ROOT / cfg["files"]["locked_test"]
    if locked_path.exists() and not force:
        raise SystemExit(
            f"{locked_path} already exists - splits are immutable. "
            "Use --force only if you understand this invalidates all results."
        )
    df = ingest.load_dataset()
    quarantined = load_quarantine_list()
    meta = split_mod.build_all_splits(df, quarantined)
    lt = meta["locked_test"]
    print(
        f"SPLITS OK dev={lt['n_dev_rows']} (pos={lt['n_dev_positives']}) "
        f"test={lt['n_test_rows']} (pos={lt['n_test_positives']}) "
        f"prevalence dev={lt['dev_prevalence']:.4%} test={lt['test_prevalence']:.4%} "
        f"dup_groups={meta['duplicate_feature_rows']['n_groups_gt1']} "
        f"cv={meta['cv']['n_repeats']}x{meta['cv']['n_splits']}"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    main(force=ap.parse_args().force)
