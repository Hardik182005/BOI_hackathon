"""Build the dev calibrated-risk reference used to rank a new score.

``risk_percentile`` answers "where does this account sit against accounts we
have already seen?". That question needs a stated reference distribution, and
the only distribution allowed to serve as one is DEV - the locked test may not
be read for any purpose, including this.

This is inference over the frozen bundle. Nothing is fitted, and the model is
not touched. Run it after any bundle change:

    python -m muleguard.cli.build_risk_reference
"""
from __future__ import annotations

import argparse
import datetime as dt

import numpy as np

from muleguard import settings
from muleguard.data.split import load_locked_test_mask
from muleguard.features.frame import raw_with_meta
from muleguard.logging import get_logger
from muleguard.models.scoring import RISK_REFERENCE_PATH, load_bundle, score_rows
from muleguard.utils import save_json

log = get_logger("cli.build_risk_reference")

CHUNK = 512


def build(max_grid: int = 4000) -> dict:
    bundle = load_bundle()
    frame = raw_with_meta()
    locked = load_locked_test_mask()
    if len(locked) != frame.height:
        raise SystemExit(
            f"locked-test mask covers {len(locked)} rows but the frame has "
            f"{frame.height}; splits and dataset are out of step")

    dev_idx = np.flatnonzero(~locked)
    dev = frame[dev_idx.tolist()]
    log.info("scoring %d dev rows (locked test excluded)", dev.height)

    scores: list[float] = []
    for start in range(0, dev.height, CHUNK):
        chunk = dev.slice(start, CHUNK)
        recs = score_rows(chunk, bundle=bundle, with_explanations=False,
                          with_counterfactual=False)
        scores.extend(float(r["calibrated_risk"]) for r in recs)

    arr = np.sort(np.asarray(scores, dtype=float))
    # A grid is enough to rank against and keeps the artifact small; the
    # quantile positions are preserved exactly at the sampled points.
    if arr.size > max_grid:
        pos = np.linspace(0, arr.size - 1, max_grid).round().astype(int)
        grid = arr[pos]
    else:
        grid = arr

    ref = {
        "built_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model_version": bundle["version"],
        "winner_family": bundle.get("winner_family"),
        "reference_population": "dev rows only (locked test excluded)",
        "n_dev_rows_scored": int(dev.height),
        "grid_points": int(grid.size),
        "sorted_scores": [float(x) for x in grid],
        "quantiles": {
            f"p{q}": float(np.quantile(arr, q / 100.0))
            for q in (50, 75, 90, 95, 99, 99.9)
        },
        "retraining_performed": False,
        "notes": [
            "percentiles rank a new score against DEV only",
            "the locked test was never read to build this reference",
        ],
    }
    settings.REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    save_json(ref, RISK_REFERENCE_PATH)
    log.info("wrote %s (%d grid points)", RISK_REFERENCE_PATH, grid.size)
    return ref


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-grid", type=int, default=4000,
                    help="maximum stored grid points (default 4000)")
    args = ap.parse_args()
    ref = build(max_grid=args.max_grid)
    print(f"dev rows scored : {ref['n_dev_rows_scored']}")
    print(f"grid points     : {ref['grid_points']}")
    for k, v in ref["quantiles"].items():
        print(f"  {k:<6} {v:.6f}")


if __name__ == "__main__":
    main()
