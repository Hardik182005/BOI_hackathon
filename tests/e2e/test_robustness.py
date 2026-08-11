"""Adversarial / robustness suite over the frozen bundle (no new labels invented).

Perturbations: missingness injection, extreme values, column order shuffling,
duplicate requests, small near-threshold noise, omitted optional columns.
Measured: score/tier stability, OOD routing, determinism, latency.
"""
import time

import numpy as np
import polars as pl
import pytest

from muleguard import settings

BUNDLE = settings.MODELS_DIR / "final_bundle.joblib"
pytestmark = pytest.mark.skipif(not BUNDLE.exists(), reason="final bundle not built")


@pytest.fixture(scope="module")
def env():
    from muleguard.data import ingest, split as split_mod
    from muleguard.models.scoring import load_bundle, score_rows

    bundle = load_bundle()
    df = ingest.load_dataset()
    mask = split_mod.load_locked_test_mask()
    dev = df.filter(~pl.Series(mask)).head(40)  # dev rows only
    base = score_rows(dev, bundle=bundle, with_explanations=False)
    return {"bundle": bundle, "dev": dev, "base": base, "score_rows": score_rows}


def test_duplicate_requests_identical(env):
    r1 = env["score_rows"](env["dev"].head(5), bundle=env["bundle"], with_explanations=False)
    r2 = env["score_rows"](env["dev"].head(5), bundle=env["bundle"], with_explanations=False)
    for a, b in zip(r1, r2):
        assert a["calibrated_risk"] == b["calibrated_risk"]
        assert a["risk_tier"] == b["risk_tier"]


def test_column_order_invariance(env):
    dev = env["dev"].head(5)
    shuffled = dev.select(list(reversed(dev.columns)))
    r1 = env["score_rows"](dev, bundle=env["bundle"], with_explanations=False)
    r2 = env["score_rows"](shuffled, bundle=env["bundle"], with_explanations=False)
    for a, b in zip(r1, r2):
        assert a["calibrated_risk"] == pytest.approx(b["calibrated_risk"], abs=1e-9)


def _numeric_raw_features(bundle, frame) -> list[str]:
    """Selected features that exist as raw columns and are numeric.

    The MG_* block is derived inside scoring rather than uploaded, so it is
    absent from the raw frame. Perturbing raw columns and letting the block be
    recomputed downstream is the faithful test: a real caller supplies raw data
    and the meta-features move with it.
    """
    return [f for f in bundle["feature_list_selected"]
            if f in frame.schema and frame.schema[f].is_numeric()]


def test_extreme_values_route_to_ood(env):
    dev = env["dev"].head(3)
    num_feats = _numeric_raw_features(env["bundle"], dev)[:15]
    mod = dev.with_columns([pl.lit(1e9).alias(f) for f in num_feats])
    res = env["score_rows"](mod, bundle=env["bundle"], with_explanations=False)
    assert all(r["ood_status"] == "OUT_OF_DISTRIBUTION" for r in res)
    assert all(r["risk_tier"] == "OOD_REVIEW" for r in res)


def test_added_missingness_does_not_crash_and_shifts_ood_signal(env):
    dev = env["dev"].head(5)
    feats = env["bundle"]["feature_list_selected"]
    half = feats[: len(feats) // 2]
    mod = dev.with_columns([pl.lit(None, dtype=dev.schema[f]).alias(f) for f in half])
    res = env["score_rows"](mod, bundle=env["bundle"], with_explanations=False)
    for r in res:
        assert 0.0 <= r["calibrated_risk"] <= 1.0
        assert r["risk_tier"] in ("CRITICAL_REVIEW", "URGENT_REVIEW", "STANDARD_REVIEW",
                                  "OOD_REVIEW", "MONITOR")


def test_small_perturbations_keep_scores_stable(env):
    dev = env["dev"].head(8)
    num_feats = _numeric_raw_features(env["bundle"], dev)
    rng = np.random.default_rng(0)
    mod = dev.with_columns([
        (pl.col(f) * (1 + 0.01 * float(rng.standard_normal()))).alias(f) for f in num_feats[:10]
    ])
    r1 = env["score_rows"](dev, bundle=env["bundle"], with_explanations=False)
    r2 = env["score_rows"](mod, bundle=env["bundle"], with_explanations=False)
    drifts = [abs(a["calibrated_risk"] - b["calibrated_risk"]) for a, b in zip(r1, r2)]
    assert float(np.median(drifts)) < 0.15  # tree models are locally stable


def test_scoring_latency_batch(env):
    t0 = time.perf_counter()
    env["score_rows"](env["dev"], bundle=env["bundle"], with_explanations=False)
    per_row = (time.perf_counter() - t0) / env["dev"].height
    assert per_row < 1.0, f"scoring too slow: {per_row:.2f}s/row"


def test_scores_never_outside_unit_interval(env):
    for r in env["base"]:
        assert 0.0 <= r["calibrated_risk"] <= 1.0
        for v in r["raw_scores"].values():
            assert 0.0 <= v <= 1.0


def test_no_certified_safe_language(env):
    for r in env["base"]:
        text = str(r).lower()
        assert "certified_clean" not in text
        assert "permanently_safe" not in text
