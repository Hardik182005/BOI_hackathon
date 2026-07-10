"""Model-level guards that must FAIL the build when violated.

These tests exercise the real artifacts (splits, quarantine, predictions)
and are skipped only when the corresponding artifact does not exist yet.
"""
import numpy as np
import polars as pl
import pytest

from muleguard import settings
from muleguard.utils import load_json

QUAR = settings.FEATURES_DIR / "quarantined_features.json"
LOCKED = settings.SPLITS_DIR / "locked_test_indices.parquet"
FOLDS = settings.SPLITS_DIR / "cv_folds.parquet"
OOF = settings.PREDICTIONS_DIR / "oof_predictions.parquet"

needs = pytest.mark.skipif


@needs(not QUAR.exists(), reason="audit not run")
def test_quarantine_contains_mandatory_entries():
    q = {e["feature"] for e in load_json(QUAR)["quarantine"]}
    assert settings.TARGET_COLUMN in q
    assert "F3912" in q
    assert "F2230" in q          # snapshot-month label artifact
    assert "__UNNAMED__0" in q   # row index


@needs(not (QUAR.exists() and LOCKED.exists()), reason="splits not built")
def test_locked_test_and_dev_disjoint():
    locked = pl.read_parquet(LOCKED)
    folds = pl.read_parquet(FOLDS)
    test_rows = set(locked.filter(pl.col("is_locked_test"))["row_index"].to_list())
    dev_rows = set(folds["row_index"].to_list())
    assert test_rows.isdisjoint(dev_rows)
    assert len(test_rows) + len(dev_rows) == locked.height


@needs(not LOCKED.exists(), reason="splits not built")
def test_locked_test_prevalence_preserved():
    locked = pl.read_parquet(LOCKED)
    overall = locked["target"].mean()
    test = locked.filter(pl.col("is_locked_test"))["target"].mean()
    # natural prevalence within a factor consistent with stratified small-n
    assert abs(test - overall) / overall < 0.5
    assert locked.filter(pl.col("is_locked_test"))["target"].sum() >= 12


@needs(not FOLDS.exists(), reason="splits not built")
def test_split_rebuild_is_reproducible():
    from muleguard.data import ingest, split as split_mod
    from muleguard.features.preprocessing import load_quarantine_list

    df = ingest.load_dataset()
    groups = split_mod.feature_row_groups(df, load_quarantine_list())
    mask, _ = split_mod.make_locked_test_split(df, groups)
    saved = pl.read_parquet(LOCKED).sort("row_index")["is_locked_test"].to_numpy()
    assert np.array_equal(mask, saved)


@needs(not OOF.exists(), reason="no OOF predictions yet")
def test_oof_probabilities_in_bounds_and_complete():
    oof = pl.read_parquet(OOF)
    assert float(oof["score"].min()) >= 0.0
    assert float(oof["score"].max()) <= 1.0
    assert oof["score"].null_count() == 0
    # every (model, repeat) covers every dev row exactly once
    folds = pl.read_parquet(FOLDS)
    n_dev = folds.height
    counts = oof.group_by(["model", "repeat"]).len()
    assert (counts["len"] == n_dev).all()


@needs(not OOF.exists(), reason="no OOF predictions yet")
def test_oof_never_contains_locked_test_rows():
    oof = pl.read_parquet(OOF)
    locked = pl.read_parquet(LOCKED)
    test_rows = set(locked.filter(pl.col("is_locked_test"))["row_index"].to_list())
    assert set(oof["row_index"].unique().to_list()).isdisjoint(test_rows)


@needs(not (settings.METRICS_DIR / "oof_metrics.json").exists(), reason="no metrics")
def test_metrics_trace_to_prediction_files():
    """Recompute one headline number from the saved predictions."""
    from sklearn.metrics import average_precision_score

    metrics = load_json(settings.METRICS_DIR / "oof_metrics.json")["models"]
    oof = pl.read_parquet(OOF)
    for model, m in metrics.items():
        sub = oof.filter((pl.col("model") == model) & (pl.col("repeat") == 0))
        if sub.height == 0:
            continue
        ap = average_precision_score(sub["target"].to_numpy(), sub["score"].to_numpy())
        assert abs(ap - m["pr_auc_per_repeat"][0]) < 1e-9, model


@needs(not (settings.MODELS_DIR / "final_bundle.joblib").exists(), reason="no bundle")
def test_bundle_features_exclude_quarantine_and_target():
    from muleguard.models.scoring import load_bundle

    b = load_bundle()
    q = {e["feature"] for e in load_json(QUAR)["quarantine"]}
    used = set(b["feature_list_selected"]) | set(b["feature_list_kept"])
    assert used.isdisjoint(q), used & q


@needs(not (settings.MODELS_DIR / "final_bundle.joblib").exists(), reason="no bundle")
def test_bundle_save_load_identical_predictions():
    import joblib

    from muleguard.data import ingest
    from muleguard.models.scoring import load_bundle, score_rows

    df = ingest.load_dataset().head(25)
    b1 = load_bundle()
    r1 = score_rows(df, bundle=b1, with_explanations=False)
    b2 = joblib.load(settings.MODELS_DIR / "final_bundle.joblib")
    r2 = score_rows(df, bundle=b2, with_explanations=False)
    for a, c in zip(r1, r2):
        assert a["calibrated_risk"] == c["calibrated_risk"]
        assert a["risk_tier"] == c["risk_tier"]


@needs(not (settings.MODELS_DIR / "final_bundle.joblib").exists(), reason="no bundle")
def test_missing_selected_feature_raises_schema_error():
    from muleguard.data import ingest
    from muleguard.models.scoring import SchemaError, load_bundle, score_rows

    b = load_bundle()
    df = ingest.load_dataset().head(3)
    df = df.drop(b["feature_list_selected"][0])
    with pytest.raises(SchemaError):
        score_rows(df, bundle=b, with_explanations=False)
