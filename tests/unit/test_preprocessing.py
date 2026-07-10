"""Fold-safe preprocessing behaviour."""
import numpy as np
import polars as pl
import pytest

from muleguard import settings
from muleguard.features.preprocessing import (
    FoldPreprocessor,
    candidate_feature_columns,
    encode_dataframe,
)


def test_target_never_in_candidates(synth):
    cands = candidate_feature_columns(synth["df"], quarantined=["F3912"])
    assert settings.TARGET_COLUMN not in cands
    assert "F3912" not in cands


def test_quarantine_respected(synth):
    cands = candidate_feature_columns(synth["df"], quarantined=["F3912", "F7"])
    assert "F7" not in cands and "F3912" not in cands


def test_constant_and_duplicate_removed_in_fold(synth):
    df = synth["df"]
    cands = candidate_feature_columns(df, quarantined=["F3912", "F7"])
    X, names, _ = encode_dataframe(df, cands)
    prep = FoldPreprocessor(mode="tree").fit(X, names)
    assert "F4" in prep.removed_constant           # constant column dropped
    assert prep.removed_duplicate.get("F5") == "F1"  # duplicate maps to first
    assert "F1" in prep.kept_features


def test_linear_mode_imputes_and_scales(synth):
    df = synth["df"]
    cands = candidate_feature_columns(df, quarantined=["F3912", "F7"])
    X, names, _ = encode_dataframe(df, cands)
    prep = FoldPreprocessor(mode="linear")
    Xt = prep.fit_transform(X, names)
    assert not np.isnan(Xt).any()
    assert abs(Xt.mean(axis=0)).max() < 0.5  # roughly centred


def test_transform_before_fit_raises():
    prep = FoldPreprocessor()
    with pytest.raises(RuntimeError):
        prep.transform(np.zeros((3, 2)))


def test_train_only_statistics_leak_free():
    """Imputation medians must come from the train part only."""
    Xtr = np.array([[1.0], [2.0], [3.0]])
    Xva = np.array([[np.nan], [100.0]])
    prep = FoldPreprocessor(mode="linear").fit(Xtr, ["a"])
    out = prep.transform(Xva)
    # imputed value equals train median (2.0) standardised, not anything from val
    expected = (2.0 - prep.means[0]) / prep.stds[0]
    assert out[0, 0] == pytest.approx(expected)
