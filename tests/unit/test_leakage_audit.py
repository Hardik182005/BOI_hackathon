"""Leakage firewall catches planted leaks and never misses the target."""
import numpy as np
import polars as pl

from muleguard import settings
from muleguard.data.leakage import (
    build_quarantine,
    exact_label_reconstruction,
    pointbiserial_with_target,
    run_leakage_audit,
)


def test_exact_copy_and_recoding_detected(synth):
    df = synth["df"]
    y = df[settings.TARGET_COLUMN].to_numpy()
    cols = ["F1", "F7", "F3912"]
    X = df.select(cols).to_numpy().astype(np.float64)
    rate = exact_label_reconstruction(X, y)
    assert rate[cols.index("F7")] == 1.0        # exact copy
    assert rate[cols.index("F3912")] == 1.0     # affine recoding
    assert rate[cols.index("F1")] < 0.9         # continuous informative feature is not flagged


def test_pointbiserial_sign_and_magnitude(synth):
    df = synth["df"]
    y = df[settings.TARGET_COLUMN].to_numpy()
    X = df.select(["F1", "F3"]).to_numpy().astype(np.float32)
    r = pointbiserial_with_target(X, y)
    assert r[0] > 0.3      # informative
    assert abs(r[1]) < 0.15  # noise


def test_full_audit_flags_planted_leaks(synth):
    audit, summary = run_leakage_audit(synth["df"])
    flagged = set(audit.filter(pl.col("suspicious"))["feature"].to_list())
    assert "F7" in flagged and "F3912" in flagged
    assert "F3" not in flagged


def test_quarantine_always_contains_target_and_f3912(synth):
    audit, _ = run_leakage_audit(synth["df"])
    q = build_quarantine(audit, synth["df"], index_candidates=[])
    names = {e["feature"] for e in q["quarantine"]}
    assert settings.TARGET_COLUMN in names
    assert "F3912" in names
    assert all(e["reason"] for e in q["quarantine"])  # every entry justified
