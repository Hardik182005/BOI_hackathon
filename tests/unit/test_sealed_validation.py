"""Sealed Validation Protocol and the three-step Validation Lab."""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from muleguard import settings
from muleguard.validation import lab, sealed


@pytest.fixture()
def seal_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(sealed, "SEAL_DIR", tmp_path / "seals")
    return tmp_path / "seals"


def _upload(n=60, with_target=True, seed=0):
    rng = np.random.default_rng(seed)
    cols = {f"F{i}": rng.normal(size=n) for i in range(1, 9)}
    if with_target:
        cols[settings.TARGET_COLUMN] = (rng.random(n) < 0.2).astype(int)
    return pl.DataFrame(cols)


# --------------------------------------------------------------------------
# sealing
# --------------------------------------------------------------------------


def test_target_is_withheld_before_scoring(seal_dir):
    df = _upload()
    stripped, y, name = sealed.withhold_target(df)
    assert name == settings.TARGET_COLUMN
    assert name not in stripped.columns
    assert y is not None and len(y) == df.height


def test_withhold_target_tolerates_a_file_without_labels():
    df = _upload(with_target=False)
    stripped, y, name = sealed.withhold_target(df)
    assert name is None and y is None
    assert stripped.columns == df.columns


def test_seal_records_hash_timestamp_and_version(seal_dir):
    df = _upload()
    rec = sealed.seal_predictions(df, np.linspace(0, 1, df.height),
                                  model_version="v9")
    assert rec.state == sealed.STATE_SEALED
    assert len(rec.prediction_sha256) == 64
    assert len(rec.input_sha256) == 64
    assert rec.model_version == "v9"
    assert rec.sealed_utc.endswith("+00:00")
    assert rec.target_column_detected == settings.TARGET_COLUMN
    assert rec.manifest_path.exists()


def test_reveal_requires_an_intact_seal(seal_dir):
    df = _upload()
    rec = sealed.seal_predictions(df, np.linspace(0, 1, df.height),
                                  model_version="v9")
    y = (np.arange(df.height) % 5 == 0).astype(int)
    out = sealed.reveal_metrics(rec.seal_id, y)
    assert out["verification"]["verified"] is True
    assert out["metrics"]["status"] == "OK"
    assert 0.0 <= out["metrics"]["pr_auc"] <= 1.0
    assert sealed.load_seal(rec.seal_id).state == sealed.STATE_REVEALED


def test_tampered_predictions_are_refused(seal_dir):
    """The whole point: predictions edited after sealing cannot be scored."""
    df = _upload()
    rec = sealed.seal_predictions(df, np.linspace(0, 1, df.height),
                                  model_version="v9")
    p = pl.read_csv(rec.prediction_path)
    p.with_columns(pl.col("risk_score").reverse()).write_csv(rec.prediction_path)

    with pytest.raises(sealed.SealBroken):
        sealed.reveal_metrics(rec.seal_id, np.zeros(df.height, dtype=int))
    assert sealed.load_seal(rec.seal_id).state == sealed.STATE_BROKEN


def test_score_count_must_match_row_count(seal_dir):
    df = _upload()
    with pytest.raises(ValueError):
        sealed.seal_predictions(df, [0.1, 0.2], model_version="v9")


def test_single_class_labels_report_rather_than_crash(seal_dir):
    df = _upload()
    rec = sealed.seal_predictions(df, np.linspace(0, 1, df.height),
                                  model_version="v9")
    out = sealed.reveal_metrics(rec.seal_id, np.zeros(df.height, dtype=int))
    assert out["metrics"]["status"] == "SINGLE_CLASS_ONLY"


def test_fingerprint_is_content_sensitive():
    a = _upload(seed=1)
    assert sealed.frame_fingerprint(a) == sealed.frame_fingerprint(a.clone())
    assert sealed.frame_fingerprint(a) != sealed.frame_fingerprint(_upload(seed=2))


# --------------------------------------------------------------------------
# the lab
# --------------------------------------------------------------------------


def test_step_1_fails_on_missing_required_features():
    df = _upload(with_target=False)
    out = lab.step_1_schema_integrity(df, ["F1", "F2", "F_ABSENT"])
    assert out["verdict"] == lab.STEP_FAIL
    assert out["n_missing_required"] == 1
    assert "F_ABSENT" in out["missing_required"]


def test_step_1_passes_on_a_complete_file():
    df = _upload(with_target=False)
    out = lab.step_1_schema_integrity(df, [f"F{i}" for i in range(1, 9)])
    assert out["verdict"] == lab.STEP_PASS
    assert out["schema_completeness"] == 1.0


def test_step_2_refuses_to_run_after_a_schema_failure():
    """UPDATE 11 fixes the order; the code enforces it rather than trusting it."""
    bad = {"verdict": lab.STEP_FAIL, "schema_completeness": 0.3}
    with pytest.raises(lab.StepOrderError):
        lab.step_2_hidden_validation_shield(
            np.zeros((30, 3)), np.zeros((30, 3)), ["A", "B", "C"], bad)


def test_step_3_refuses_without_the_shield(seal_dir):
    df = _upload(with_target=False)
    ok = {"verdict": lab.STEP_PASS, "schema_completeness": 1.0}
    with pytest.raises(lab.StepOrderError):
        lab.step_3_predictions(df, np.zeros(df.height), model_version="v1",
                               schema=ok, shield={"step": 1})


def test_compatibility_score_rewards_indistinguishable_samples():
    schema = {"schema_completeness": 1.0}
    same = lab.compatibility_score(schema, {"adversarial_auc": 0.50}, 1.0, 1.0)
    shifted = lab.compatibility_score(schema, {"adversarial_auc": 0.95}, 1.0, 1.0)
    assert same["score"] == 100.0
    assert same["band"] == "HIGH_COMPATIBILITY"
    assert shifted["score"] < same["score"]
    assert sum(lab.COMPAT_WEIGHTS.values()) == 100


def test_compatibility_never_changes_predictions():
    out = lab.compatibility_score({"schema_completeness": 0.4},
                                  {"adversarial_auc": 0.99}, 0.2, 0.1)
    assert out["band"] == "INCOMPATIBLE"
    assert "never modifies" in out["note"]


def test_full_lab_run_seals_and_reports(seal_dir):
    rng = np.random.default_rng(3)
    names = [f"F{i}" for i in range(1, 9)]
    df = _upload(n=80, with_target=False, seed=3)
    train_X = rng.normal(size=(300, 8))
    upload_X = df.select(names).to_numpy()

    out = lab.run_lab(
        df, required_features=names, train_X=train_X, upload_X=upload_X,
        feature_names=names, scores=rng.random(df.height), model_version="v9",
    )
    assert out["overall"] in (lab.STEP_PASS, lab.STEP_WARN)
    assert [s["step"] for s in out["steps"]] == [1, 2, 3]
    assert out["ui_state"] == sealed.STATE_SEALED
    assert out["seal_id"]
    assert 0 <= out["compatibility_score"] <= 100


def test_lab_stops_at_step_1_on_a_broken_schema(seal_dir):
    df = _upload(n=40, with_target=False)
    out = lab.run_lab(df, required_features=["F1", "NOPE"],
                      train_X=np.zeros((40, 2)), upload_X=np.zeros((40, 2)),
                      feature_names=["F1", "NOPE"], scores=np.zeros(40),
                      model_version="v9")
    assert out["overall"] == lab.STEP_FAIL
    assert out["stopped_at_step"] == 1
    assert len(out["steps"]) == 1  # nothing was sealed
