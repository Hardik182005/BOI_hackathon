"""The export layer must not be able to corrupt a submission.

These tests are mostly about refusal. An export that silently drops rows,
reorders them, or leaks the target column back into the file is worse than a
crash, because nobody finds out until the score comes back wrong.
"""
from __future__ import annotations

import polars as pl
import pytest

from muleguard.action.submission import (
    ExportValidationError,
    SubmissionFormat,
    build_analyst_export,
    build_minimal_submission,
    resolve_row_ids,
    validate_minimal_submission,
    write_exports,
)


@pytest.fixture()
def fmt() -> SubmissionFormat:
    return SubmissionFormat()


def test_minimal_export_is_exactly_two_columns(fmt):
    df = build_minimal_submission([0.1, 0.9, 0.5], fmt=fmt)
    assert df.columns == [fmt.id_column, fmt.prediction_column]
    assert df.height == 3


def test_row_ids_are_positional_and_ordered(fmt):
    df = build_minimal_submission([0.3, 0.2, 0.1], fmt=fmt)
    assert df[fmt.id_column].to_list() == [1, 2, 3]


def test_row_order_is_never_sorted_by_score(fmt):
    """A submission joined positionally is destroyed by reordering."""
    scores = [0.9, 0.1, 0.5]
    df = build_minimal_submission(scores, fmt=fmt)
    assert df[fmt.prediction_column].to_list() == pytest.approx(scores)


def test_score_count_mismatch_is_refused(fmt):
    with pytest.raises(ExportValidationError, match="refusing"):
        build_minimal_submission([0.1, 0.2], n_rows=5, fmt=fmt)


def test_binary_mode_emits_only_zero_and_one():
    cfg = {
        "minimal": {"id_column": "row_id", "prediction_column": "F3924",
                    "prediction_type": "binary", "binary_threshold": 0.5},
        "validation": {},
    }
    fmt = SubmissionFormat(cfg)
    df = build_minimal_submission([0.9, 0.1, 0.5], fmt=fmt)
    assert set(df["F3924"].to_list()) <= {0, 1}
    assert df["F3924"].to_list() == [1, 0, 1]


def test_binary_policy_threshold_without_a_snapshot_is_refused():
    cfg = {
        "minimal": {"prediction_type": "binary", "binary_threshold": "policy"},
        "validation": {},
    }
    fmt = SubmissionFormat(cfg)
    with pytest.raises(ExportValidationError, match="refusing to invent"):
        build_minimal_submission([0.9], fmt=fmt, policy_threshold=None)


def test_validation_catches_a_dropped_row(fmt):
    df = build_minimal_submission([0.1, 0.2], fmt=fmt)
    with pytest.raises(ExportValidationError, match="row count"):
        validate_minimal_submission(df, n_input_rows=3, fmt=fmt)


def test_validation_catches_broken_row_order(fmt):
    df = pl.DataFrame({fmt.id_column: [2, 1], fmt.prediction_column: [0.1, 0.2]})
    with pytest.raises(ExportValidationError, match="1..N"):
        validate_minimal_submission(df, n_input_rows=2, fmt=fmt)


def test_validation_catches_out_of_range_probabilities(fmt):
    df = pl.DataFrame({fmt.id_column: [1, 2], fmt.prediction_column: [0.5, 1.7]})
    with pytest.raises(ExportValidationError, match=r"\[0,1\]"):
        validate_minimal_submission(df, n_input_rows=2, fmt=fmt)


def test_validation_catches_nulls(fmt):
    df = pl.DataFrame({fmt.id_column: [1, 2],
                       fmt.prediction_column: [0.5, None]})
    with pytest.raises(ExportValidationError, match="null"):
        validate_minimal_submission(df, n_input_rows=2, fmt=fmt)


def test_extra_columns_are_refused(fmt):
    df = pl.DataFrame({fmt.id_column: [1], fmt.prediction_column: [0.5],
                       "F3912": [0.9]})
    with pytest.raises(ExportValidationError):
        validate_minimal_submission(df, n_input_rows=1, fmt=fmt)


def test_quarantined_feature_cannot_ride_along_in_an_export():
    """F3912 reconstructs the label; it must never leave in a submission."""
    cfg = {
        "minimal": {"id_column": "row_id", "prediction_column": "F3924",
                    "prediction_type": "probability"},
        "validation": {"forbid_extra_columns": False,
                       "forbid_leakage_columns": True},
    }
    fmt = SubmissionFormat(cfg)
    df = pl.DataFrame({"row_id": [1], "F3924": [0.5], "F3912": [1.0]})
    with pytest.raises(ExportValidationError, match="quarantined"):
        validate_minimal_submission(df, n_input_rows=1, fmt=fmt)


def test_target_name_is_legal_as_the_prediction_column(fmt):
    """F3924 is what we were asked to predict - naming the output that is
    correct, and must not be confused with echoing the input label back."""
    df = build_minimal_submission([0.5], fmt=fmt)
    assert "F3924" in df.columns
    validate_minimal_submission(df, n_input_rows=1, fmt=fmt)


def test_upload_identifier_column_is_preferred_over_positional(fmt):
    frame = pl.DataFrame({"row_id": [11, 22, 33], "F1": [1.0, 2.0, 3.0]})
    ids, source = resolve_row_ids(frame, fmt)
    assert ids == [11, 22, 33]
    assert source.startswith("upload_column")


def test_duplicate_identifier_falls_back_to_positional(fmt):
    frame = pl.DataFrame({"row_id": [7, 7], "F1": [1.0, 2.0]})
    ids, source = resolve_row_ids(frame, fmt)
    assert ids == [1, 2]
    assert source == "positional"


def test_analyst_export_carries_reasons_and_uncertainty(fmt):
    records = [{
        "calibrated_risk": 0.8, "risk_tier": "CRITICAL", "decision": "REVIEW",
        "model_agreement": 0.9, "prediction_std": 0.02,
        "conformal_status": "MULE", "ood_status": "IN_DISTRIBUTION",
        "model_version": "2.0.0",
        "top_reasons": [
            {"feature": "MG_PASSTHROUGH_7D", "verified_semantic_name": None,
             "direction": "INCREASES_RISK", "legitimate_percentile": 0.97},
        ],
    }]
    out = build_analyst_export(records, fmt=fmt, policy_threshold=0.5)
    assert out["binary_prediction"].to_list() == [1]
    assert out["uncertainty"].to_list() == [0.02]
    assert "MG_PASSTHROUGH_7D" in out["top_reason_1"][0]
    # Absent reasons render empty rather than as a fabricated explanation.
    assert out["top_reason_3"][0] == ""


def test_write_exports_records_hashes_and_checks(tmp_path, fmt):
    manifest = write_exports(
        tmp_path, scores=[0.1, 0.2, 0.3],
        records=[{"calibrated_risk": s} for s in (0.1, 0.2, 0.3)],
        fmt=fmt, policy_threshold=0.5, model_version="2.0.0")
    assert (tmp_path / "submission.csv").exists()
    assert (tmp_path / "analyst_export.csv").exists()
    assert manifest["retraining_performed"] is False
    assert "row_count_matches_input" in manifest["validation_checks_passed"]
    assert len(manifest["minimal_sha256"]) == 64


def test_shipped_config_is_the_minimal_competition_shape(fmt):
    """The repo default must be the two-column file the spec names."""
    assert fmt.id_column == "row_id"
    assert fmt.prediction_column == "F3924"
    assert fmt.prediction_type in {"probability", "binary"}
