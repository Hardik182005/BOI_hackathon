"""Security-focused tests: secrets, log redaction, CSV injection, path traversal."""
import re
import subprocess
from pathlib import Path

import pytest

from muleguard import settings
from muleguard.logging import _redact

REPO = settings.REPO_ROOT


def test_env_file_not_committed():
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=REPO, capture_output=True, text=True
    ).stdout.splitlines()
    assert ".env" not in tracked
    assert ".env.example" in tracked


def test_no_hardcoded_secrets_in_source():
    pattern = re.compile(
        r"(api[_-]?key|secret|password|token)\s*[:=]\s*['\"][A-Za-z0-9+/]{16,}['\"]", re.I
    )
    offenders = []
    for py in (REPO / "src").rglob("*.py"):
        if pattern.search(py.read_text(encoding="utf-8")):
            offenders.append(str(py))
    assert not offenders, offenders


def test_log_redaction_masks_sensitive_keys():
    out = _redact({"api_key": "abc123", "nested": {"password": "x", "safe": 1}})
    assert out["api_key"] == "***REDACTED***"
    assert out["nested"]["password"] == "***REDACTED***"
    assert out["nested"]["safe"] == 1


def test_raw_dataset_not_tracked_by_git():
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=REPO, capture_output=True, text=True
    ).stdout.splitlines()
    assert "DataSet.xlsx" not in tracked
    assert not any(t.startswith("data/raw/DataSet") for t in tracked)


def test_csv_export_formula_injection_guard():
    """Any CSV we export for analysts must not begin cells with =,+,-,@."""
    from muleguard.explain.evidence_packet import csv_safe

    assert csv_safe("=cmd|' /C calc'!A0") == "'=cmd|' /C calc'!A0"
    assert csv_safe("+SUM(A1)") == "'+SUM(A1)"
    assert csv_safe("normal text") == "normal text"
    assert csv_safe(12.5) == 12.5


def test_masked_account_reference_pattern():
    from muleguard.api.main import MASKED_REF

    assert MASKED_REF.match("ACC-1029")
    assert not MASKED_REF.match("raj kumar 9876543210")  # PII-looking
    assert not MASKED_REF.match("a" * 100)


def test_raw_data_files_readonly_copy_exists_and_unmodified():
    fp_path = REPO / "data/interim/data_fingerprint.json"
    if not fp_path.exists():
        pytest.skip("fingerprint not built")
    import json

    from muleguard.utils import sha256_file
    fp = json.loads(fp_path.read_text())
    raw_copy = Path(fp["raw_file"]["raw_copy_path"])
    if not raw_copy.exists():
        pytest.skip("raw copy absent")
    assert sha256_file(raw_copy) == fp["raw_file"]["sha256"]
