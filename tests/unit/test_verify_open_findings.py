"""The check that stops an inconvenient result from going quiet.

`check_open_findings_declared` exists because of a specific near-miss: the
subset sweep returned REPLACES_FULL_CLEAN for an arm the shipped model does not
use, and nothing in the pipeline would have noticed if that verdict had simply
been left in a JSON file while the reports kept quoting the old number.

The behaviour under test is deliberately not "no arm may win". An arm winning is
a scheduling problem, not a defect. The check fails on *silence*.
"""
from __future__ import annotations

import json

import pytest

from muleguard import settings
from muleguard.cli.verify_metrics import Check, check_open_findings_declared

FINDING = "docs/FEATURE_SUBSET_SIZE_FINDING.md"


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """A repo root and metrics dir the check will read instead of the real ones."""
    (tmp_path / "docs").mkdir()
    monkeypatch.setattr(settings, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(settings, "METRICS_DIR", tmp_path)

    def write(verdicts: dict[str, str] | None, doc: str | None) -> None:
        if verdicts is not None:
            (tmp_path / "nested_feature_family_arms.json").write_text(
                json.dumps({"rule": {"verdicts": verdicts}}), encoding="utf-8")
        if doc is not None:
            (tmp_path / FINDING).write_text(doc, encoding="utf-8")

    return write


def _run() -> Check:
    return check_open_findings_declared(Check("open_findings_declared", "why"))


def test_a_winning_arm_named_in_the_writeup_passes(sandbox):
    sandbox({"full_clean_top200": "REPLACES_FULL_CLEAN",
             "full_clean_top60": "NO_CHANGE"},
            "The shipped size may be cut too deep: full_clean_top200 wins.")
    c = _run()
    assert c.ok
    assert c.detail["winning_arms"] == ["full_clean_top200"]
    assert c.detail["undeclared"] == []


def test_a_winning_arm_with_no_writeup_at_all_fails(sandbox):
    sandbox({"full_clean_top200": "REPLACES_FULL_CLEAN"}, None)
    c = _run()
    assert not c.ok
    assert c.detail["undeclared"] == ["full_clean_top200"]


def test_a_writeup_that_does_not_name_the_arm_fails(sandbox):
    """A document that exists but discusses something else is still silence."""
    sandbox({"full_clean_top200": "REPLACES_FULL_CLEAN"},
            "Some general prose about feature counts that names no arm.")
    assert not _run().ok


def test_declaring_one_winner_does_not_cover_a_second(sandbox):
    sandbox({"full_clean_top200": "REPLACES_FULL_CLEAN",
             "behavior_profile": "REPLACES_FULL_CLEAN"},
            "Only full_clean_top200 is discussed here.")
    c = _run()
    assert not c.ok
    assert c.detail["undeclared"] == ["behavior_profile"]


def test_no_winner_passes_without_needing_a_document(sandbox):
    sandbox({"full_clean_top60": "NO_CHANGE"}, None)
    c = _run()
    assert c.ok
    assert c.detail["winning_arms"] == []


def test_a_missing_sweep_is_not_treated_as_a_failure(sandbox):
    """Absence of the experiment is `ablations_present`'s job to complain about;
    this check must not double-report it as a hidden finding."""
    sandbox(None, None)
    c = _run()
    assert c.ok
    assert "nothing to declare" in c.note


def test_the_real_repo_state_is_declared():
    """Guards the shipped tree, not a fixture: whatever the sweep currently says,
    the write-up must keep up with it."""
    assert _run().ok
