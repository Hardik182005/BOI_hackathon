"""Header resolution must be exact, or it must fail.

The failure mode this guards against is not a rejected file - it is a file
that is accepted after a column was mapped into the wrong feature slot. Those
scores look perfectly normal and are wrong for every row.
"""
from __future__ import annotations

import polars as pl
import pytest

from muleguard.validation.column_mapping import (
    apply_column_mapping,
    build_alias_index,
    resolve_columns,
)


@pytest.fixture(scope="module")
def registry() -> dict:
    return {
        "features": {
            "F0001": {"variable_name": "TOTAL_CR_AMT_30D"},
            "F0002": {"variable_name": "TOTAL_DB_AMT_30D"},
            "F0003": {"variable_name": "AVG_BAL_7D"},
            "F0004": {"variable_name": "TOTAL_CR_AMT_30D"},  # deliberate clash
        }
    }


def test_canonical_ids_are_left_alone(registry):
    plan = resolve_columns(["F0001", "F0003"], registry)
    assert plan["mapping"] == {}
    assert plan["n_already_canonical"] == 2


def test_variable_names_resolve_to_feature_ids(registry):
    plan = resolve_columns(["AVG_BAL_7D"], registry)
    assert plan["mapping"] == {"AVG_BAL_7D": "F0003"}


@pytest.mark.parametrize("header", ["avg bal 7d", "Avg-Bal-7D", "avg.bal.7d",
                                    "AVG_BAL_7D", "AvgBal7d"])
def test_case_and_separator_drift_is_tolerated(registry, header):
    """One drifted spelling resolves; the data is right, only the header differs."""
    plan = resolve_columns([header], registry)
    assert plan["mapping"] == {header: "F0003"}


def test_several_spellings_of_one_feature_are_all_refused(registry):
    """Three headers targeting F0003 would silently destroy two of them."""
    plan = resolve_columns(["avg bal 7d", "Avg-Bal-7D", "avg.bal.7d"], registry)
    assert plan["mapping"] == {}
    assert plan["collisions"][0]["target"] == "F0003"
    assert len(plan["collisions"][0]["sources"]) == 3


def test_an_ambiguous_name_is_never_guessed(registry):
    """TOTAL_CR_AMT_30D maps to two features here; picking one is unacceptable."""
    plan = resolve_columns(["TOTAL_CR_AMT_30D"], registry)
    assert plan["mapping"] == {}
    assert plan["ambiguous"][0]["candidates"] == ["F0001", "F0004"]
    assert "TOTAL_CR_AMT_30D" in plan["unmapped"]


def test_no_fuzzy_matching(registry):
    """A near-miss stays unmapped. Edit distance is how columns get swapped."""
    plan = resolve_columns(["AVG_BAL_8D", "TOTL_CR_AMT_30D"], registry)
    assert plan["mapping"] == {}
    assert set(plan["unmapped"]) == {"AVG_BAL_8D", "TOTL_CR_AMT_30D"}


def test_rename_colliding_with_an_existing_column_is_refused(registry):
    """If F0003 is already present, renaming AVG_BAL_7D onto it would drop one."""
    plan = resolve_columns(["F0003", "AVG_BAL_7D"], registry)
    assert plan["mapping"] == {}
    assert plan["collisions"][0]["target"] == "F0003"
    assert plan["collisions"][0]["already_in_upload"] is True


def test_apply_renames_the_frame_and_returns_the_audit(registry):
    frame = pl.DataFrame({"AVG_BAL_7D": [1.0, 2.0], "unknown_col": [3, 4]})
    out, plan = apply_column_mapping(frame, registry)
    assert "F0003" in out.columns
    assert "AVG_BAL_7D" not in out.columns
    assert out.height == 2
    assert plan["n_renamed"] == 1
    assert "unknown_col" in plan["unmapped"]


def test_values_survive_the_rename_unchanged(registry):
    frame = pl.DataFrame({"AVG_BAL_7D": [1.5, 2.5]})
    out, _ = apply_column_mapping(frame, registry)
    assert out["F0003"].to_list() == [1.5, 2.5]


def test_alias_index_covers_ids_and_variable_names(registry):
    alias, varnames = build_alias_index(registry)
    assert "F0003" in varnames
    assert alias["AVGBAL7D"] == {"F0003"}
