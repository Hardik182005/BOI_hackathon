"""The organiser dry-run's input variants, checked without a backend.

The dry run itself needs a live API, so these tests exercise the part that can
go wrong quietly: the payloads. Two of them exist to be *refused*, and a
refusal-expected payload that is accidentally well-formed proves nothing while
still reporting a pass.
"""
from __future__ import annotations

import io

import polars as pl
import pytest

from muleguard.cli.dry_run import (
    _raw_variants,
    _required_features,
    _variable_name_headers,
    _variants,
)


@pytest.fixture(scope="module")
def df() -> pl.DataFrame:
    """A frame the variant builder will accept: it must contain at least one of
    the champion's own features, because several variants perturb one."""
    req = sorted(_required_features())[:6]
    cols: dict[str, list] = {"__UNNAMED__0": [1, 2, 3]}
    for i, name in enumerate(req):
        cols[name] = [float(i), float(i) + 1.0, float(i) + 2.0]
    cols["F_unused_optional"] = [7.0, 8.0, 9.0]
    return pl.DataFrame(cols)


def test_every_frame_variant_keeps_the_row_count(df):
    """A variant is a different *spelling* of the same file. Losing a row would
    make the prediction-invariance claim meaningless rather than false."""
    variants = _variants(df)
    assert variants, "no variants produced"
    for name, v in variants.items():
        assert v.height == df.height, name


def test_variable_name_headers_neither_adds_nor_drops_a_column(df):
    """A header the resolver cannot map back is worse than an F-number: it turns
    a rename test into an unmapped-column test, so unmappable names are left
    exactly as they were."""
    out = _variable_name_headers(df)
    assert out.height == df.height
    assert out.width == df.width
    assert len(set(out.columns)) == out.width, "a rename collided"


def test_the_two_refusal_payloads_are_present(df):
    """These are the only variants where a 4xx is the passing outcome."""
    assert set(_raw_variants(df)) == {"corrupted_xlsx", "duplicate_columns"}


def test_both_refusal_payloads_say_so_and_say_why(df):
    for name, spec in _raw_variants(df).items():
        assert spec["must_be_refused"] is True, name
        assert spec["why"].strip(), f"{name} refuses without stating a reason"
        assert isinstance(spec["payload"], (bytes, bytearray)), name


def test_the_corrupted_workbook_is_actually_unreadable(df):
    """Truncating the bytes must break the container. If the reader can still
    open it, the variant is testing nothing and would pass on a lie."""
    body = bytes(_raw_variants(df)["corrupted_xlsx"]["payload"])
    assert body, "empty payload"
    with pytest.raises(Exception):
        pl.read_excel(io.BytesIO(body))


def test_the_duplicate_column_payload_really_repeats_a_header(df):
    """polars will not build a frame with two identical column names, which is
    exactly why this one is hand-written CSV text rather than a frame."""
    text = bytes(_raw_variants(df)["duplicate_columns"]["payload"]).decode()
    names = [n.strip() for n in text.splitlines()[0].split(",")]
    assert len(names) != len(set(names)), "no header is actually duplicated"


def test_the_duplicate_column_payload_is_otherwise_well_formed(df):
    """The repeated header must be the *only* defect, or a refusal could be
    caused by something else and still be scored as this test passing."""
    lines = bytes(_raw_variants(df)["duplicate_columns"]["payload"]).decode().splitlines()
    widths = {len(ln.split(",")) for ln in lines if ln.strip()}
    assert len(widths) == 1, f"ragged rows: {widths}"
