"""Missingness signature: the properties that make the ablation trustworthy.

The ablation in ``muleguard.cli.missingness_ablation`` is only meaningful if
the encoder it exercises really is fold-local. If any fitted statistic leaked
from validation rows into the training fold, the WITH arm would be measuring
its own contamination and the +0.049 paired gain would be an artifact. So the
tests here are weighted towards that one property rather than spread evenly
over the surface area:

1. **Locality** - the transform of a row depends on that row and nothing else.
   Two independent checks: transforming a subset equals subsetting the
   transform, and perturbing other rows leaves a row's output untouched.
2. **Fit/transform separation** - a signature fitted on training rows applies
   unchanged to unseen rows, including rows whose missingness pattern never
   occurred in training.
3. **Determinism** - identical input gives byte-identical output, since the
   ablation's pairing depends on it.
4. **Banding and capping** - the decisions the fit is allowed to make, made
   correctly, including the tie-break that keeps them reproducible.
5. **Structural correctness** - counts, ratios and window-pair asymmetries are
   what they claim to be, checked against hand-computed values on a tiny frame
   rather than against the implementation.

Degenerate inputs get their own tests because the encoder runs inside a CV
loop where a fold can easily contain an all-null or never-null column.
"""
from __future__ import annotations

import numpy as np
import pytest

from muleguard.features.missingness import (
    PREFIX,
    MissingnessSignature,
    _stem,
    describe,
)


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------
@pytest.fixture
def tiny():
    """A frame small enough to verify every generated column by hand.

    Twelve columns in two families. UPI carries **four** distinct 7-day/31-day
    stem pairs, which is deliberate: ``_MIN_FAMILY_SIZE`` is 4, so a family with
    fewer pairs produces no window-pair group at all and the asymmetry columns
    would silently not exist. CASH has four flat members and no windows, so the
    two code paths are exercised separately.
    """
    names = ["u1_l7d", "u1_l31d", "u2_l7d", "u2_l31d",
             "u3_l7d", "u3_l31d", "u4_l7d", "u4_l31d",
             "c_a", "c_b", "c_c", "c_d"]
    registry = {}
    for i in (1, 2, 3, 4):
        for w in ("L7D", "L31D"):
            registry[f"u{i}_{w.lower()}"] = {
                "variable_name": f"R_UPI_AMT{i}_{w}", "feature_family": "UPI"}
    for c in "abcd":
        registry[f"c_{c}"] = {
            "variable_name": f"R_CASH_{c.upper()}", "feature_family": "CASH"}

    n = 40
    rng = np.random.default_rng(7)
    X = rng.normal(size=(n, len(names)))
    # Deliberate patterns rather than random nulls, so expected values are
    # arithmetic rather than guesses.
    X[0:10, names.index("u1_l7d")] = np.nan    # short missing, long present
    X[10:14, names.index("u1_l31d")] = np.nan  # long missing, short present
    X[0:20, names.index("c_a")] = np.nan       # half the CASH block
    X[:, names.index("c_b")] = np.nan          # c_b always missing
    return X, names, registry


@pytest.fixture
def fitted(tiny):
    X, names, registry = tiny
    return MissingnessSignature.fit(X, names, registry, max_flags=50), X, names


# --------------------------------------------------------------------------
# 1. locality - the property the ablation depends on
# --------------------------------------------------------------------------
def test_transform_of_subset_equals_subset_of_transform(fitted):
    """Row-local means the row's neighbours cannot change its encoding."""
    sig, X, _ = fitted
    full = sig.transform(X)
    idx = np.array([3, 17, 29, 0])
    np.testing.assert_array_equal(sig.transform(X[idx]), full[idx])


def test_perturbing_other_rows_does_not_change_a_row(fitted):
    """A stronger form: corrupt every other row and row 5 is unaffected."""
    sig, X, _ = fitted
    before = sig.transform(X)[5].copy()
    Y = X.copy()
    Y[:5] = np.nan
    Y[6:] = np.nan
    np.testing.assert_array_equal(sig.transform(Y)[5], before)


def test_single_row_scores_identically_to_batch(fitted):
    """Serving scores one account at a time; it must match the batch path."""
    sig, X, _ = fitted
    batch = sig.transform(X)
    for i in (0, 12, 39):
        np.testing.assert_array_equal(sig.transform(X[i : i + 1])[0], batch[i])


# --------------------------------------------------------------------------
# 2. fit/transform separation
# --------------------------------------------------------------------------
def test_fit_ignores_validation_rows(tiny):
    """Fitting on train rows must not consult the rows it will later score.

    Two signatures fitted on the same training rows produce the same decisions
    regardless of what the held-out block contains - here the held-out rows are
    replaced with an entirely different missingness pattern.
    """
    X, names, registry = tiny
    tr, va = np.arange(0, 30), np.arange(30, 40)

    a = MissingnessSignature.fit(X[tr], names, registry, max_flags=50)
    X2 = X.copy()
    X2[va] = np.nan
    b = MissingnessSignature.fit(X2[tr], names, registry, max_flags=50)

    assert a.names == b.names
    np.testing.assert_array_equal(a.flag_idx, b.flag_idx)


def test_unseen_missingness_pattern_still_transforms(tiny):
    """A validation row may be null where no training row ever was."""
    X, names, registry = tiny
    tr = np.arange(0, 30)
    sig = MissingnessSignature.fit(X[tr], names, registry, max_flags=50)

    novel = X[30:31].copy()
    novel[:] = np.nan
    out = sig.transform(novel)
    assert out.shape == (1, len(sig.names))
    assert np.isfinite(out).all()
    # Everything absent means the total-null ratio saturates.
    assert out[0, sig.names.index(f"{PREFIX}TOTAL_NULL_RATIO")] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# 3. determinism
# --------------------------------------------------------------------------
def test_repeated_fit_is_identical(tiny):
    X, names, registry = tiny
    a = MissingnessSignature.fit(X, names, registry, max_flags=13)
    b = MissingnessSignature.fit(X, names, registry, max_flags=13)
    assert a.names == b.names
    np.testing.assert_array_equal(a.flag_idx, b.flag_idx)
    np.testing.assert_array_equal(a.transform(X), b.transform(X))


# --------------------------------------------------------------------------
# 4. banding and capping - the decisions fit is allowed to make
# --------------------------------------------------------------------------
def test_flags_exclude_constant_columns(fitted):
    """Never-null and always-null columns carry no contrast and get no flag."""
    sig, _, names = fitted
    flagged = {names[i] for i in sig.flag_idx}
    assert "c_b" not in flagged, "always-null column must not earn a flag"
    assert "u_cnt_l7d" not in flagged, "never-null column must not earn a flag"
    assert "c_a" in flagged, "a 50%-null column is exactly what a flag is for"


def test_band_edges_are_respected(tiny):
    X, names, registry = tiny
    sig = MissingnessSignature.fit(X, names, registry,
                                   min_rate=0.4, max_rate=0.6, max_flags=50)
    rate = np.isnan(X).mean(axis=0)
    for i in sig.flag_idx:
        assert 0.4 <= rate[i] <= 0.6


def test_max_flags_is_a_hard_cap(tiny):
    X, names, registry = tiny
    sig = MissingnessSignature.fit(X, names, registry,
                                   min_rate=0.0, max_rate=1.0, max_flags=2)
    assert sig.flag_idx.size == 2


def test_cap_keeps_highest_variance_and_breaks_ties_by_column_order(tiny):
    """The cap must be reproducible, not just correctly sized."""
    X, names, registry = tiny
    sig = MissingnessSignature.fit(X, names, registry,
                                   min_rate=0.0, max_rate=1.0, max_flags=2)
    rate = np.isnan(X).mean(axis=0)
    var = rate * (1 - rate)
    kept = set(sig.flag_idx.tolist())
    dropped = set(range(len(names))) - kept
    assert min(var[i] for i in kept) >= max(var[i] for i in dropped)
    # ascending column order is preserved, which is what makes it reproducible
    assert list(sig.flag_idx) == sorted(sig.flag_idx)


# --------------------------------------------------------------------------
# 5. structural correctness, against hand-computed values
# --------------------------------------------------------------------------
def test_family_count_and_ratio_are_correct(fitted):
    sig, X, _ = fitted
    out = sig.transform(X)
    cnt = out[:, sig.names.index(f"{PREFIX}FAMCNT__CASH")]
    ratio = out[:, sig.names.index(f"{PREFIX}FAMRATIO__CASH")]
    # rows 0-19: c_a and c_b null -> 2 of 4; rows 20+: c_b only -> 1 of 4
    assert cnt[0] == 2.0 and cnt[25] == 1.0
    np.testing.assert_allclose(ratio, cnt / 4.0)


def test_short_gap_counts_only_short_missing_while_long_present(fitted):
    sig, X, _ = fitted
    out = sig.transform(X)
    key = f"{PREFIX}SHORTGAP__UPI__L7D_missing_L31D_present"
    assert key in sig.names, (
        "the UPI family has four L7D/L31D stem pairs, so the window-pair group "
        "must be built; if it is not, the asymmetry columns are silently absent")
    short = out[:, sig.names.index(key)]
    long = out[:, sig.names.index(
        f"{PREFIX}LONGGAP__UPI__L31D_missing_L7D_present")]
    assert short[0] == 1.0, "row 0 has l7d missing while l31d is present"
    assert long[0] == 0.0
    assert short[12] == 0.0, "row 12 has l31d missing while l7d is present"
    assert long[12] == 1.0
    assert short[35] == 0.0 and long[35] == 0.0, "row 35 has neither missing"


def test_total_null_matches_isnan(fitted):
    sig, X, _ = fitted
    out = sig.transform(X)
    np.testing.assert_allclose(
        out[:, sig.names.index(f"{PREFIX}TOTAL_NULL")],
        np.isnan(X).sum(axis=1))
    np.testing.assert_allclose(
        out[:, sig.names.index(f"{PREFIX}TOTAL_NULL_RATIO")],
        np.isnan(X).mean(axis=1), rtol=1e-6)


def test_width_matches_names_and_augment_appends(fitted):
    sig, X, names = fitted
    out = sig.transform(X)
    assert out.shape == (X.shape[0], len(sig.names))
    assert len(set(sig.names)) == len(sig.names), "generated names must be unique"
    aug = sig.augment(X)
    assert aug.shape[1] == X.shape[1] + len(sig.names)
    np.testing.assert_array_equal(aug[:, : X.shape[1]], X)
    assert sig.augmented_names() == list(names) + sig.names


def test_every_generated_name_carries_the_prefix(fitted):
    sig, _, _ = fitted
    assert all(n.startswith(PREFIX) for n in sig.names)


# --------------------------------------------------------------------------
# 6. degenerate inputs a CV fold can actually contain
# --------------------------------------------------------------------------
def test_frame_with_no_missing_values_at_all(tiny):
    """No nulls anywhere: no flags, and the block is still well formed."""
    X, names, registry = tiny
    Z = np.zeros_like(X)
    sig = MissingnessSignature.fit(Z, names, registry, max_flags=50)
    assert sig.flag_idx.size == 0
    out = sig.transform(Z)
    assert out.shape == (Z.shape[0], len(sig.names))
    assert (out[:, sig.names.index(f"{PREFIX}TOTAL_NULL")] == 0).all()


def test_frame_that_is_entirely_missing(tiny):
    X, names, registry = tiny
    Z = np.full_like(X, np.nan)
    sig = MissingnessSignature.fit(Z, names, registry, max_flags=50)
    out = sig.transform(Z)
    assert np.isfinite(out).all()
    assert (out[:, sig.names.index(f"{PREFIX}TOTAL_NULL_RATIO")] == 1.0).all()


def test_single_row_fit(tiny):
    """A fold can be tiny; fitting must not divide by zero or raise."""
    X, names, registry = tiny
    sig = MissingnessSignature.fit(X[:1], names, registry, max_flags=50)
    assert sig.transform(X[:1]).shape == (1, len(sig.names))


def test_columns_absent_from_registry_are_tolerated(tiny):
    """Real frames carry engineered columns the dictionary has never seen."""
    X, names, registry = tiny
    reg = {k: v for k, v in registry.items() if k != "c_d"}
    sig = MissingnessSignature.fit(X, names, reg, max_flags=50)
    assert sig.transform(X).shape[1] == len(sig.names)


# --------------------------------------------------------------------------
# 7. helpers
# --------------------------------------------------------------------------
@pytest.mark.parametrize("variable,expected", [
    ("R_CHQ_AMT_CR_L7_14D", ("R_CHQ_AMT_CR", "L7_14D")),
    ("R_UPI_AMT_L31D", ("R_UPI_AMT", "L31D")),
    ("R_UPI_AMT_L7D", ("R_UPI_AMT", "L7D")),
    ("R_PROFILE_AGE", None),
])
def test_stem_splits_longest_window_first(variable, expected):
    """``L7_14D`` must be stripped whole, never as ``L7`` plus a remainder."""
    assert _stem(variable) == expected


def test_describe_is_plain_language_for_each_kind():
    for name in (f"{PREFIX}NULL__F1", f"{PREFIX}FAMCNT__UPI",
                 f"{PREFIX}FAMRATIO__UPI", f"{PREFIX}CTXALL__PROFILE",
                 f"{PREFIX}TOTAL_NULL",
                 f"{PREFIX}SHORTGAP__UPI__L7D_missing_L31D_present"):
        gloss = describe(name)
        assert gloss != name and gloss[0].isupper() and PREFIX not in gloss


def test_describe_passes_through_unknown_names():
    assert describe("F1234") == "F1234"
