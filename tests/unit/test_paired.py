"""Tests for muleguard.models.paired.

The point of this module is that it cannot be talked into a friendlier answer,
so the tests check the arithmetic against a published result and check that all
three tests are always present, including when they disagree.
"""
from __future__ import annotations

import numpy as np
import pytest

from muleguard.models.paired import detectable_effect, paired_report, sign_test_p


def test_sign_test_matches_binomial_by_hand():
    # 11 improvements out of 15, no ties. Two-sided exact binomial tail.
    d = np.array([1.0] * 11 + [-1.0] * 4)
    p, up, dn, tied = sign_test_p(d)
    assert (up, dn, tied) == (11, 4, 0)
    assert p == pytest.approx(0.11847, abs=1e-5)


def test_sign_test_drops_exact_zeros_and_reports_them():
    d = np.array([1.0, 1.0, 0.0, 0.0, -1.0])
    p, up, dn, tied = sign_test_p(d)
    assert (up, dn, tied) == (2, 1, 2)
    # the p-value is computed on n=3, not n=5
    assert p == pytest.approx(1.0, abs=1e-9)


def test_sign_test_all_zero_is_not_significant():
    p, up, dn, tied = sign_test_p(np.zeros(15))
    assert (p, up, dn, tied) == (1.0, 0, 0, 15)


def test_reproduces_the_published_missingness_result():
    """docs/MISSINGNESS_SIGNATURE.md is the fixed point for this arithmetic.

    That document reports, for the 15 per-fold gains stored in
    artifacts/metrics/missingness_ablation.json: mean +0.04945, 11/15 improved,
    sign p=0.11847, Wilcoxon p=0.03534, paired t p=0.02495 (t=2.5106) and a 95 %
    CI of [0.00721, 0.09170]. Those numbers were produced by a different code
    path, so recovering them from the stored per-fold vector is an independent
    check on this module rather than a restatement of it.
    """
    import json
    import pathlib

    from muleguard import settings

    art = pathlib.Path(settings.METRICS_DIR) / "missingness_ablation.json"
    if not art.exists():
        pytest.skip("missingness_ablation.json not present")
    blk = json.loads(art.read_text(encoding="utf-8"))["paired"]
    diff = np.asarray(blk["per_fold_gain"], dtype=float)
    r = paired_report(np.zeros_like(diff), diff)

    assert r.mean == pytest.approx(blk["mean_gain"], abs=1e-5)
    assert r.median == pytest.approx(blk["median_gain"], abs=1e-5)
    assert r.n_improved == blk["n_folds_improved"]

    # The stored std_of_paired_diff is a POPULATION sd (ddof=0); this module
    # uses the sample sd (ddof=1), which is the one the published CI was
    # actually built from - the CI assertions below reproduce it exactly. The
    # older artifact therefore mixes the two conventions by a factor
    # sqrt(15/14) = 1.035. Recorded here rather than papered over.
    from math import sqrt

    assert r.sd == pytest.approx(blk["std_of_paired_diff"] * sqrt(15 / 14),
                                 abs=1e-5)

    assert r.p_sign == pytest.approx(blk["sign_test_p_two_sided"], abs=1e-5)
    # published in docs/MISSINGNESS_SIGNATURE.md but not in the JSON
    assert r.p_wilcoxon == pytest.approx(0.03534, abs=1e-5)
    assert r.p_ttest == pytest.approx(0.02495, abs=1e-5)
    assert r.t_stat == pytest.approx(2.5106, abs=1e-4)
    assert r.ci95[0] == pytest.approx(0.00721, abs=1e-5)
    assert r.ci95[1] == pytest.approx(0.09170, abs=1e-5)


def test_ci_is_the_t_interval():
    rng = np.random.default_rng(0)
    a = rng.normal(size=15)
    b = a + rng.normal(scale=0.1, size=15)
    r = paired_report(a, b)
    from math import sqrt

    from scipy import stats

    d = b - a
    half = stats.t.ppf(0.975, 14) * d.std(ddof=1) / sqrt(15)
    assert r.ci95[0] == pytest.approx(d.mean() - half)
    assert r.ci95[1] == pytest.approx(d.mean() + half)


def test_all_three_tests_are_always_reported():
    rng = np.random.default_rng(1)
    a = rng.normal(size=15)
    b = a + 0.5
    out = paired_report(a, b).to_dict()
    for key in ("sign_test_p_two_sided", "wilcoxon_p_two_sided",
                "paired_t_p_two_sided", "mean_paired_diff", "ci95_of_mean",
                "std_of_paired_diff", "min_detectable_effect_80pct_power"):
        assert key in out, key
    assert out["n_folds"] == 15


def test_disagreement_is_detected_not_hidden():
    """Two large wins against thirteen small losses - the tests point opposite ways.

    The mean is positive, so a magnitude-based reading says the arm helped. But
    13 of 15 folds got worse, and the sign test says so at p = 0.007 - pointing
    the other way. This is exactly the situation the write-up rule exists for:
    the disagreement is the finding, and quoting either number alone would be a
    misrepresentation.
    """
    d = np.array([0.40, 0.35] + [-0.01] * 13)
    r = paired_report(np.zeros(15), d)
    assert r.mean > 0                     # magnitude favours the arm
    assert (r.n_improved, r.n_worse) == (2, 13)
    assert r.p_sign < 0.05                # direction is significant AGAINST it
    assert r.p_ttest > 0.05               # magnitude is not significant at all
    assert r.tests_disagree
    assert r.to_dict()["tests_disagree"] is True


def test_direction_is_arm_minus_baseline():
    r = paired_report([0.1] * 5, [0.2] * 5)
    assert r.mean == pytest.approx(0.1)
    assert r.n_improved == 5 and r.n_worse == 0


def test_identical_arms_are_not_significant():
    a = [0.1, 0.2, 0.3, 0.4, 0.5]
    r = paired_report(a, a)
    assert r.mean == 0.0
    assert r.p_sign == 1.0 and r.p_ttest == 1.0
    assert r.ci95 == (0.0, 0.0)
    # scipy >= 1.18 returns p = 1.0 for an all-zero difference vector; older
    # versions raise, which the module catches and turns into NaN. Either is
    # acceptable, "significant" is not.
    assert np.isnan(r.p_wilcoxon) or r.p_wilcoxon == 1.0
    assert r.tests_significant == 0


def test_unpaired_inputs_are_rejected():
    with pytest.raises(ValueError):
        paired_report([1.0, 2.0, 3.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        paired_report([1.0, np.nan], [1.0, 2.0])


def test_yardstick_note_names_the_paired_spread_and_disowns_0905():
    out = paired_report(np.arange(15.0), np.arange(15.0) + 1).to_dict()
    assert "0.0905" in out["yardstick"]
    assert "NOT" in out["yardstick"]
    assert "std_of_paired_diff" in out["yardstick"]


def test_detectable_effect_scales_with_spread_and_n():
    small = detectable_effect(np.full(15, 0.01) * np.arange(15))
    big = detectable_effect(np.full(15, 0.10) * np.arange(15))
    assert big > small
    # 80 % power at n=15 with sd=0.05 needs a mean shift of roughly 0.036
    d = np.linspace(-0.05, 0.05, 15)
    mde = detectable_effect(d)
    assert 0.02 < mde < 0.06


def test_report_is_deterministic():
    rng = np.random.default_rng(7)
    a, b = rng.normal(size=15), rng.normal(size=15)
    assert paired_report(a, b).to_dict() == paired_report(a, b).to_dict()
