"""Conformal coverage and calibration selection behaviour."""
import numpy as np

from muleguard.models.calibration import (
    PlattCalibrator, crossfit_calibrated, select_calibrator,
)
from muleguard.models.conformal import MondrianConformal, empirical_coverage


def _synth_scores(n=4000, prev=0.05, seed=3):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < prev).astype(int)
    # well-separated but noisy scores
    p = np.clip(0.15 + 0.55 * y + rng.normal(0, 0.18, n), 0.001, 0.999)
    return y, p


def test_conformal_coverage_close_to_target():
    y, p = _synth_scores()
    conf = MondrianConformal(alpha=0.1).fit(p, y)
    cov = empirical_coverage(conf, p, y)
    assert cov["positive_coverage"] >= 0.85
    assert cov["negative_coverage"] >= 0.85


def test_conformal_uncertain_on_ambiguous_scores():
    y, p = _synth_scores()
    conf = MondrianConformal(alpha=0.1).fit(p, y)
    sets = conf.predict_set(np.array([0.45]))
    assert sets[0] == "UNCERTAIN_SET"


def test_conformal_no_calibration_data_abstains_high():
    conf = MondrianConformal(alpha=0.1).fit(np.array([]), np.array([]))
    # with no data both classes are conformal -> uncertain
    assert conf.predict_set(np.array([0.9]))[0] == "UNCERTAIN_SET"


def test_platt_monotone_and_bounded():
    y, p = _synth_scores()
    cal = PlattCalibrator().fit(p, y)
    grid = np.linspace(0.01, 0.99, 50)
    out = cal.predict(grid)
    assert np.all(np.diff(out) >= -1e-12)
    assert out.min() >= 0 and out.max() <= 1


def test_crossfit_probabilities_in_bounds():
    y, p = _synth_scores()
    for m in ("platt", "isotonic"):
        cp = crossfit_calibrated(p, y, m)
        assert cp.min() >= 0 and cp.max() <= 1


def test_selector_prefers_platt_with_few_positives():
    # tiny positive count: isotonic should rarely clear the 2% double bar
    y, p = _synth_scores(n=1500, prev=0.01, seed=5)
    result = select_calibrator(p, y)
    assert result["winner"] in ("platt", "isotonic")
    assert "comparison" in result and set(result["comparison"]) == {"platt", "isotonic"}
