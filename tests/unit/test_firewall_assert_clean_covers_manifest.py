"""`assert_clean` must refuse everything the release gate refuses.

Until 2026-08-14 it refused 9 of the manifest's 13 columns. The four it let
through - `F3892` (GENDER, excluded on fairness grounds rather than
availability) and `F3916`/`F3917`/`F3918` (`PRE_EXISTING_RISK_CONTEXT`, a class
that is not forbidden) - were caught downstream by
`release_gate.no_target_or_f3912_leakage`, so nothing leaked. But `assert_clean`
is called at bundle-freeze time and by the scoring service, and its docstring
promises a second line of defence. For those four there was only ever one.

The two checks must agree *by construction*, which is why this reads the shipped
manifest rather than restating a list of names that could drift apart from it.
"""
from __future__ import annotations

import json

import pytest

from muleguard import settings
from muleguard.features import firewall


def _manifest() -> list[str]:
    path = settings.FEATURES_DIR / "quarantined_features.json"
    return sorted({e["feature"]
                   for e in json.loads(path.read_text())["quarantine"]})


def test_the_manifest_still_has_thirteen_entries():
    """A canary: if this changes, the change was deliberate and the reports that
    say "13 quarantined columns" need to change with it."""
    assert len(_manifest()) == 13


@pytest.mark.parametrize("col", _manifest())
def test_every_manifest_column_is_refused(col):
    with pytest.raises(firewall.LeakageViolation):
        firewall.assert_clean([col], context="test")


def test_the_refusal_says_which_column_and_why():
    """An exception that names no column cannot be acted on by whoever sees it."""
    with pytest.raises(firewall.LeakageViolation) as exc:
        firewall.assert_clean(["F3892"], context="test")
    msg = str(exc.value)
    assert "F3892" in msg
    assert "fairness" in msg.lower()


def test_all_offending_columns_are_reported_not_just_the_first():
    with pytest.raises(firewall.LeakageViolation) as exc:
        firewall.assert_clean(["F3892", "F3924", "F3916"], context="test")
    msg = str(exc.value)
    assert all(c in msg for c in ("F3892", "F3924", "F3916"))


def test_the_shipped_bundle_still_passes():
    """The tightening must refuse more without refusing the model that ships -
    otherwise the scoring service stops loading and this is a regression, not a
    fix."""
    import joblib
    b = joblib.load(settings.MODELS_DIR / "final_bundle.joblib")
    firewall.assert_clean(b["feature_list_selected"], context="bundle")
    firewall.assert_clean(b["feature_list_kept"], context="bundle")


def test_a_quarantined_column_hidden_among_legitimate_ones_is_still_caught():
    import joblib
    b = joblib.load(settings.MODELS_DIR / "final_bundle.joblib")
    smuggled = list(b["feature_list_selected"][:50]) + ["F3917"]
    with pytest.raises(firewall.LeakageViolation):
        firewall.assert_clean(smuggled, context="test")
