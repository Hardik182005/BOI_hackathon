"""Feature Availability Firewall - the single gate every feature must pass.

Nothing in this project selects, trains, calibrates, explains or scores on a
column that has not been admitted here. The firewall answers one question per
column - *was this value knowable when the analyst had to decide?* - and
refuses everything else.

Two guarantees follow, and both are asserted by the release gate:

  1. No accepted artifact contains a POST_RESOLUTION_LEAKAGE, TARGET or
     INDEX_OR_ID column.
  2. Every admitted column carries a recorded availability class, so a model
     manifest can state exactly which classes of evidence produced a score.

The firewall is intentionally the *narrow* path: `admitted_features()` starts
from the full column list and removes, never adds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from muleguard import settings
from muleguard.features import dictionary as fd
from muleguard.logging import get_logger

log = get_logger("features.firewall")

# Availability classes that may ever reach a model.
ADMISSIBLE_CLASSES = (fd.BEHAVIORAL, fd.PROFILE, fd.ALERT_CONTEXT)

# Classes that make the release gate fail if found in an accepted artifact.
FORBIDDEN_CLASSES = (fd.POST_RESOLUTION_LEAKAGE, fd.TARGET, fd.INDEX_OR_ID)


@dataclass
class FirewallDecision:
    """Outcome of applying the firewall to one column list."""

    admitted: list[str]
    rejected: dict[str, str]  # feature -> reason
    classes_used: dict[str, int] = field(default_factory=dict)
    families_used: dict[str, int] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "n_admitted": len(self.admitted),
            "n_rejected": len(self.rejected),
            "availability_classes_used": self.classes_used,
            "feature_families_used": dict(
                sorted(self.families_used.items(), key=lambda kv: -kv[1])
            ),
        }


@dataclass
class FirewallConfig:
    """Materialised policy from configs/feature_availability.yaml."""

    policy_version: str
    hard_quarantine: dict[str, str]
    conditional_quarantine: dict[str, str]
    fairness_excluded: dict[str, str]
    contextual_only: list[str]
    views: dict[str, dict[str, Any]]

    @classmethod
    def load(cls) -> "FirewallConfig":
        cfg = settings.load_config("feature_availability")
        return cls(
            policy_version=str(cfg.get("policy_version", "0")),
            hard_quarantine={
                e["feature"]: e["reason"] for e in cfg.get("hard_quarantine", [])
            },
            conditional_quarantine={
                e["feature"]: e["reason"] for e in cfg.get("conditional_quarantine", [])
            },
            fairness_excluded={
                e["feature"]: e["reason"]
                for e in cfg.get("fairness", {}).get("excluded_by_default", [])
            },
            contextual_only=list(cfg.get("fairness", {}).get("contextual_only", [])),
            views=dict(cfg.get("views", {})),
        )


_CONFIG_CACHE: FirewallConfig | None = None


def config() -> FirewallConfig:
    global _CONFIG_CACHE
    if _CONFIG_CACHE is None:
        _CONFIG_CACHE = FirewallConfig.load()
    return _CONFIG_CACHE


def admitted_features(
    columns: Sequence[str],
    *,
    allow_classes: Iterable[str] = ADMISSIBLE_CLASSES,
    include_conditional: bool = False,
    include_gender: bool = False,
    include_demographics: bool = True,
    families: Iterable[str] | None = None,
    bank_finalized_only: bool = False,
    registry: dict[str, Any] | None = None,
) -> FirewallDecision:
    """Filter ``columns`` down to what may legitimately enter a model.

    Args:
        columns: candidate column names (raw dataset order preserved).
        allow_classes: availability classes permitted for this run.
        include_conditional: admit the L1/L2/L3 risk flags. Only ever true for
            an explicitly labelled ablation, never for the accepted model.
        include_gender: admit F3892. Requires a documented justification.
        include_demographics: admit age/occupation/area contextual fields.
        families: restrict to these semantic families (view E).
        bank_finalized_only: restrict to the bank's finalized variables (view C).
    """
    reg = registry or fd.load_registry()
    cfg = config()
    allow = set(allow_classes)
    family_filter = set(families) if families else None

    admitted: list[str] = []
    rejected: dict[str, str] = {}
    classes_used: dict[str, int] = {}
    families_used: dict[str, int] = {}

    for col in columns:
        rec = fd.describe(col, reg)
        cls = rec["availability_class"]

        if col in cfg.hard_quarantine:
            rejected[col] = f"HARD_QUARANTINE: {cfg.hard_quarantine[col]}"
            continue
        if cls in FORBIDDEN_CLASSES:
            rejected[col] = f"FORBIDDEN_CLASS {cls}: {rec['variable_name']}"
            continue
        if col in cfg.conditional_quarantine and not include_conditional:
            rejected[col] = f"CONDITIONAL_QUARANTINE: {cfg.conditional_quarantine[col]}"
            continue
        if col in cfg.fairness_excluded and not include_gender:
            rejected[col] = f"FAIRNESS_EXCLUDED: {cfg.fairness_excluded[col]}"
            continue
        if not include_demographics and col in cfg.contextual_only:
            rejected[col] = "DEMOGRAPHICS_DISABLED for this run"
            continue
        if cls not in allow and not (include_conditional and col in cfg.conditional_quarantine):
            rejected[col] = f"CLASS_NOT_IN_VIEW {cls}"
            continue
        if bank_finalized_only and not rec["bank_finalized"]:
            rejected[col] = "NOT_BANK_FINALIZED"
            continue
        if family_filter is not None and rec["feature_family"] not in family_filter:
            rejected[col] = f"FAMILY_NOT_IN_VIEW {rec['feature_family']}"
            continue

        admitted.append(col)
        classes_used[cls] = classes_used.get(cls, 0) + 1
        fam = rec["feature_family"]
        families_used[fam] = families_used.get(fam, 0) + 1

    return FirewallDecision(admitted, rejected, classes_used, families_used)


def view_features(view_name: str, columns: Sequence[str],
                  registry: dict[str, Any] | None = None) -> FirewallDecision:
    """Apply a named view from configs/feature_availability.yaml."""
    cfg = config()
    if view_name not in cfg.views:
        raise KeyError(f"unknown view {view_name!r}; known: {sorted(cfg.views)}")
    spec = cfg.views[view_name]
    return admitted_features(
        columns,
        allow_classes=spec.get("classes", ADMISSIBLE_CLASSES),
        families=spec.get("families"),
        bank_finalized_only=bool(spec.get("restrict_to_bank_finalized", False)),
        registry=registry,
    )


def assert_clean(features: Sequence[str], *, context: str = "model",
                 registry: dict[str, Any] | None = None) -> None:
    """Raise if any feature would violate the release blockers.

    Called by the release gate, by bundle freezing and by the scoring service
    so a leaking artifact cannot be shipped even if a training script is wrong.

    The three grounds are checked here, not two. Until 2026-08-14 this function
    covered ``hard_quarantine`` and the forbidden availability classes only,
    which left **4 of the manifest's 13 columns** able to pass: ``F3892``
    (GENDER, excluded on fairness grounds rather than availability) and
    ``F3916``/``F3917``/``F3918`` (the L1-L3 alert flags, whose class is
    ``PRE_EXISTING_RISK_CONTEXT`` and therefore not forbidden). Nothing shipped
    with them - ``release_gate.no_target_or_f3912_leakage`` tests the bundle
    against the full manifest and is the check that was actually holding - but
    the docstring above promises a second line of defence, and for those four
    columns there was only ever one. A guarantee with a hole in it that happens
    not to be exercised is still a guarantee with a hole in it.

    The three config lists below are exactly the three that ``quarantine_manifest``
    writes into the 13-entry file, so this check and the release gate now refuse
    the same set by construction rather than by two lists agreeing.
    """
    reg = registry or fd.load_registry()
    cfg = config()
    violations: list[str] = []
    grounds = (
        (cfg.hard_quarantine, ""),
        (cfg.conditional_quarantine, "quarantined until proven pre-decision: "),
        (cfg.fairness_excluded, "excluded by fairness policy: "),
    )
    for col in features:
        for listing, prefix in grounds:
            if col in listing:
                violations.append(f"{col} ({prefix}{listing[col]})")
                break
        else:
            rec = fd.describe(col, reg)
            if rec["availability_class"] in FORBIDDEN_CLASSES:
                violations.append(f"{col} ({rec['variable_name']}: "
                                  f"{rec['availability_class']})")
    if violations:
        raise LeakageViolation(
            f"{context} contains quarantined feature(s): " + "; ".join(violations)
        )


class LeakageViolation(RuntimeError):
    """Raised when a quarantined column reaches an accepted artifact."""


def quarantine_manifest(columns: Sequence[str] | None = None,
                        registry: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build artifacts/features/quarantined_features.json content."""
    reg = registry or fd.load_registry()
    cfg = config()
    entries = []
    for feat, reason in cfg.hard_quarantine.items():
        rec = fd.describe(feat, reg)
        entries.append({
            "feature": feat,
            "variable_name": rec["variable_name"],
            "availability_class": rec["availability_class"],
            "reason": reason,
            "disposition": "EXCLUDED_FROM_ALL_TRAINING",
        })
    for feat, reason in cfg.conditional_quarantine.items():
        rec = fd.describe(feat, reg)
        entries.append({
            "feature": feat,
            "variable_name": rec["variable_name"],
            "availability_class": rec["availability_class"],
            "reason": reason,
            "disposition": "QUARANTINE_UNTIL_PROVEN_PRE_DECISION",
        })
    for feat, reason in cfg.fairness_excluded.items():
        rec = fd.describe(feat, reg)
        entries.append({
            "feature": feat,
            "variable_name": rec["variable_name"],
            "availability_class": rec["availability_class"],
            "reason": reason,
            "disposition": "EXCLUDED_BY_FAIRNESS_POLICY",
        })
    payload: dict[str, Any] = {
        "policy_version": cfg.policy_version,
        "policy": (
            "Quarantined features are excluded from every model, ensemble, "
            "selector, calibrator, explanation and export. Ablations that "
            "include them are labelled REJECTED LEAKAGE evidence and are never "
            "candidate models."
        ),
        "quarantine": entries,
    }
    if columns is not None:
        present = [e["feature"] for e in entries if e["feature"] in set(columns)]
        payload["present_in_dataset"] = present
    return payload
