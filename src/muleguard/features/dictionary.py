"""Description.xlsx -> machine-readable semantic feature registry.

`Description.xlsx` is a first-class input: it is the ONLY source of feature
meaning in this project. Nothing downstream is allowed to invent a domain
interpretation for an ``Fxxxx`` column - every human-readable label shown in
the UI, in a ProofGraph edge or in an LLM prompt is derived here.

The parser is deliberately mechanical. It reads the four supplied columns
(Feature / Variable Name / Description / Bank_Finalized_Variables) and derives,
by regular expression over the *variable name*, four orthogonal facets:

  feature_family      what is being measured  (CASH, UPI, BAL, ALERT, ...)
  transform_family    how it is measured      (RATIO_WINDOW, DEVIATION, MIN, ...)
  window              the observation window  (L7D, L7_14D, L14_31D, ...)
  direction           credit / debit / both

plus ``availability_class`` (when the value is knowable relative to the alert
decision) and ``leakage_status``. Those last two drive the firewall in
``muleguard.features.firewall`` and are the reason the accepted model can be
proven free of post-resolution information.

Nothing here reads DataSet.xlsx.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from muleguard import settings
from muleguard.logging import get_logger
from muleguard.utils import save_json, sha256_file

log = get_logger("features.dictionary")

DESCRIPTION_SHEET = "Data_Dicitionary"  # organiser's spelling, kept verbatim

# --------------------------------------------------------------------------
# Availability classes (section 6 of the build spec).
# --------------------------------------------------------------------------
BEHAVIORAL = "BEHAVIORAL"
PROFILE = "PROFILE"
ALERT_CONTEXT = "ALERT_CONTEXT"
PRE_EXISTING_RISK_CONTEXT = "PRE_EXISTING_RISK_CONTEXT"
POST_RESOLUTION_LEAKAGE = "POST_RESOLUTION_LEAKAGE"
TARGET = "TARGET"
INDEX_OR_ID = "INDEX_OR_ID"
UNKNOWN_REVIEW = "UNKNOWN_REVIEW"

SAFE = "SAFE"
QUARANTINED = "QUARANTINED"
REVIEW = "REVIEW"

# Explicit per-feature overrides. These come from the organiser's own
# description text ("Resolution status flag", "Customer risk level flag",
# "Target variable") - they are read from the file, not guessed, but pinned
# here so a typo upstream cannot silently reopen a leak.
EXPLICIT_AVAILABILITY: dict[str, str] = {
    # --- post-outcome: only knowable AFTER an analyst resolved the alert ---
    "F3898": POST_RESOLUTION_LEAKAGE,  # MIN_RESOLVE_DAYS
    "F3899": POST_RESOLUTION_LEAKAGE,  # MAX_RESOLVE_DAYS
    "F3912": POST_RESOLUTION_LEAKAGE,  # FRAUD_SUSPECTED
    "F3913": POST_RESOLUTION_LEAKAGE,  # OTHER_RESOLUTION
    "F3914": POST_RESOLUTION_LEAKAGE,  # FALSE_POSITIVE
    "F3915": POST_RESOLUTION_LEAKAGE,  # UNATTENDED
    # --- ambiguous timing: quarantined until proven pre-decision ---
    "F3916": PRE_EXISTING_RISK_CONTEXT,  # L3_FLG
    "F3917": PRE_EXISTING_RISK_CONTEXT,  # L2_FLG
    "F3918": PRE_EXISTING_RISK_CONTEXT,  # L1_FLG
    # --- static customer/account profile ---
    "F3886": PROFILE, "F3887": PROFILE, "F3888": PROFILE, "F3889": PROFILE,
    "F3890": PROFILE, "F3891": PROFILE, "F3892": PROFILE, "F3893": PROFILE,
    "F3894": PROFILE,
    # --- alert context available at/ before the decision point ---
    "F3895": ALERT_CONTEXT, "F3896": ALERT_CONTEXT, "F3897": ALERT_CONTEXT,
    "F3900": ALERT_CONTEXT, "F3901": ALERT_CONTEXT, "F3902": ALERT_CONTEXT,
    "F3903": ALERT_CONTEXT, "F3904": ALERT_CONTEXT, "F3905": ALERT_CONTEXT,
    "F3906": ALERT_CONTEXT, "F3907": ALERT_CONTEXT, "F3908": ALERT_CONTEXT,
    "F3909": ALERT_CONTEXT, "F3910": ALERT_CONTEXT, "F3911": ALERT_CONTEXT,
    "F3919": ALERT_CONTEXT, "F3920": ALERT_CONTEXT, "F3921": ALERT_CONTEXT,
    "F3922": ALERT_CONTEXT, "F3923": ALERT_CONTEXT,
    # --- target ---
    "F3924": TARGET,
    # MNTH is the data snapshot month, not account behaviour. In the supplied
    # extract it separates the label classes perfectly (every negative is the
    # 2025-10 snapshot; every positive is Sep/Nov/Dec), so it is a sampling
    # artifact of how the extract was assembled - never a predictive feature.
    "F2230": INDEX_OR_ID,
}

# Sensitive / protected attributes (section 24).
SENSITIVE_FEATURES = {
    "F3892": "gender",
    "F3894": "age",
    "F3891": "occupation",
    "F3890": "geography",
}

# --------------------------------------------------------------------------
# Semantic families, matched against the VARIABLE NAME only.
# Order matters: first match wins, so specific rails precede generic ones.
# --------------------------------------------------------------------------
def _tok(*words: str) -> str:
    """Token-boundary regex: underscores are word characters, so ``\\b`` is
    useless here. Matches WORD at a start/underscore boundary."""
    body = "|".join(words)
    return rf"(?:^|_)(?:{body})(?:_|$)"


_RAIL_PATTERNS: list[tuple[str, str]] = [
    ("UPI_XFER", r"UPI_XFER"),
    ("UPI", _tok("UPI")),
    ("NON_CASH_CHQ", r"NON_CASH_CHQ"),
    ("CHEQUE", _tok("CHQ", "CHEQUE")),
    ("CASH", _tok("CASH")),
    ("ELEC_XFER", r"ELEC_XFER"),
    ("NET_BANKING", r"NET_BNKING|NETBNKING"),
    ("MOBILE_BANKING", r"MOB_BNKING|MOBILE_BNKING|MBNKING"),
    ("ATM", _tok("ATM")),
    ("POS_MERCHANT", r"POS_PYMT|" + _tok("POS")),
    ("BBPS", _tok("BBPS")),
    ("GST", _tok("GST")),
    ("LOAN", _tok("LOAN")),
    ("STANDING_INSTRUCTION", r"STDNG_INSTR"),
    ("AADHAAR_PAYMENT_BRIDGE", _tok("APB")),
    ("FEES_AND_CHARGES", r"FEES_CHRGS"),
    ("CASH_INTENSIVE", _tok("CI")),
    ("IMPS", _tok("IMPS")),
    ("NEFT", _tok("NEFT")),
    ("RTGS", _tok("RTGS")),
    ("BALANCE", _tok("BAL") + r"|BALANCE"),
    ("TRANSFER", _tok("XFER") + r"|TRANSFER"),
    ("TOTAL_ALL_RAILS", _tok("TOT") + r"|TOT_TXNAMT"),
]

_TRANSFORM_PATTERNS: list[tuple[str, str]] = [
    ("RANGE_MAX_MINUS_MIN", r"^MM_|^MAXMIN_|^MMX_|MAX_MINUS_MIN"),
    ("RATIO_CREDIT_INTENSITY", r"^R_CI_|^RA_CI_"),
    ("RATIO_OF_AVERAGES", r"^RA_"),
    ("RATIO_WINDOW", r"^R_"),
    ("DEV_TOTAL_VS_AVG", r"^D_TA_"),
    ("DEVIATION_OF_AVERAGE", r"^DA_"),
    ("DEVIATION", r"^D_"),
    ("AVERAGE", r"^AVG_"),
    ("MINIMUM", r"^MIN_"),
    ("MAXIMUM", r"^MAX_"),
    ("COUNT", r"^CNT_|^COUNT_|_TXNS?(_|$)"),
]

# Windows such as L7D, L14D, L31D, L7_14D, L14_31D, L7_31D, 7DAYS, 14DAYS ...
_WINDOW_RE = re.compile(r"L(\d+)(?:_(\d+))?D\b|(\d+)\s*TO\s*(\d+)DAYS|\b(\d+)DAYS\b")

_DIRECTION_RE = [
    ("CREDIT", re.compile(r"_CR(_|$)|_CR\b|CREDIT")),
    ("DEBIT", re.compile(r"_DB(_|$)|_DB\b|DEBIT")),
]


@dataclass
class FeatureRecord:
    """One row of the semantic registry."""

    feature: str
    variable_name: str
    description: str
    bank_finalized: bool
    feature_family: str
    transform_family: str
    window: str
    direction: str
    availability_class: str
    sensitive: bool
    sensitive_kind: str | None
    leakage_status: str
    semantic_tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _match_first(name: str, patterns: list[tuple[str, str]], default: str) -> str:
    for label, pattern in patterns:
        if re.search(pattern, name):
            return label
    return default


def _parse_window(name: str) -> str:
    m = _WINDOW_RE.search(name)
    if not m:
        return "NONE"
    a, b, c, d, e = m.groups()
    if a and b:
        return f"L{a}_{b}D"
    if a:
        return f"L{a}D"
    if c and d:
        return f"L{c}_{d}D"
    if e:
        return f"L{e}D"
    return "NONE"


def _parse_direction(name: str) -> str:
    hits = [label for label, rx in _DIRECTION_RE if rx.search(name)]
    if len(hits) == 1:
        return hits[0]
    return "BOTH"


def _semantic_tags(rec_name: str, description: str, family: str,
                   transform: str, window: str, direction: str) -> list[str]:
    tags = {family, transform}
    if window != "NONE":
        tags.add(f"WINDOW_{window}")
    if direction != "BOTH":
        tags.add(direction)
    text = f"{rec_name} {description}".upper()
    if "AMT" in text or "AMOUNT" in text:
        tags.add("AMOUNT")
    if re.search(r"\bTXN|TRANSACTION", text):
        tags.add("TRANSACTION_COUNT")
    if "ALERT" in text:
        tags.add("ALERT")
    if "OCC" in text or "OCCUPATION" in text:
        tags.add("OCCUPATION_SEGMENT")
    if "SEGMENT" in text:
        tags.add("SEGMENT")
    if "SHORT" in window or window in {"L7D", "L7_14D"}:
        tags.add("SHORT_WINDOW")
    if window in {"L31D", "L14_31D", "L7_31D"}:
        tags.add("LONG_WINDOW")
    return sorted(t for t in tags if t)


def _availability_for(feature: str, variable_name: str, description: str) -> str:
    if feature in EXPLICIT_AVAILABILITY:
        return EXPLICIT_AVAILABILITY[feature]
    desc = (description or "").strip().lower()
    if "resolution status" in desc:
        return POST_RESOLUTION_LEAKAGE
    if "target variable" in desc:
        return TARGET
    if "customer risk level" in desc:
        return PRE_EXISTING_RISK_CONTEXT
    if "alert description flag" in desc or desc.startswith("count of alerts"):
        return ALERT_CONTEXT
    # Everything else in this dictionary is a windowed transaction/balance
    # aggregate measured before the alert decision.
    return BEHAVIORAL


def _leakage_status_for(availability: str) -> str:
    if availability in (POST_RESOLUTION_LEAKAGE, TARGET, INDEX_OR_ID):
        return QUARANTINED
    if availability in (PRE_EXISTING_RISK_CONTEXT, UNKNOWN_REVIEW):
        return REVIEW
    return SAFE


def parse_description_workbook(path: Path) -> list[FeatureRecord]:
    """Parse Description.xlsx into ordered ``FeatureRecord`` objects."""
    raw = pd.read_excel(path, sheet_name=DESCRIPTION_SHEET)
    expected = {"Feature", "Variable Name", "Description", "Bank_Finalized_Variables"}
    missing = expected - set(raw.columns)
    if missing:
        raise ValueError(f"Description.xlsx missing expected columns: {sorted(missing)}")

    bank_finalized_names = {
        str(v).strip()
        for v in raw["Bank_Finalized_Variables"].dropna().tolist()
        if str(v).strip() and str(v).strip().lower() != "target variable"
    }

    records: list[FeatureRecord] = []
    for row in raw.itertuples(index=False):
        feature = str(row.Feature).strip()
        variable_name = str(getattr(row, "_1")).strip()  # "Variable Name"
        description = str(getattr(row, "Description") or "").strip()
        upper = variable_name.upper()

        family = _match_first(upper, _RAIL_PATTERNS, "OTHER")
        if family == "OTHER" and re.search(r"BALANCE|MINANCE", description.upper()):
            # Balance-profile variables whose names drop the BAL token
            # (RT_AVG_*, DEV_AVG_*, MAX_MIN_*) - the description carries it.
            family = "BALANCE"
        transform = _match_first(upper, _TRANSFORM_PATTERNS, "RAW_OR_META")
        window = _parse_window(upper)
        direction = _parse_direction(upper)
        availability = _availability_for(feature, variable_name, description)

        # Profile/alert/target rows are not transaction aggregates; label their
        # family from the availability class so the UI groups them sensibly.
        if availability in (PROFILE, ALERT_CONTEXT, POST_RESOLUTION_LEAKAGE,
                            PRE_EXISTING_RISK_CONTEXT, TARGET, INDEX_OR_ID):
            if family == "OTHER":
                family = availability

        records.append(
            FeatureRecord(
                feature=feature,
                variable_name=variable_name,
                description=description,
                bank_finalized=variable_name in bank_finalized_names,
                feature_family=family,
                transform_family=transform,
                window=window,
                direction=direction,
                availability_class=availability,
                sensitive=feature in SENSITIVE_FEATURES,
                sensitive_kind=SENSITIVE_FEATURES.get(feature),
                leakage_status=_leakage_status_for(availability),
                semantic_tags=_semantic_tags(variable_name, description, family,
                                             transform, window, direction),
            )
        )
    return records


def build_registry(description_path: Path | None = None,
                   out_dir: Path | None = None) -> dict[str, Any]:
    """Parse the workbook and persist the JSON + CSV registry artifacts."""
    description_path = Path(description_path or _default_description_path())
    out_dir = Path(out_dir or settings.FEATURES_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = parse_description_workbook(description_path)
    payload = {
        "source_file": description_path.name,
        "source_sha256": sha256_file(description_path),
        "n_features": len(records),
        "availability_counts": _counts(r.availability_class for r in records),
        "leakage_counts": _counts(r.leakage_status for r in records),
        "family_counts": _counts(r.feature_family for r in records),
        "transform_counts": _counts(r.transform_family for r in records),
        "bank_finalized": [r.feature for r in records if r.bank_finalized],
        "sensitive": [r.feature for r in records if r.sensitive],
        "features": {r.feature: r.to_dict() for r in records},
    }
    save_json(payload, out_dir / "feature_dictionary.json")
    frame = pd.DataFrame([r.to_dict() for r in records])
    frame["semantic_tags"] = frame["semantic_tags"].map(lambda t: "|".join(t))
    frame.to_csv(out_dir / "feature_dictionary.csv", index=False)
    log.info("feature dictionary built: %d features -> %s", len(records), out_dir)
    return payload


def _counts(values) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def _default_description_path() -> Path:
    """Locate Description.xlsx: repo root first, then the parent upload dir."""
    for candidate in (
        settings.REPO_ROOT / "Description.xlsx",
        settings.RAW_DIR / "Description.xlsx",
        settings.REPO_ROOT.parent / "Description.xlsx",
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Description.xlsx not found in repo root, data/raw/ or the parent directory"
    )


_REGISTRY_CACHE: dict[str, Any] | None = None


def load_registry(path: Path | None = None) -> dict[str, Any]:
    """Load the persisted registry, building it on first use if absent."""
    global _REGISTRY_CACHE
    if _REGISTRY_CACHE is not None and path is None:
        return _REGISTRY_CACHE
    p = Path(path or settings.FEATURES_DIR / "feature_dictionary.json")
    if not p.exists():
        payload = build_registry()
    else:
        from muleguard.utils import load_json

        payload = load_json(p)
    if path is None:
        _REGISTRY_CACHE = payload
    return payload


def describe(feature: str, registry: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the registry record for one feature.

    Callers must use this (never a hand-written string) whenever a feature is
    shown to a human. Unknown columns get an explicit UNKNOWN_REVIEW record so
    that a missing dictionary entry is visible rather than silently invented.
    """
    reg = registry or load_registry()
    rec = reg["features"].get(feature)
    if rec is not None:
        return rec
    return {
        "feature": feature,
        "variable_name": feature,
        "description": "No entry in Description.xlsx - meaning not established.",
        "bank_finalized": False,
        "feature_family": "OTHER",
        "transform_family": "RAW_OR_META",
        "window": "NONE",
        "direction": "BOTH",
        "availability_class": UNKNOWN_REVIEW,
        "sensitive": False,
        "sensitive_kind": None,
        "leakage_status": REVIEW,
        "semantic_tags": [],
    }
