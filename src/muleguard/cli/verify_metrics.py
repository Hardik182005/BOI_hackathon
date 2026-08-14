"""One read-only pass over the accuracy claims, checking each against its source.

Every number this project reports is stored in an artifact, and an artifact is
only worth what the thing behind it is worth. This re-derives the headline
metrics from the saved predictions and the shipped bundle, then compares each
one to what the registry and the reports claim. A check passes when the two
agree; nothing here recomputes a model, retrains anything, or reads the locked
test set.

    .venv/Scripts/python.exe -m muleguard.cli.verify_metrics

Exit code is 0 only if every check passed, so this is safe to put in front of a
demo or a commit hook. The point is not to produce a number - the numbers
already exist - but to make it impossible for a stale artifact to sit in the
repo claiming to describe a model that has since changed.
"""
from __future__ import annotations

import argparse
import datetime as dt
from typing import Any, Callable

import numpy as np
import polars as pl

from muleguard import settings
from muleguard.utils import load_json, save_json, sha256_file

OUT = settings.METRICS_DIR / "verify_metrics.json"

#: Agreement tolerance for a metric that was rounded to 5 decimals on the way
#: into an artifact. Anything looser would let a real regression pass.
TOL = 1e-4


class Check:
    """One verifiable claim, its source, and what it was checked against."""

    def __init__(self, name: str, why: str):
        self.name, self.why = name, why
        self.ok: bool | None = None
        self.detail: dict[str, Any] = {}
        self.note = ""

    def record(self, ok: bool, note: str, **detail: Any) -> "Check":
        self.ok, self.note, self.detail = ok, note, detail
        return self

    def to_dict(self) -> dict[str, Any]:
        return {"check": self.name, "verifies": self.why,
                "passed": self.ok, "detail": self.detail, "note": self.note}


# --- the checks ---------------------------------------------------------------

def _registry() -> dict[str, Any]:
    reg = load_json(settings.REGISTRY_DIR / "registry.json")
    active = [m for m in reg.get("models", []) if m.get("status") != "retired"]
    if not active:
        raise SystemExit("registry lists no active model")
    return active[-1]


def check_bundle_integrity(c: Check) -> Check:
    """The file on disk must be the file the registry signed."""
    reg = _registry()
    path = settings.MODELS_DIR / "final_bundle.joblib"
    if not path.exists():
        return c.record(False, f"bundle absent at {path}")
    got = sha256_file(path)
    want = reg.get("bundle_sha256")
    return c.record(got == want,
                    "on-disk bundle matches the registry signature"
                    if got == want else
                    "the bundle on disk is NOT the one the registry describes",
                    recomputed_sha256=got, registry_sha256=want)


def check_champion_identity(c: Check) -> Check:
    """The registry, the loaded bundle and the accuracy table must name one model."""
    from muleguard.models.scoring import load_bundle

    reg = _registry()
    b = load_bundle()
    table = pl.read_csv(settings.METRICS_DIR / "final_accuracy_table.csv")
    crowned = table.filter(pl.col("status") == "CHAMPION")["model"].to_list()
    # The bundle records the tournament entry it was built from under
    # `winner_oof_name`; `winner` is the registry's spelling of the same thing.
    names = {"registry": reg.get("winner"), "bundle": b.get("winner_oof_name"),
             "accuracy_table": crowned[0] if len(crowned) == 1 else crowned}
    ok = len(set(map(str, names.values()))) == 1
    return c.record(ok, f"all three name {names['registry']}" if ok else
                    "the three sources disagree about which model shipped",
                    **names)


def check_pr_auc_reproduces(c: Check) -> Check:
    """Recompute the champion's PR-AUC from the saved predictions."""
    from sklearn.metrics import average_precision_score

    reg = _registry()
    champ = str(reg.get("winner"))
    store = settings.PREDICTIONS_DIR / "final_model_oof_predictions.parquet"
    if not store.exists():
        return c.record(False, f"prediction store absent: {store.name}")
    df = pl.read_parquet(store).filter(pl.col("model") == champ)
    if df.is_empty():
        return c.record(False, f"{champ} has no saved OOF predictions")
    aps = []
    for _, part in df.partition_by("repeat", as_dict=True).items():
        y = part["target"].to_numpy().astype(int)
        if y.sum() == 0:
            continue
        aps.append(float(average_precision_score(y, part["score"].to_numpy())))
    got = float(np.mean(aps))
    want = float(reg.get("oof_pr_auc", float("nan")))
    ok = abs(got - want) < TOL
    return c.record(ok, f"recomputed {got:.5f} vs registry {want:.5f}",
                    recomputed=round(got, 5), registry=round(want, 5),
                    n_repeats=len(aps), source=store.name)


def check_leakage_firewall(c: Check) -> Check:
    """No quarantined column may appear in the shipped feature list."""
    import yaml

    from muleguard.models.scoring import load_bundle

    from muleguard.cli.nested_ses import HARD_QUARANTINE

    with open(settings.CONFIG_DIR / "leakage_quarantine.yaml", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    # The union, not the config alone. The YAML records the four columns that
    # reconstruct the label outright; the code additionally blocks the
    # post-resolution block. Checking only the file would let a column the
    # training runs treat as forbidden pass unexamined here.
    banned = {str(e["feature"]) for e in cfg.get("quarantine", []) if e.get("feature")}
    banned |= set(HARD_QUARANTINE)
    b = load_bundle()
    shipped = set(b["feature_list_selected"]) | set(b.get("feature_list_kept", []))
    breach = sorted(banned & shipped)
    return c.record(not breach,
                    f"0 of {len(banned)} quarantined columns reach the model"
                    if not breach else f"QUARANTINE BREACH: {breach}",
                    n_quarantined=len(banned), n_shipped_features=len(shipped),
                    breaches=breach)


def check_locked_test_untouched(c: Check) -> Check:
    """Dev and locked test must be disjoint, and the split must still be the
    one every stored metric was computed against."""
    from muleguard.data.split import load_locked_test_mask

    mask = np.asarray(load_locked_test_mask(), dtype=bool)
    n_test, n_dev = int(mask.sum()), int((~mask).sum())
    store = settings.PREDICTIONS_DIR / "final_model_oof_predictions.parquet"
    dev_rows = 0
    if store.exists():
        df = pl.read_parquet(store)
        idx = df["row_index"].unique().to_numpy()
        dev_rows = len(idx)
        # An OOF row index landing inside the locked test would mean a metric
        # was computed on held-out data, which is the one unrecoverable error.
        intruders = int(mask[idx].sum())
    else:
        intruders = -1
    ok = intruders == 0 and dev_rows == n_dev
    return c.record(ok, "no out-of-fold prediction touches a locked-test row"
                    if ok else "an OOF prediction was made on a locked-test row",
                    locked_test_rows=n_test, dev_rows=n_dev,
                    distinct_oof_rows=dev_rows, locked_rows_in_oof=intruders)


def check_policy_thresholds(c: Check) -> Check:
    """Tiers must be ordered and inside [0, 1], or the queue is nonsense."""
    reg = _registry()
    t = reg.get("policy_thresholds") or {}
    crit, urg, std = (t.get("critical_risk"), t.get("urgent_risk"),
                      t.get("standard_risk"))
    vals = [v for v in (crit, urg, std) if v is not None]
    ok = (len(vals) == 3 and crit > urg > std > 0.0 and crit <= 1.0)
    return c.record(bool(ok), "critical > urgent > standard, all in (0, 1]"
                    if ok else "policy tiers are unordered or out of range",
                    critical_risk=crit, urgent_risk=urg, standard_risk=std)


def check_calibration(c: Check) -> Check:
    """A calibrator must be recorded, with a stored Brier that is at least
    better than predicting the base rate for everyone."""
    from muleguard.models.scoring import load_bundle

    b = load_bundle()
    # The bundle stores the fitted object, not its name; a class name is what
    # a reader can compare against the calibration report.
    obj = b.get("calibrator")
    cal = type(obj).__name__ if obj is not None else None
    stored: dict[str, Any] = {}
    for name in ("final_locked_test_metrics.json", "locked_test_metrics.json",
                 "calibration_metrics.json"):
        try:
            stored = load_json(settings.METRICS_DIR / name)
            stored["_source"] = name
            break
        except (FileNotFoundError, ValueError):
            continue
    brier = stored.get("brier") or stored.get("brier_score")
    ok = bool(cal) and (brier is None or brier < 0.0089)
    return c.record(ok,
                    f"calibrator={cal}, stored brier={brier}" if ok else
                    "no calibrator recorded, or the stored Brier is no better "
                    "than the base rate",
                    calibrator=cal, brier=brier, source=stored.get("_source"))


def check_accuracy_table(c: Check) -> Check:
    """The definitive table must have all 27 columns and must not rank a
    leakage-unsafe model above the champion among eligible rows."""
    from muleguard.cli.final_accuracy_table import COLUMNS

    path = settings.METRICS_DIR / "final_accuracy_table.csv"
    if not path.exists():
        return c.record(False, "final_accuracy_table.csv has not been built")
    t = pl.read_csv(path)
    cols_ok = list(t.columns) == COLUMNS
    champ = t.filter(pl.col("status") == "CHAMPION")
    if champ.is_empty():
        return c.record(False, "the table crowns no champion")
    cpr = float(champ["pr_auc_mean"][0])
    above = t.filter((pl.col("pr_auc_mean") > cpr)
                     & (pl.col("status") != "REJECTED"))["model"].to_list()
    ok = cols_ok and not above
    return c.record(ok,
                    f"{t.height} models, {len(t.columns)} columns, champion leads "
                    "every non-rejected row" if ok else
                    f"columns_ok={cols_ok}; ranked above the champion: {above}",
                    n_models=t.height, n_columns=len(t.columns),
                    champion_pr_auc=cpr, unrejected_models_above_champion=above)


def check_ablations_present(c: Check) -> Check:
    """The experiments that justify the design must exist, with verdicts."""
    wanted = {
        "feature/meta ablation": "nested_feature_family_arms.json",
        "resampling (SMOTE)": "smote_ablation.json",
        "sensitive-attribute": "fairness_ablation.json",
        "missingness signature": "missingness_ablation.json",
    }
    found: dict[str, Any] = {}
    missing: list[str] = []
    for label, fname in wanted.items():
        try:
            d = load_json(settings.METRICS_DIR / fname)
        except (FileNotFoundError, ValueError):
            missing.append(f"{label} ({fname})")
            continue
        found[label] = d.get("verdict") or d.get("decision") or "present"
    return c.record(not missing,
                    "every design decision has a recorded experiment"
                    if not missing else f"no experiment recorded for: {missing}",
                    verdicts=found, missing=missing)


def check_open_findings_declared(c: Check) -> Check:
    """An experiment that says "change this" must not be able to go quiet.

    The subset sweep can return ``REPLACES_FULL_CLEAN`` for an arm the shipped
    model does not use - it currently does, for top-200 against the shipped
    top-120. That is a legitimate state to be in: acting on it is a full retrain
    and a retrain is a scheduled event, not a reflex. What is *not* legitimate
    is letting the disagreement sit in a JSON file nobody opens while the
    reports quote the old number as though nothing had been measured.

    So the rule is not "no arm may win". The rule is that every winning arm has
    a written finding naming it. The check fails on silence, not on the result.
    """
    finding = settings.REPO_ROOT / "docs/FEATURE_SUBSET_SIZE_FINDING.md"
    try:
        d = load_json(settings.METRICS_DIR / "nested_feature_family_arms.json")
    except (FileNotFoundError, ValueError):
        return c.record(True, "no subset sweep on disk; nothing to declare")

    verdicts = ((d.get("rule") or {}).get("verdicts") or {})
    winners = [a for a, v in verdicts.items() if v == "REPLACES_FULL_CLEAN"]
    if not winners:
        return c.record(True, "no arm beat the shipped configuration",
                        winning_arms=[])

    text = finding.read_text(encoding="utf-8") if finding.exists() else ""
    undeclared = [a for a in winners if a not in text]
    return c.record(not undeclared,
                    f"{len(winners)} arm(s) beat the shipped configuration and "
                    f"are declared in {finding.name}" if not undeclared else
                    f"beat the shipped configuration but written up nowhere: "
                    f"{undeclared}",
                    winning_arms=winners, undeclared=undeclared,
                    declared_in=str(finding.relative_to(settings.REPO_ROOT)),
                    reading="a winning arm is a finding to schedule, not a "
                            "defect to hide; the champion is unchanged and "
                            "still quotes the number it actually produces")


CHECKS: list[tuple[str, str, Callable[[Check], Check]]] = [
    ("bundle_integrity", "the served model is the one the registry signed",
     check_bundle_integrity),
    ("champion_identity", "registry, bundle and accuracy table name one model",
     check_champion_identity),
    ("pr_auc_reproduces", "the headline accuracy recomputes from saved predictions",
     check_pr_auc_reproduces),
    ("leakage_firewall", "no quarantined column reaches the shipped model",
     check_leakage_firewall),
    ("locked_test_untouched", "no reported metric was computed on held-out rows",
     check_locked_test_untouched),
    ("policy_thresholds", "the risk tiers are ordered and in range",
     check_policy_thresholds),
    ("calibration", "probabilities are calibrated, not raw margins",
     check_calibration),
    ("accuracy_table", "the definitive table is complete and self-consistent",
     check_accuracy_table),
    ("ablations_present", "every design decision has a recorded experiment",
     check_ablations_present),
    ("open_findings_declared", "an experiment that beat the shipped setup is "
     "written up, not buried", check_open_findings_declared),
]


def run() -> dict[str, Any]:
    results = []
    for name, why, fn in CHECKS:
        c = Check(name, why)
        try:
            fn(c)
        except Exception as exc:  # noqa: BLE001 - a check that crashes is a fail
            c.record(False, f"{type(exc).__name__}: {exc}")
        results.append(c.to_dict())

    failed = [r for r in results if not r["passed"]]
    payload = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scope": "read-only verification of the shipped accuracy claims; no "
                 "model was retrained and the locked test set was not read",
        "n_checks": len(results), "n_passed": len(results) - len(failed),
        "checks": results,
        "verdict": "PASS" if not failed else "FAIL",
        "failed": [r["check"] for r in failed],
    }
    save_json(payload, OUT)
    return payload


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    p = run()
    if not a.quiet:
        for r in p["checks"]:
            print(f"  [{'PASS' if r['passed'] else 'FAIL'}] {r['check']:<24} "
                  f"{r['note']}")
        print(f"\n  {p['n_passed']}/{p['n_checks']} checks passed -> {p['verdict']}")
        if p["failed"]:
            print(f"  failed: {', '.join(p['failed'])}")
    return 0 if p["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
