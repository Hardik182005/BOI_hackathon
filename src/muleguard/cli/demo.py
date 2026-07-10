"""End-to-end demo scenarios (judge narrative, Scenes 3-6).

Selects real accounts from the DEV set (never locked test) that exemplify:
  1. high-risk mule-like account            (Lens 1)
  2. legitimate business look-alike          (Lens 2 protection)
  3. model-disagreement case                 (Lens 3 second opinion)
  4. OOD case (synthetic perturbation, labelled as such)
  5. low-current-risk monitoring case
  6. Ollama disconnected -> deterministic narrative
  7. hallucinated LLM output -> validator rejection (simulated malformed text)

Writes artifacts/evidence/demo_scenarios.json and, when the API is running,
scores each scenario through HTTP so cases appear in the dashboard queue.
"""
from __future__ import annotations

import argparse
import datetime as dt

import numpy as np
import polars as pl

from muleguard import settings
from muleguard.data import ingest, split as split_mod
from muleguard.llm.deterministic_fallback import deterministic_narrative
from muleguard.llm.schemas import NarratorInput, ReasonFact
from muleguard.llm.validator import validate_llm_output
from muleguard.logging import get_logger
from muleguard.models.scoring import load_bundle, score_rows
from muleguard.utils import save_json, set_global_seed

log = get_logger("cli.demo")


def pick_scenarios() -> dict:
    set_global_seed(settings.GLOBAL_SEED)
    bundle = load_bundle()
    df = ingest.load_dataset()
    test_mask = split_mod.load_locked_test_mask()
    dev = df.filter(~pl.Series(test_mask))
    y = dev[settings.TARGET_COLUMN].cast(pl.Int32).to_numpy()

    log.info("scoring dev set for scenario selection...")
    results = score_rows(dev, bundle=bundle, with_explanations=False)
    risk = np.array([r["calibrated_risk"] for r in results])
    agree = np.array([r["model_agreement"] for r in results])
    anom = np.array([r["anomaly_percentile"] for r in results])
    verifier = np.array([r["verifier_confirms_risk"] for r in results])

    scenarios: dict[str, dict] = {}

    # 1. true mule with highest calibrated risk
    pos_idx = np.where(y == 1)[0]
    s1 = pos_idx[np.argmax(risk[pos_idx])]
    scenarios["high_risk_mule"] = {"dev_row": int(s1), "why": "labelled mule with highest calibrated risk"}

    # 2. business look-alike: legitimate, high screener risk, verifier says look-alike
    neg_idx = np.where((y == 0) & (~verifier))[0]
    s2 = neg_idx[np.argmax(risk[neg_idx])] if len(neg_idx) else int(np.argmax(risk * (y == 0)))
    scenarios["business_lookalike"] = {
        "dev_row": int(s2),
        "why": "legitimate account with high screener score where the hard-negative "
               "verifier disagrees - Lens 2 routes to review instead of punishment",
    }

    # 3. disagreement: supervised low, anomaly high
    low_sup = (risk < np.quantile(risk, 0.5)) & (anom > 99.0) & (y == 0)
    cand = np.where(low_sup)[0]
    s3 = int(cand[0]) if len(cand) else int(np.argmax(anom * (y == 0)))
    scenarios["model_disagreement"] = {
        "dev_row": s3,
        "why": "supervised score low but anomaly challenger in the top percentile - "
               "escalated to review, never certified clean",
    }

    # 5. monitoring case: low risk, high agreement, low anomaly
    calm = np.where((risk < np.quantile(risk, 0.2)) & (agree > 0.9) & (y == 0) & (anom < 50))[0]
    s5 = int(calm[0]) if len(calm) else int(np.argmin(risk))
    scenarios["monitor_low_risk"] = {
        "dev_row": s5,
        "why": "not currently flagged; monitoring continues (never 'certified safe')",
    }

    # 4. OOD: perturb the monitoring case with extreme values (synthetic, labelled)
    scenarios["ood_synthetic"] = {
        "base_dev_row": s5,
        "why": "SYNTHETIC perturbation of a real account: selected features pushed far "
               "outside the training range - system must route to OOD_REVIEW",
        "synthetic": True,
    }
    return scenarios


def run_llm_scenes() -> dict:
    """Scenes 6b/6c: deterministic fallback + hallucination rejection."""
    ctx = NarratorInput(
        account_reference="DEMO-ACC-6",
        calibrated_risk=0.91,
        risk_tier="CRITICAL_REVIEW",
        model_agreement=0.86,
        conformal_status="HIGH_RISK_SET",
        ood_status="IN_DISTRIBUTION",
        top_reasons=[ReasonFact(feature="F1702", value=10.4, legitimate_percentile=99.2,
                                direction="INCREASES_RISK", shap_contribution=0.21)],
    )
    det = deterministic_narrative(ctx)
    # schema-valid on purpose so every CONTENT rule fires and is recorded:
    # altered score, altered tier, invented feature, invented amount, guilt
    # claim, disallowed action, missing required limitations
    hallucinated = (
        '{"summary": "Account DEMO-ACC-6 moved Rs. 4,50,000 through F999 and the '
        'holder is guilty of laundering.", "risk_tier": "MONITOR", '
        '"verified_risk_score": 0.05, "reason_codes": [], '
        '"recommended_checks": ["FREEZE_ACCOUNT_NOW"], "limitations": ["none"]}'
    )
    out, reasons = validate_llm_output(hallucinated, ctx)
    assert out is None, "validator must reject the planted hallucination"
    return {
        "deterministic_narrative_works_without_ollama": det.model_dump(),
        "hallucination_rejected": {"planted_output": hallucinated, "rejection_reasons": reasons},
    }


def main(via_api: bool = False) -> None:
    scenarios = pick_scenarios()
    llm_scenes = run_llm_scenes()

    bundle = load_bundle()
    df = ingest.load_dataset()
    test_mask = split_mod.load_locked_test_mask()
    dev = df.filter(~pl.Series(test_mask))

    detailed = {}
    for name, spec in scenarios.items():
        row_i = spec.get("dev_row", spec.get("base_dev_row"))
        row = dev.slice(row_i, 1)
        if spec.get("synthetic"):
            # push 10 selected numeric features to 1000x their dev max
            mod = {}
            for f in bundle["feature_list_selected"][:10]:
                if row.schema[f].is_numeric():
                    mod[f] = 1e9
            row = row.with_columns([pl.lit(v).alias(k) for k, v in mod.items()])
        res = score_rows(row, bundle=bundle, with_explanations=True)[0]
        detailed[name] = {
            **spec,
            "result": {k: v for k, v in res.items() if k not in ("ood_detail",)},
        }
        log.info("%s -> tier=%s risk=%.3f", name, res["risk_tier"], res["calibrated_risk"])

    packet = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "note": "scenario rows come from the DEV set only; the OOD case is an "
                "explicitly labelled synthetic perturbation",
        "scenarios": detailed,
        "llm_scenes": llm_scenes,
    }
    save_json(packet, settings.EVIDENCE_DIR / "demo_scenarios.json")

    if via_api:
        import httpx

        for name, d in detailed.items():
            row_i = d.get("dev_row", d.get("base_dev_row"))
            row = dev.slice(row_i, 1).to_dicts()[0]
            row.pop(settings.TARGET_COLUMN, None)
            feats = {k: (str(v) if hasattr(v, "isoformat") else v) for k, v in row.items()}
            if d.get("synthetic"):
                for f in bundle["feature_list_selected"][:10]:
                    if isinstance(feats.get(f), (int, float)):
                        feats[f] = 1e9
            r = httpx.post("http://127.0.0.1:8001/v1/score",
                           json={"account_reference": f"DEMO-{name.upper()[:20]}",
                                 "features": feats}, timeout=60)
            log.info("API %s -> %s", name, r.status_code)

    checks = {n: d["result"]["risk_tier"] for n, d in detailed.items()}
    print("DEMO SCENARIOS:", checks)
    ood_ok = detailed["ood_synthetic"]["result"]["ood_status"] == "OUT_OF_DISTRIBUTION"
    print(f"OOD routing works: {ood_ok}; hallucination rejected: True; "
          f"deterministic fallback: True")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--via-api", action="store_true",
                    help="also push scenarios through the running API")
    main(via_api=ap.parse_args().via_api)
