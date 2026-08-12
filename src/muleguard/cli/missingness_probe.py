"""Evidence behind the missingness write-up - an exploratory probe, not a test.

Everything this CLI computes is measured on **all** development rows at once.
That makes it useful for understanding the data and useless for deciding
anything: a univariate AUC computed on the same rows it is reported for is
in-sample, and the strongest of 358 candidate columns is the maximum of 358
noisy statistics. So the output artifact is stamped

    EXPLORATORY - NOT A SELECTION INSTRUMENT

and the decision about whether the missingness block enters the model is taken
elsewhere, by the paired nested ablation in
:mod:`muleguard.cli.missingness_ablation`. The two are deliberately separate
files so that the tempting shortcut - "column X looked good here, let me add
it" - has to be typed out on purpose rather than happening by accident.

What the probe is *for* is the three questions that the ablation cannot answer,
because they are about provenance rather than performance:

1. **Is the missingness structural or incidental?** Volume, spread across
   columns, and how much of it sits in a band where an indicator could carry
   contrast at all.
2. **Is any apparent signal actually leakage?** The strongest indicator is
   checked against the target *and* against every quarantined post-resolution
   marker. If a missingness column tracks the bank's investigation workflow
   rather than the account's behaviour, it has to be rejected regardless of how
   well it scores - and the only way to know is to look.
3. **Could the pattern be a sample-size artifact?** With 64 positives, "the
   mules only take 6 distinct values here" is exactly what 64 draws from a wide
   distribution look like some of the time. That claim needs a null model
   before it is worth repeating, so this builds one.

Writes ``artifacts/metrics/missingness_probe.json``.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from collections import Counter
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from muleguard import settings
from muleguard.data import ingest
from muleguard.features import frame as frame_mod
from muleguard.features.missingness import PREFIX, MissingnessSignature, describe
from muleguard.logging import configure, get_logger
from muleguard.models import harness

log = get_logger("cli.missingness_probe")

#: Post-resolution markers held out of the model. F3924 is the target itself;
#: the rest are the investigation-workflow columns quarantined by the firewall.
#: They are loaded here *only* as audit references and never as inputs.
_MARKERS = ("F3924", "F3912", "F3913", "F3914", "F3898", "F3899", "F3915")

#: F3898 and F3899 are small-count integers rather than flags, so "the event
#: happened" is ``> 0`` for them and ``== 1`` for the binary markers.
_MULTI_VALUED = ("F3898", "F3899")


def _binarize(col: np.ndarray, name: str) -> np.ndarray:
    return (col > 0).astype(np.int8) if name in _MULTI_VALUED else (col == 1).astype(np.int8)


def _safe_auc(y: np.ndarray, s: np.ndarray) -> float | None:
    """ROC-AUC, or ``None`` when the label or the score is constant."""
    if len(np.unique(y)) < 2 or len(np.unique(s)) < 2:
        return None
    return float(roc_auc_score(y, s))


def _cohort_shape(values: np.ndarray) -> dict[str, Any]:
    """How concentrated one group's values are on a discrete column."""
    counts = Counter(values.tolist())
    modal_value, modal_count = counts.most_common(1)[0]
    return {
        "n_rows": int(values.size),
        "n_distinct": len(counts),
        "modal_value": float(modal_value),
        "modal_share": round(modal_count / values.size, 5),
    }


def _sample_size_control(legit: np.ndarray, n_draw: int, observed: dict[str, Any],
                         *, n_draws: int, seed: int) -> dict[str, Any]:
    """Null model for the concentration claim.

    Draws ``n_draw`` legitimate rows repeatedly and asks how often a group that
    size looks as concentrated as the mules do. If the answer is "often", the
    fingerprint is an artifact of having 64 positives and nothing more.
    """
    rng = np.random.default_rng(seed)
    distinct = np.empty(n_draws, dtype=np.int32)
    modal = np.empty(n_draws, dtype=np.float64)
    for i in range(n_draws):
        draw = legit[rng.choice(legit.size, size=n_draw, replace=False)]
        c = Counter(draw.tolist())
        distinct[i] = len(c)
        modal[i] = c.most_common(1)[0][1] / n_draw
    return {
        "n_draws": n_draws,
        "seed": seed,
        "rows_per_draw": int(n_draw),
        "distinct_min": int(distinct.min()),
        "distinct_mean": round(float(distinct.mean()), 3),
        "modal_share_max": round(float(modal.max()), 5),
        "modal_share_mean": round(float(modal.mean()), 5),
        # One-sided empirical p-values: how many equally-sized draws of ordinary
        # accounts were at least as concentrated as the mules.
        "p_distinct_le_observed": round(
            float((distinct <= observed["n_distinct"]).mean()), 5),
        "p_modal_share_ge_observed": round(
            float((modal >= observed["modal_share"]).mean()), 5),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repeats", type=int, default=3,
                    help="only selects which dev split to describe; no CV is run")
    ap.add_argument("--max-flags", type=int, default=200)
    ap.add_argument("--top", type=int, default=25, help="indicators to list")
    ap.add_argument("--control-draws", type=int, default=2000)
    ap.add_argument("--control-seed", type=int, default=42)
    args = ap.parse_args(argv)

    configure()

    raw = ingest.load_dataset()
    frame = frame_mod.build_model_frame()
    registry = frame_mod.augmented_registry()["features"]
    dev = harness.dev_split(args.repeats)
    rows = dev.row_index

    Xdev = frame.X[rows]
    ydev = np.asarray(frame.y)[rows].astype(np.int8)
    n_pos = int(ydev.sum())

    # --- 1. is the missingness structural? ---------------------------------
    nan = np.isnan(Xdev)
    rate = nan.mean(axis=0)
    row_ratio = nan.mean(axis=1)
    total_null = nan.sum(axis=1).astype(np.float64)

    # The admitted width happens to equal the raw width: the firewall removes 13
    # quarantined columns and the MG_* block adds 13, so 3,925 goes in and 3,925
    # comes out. That coincidence reads like "nothing was removed", so the
    # artifact carries the check rather than the reader having to trust the count.
    admitted = set(frame.feature_names)
    still_present = [m for m in _MARKERS + ("F3916", "F3917", "F3918", "F2230",
                                            "F3892", "__UNNAMED__0")
                     if m in admitted]

    structure = {
        "n_dev_rows": int(Xdev.shape[0]),
        "n_dev_positives": n_pos,
        "n_admitted_columns": int(Xdev.shape[1]),
        "admitted_width_note": (
            "equal to the raw width by coincidence: 13 quarantined columns "
            "removed, 13 MG_* meta-features added"),
        "quarantined_columns_found_in_admitted_set": still_present,
        "cell_missing_fraction": round(float(nan.mean()), 6),
        "n_columns_with_any_missing": int((rate > 0).sum()),
        "n_columns_fully_missing": int((rate == 1.0).sum()),
        "n_columns_in_1_to_99pct_band": int(((rate >= 0.01) & (rate <= 0.99)).sum()),
        "row_missing_ratio_negatives": round(float(row_ratio[ydev == 0].mean()), 5),
        "row_missing_ratio_positives": round(float(row_ratio[ydev == 1].mean()), 5),
        "naive_total_null_roc_auc": round(_safe_auc(ydev, total_null) or float("nan"), 5),
        "interpretation": (
            "Missingness is structural, not incidental - it touches almost every "
            "column and a large minority of columns sit in a band where an "
            "indicator can carry contrast. The naive volume count separates the "
            "classes only slightly, which is the reason to encode the shape "
            "rather than the amount."),
    }

    # --- 2. univariate view of the signature block -------------------------
    sig = MissingnessSignature.fit(Xdev, frame.feature_names, registry,
                                   max_flags=args.max_flags)
    block = sig.transform(Xdev)
    aucs: list[tuple[str, float]] = []
    for j, name in enumerate(sig.names):
        a = _safe_auc(ydev, block[:, j].astype(np.float64))
        if a is not None:
            aucs.append((name, a))
    aucs.sort(key=lambda kv: -kv[1])
    top = [{"column": n, "roc_auc": round(a, 5), "gloss": describe(n)}
           for n, a in aucs[: args.top]]

    # --- 3. leakage audit on the strongest indicator ------------------------
    lead_name, lead_auc = aucs[0]
    lead = block[:, sig.names.index(lead_name)].astype(np.float64)

    markers = {m: _binarize(raw[m].to_numpy()[rows], m)
               for m in _MARKERS if m in raw.columns}
    target = markers["F3924"]

    marker_audit = []
    for m, mb in markers.items():
        marker_audit.append({
            "marker": m,
            "is_target": m == "F3924",
            "prevalence": round(float(mb.mean()), 5),
            "auc_from_leading_indicator": (
                lambda v: round(v, 5) if v is not None else None)(_safe_auc(mb, lead)),
            "agreement_with_target": round(float((mb == target).mean()), 5),
            "positives_shared_with_target": int(((mb == 1) & (target == 1)).sum()),
        })
    marker_audit.sort(key=lambda d: -(d["auc_from_leading_indicator"] or 0.0))

    # Agreement alone cannot identify a target proxy. At 0.88 % prevalence, two
    # unrelated rare columns agree on ~99 % of rows simply by both being almost
    # always zero: F3915 agrees with the target on 98.885 % of rows while sharing
    # *none* of its 64 positives. A proxy has to actually cover the positives, so
    # the test is agreement AND positive coverage.
    n_target_pos = int(target.sum())

    def _is_proxy(d: dict[str, Any]) -> bool:
        if d["is_target"] or d["agreement_with_target"] < 0.99:
            return False
        return d["positives_shared_with_target"] >= 0.5 * n_target_pos

    proxies = [d for d in marker_audit if _is_proxy(d)]
    independent = [d for d in marker_audit
                   if not d["is_target"] and not _is_proxy(d)]
    ind_aucs = [d["auc_from_leading_indicator"] for d in independent
                if d["auc_from_leading_indicator"] is not None]

    leakage = {
        "leading_indicator": lead_name,
        "leading_indicator_roc_auc_vs_target": round(lead_auc, 5),
        "markers": marker_audit,
        "proxy_rule": (
            "agreement with target >= 0.99 AND at least half the target's "
            "positives shared - agreement alone is satisfied by any two rare "
            "columns and would misclassify a merely co-rare marker as a proxy"),
        "n_target_positives": n_target_pos,
        "target_proxies_found": [
            {"marker": d["marker"],
             "agreement_with_target": d["agreement_with_target"],
             "positives_shared_with_target": d["positives_shared_with_target"]}
            for d in proxies],
        "co_rare_not_proxy": [
            {"marker": d["marker"],
             "agreement_with_target": d["agreement_with_target"],
             "positives_shared_with_target": d["positives_shared_with_target"],
             "prevalence": d["prevalence"]}
            for d in independent
            if d["agreement_with_target"] >= 0.95
            and d["positives_shared_with_target"] < 0.5 * n_target_pos],
        "independent_marker_auc_min": round(min(ind_aucs), 5) if ind_aucs else None,
        "independent_marker_auc_max": round(max(ind_aucs), 5) if ind_aucs else None,
        "verdict": (
            "A high AUC against a marker that is itself a near-perfect copy of "
            "the target is the target restated, not a second, independent leak - "
            "so it is not evidence of workflow leakage. The markers that are "
            "genuinely independent of the target are the ones that matter here, "
            "and if they sit near 0.5 the indicator is not tracking the bank's "
            "investigation process."),
    }

    # --- 4. cohort concentration and its null model -------------------------
    fingerprint: dict[str, Any] = {"column": lead_name}
    if lead_name.startswith(f"{PREFIX}FAMCNT__") or lead_name.startswith(f"{PREFIX}CTXCNT__"):
        pos_vals, neg_vals = lead[ydev == 1], lead[ydev == 0]
        obs_pos = _cohort_shape(pos_vals)
        fingerprint.update({
            "mules": obs_pos,
            "legitimate": _cohort_shape(neg_vals),
            "sample_size_control": _sample_size_control(
                neg_vals, n_pos, obs_pos,
                n_draws=args.control_draws, seed=args.control_seed),
        })
    else:
        fingerprint["skipped"] = (
            "leading indicator is not a count column, so a distinct-value "
            "concentration statistic would not be meaningful")

    out = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "EXPLORATORY - NOT A SELECTION INSTRUMENT",
        "measured_on": (
            "all development rows at once, in-sample; these numbers describe the "
            "data and must not be used to choose columns. The decision instrument "
            "is muleguard.cli.missingness_ablation."),
        "signature_shape": {
            "n_flags": int(sig.flag_idx.size),
            "n_families": len(sig.family_idx),
            "n_window_pair_groups": len(sig.pair_idx),
            "n_context_blocks": len(sig.context_idx),
            "n_columns_total": len(sig.names),
        },
        "structure": structure,
        "top_indicators_in_sample": top,
        "leakage_audit": leakage,
        "cohort_fingerprint": fingerprint,
    }

    settings.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    path = settings.METRICS_DIR / "missingness_probe.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    log.info("cells missing %.4f%%  band columns %d  naive AUC %.5f",
             100 * structure["cell_missing_fraction"],
             structure["n_columns_in_1_to_99pct_band"],
             structure["naive_total_null_roc_auc"])
    log.info("leading indicator %s AUC %.5f", lead_name, lead_auc)
    if proxies:
        log.info("target proxies among markers: %s",
                 ", ".join(f"{d['marker']} ({d['agreement_with_target']:.5f})"
                           for d in proxies))
    log.info("independent marker AUC range %s..%s",
             leakage["independent_marker_auc_min"],
             leakage["independent_marker_auc_max"])
    log.info("wrote %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
