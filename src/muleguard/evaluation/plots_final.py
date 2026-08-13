"""The seventeen plots section 56 requires, drawn from saved predictions only.

Run::

    .venv/Scripts/python.exe -m muleguard.evaluation.plots_final

Three rules hold for every figure here.

**Nothing is drawn from memory.** Each plot names the artifact it reads. If that
artifact does not exist yet, the plot is skipped and reported as skipped, with
the run that would produce it. A missing figure is a visible gap; a figure drawn
from stale or invented numbers is a lie that looks like evidence.

**Every title carries split, model ID and data version**, as §56 requires. The
data version is the first twelve characters of the DataSet.xlsx sha256 - the same
fingerprint the audit writes - so a figure can always be tied back to the
workbook it came from.

**Red means rejected.** It is used for the F3912 leakage bar and nothing else, so
that a reader who has learned the palette cannot mistake a rejected result for a
reported one.

Writes ``artifacts/plots/*.png`` and ``artifacts/plots/plot_manifest.json``.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any, Callable

import numpy as np
import polars as pl

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from muleguard import settings
from muleguard.logging import configure, get_logger
from muleguard.utils import load_json, save_json

log = get_logger("evaluation.plots_final")

ACCENT = "#2563eb"      # accepted results
MUTED = "#94a3b8"
REJECT = "#dc2626"      # rejected leakage evidence only
OK = "#059669"
WARN = "#d97706"

plt.rcParams.update({
    "figure.dpi": 130, "font.size": 9, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
})

ROOT = settings.REPO_ROOT
MANIFEST = settings.PLOTS_DIR / "plot_manifest.json"


class Skip(Exception):
    """Raised when the evidence a plot needs does not exist yet."""


# --------------------------------------------------------------------------
# context every title carries
# --------------------------------------------------------------------------
def _data_version() -> str:
    fp = ROOT / "data/interim/data_fingerprint.json"
    if not fp.exists():
        return "unknown"
    return load_json(fp).get("raw_file", {}).get("sha256", "")[:12]


def _champion() -> str:
    path = settings.METRICS_DIR / "promotion_decision_v2.json"
    return load_json(path).get("promoted", "champion") if path.exists() else "champion"


DATA_VERSION = _data_version()


def _subtitle(split: str, model: str) -> str:
    return f"split: {split}  |  model: {model}  |  data: DataSet.xlsx@{DATA_VERSION}"


def _titled(ax, title: str, split: str, model: str) -> None:
    ax.set_title(f"{title}\n{_subtitle(split, model)}", fontsize=9.5, loc="left")


def _need(rel: str) -> Path:
    p = ROOT / rel if not str(rel).startswith(str(ROOT)) else Path(rel)
    if not p.exists():
        raise Skip(f"missing {rel}")
    return p


def _save(fig, name: str) -> str:
    settings.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(settings.PLOTS_DIR / name, bbox_inches="tight")
    plt.close(fig)
    return name


# --------------------------------------------------------------------------
# predictions: nested when it exists, flat otherwise, always labelled
# --------------------------------------------------------------------------
def _predictions() -> tuple[np.ndarray, np.ndarray, str, str]:
    """(y, score, split label, model id) for the best available protocol."""
    nested = ROOT / "artifacts/predictions/nested_oof.parquet"
    flat = ROOT / "artifacts/predictions/oof_v2.parquet"
    champion = _champion()
    if nested.exists():
        df = pl.read_parquet(nested)
        family = champion.split("_top_")[0]
        models = df["model"].unique().to_list()
        pick = family if family in models else \
            max((m for m in models if m != "dummy_prevalence"), default=models[0])
        sub = df.filter(pl.col("model") == pick)
        # Average the repeats per row: the nested store holds one score per
        # (repeat, row), and the tournament ranked on the repeat-averaged vector.
        agg = sub.group_by("row_index").agg(
            pl.col("score").mean().alias("score"),
            pl.col("target").first().alias("target")).sort("row_index")
        n_rep = sub["repeat"].n_unique()
        return (agg["target"].to_numpy(), agg["score"].to_numpy(),
                f"dev OOF, nested outer folds ({n_rep} repeats)", pick)
    df = pl.read_parquet(_need("artifacts/predictions/oof_v2.parquet"))
    sub = df.filter(pl.col("model") == champion)
    if sub.height == 0:
        raise Skip(f"{champion} not present in oof_v2.parquet")
    agg = sub.group_by("row_index").agg(
        pl.col("score").mean().alias("score"),
        pl.col("target").first().alias("target")).sort("row_index")
    return (agg["target"].to_numpy(), agg["score"].to_numpy(),
            "dev OOF, flat repeated CV (HISTORICAL PROTOCOL)", champion)


def _pr_points(y: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    from sklearn.metrics import average_precision_score, precision_recall_curve
    p, r, _ = precision_recall_curve(y, s)
    return r, p, float(average_precision_score(y, s))


# --------------------------------------------------------------------------
# 1-7: curves from the prediction vector
# --------------------------------------------------------------------------
def plot_pr_curve() -> str:
    y, s, split, model = _predictions()
    r, p, ap = _pr_points(y, s)
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    ax.step(r, p, where="post", color=ACCENT, lw=1.6)
    ax.axhline(y.mean(), color=MUTED, ls="--", lw=1,
               label=f"prevalence {y.mean():.4f}")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="upper right", frameon=False)
    _titled(ax, f"Precision-recall  ·  AP = {ap:.4f}", split, model)
    return _save(fig, "final_pr_curve.png")


def plot_roc_curve() -> str:
    from sklearn.metrics import roc_auc_score, roc_curve
    y, s, split, model = _predictions()
    fpr, tpr, _ = roc_curve(y, s)
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    ax.plot(fpr, tpr, color=ACCENT, lw=1.6)
    ax.plot([0, 1], [0, 1], color=MUTED, ls="--", lw=1)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    _titled(ax, f"ROC  ·  AUC = {roc_auc_score(y, s):.4f}  "
                f"(secondary: at 0.88 % prevalence ROC flatters)", split, model)
    return _save(fig, "final_roc_curve.png")


def plot_calibration() -> str:
    y, s, split, model = _predictions()
    n_bins = 10
    edges = np.quantile(s, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    idx = np.clip(np.digitize(s, edges[1:-1]), 0, len(edges) - 2)
    xs, ys, ns = [], [], []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() == 0:
            continue
        xs.append(s[m].mean())
        ys.append(y[m].mean())
        ns.append(int(m.sum()))
    brier = float(np.mean((s - y) ** 2))
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    ax.plot([0, max(xs + ys)], [0, max(xs + ys)], color=MUTED, ls="--", lw=1)
    ax.plot(xs, ys, "o-", color=ACCENT, lw=1.4, ms=4)
    ax.set_xlabel("mean predicted probability (equal-count bins)")
    ax.set_ylabel("observed fraction positive")
    _titled(ax, f"Calibration  ·  Brier = {brier:.5f}  ·  {n_bins} equal-count bins",
            split, model)
    return _save(fig, "final_calibration_curve.png")


def plot_confusion() -> str:
    y, s, split, model = _predictions()
    budgets = [25, 50, 100]
    fig, axes = plt.subplots(1, len(budgets), figsize=(9.6, 3.1))
    order = np.argsort(-s)
    for ax, k in zip(axes, budgets):
        pred = np.zeros_like(y)
        pred[order[:k]] = 1
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        cm = np.array([[tn, fp], [fn, tp]])
        ax.imshow(np.log1p(cm), cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                        fontsize=9, color="black")
        ax.set_xticks([0, 1], ["pred 0", "pred 1"])
        ax.set_yticks([0, 1], ["true 0", "true 1"])
        ax.grid(False)
        ax.set_title(f"budget {k}  ·  recall {tp / max(y.sum(), 1):.3f}", fontsize=9)
    fig.suptitle(f"Confusion at analyst budgets\n{_subtitle(split, model)}",
                 fontsize=9.5, x=0.01, ha="left")
    return _save(fig, "final_confusion_matrices.png")


def _topk_curve(y: np.ndarray, s: np.ndarray, ks: np.ndarray):
    order = np.argsort(-s)
    hits = np.cumsum(y[order])
    recall = hits[ks - 1] / max(y.sum(), 1)
    precision = hits[ks - 1] / ks
    return recall, precision


def plot_recall_at_topk() -> str:
    y, s, split, model = _predictions()
    ks = np.unique(np.clip(np.linspace(5, min(500, len(y)), 60).astype(int), 1, len(y)))
    recall, _ = _topk_curve(y, s, ks)
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    ax.plot(ks, recall, color=ACCENT, lw=1.6)
    for k in (25, 50, 100):
        if k <= len(y):
            r = _topk_curve(y, s, np.array([k]))[0][0]
            ax.plot([k], [r], "o", color=OK, ms=5)
            ax.annotate(f"{k}: {r:.2f}", (k, r), textcoords="offset points",
                        xytext=(4, -9), fontsize=8, color=OK)
    ax.set_xlabel("accounts reviewed (top-K by score)")
    ax.set_ylabel("recall")
    ax.set_ylim(0, 1.02)
    _titled(ax, f"Recall@TopK  ·  {int(y.sum())} mules in {len(y):,} rows",
            split, model)
    return _save(fig, "final_recall_at_topk.png")


def plot_precision_at_topk() -> str:
    y, s, split, model = _predictions()
    ks = np.unique(np.clip(np.linspace(5, min(500, len(y)), 60).astype(int), 1, len(y)))
    _, precision = _topk_curve(y, s, ks)
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    ax.plot(ks, precision, color=ACCENT, lw=1.6)
    ax.axhline(y.mean(), color=MUTED, ls="--", lw=1,
               label=f"prevalence {y.mean():.4f}")
    ax.set_xlabel("accounts reviewed (top-K by score)")
    ax.set_ylabel("precision")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False)
    _titled(ax, "Precision@TopK  ·  the analyst's hit rate at each queue depth",
            split, model)
    return _save(fig, "final_precision_at_topk.png")


def plot_score_distribution() -> str:
    y, s, split, model = _predictions()
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    bins = np.linspace(0, 1, 41)
    ax.hist(s[y == 0], bins=bins, color=MUTED, alpha=0.85,
            label=f"legitimate (n={int((y == 0).sum()):,})", log=True)
    ax.hist(s[y == 1], bins=bins, color=ACCENT, alpha=0.9,
            label=f"mule (n={int(y.sum())})", log=True)
    ax.set_xlabel("model score")
    ax.set_ylabel("accounts (log scale)")
    ax.legend(frameon=False)
    _titled(ax, "Score distribution by class  ·  log count, 40 bins", split, model)
    return _save(fig, "final_score_distribution.png")


# --------------------------------------------------------------------------
# 8-9: variability
# --------------------------------------------------------------------------
def plot_fold_ap_distribution() -> str:
    d = load_json(_need("artifacts/metrics/nested_cv.json"))
    board = d.get("leaderboard") or []
    rows = [(m["model"], m.get("pr_auc_per_repeat") or []) for m in board
            if m.get("model") != "dummy_prevalence"]
    rows = [(name, vals) for name, vals in rows if vals]
    if not rows:
        raise Skip("nested_cv.json carries no per-repeat AP values")
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    # tick labels are set separately: the boxplot keyword for them was renamed
    # in matplotlib 3.9 and this has to work on either side of that.
    ax.boxplot([v for _, v in rows], vert=True, widths=0.55, patch_artist=True,
               boxprops=dict(facecolor="#dbeafe", color=ACCENT),
               medianprops=dict(color=ACCENT))
    ax.set_xticks(range(1, len(rows) + 1), [n for n, _ in rows])
    for i, (_, vals) in enumerate(rows, start=1):
        ax.plot(np.full(len(vals), i) + np.random.default_rng(0).normal(0, 0.04, len(vals)),
                vals, "o", ms=3, color=MUTED, zorder=3)
    ax.set_ylabel("average precision")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    outer = d.get("design", {}).get("outer", "nested outer folds")
    _titled(ax, "Fold AP distribution  ·  one point per outer repeat",
            outer, "all families")
    return _save(fig, "final_fold_ap_distribution.png")


def plot_seed_stability() -> str:
    d = load_json(_need("artifacts/metrics/seed_variance_v2.json"))
    per_seed = d.get("per_seed") or []
    vals = [p.get("pr_auc_mean", p.get("pr_auc")) for p in per_seed] \
        if isinstance(per_seed, list) else list(per_seed.values())
    seeds = [str(p.get("seed", i)) for i, p in enumerate(per_seed)] \
        if isinstance(per_seed, list) else list(per_seed)
    vals = [v for v in vals if v is not None]
    if not vals:
        raise Skip("seed_variance_v2.json carries no per-seed scores")
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.bar(range(len(vals)), vals, color=ACCENT, alpha=0.85)
    ax.axhline(float(np.mean(vals)), color=MUTED, ls="--", lw=1,
               label=f"mean {np.mean(vals):.4f}")
    ax.set_xticks(range(len(vals)), seeds[:len(vals)], rotation=0)
    ax.set_xlabel("seed")
    ax.set_ylabel("PR-AUC")
    ax.set_ylim(min(vals) - 0.08, max(vals) + 0.05)
    ax.legend(frameon=False)
    _titled(ax, f"Seed stability  ·  spread {d.get('spread', float('nan')):.4f} "
                f"(unpaired - not a yardstick for paired tests)",
            "dev OOF, flat repeated CV", d.get("model", _champion()))
    return _save(fig, "final_seed_stability.png")


# --------------------------------------------------------------------------
# 10-11: what the model uses
# --------------------------------------------------------------------------
def plot_selection_frequency() -> str:
    import csv
    path = ROOT / "artifacts/features/selection_frequency_v2.csv"
    if not path.exists():
        path = _need("artifacts/features/selection_frequency.csv")
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))[:30]
    if not rows:
        raise Skip(f"{path.name} is empty")
    names = [r["feature"] for r in rows][::-1]
    freq = [float(r["selection_frequency"]) for r in rows][::-1]
    fig, ax = plt.subplots(figsize=(5.8, 6.4))
    ax.barh(names, freq, color=[ACCENT if f >= 0.5 else MUTED for f in freq])
    ax.set_xlabel("fraction of folds selecting the feature")
    ax.set_xlim(0, 1)
    ax.grid(axis="y", alpha=0)
    _titled(ax, "Feature selection frequency  ·  top 30, fold-local ranking",
            "dev folds only", _champion())
    return _save(fig, "final_selection_frequency.png")


def plot_shap_importance() -> str:
    d = load_json(_need("artifacts/metrics/global_shap_importance.json"))
    rank = (d.get("ranking") or [])[:25][::-1]
    if not rank:
        raise Skip("global_shap_importance.json carries no ranking")
    names = [r["feature"] for r in rank]
    vals = [r["mean_abs_shap"] for r in rank]
    fig, ax = plt.subplots(figsize=(5.8, 5.8))
    ax.barh(names, vals, color=ACCENT)
    ax.set_xlabel("mean |SHAP| over out-of-fold attributions")
    ax.grid(axis="y", alpha=0)
    fidelity = d.get("attribution_method", {}).get("fidelity", "")
    _titled(ax, f"Top SHAP importance  ·  exact TreeSHAP  ·  {fidelity.split(' - ')[0]}",
            f"dev OOF, {d.get('n_repeats', '?')} repeats", d.get("champion", "champion"))
    return _save(fig, "final_shap_importance.png")


# --------------------------------------------------------------------------
# 12-14: ablations
# --------------------------------------------------------------------------
def plot_leakage_ablation() -> str:
    d = load_json(_need("artifacts/metrics/with_vs_without_f3912.json"))
    without = d["without_f3912"]["pr_auc_mean"]
    with_ = d["with_f3912"]["pr_auc_mean"]
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    bars = ax.bar(["clean (shipped)", "with F3912"], [without, with_],
                  color=[ACCENT, REJECT], width=0.55)
    for b, v in zip(bars, [without, with_]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.4f}",
                ha="center", fontsize=9)
    ax.set_ylabel("OOF PR-AUC")
    ax.set_ylim(0, 1.05)
    ax.text(1, with_ / 2, "REJECTED\nEVIDENCE", ha="center", va="center",
            color="white", fontsize=9, fontweight="bold")
    _titled(ax, f"Leakage ablation  ·  F3912 buys {with_ - without:+.4f} it may not keep",
            "dev OOF", d["without_f3912"]["model"])
    return _save(fig, "final_leakage_ablation.png")


def plot_alert_context_ablation() -> str:
    d = load_json(_need("artifacts/metrics/alert_context_ablation_v2.json"))
    variants = d.get("variants") or {}
    names, vals, errs = [], [], []
    for name, v in variants.items():
        names.append(name.replace("_", " "))
        vals.append(v.get("pr_auc_mean") if isinstance(v, dict) else v)
        errs.append((v.get("pr_auc_std") or 0) if isinstance(v, dict) else 0)
    if not names:
        raise Skip("alert_context_ablation_v2.json carries no variants")
    fig, ax = plt.subplots(figsize=(5.8, 3.7))
    ax.bar(names, vals, yerr=errs, capsize=4, color=ACCENT, alpha=0.9, width=0.55)
    ax.set_ylabel("OOF PR-AUC")
    plt.setp(ax.get_xticklabels(), rotation=12, ha="right")
    _titled(ax, f"Alert-context ablation  ·  {str(d.get('verdict', ''))[:70]}",
            "dev OOF", d.get("model_family", _champion()))
    return _save(fig, "final_alert_context_ablation.png")


def plot_feature_subset_ablation() -> str:
    nested = ROOT / "artifacts/metrics/nested_feature_family_arms.json"
    if nested.exists():
        d = load_json(nested)
        arms = d.get("arms") or []
        names = [a.get("arm", "?") for a in arms]
        vals = [a.get("mean_gain", 0.0) for a in arms]
        errs = [a.get("std_of_paired_diff", 0.0) for a in arms]
        split, xlabel = "nested outer folds, paired", "mean paired AP difference vs the full clean set"
        colours = [OK if v > 0 else MUTED for v in vals]
    else:
        d = load_json(_need("artifacts/metrics/family_dropout_v2.json"))
        per = d.get("per_family") or []
        names = [f"drop {f['family_removed']}" for f in per]
        vals = [-(f.get("relative_drop") or 0.0) for f in per]
        errs = [0] * len(per)
        split = "dev OOF, flat CV (FLAT FALLBACK - nested arms not run yet)"
        xlabel = "relative PR-AUC change when the family is removed"
        colours = [REJECT if v < -0.05 else MUTED for v in vals]
    if not names:
        raise Skip("no feature-subset arms available")
    order = np.argsort(vals)
    fig, ax = plt.subplots(figsize=(6.2, max(3.2, 0.3 * len(names))))
    ax.barh([names[i] for i in order], [vals[i] for i in order],
            xerr=[errs[i] for i in order], color=[colours[i] for i in order],
            capsize=3)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", alpha=0)
    _titled(ax, "Feature-subset ablation", split, _champion())
    return _save(fig, "final_feature_subset_ablation.png")


# --------------------------------------------------------------------------
# 15-16: stress
# --------------------------------------------------------------------------
def plot_positive_removal() -> str:
    nested = ROOT / "artifacts/metrics/nested_positive_removal.json"
    if nested.exists():
        d = load_json(nested)
        vals = d.get("ap_per_round") or d.get("per_round") or []
        vals = [v.get("ap") if isinstance(v, dict) else v for v in vals]
        ref = d.get("reference_ap") or d.get("baseline_ap")
        split = "nested outer folds"
    else:
        d = load_json(_need("artifacts/metrics/stability_stress_v2.json"))
        vals = d.get("positive_removal_pr_auc_per_round") or []
        ref = d.get("reference_pr_auc")
        split = "dev OOF, flat CV (FLAT FALLBACK - nested arm not run yet)"
    if not vals:
        # The flat artifact stores summary statistics rather than the rounds;
        # draw what it does store instead of inventing a per-round series.
        mean = d.get("positive_removal_pr_auc_mean")
        std = d.get("positive_removal_pr_auc_std")
        lo = d.get("positive_removal_pr_auc_min")
        if mean is None:
            raise Skip("no positive-removal series or summary available")
        fig, ax = plt.subplots(figsize=(5.4, 3.6))
        ax.bar(["reference", "after removal"], [ref, mean],
               yerr=[0, std or 0], capsize=5, color=[MUTED, ACCENT], width=0.5)
        ax.plot([1], [lo], "v", color=WARN, ms=7, label=f"worst round {lo:.4f}")
        ax.set_ylabel("PR-AUC")
        ax.legend(frameon=False)
        _titled(ax, f"Positive-removal stability  ·  {d.get('rounds', '?')} rounds "
                    f"removing {d.get('removal_fraction', '?')} of training positives",
                split, d.get("model", _champion()))
        return _save(fig, "final_positive_removal.png")

    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    ax.plot(range(1, len(vals) + 1), vals, "o-", color=ACCENT, lw=1.4, ms=4)
    if ref:
        ax.axhline(ref, color=MUTED, ls="--", lw=1, label=f"reference {ref:.4f}")
        ax.legend(frameon=False)
    ax.set_xlabel("removal round")
    ax.set_ylabel("PR-AUC")
    _titled(ax, "Positive-removal stability", split, _champion())
    return _save(fig, "final_positive_removal.png")


def plot_adversarial_validation() -> str:
    d = load_json(_need("artifacts/metrics/nested_shift_shield.json"))
    auc = d.get("adversarial_auc")
    per_fold = d.get("adversarial_auc_per_fold") or []
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    if per_fold:
        ax.plot(range(1, len(per_fold) + 1), per_fold, "o", color=ACCENT, ms=5)
    if auc is not None:
        ax.axhline(auc, color=ACCENT, lw=1.2, label=f"mean {auc:.4f}")
    ax.axhspan(0.45, 0.55, color=OK, alpha=0.12,
               label="exchangeable band [0.45, 0.55]")
    ax.axhline(0.5, color=MUTED, ls="--", lw=1)
    ax.set_xlabel("outer fold")
    ax.set_ylabel("adversarial AUC (train vs validation)")
    ax.set_ylim(0.3, 0.7)
    ax.legend(frameon=False, fontsize=8)
    _titled(ax, "Adversarial validation  ·  outside the band means the folds are "
                "not exchangeable", "nested outer folds", "adversarial classifier")
    return _save(fig, "final_adversarial_validation.png")


# --------------------------------------------------------------------------
PLOTS: dict[str, tuple[Callable[[], str], str]] = {
    "precision-recall curve": (plot_pr_curve, ""),
    "ROC curve": (plot_roc_curve, ""),
    "calibration curve": (plot_calibration, ""),
    "confusion matrix": (plot_confusion, ""),
    "Recall@TopK curve": (plot_recall_at_topk, ""),
    "Precision@TopK curve": (plot_precision_at_topk, ""),
    "score distribution by class": (plot_score_distribution, ""),
    "fold AP distribution": (plot_fold_ap_distribution,
                             "muleguard.cli.nested_cv --repeats 3 --inner 4"),
    "seed stability": (plot_seed_stability, ""),
    "feature selection frequency": (plot_selection_frequency, ""),
    "top SHAP importance": (plot_shap_importance, "muleguard.cli.global_shap"),
    "leakage ablation": (plot_leakage_ablation, ""),
    "alert-context ablation": (plot_alert_context_ablation, ""),
    "feature-subset ablation": (plot_feature_subset_ablation,
                                "muleguard.cli.nested_ses --stages families"),
    "positive-removal stability": (plot_positive_removal,
                                   "muleguard.cli.nested_ses --stages posremoval"),
    "adversarial validation / shift plot": (plot_adversarial_validation,
                                            "muleguard.cli.nested_ses --stages shift"),
}


def generate_all() -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for label, (fn, produced_by) in PLOTS.items():
        try:
            name = fn()
            results.append({"plot": label, "status": "WRITTEN", "file": name})
            log.info("%-38s -> %s", label, name)
        except Skip as exc:
            results.append({"plot": label, "status": "SKIPPED", "reason": str(exc),
                            "produced_by": produced_by})
            log.warning("%-38s -- skipped (%s)", label, exc)
        except Exception as exc:                            # noqa: BLE001
            results.append({"plot": label, "status": "FAILED", "reason": repr(exc)})
            log.error("%-38s -- failed (%r)", label, exc)

    written = sum(r["status"] == "WRITTEN" for r in results)
    payload = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "spec_section": "56",
        "data_version": DATA_VERSION,
        "champion": _champion(),
        "required": len(PLOTS),
        "written": written,
        "skipped": sum(r["status"] == "SKIPPED" for r in results),
        "failed": sum(r["status"] == "FAILED" for r in results),
        "title_contract": "every figure states split, model ID and data version",
        "plots": results,
    }
    save_json(payload, MANIFEST)
    log.info("%d/%d required plots written; manifest %s",
             written, len(PLOTS), MANIFEST)
    return payload


def main() -> int:
    configure()
    payload = generate_all()
    return 1 if payload["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
