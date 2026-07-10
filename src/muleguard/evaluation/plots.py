"""Required evidence plots. Every figure states split + metric in its title.

Style follows a restrained, consistent system: one hue for accepted results,
red reserved exclusively for rejected leakage evidence.
"""
from __future__ import annotations

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from muleguard import settings
from muleguard.utils import load_json

ACCENT = "#2563eb"      # accepted results
MUTED = "#94a3b8"
REJECT = "#dc2626"      # rejected leakage only
OK = "#059669"

plt.rcParams.update({
    "figure.dpi": 130, "font.size": 9, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
})


def _save(fig, name: str) -> None:
    settings.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(settings.PLOTS_DIR / name, bbox_inches="tight")
    plt.close(fig)


def plot_pr_curve(report: dict, name="precision_recall_curve.png") -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4))
    pr = report["pr_curve"]
    ax.plot(pr["recall"], pr["precision"], color=ACCENT, lw=1.8)
    ci = report["pr_auc"]
    ax.axhline(report["prevalence"], color=MUTED, ls="--", lw=1,
               label=f"no-skill = prevalence ({report['prevalence']:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(
        f"Precision-Recall - {report['split']}\n"
        f"PR-AUC {ci['point']:.3f} (95% CI {ci['ci_low']:.3f}-{ci['ci_high']:.3f}), "
        f"n={report['n']}, positives={report['n_positives']}"
    )
    ax.legend(loc="upper right", frameon=False)
    _save(fig, name)


def plot_recall_at_budget(report: dict, name="recall_at_budget.png") -> None:
    rows = report["recall_at_budget"]
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    budgets = [str(r["budget"]) for r in rows]
    x = np.arange(len(rows))
    ax.bar(x - 0.18, [r["recall"] for r in rows], width=0.36, color=ACCENT, label="recall")
    ax.bar(x + 0.18, [r["precision"] for r in rows], width=0.36, color=OK, label="precision")
    ax.set_xticks(x, [f"top {b}" for b in budgets])
    ax.set_ylim(0, 1.05)
    for i, r in enumerate(rows):
        ax.text(i - 0.18, r["recall"] + 0.02, f"{r['recall']:.2f}", ha="center", fontsize=8)
        ax.text(i + 0.18, r["precision"] + 0.02, f"{r['precision']:.2f}", ha="center", fontsize=8)
    ax.set_title(f"Recall / precision at analyst alert budgets - {report['split']}\n"
                 f"(metric: share of true mules caught / alert purity in top-K by score)")
    ax.legend(frameon=False)
    _save(fig, name)


def plot_calibration(report: dict, name="calibration_curve.png") -> None:
    cal = report.get("calibration")
    if not cal:
        return
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    xs = [b["mean_predicted"] for b in cal["bins"]]
    ys = [b["observed_rate"] for b in cal["bins"]]
    ns = [b["count"] for b in cal["bins"]]
    ax.plot([0, 1], [0, 1], color=MUTED, ls="--", lw=1, label="perfect calibration")
    ax.plot(xs, ys, "o-", color=ACCENT, lw=1.6, ms=4)
    for x, y, n in zip(xs, ys, ns):
        ax.annotate(f"n={n}", (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax.set_xlabel("Mean predicted risk (bin)")
    ax.set_ylabel("Observed mule rate (bin)")
    ax.set_title(f"Calibration - {report['split']}\n"
                 f"Brier {report.get('brier'):.5f}, ECE {cal['ece']:.4f} ({cal['n_bins']} bins)")
    ax.legend(frameon=False, loc="upper left")
    _save(fig, name)


def plot_leakage_ablation(name="leakage_ablation.png") -> None:
    data = load_json(settings.METRICS_DIR / "with_vs_without_f3912.json")
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    labels = ["No-skill\n(prevalence)", "LightGBM clean\n(F3912 quarantined)",
              "LightGBM + F3912\nREJECTED LEAKAGE"]
    prev = load_json(settings.REPO_ROOT / "data/interim/data_fingerprint.json")["positive_prevalence"]
    vals = [prev, data["without_f3912"]["pr_auc_mean"], data["with_f3912"]["pr_auc_mean"]]
    errs = [0, data["without_f3912"]["pr_auc_std"], data["with_f3912"]["pr_auc_std"]]
    colors = [MUTED, ACCENT, REJECT]
    bars = ax.barh(labels[::-1], vals[::-1], xerr=errs[::-1], color=colors[::-1], height=0.55)
    for bar, v in zip(bars, vals[::-1]):
        ax.text(min(v + 0.02, 1.02), bar.get_y() + bar.get_height() / 2, f"{v:.3f}",
                va="center", fontsize=9)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("PR-AUC (Average Precision), dev OOF repeated stratified 5-fold CV")
    ax.set_title("Leakage ablation - the F3912 result is REJECTED evidence, not a model score")
    ax.text(0.99, 0.02, "red = discarded leakage artifact", transform=ax.transAxes,
            ha="right", color=REJECT, fontsize=8)
    _save(fig, name)


def plot_model_comparison(name="model_comparison.png") -> None:
    m = load_json(settings.METRICS_DIR / "oof_metrics.json")["models"]
    rows = sorted(
        [(k, v) for k, v in m.items()],
        key=lambda kv: kv[1]["pr_auc_mean"],
    )
    fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(rows) + 1.6))
    for i, (k, v) in enumerate(rows):
        rejected = k.startswith("REJECTED")
        color = REJECT if rejected else ACCENT
        ax.barh(i, v["pr_auc_mean"], xerr=v["pr_auc_std"], color=color, height=0.6)
        ax.text(v["pr_auc_mean"] + 0.015, i, f"{v['pr_auc_mean']:.3f}", va="center", fontsize=8)
    ax.set_yticks(range(len(rows)),
                  [k + (" (REJECTED)" if k.startswith("REJECTED") else "") for k, _ in rows],
                  fontsize=8)
    ax.set_xlabel("PR-AUC mean +/- std across repeated 5-fold CV (dev OOF)")
    ax.set_title("Model tournament - dev OOF, natural prevalence")
    ax.set_xlim(0, 1.12)
    _save(fig, name)


def plot_feature_stability(name="feature_stability.png") -> None:
    import polars as pl

    freq = pl.read_csv(settings.FEATURES_DIR / "selection_frequency.csv").head(30)
    fig, ax = plt.subplots(figsize=(6, 6.4))
    feats = freq["feature"].to_list()[::-1]
    vals = freq["selection_frequency"].to_list()[::-1]
    ax.barh(feats, vals, color=ACCENT, height=0.62)
    ax.axvline(0.6, color=MUTED, ls="--", lw=1)
    ax.axvline(0.8, color=MUTED, ls=":", lw=1)
    ax.set_xlabel("Selection frequency (stability selection inside training folds)")
    ax.set_title("Top-30 feature stability - dev folds only\n"
                 "(LGBM gain + L1 logistic over repeated subsamples)")
    ax.tick_params(axis="y", labelsize=7)
    _save(fig, name)


def plot_confusion_matrices(name="confusion_matrices.png") -> None:
    import polars as pl

    tbl = pl.read_csv(settings.METRICS_DIR / "threshold_table.csv")
    n = tbl.height
    fig, axes = plt.subplots(1, n, figsize=(3.1 * n, 3.4))
    if n == 1:
        axes = [axes]
    for ax, row in zip(axes, tbl.iter_rows(named=True)):
        M = np.array([[row["tn"], row["fp"]], [row["fn"], row["tp"]]])
        ax.imshow(M, cmap="Blues", vmin=0, vmax=max(1, M.max()))
        for (i, j), v in np.ndenumerate(M):
            ax.text(j, i, f"{v:,}", ha="center", va="center",
                    color="white" if v > M.max() * 0.6 else "#111", fontsize=10)
        ax.set_xticks([0, 1], ["pred legit", "pred mule-like"], fontsize=8)
        ax.set_yticks([0, 1], ["true 0", "true 1"], fontsize=8)
        ax.set_title(f"{row['tier_threshold']} thr={row['threshold']:.3f}\n"
                     f"P={row['precision']:.2f} R={row['recall']:.2f}", fontsize=9)
        ax.grid(False)
    fig.suptitle("Confusion matrices at operational tier thresholds - LOCKED TEST", y=1.03)
    _save(fig, name)


def generate_all_available() -> list[str]:
    """Produce every plot whose inputs exist; return the list generated."""
    done = []
    oof_path = settings.METRICS_DIR / "oof_metrics.json"
    if oof_path.exists():
        m = load_json(oof_path)["models"]
        best = max((k for k in m if not k.startswith("REJECTED")),
                   key=lambda k: m[k]["pr_auc_mean"])
        plot_pr_curve(m[best]["per_repeat"][0] | {"split": f"dev OOF repeat 0 - {best}"},
                      name="precision_recall_curve_oof.png")
        done.append("precision_recall_curve_oof.png")
        plot_model_comparison(); done.append("model_comparison.png")
    if (settings.METRICS_DIR / "with_vs_without_f3912.json").exists():
        plot_leakage_ablation(); done.append("leakage_ablation.png")
    if (settings.FEATURES_DIR / "selection_frequency.csv").exists():
        plot_feature_stability(); done.append("feature_stability.png")
    lt = settings.METRICS_DIR / "locked_test_metrics.json"
    if lt.exists():
        rpt = load_json(lt)
        plot_pr_curve(rpt); done.append("precision_recall_curve.png")
        plot_recall_at_budget(rpt); done.append("recall_at_budget.png")
        plot_calibration(rpt); done.append("calibration_curve.png")
        if (settings.METRICS_DIR / "threshold_table.csv").exists():
            plot_confusion_matrices(); done.append("confusion_matrices.png")
    return done


if __name__ == "__main__":
    print("plots generated:", generate_all_available())
