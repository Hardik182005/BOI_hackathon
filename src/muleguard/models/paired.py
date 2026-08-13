"""Paired comparison statistics for fold-held-fixed experiments.

Every experiment in the nested stability / ensemble / shift programme compares
two arms that were scored on **byte-identical outer folds**. That design makes
the per-fold difference the unit of evidence, and this module is the one place
that turns 15 differences into a verdict.

Three tests, always all three
-----------------------------
The three tests answer slightly different questions and can disagree:

======================  ==================================  ===================
test                    what it uses                        fails when
======================  ==================================  ===================
exact sign test         direction only                      a few large wins
                                                            against many tiny
                                                            losses
Wilcoxon signed-rank    direction + rank of |difference|    one huge outlier
paired t                direction + magnitude               heavy tails, n small
======================  ==================================  ===================

Reporting only the kindest of the three is the specific failure this module
exists to prevent, so :func:`paired_report` computes all three unconditionally
and there is no argument that selects one. When they disagree, the disagreement
is the finding and the write-up has to explain the mechanism.

Yardstick
---------
The project's published seed-noise floor of 0.0905 PR-AUC is an **unpaired**
figure: it is the spread of whole-model scores across independent seeds. It is
the right test for "is model A better than model B across the board" and the
wrong test here, because a paired difference on fixed folds has its own, far
smaller, spread. ``std_of_paired_diff`` in the output is the correct yardstick,
and every artifact written by this programme carries it next to the mean so the
two cannot be confused.

Power
-----
With 15 outer folds and roughly 13 validation positives in each, this design
detects large effects and nothing else. :func:`detectable_effect` reports the
smallest true mean difference the paired t-test would find at 80 % power for
the observed spread, so that a null result can be read as "no effect this large"
rather than the much stronger claim "no effect".
"""
from __future__ import annotations

from dataclasses import dataclass
from math import comb, sqrt
from typing import Any, Sequence

import numpy as np

__all__ = ["PairedResult", "paired_report", "sign_test_p", "detectable_effect"]


def sign_test_p(diff: np.ndarray) -> tuple[float, int, int, int]:
    """Exact two-sided sign test on non-zero differences.

    Returns ``(p, n_improved, n_worse, n_tied)``. Exact zero differences are
    dropped, which is the textbook treatment; they are reported separately so a
    reader can see how many there were. Counting them as losses instead (as an
    earlier ablation in this repository did) makes the test more conservative
    still - it never makes it more permissive - and with float average-precision
    values ties are almost always absent.
    """
    d = np.asarray(diff, dtype=float)
    n_tied = int((d == 0).sum())
    nz = d[d != 0]
    n_up = int((nz > 0).sum())
    n_dn = int((nz < 0).sum())
    n = n_up + n_dn
    if n == 0:
        return 1.0, 0, 0, n_tied
    k = max(n_up, n_dn)
    tail = sum(comb(n, i) for i in range(k, n + 1))
    return float(min(1.0, 2.0 * tail / 2 ** n)), n_up, n_dn, n_tied


def detectable_effect(diff: np.ndarray, *, power: float = 0.80,
                      alpha: float = 0.05) -> float:
    """Smallest mean difference detectable at ``power``, given the observed sd.

    A normal approximation to the paired t-test: ``(z_alpha/2 + z_power) * sd /
    sqrt(n)``. It is an approximation and slightly optimistic for n = 15, which
    is stated where it is reported. Its purpose is to bound how much a null
    result is allowed to claim.
    """
    from scipy import stats

    d = np.asarray(diff, dtype=float)
    n = len(d)
    if n < 2:
        return float("nan")
    sd = float(d.std(ddof=1))
    z = float(stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power))
    return float(z * sd / sqrt(n))


@dataclass
class PairedResult:
    """The full paired picture for one comparison. Nothing is hidden."""

    name: str
    baseline: str
    arm: str
    diff: np.ndarray
    mean: float
    median: float
    sd: float
    ci95: tuple[float, float]
    p_sign: float
    p_wilcoxon: float
    p_ttest: float
    t_stat: float
    n_improved: int
    n_worse: int
    n_tied: int
    mde80: float

    @property
    def tests_significant(self) -> int:
        return sum(p < 0.05 for p in (self.p_sign, self.p_wilcoxon, self.p_ttest)
                   if p == p)  # NaN-safe

    @property
    def tests_disagree(self) -> bool:
        sig = [p < 0.05 for p in (self.p_sign, self.p_wilcoxon, self.p_ttest)
               if p == p]
        return len(set(sig)) > 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "comparison": self.name,
            "baseline_arm": self.baseline,
            "test_arm": self.arm,
            "n_folds": int(len(self.diff)),
            "mean_paired_diff": round(self.mean, 5),
            "median_paired_diff": round(self.median, 5),
            "std_of_paired_diff": round(self.sd, 5),
            "ci95_of_mean": [round(self.ci95[0], 5), round(self.ci95[1], 5)],
            "per_fold_diff": [round(float(d), 5) for d in self.diff],
            "n_folds_improved": self.n_improved,
            "n_folds_worse": self.n_worse,
            "n_folds_tied": self.n_tied,
            "sign_test_p_two_sided": round(self.p_sign, 5),
            "wilcoxon_p_two_sided": (round(self.p_wilcoxon, 5)
                                     if self.p_wilcoxon == self.p_wilcoxon else None),
            "paired_t_p_two_sided": round(self.p_ttest, 5),
            "paired_t_statistic": round(self.t_stat, 4),
            "n_tests_below_0.05": self.tests_significant,
            "tests_disagree": self.tests_disagree,
            "min_detectable_effect_80pct_power": round(self.mde80, 5),
            "yardstick": (
                "paired on identical outer folds - judge against "
                f"std_of_paired_diff={self.sd:.5f}, NOT against the project's "
                "0.0905 unpaired seed-noise floor"),
        }


def paired_report(baseline: Sequence[float], arm: Sequence[float], *,
                  name: str = "", baseline_name: str = "baseline",
                  arm_name: str = "arm") -> PairedResult:
    """All three paired tests plus the mean difference and its 95 % CI.

    ``arm`` minus ``baseline``, so a positive mean means the arm is better on a
    metric where larger is better. The CI is the ordinary t interval on the mean
    of the differences; it is deterministic, which a bootstrap interval would
    not be without pinning a seed, and reproducibility is worth more here than
    distribution-freeness at n = 15.
    """
    from scipy import stats

    a = np.asarray(baseline, dtype=float)
    b = np.asarray(arm, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"arms are not paired: {a.shape} vs {b.shape}")
    if a.ndim != 1 or len(a) < 2:
        raise ValueError("need at least 2 paired observations")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("non-finite values in a paired comparison")

    d = b - a
    n = len(d)
    mean = float(d.mean())
    sd = float(d.std(ddof=1))
    half = float(stats.t.ppf(0.975, n - 1) * sd / sqrt(n)) if sd > 0 else 0.0

    p_sign, n_up, n_dn, n_tie = sign_test_p(d)
    if sd == 0:
        t_stat, p_t = 0.0, 1.0
    else:
        t_stat, p_t = stats.ttest_rel(b, a)
    try:
        _, p_w = stats.wilcoxon(d)
    except ValueError:      # every difference is exactly zero
        p_w = float("nan")

    return PairedResult(
        name=name or f"{arm_name} vs {baseline_name}",
        baseline=baseline_name, arm=arm_name, diff=d,
        mean=mean, median=float(np.median(d)), sd=sd,
        ci95=(mean - half, mean + half),
        p_sign=float(p_sign), p_wilcoxon=float(p_w),
        p_ttest=float(p_t), t_stat=float(t_stat),
        n_improved=n_up, n_worse=n_dn, n_tied=n_tie,
        mde80=detectable_effect(d),
    )
