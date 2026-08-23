"""Trinetra Mule-Farm Cohort Radar - behaviourally similar accounts.

An analyst finds one suspicious account and asks the only question that
naturally follows: *which other accounts in this portfolio behave unusually
similarly?* This module answers it, and is careful about what the answer means.

**It is retrieval, not classification.** Nothing here produces, adjusts, blends
into or escalates a risk probability. The classifier is frozen; the radar reads
its output and never writes back. There is deliberately no
``final_risk = a*model + b*cohort`` anywhere in this file, because such a
formula would silently make the radar part of the model and invalidate every
accuracy figure the project has published.

**It is behavioural similarity, not a transaction network.** Two accounts being
neighbours here means their admitted behavioural features are close together.
It does not mean they transacted, share an owner, or belong to one criminal
group - the data contains no evidence of any of that. Edges are called
``BEHAVIORALLY_SIMILAR_TO`` and nothing else, and the language guard in
:func:`assert_language_safe` refuses to emit the alternatives.

Design choices worth stating plainly:

* **The fingerprint is the champion's own feature list.** Those 120 columns
  already passed the leakage firewall to become model inputs, so the radar
  cannot see anything the classifier could not. They are re-checked through
  ``firewall.assert_clean`` at build and query time regardless: "it was admitted
  once" is a claim, and the firewall is the check.
* **The transform is fitted on the development partition only** and frozen to
  disk. The locked test contributes nothing - not a median, not an IQR, not a
  category. A query row cannot alter it.
* **"Unusually similar" is measured, not asserted.** A threshold like
  ``similarity > 0.9`` means nothing without knowing what similarity two
  unrelated accounts normally have. The bands come from an empirical null
  distribution over random development pairs.
* **Missing is a state, not a zero.** Imputing a missing value to zero would
  make two accounts look alike precisely where the data is silent.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Sequence

import joblib
import numpy as np
import polars as pl

from muleguard import settings
from muleguard.features.frame import attach_meta
from muleguard.logging import get_logger

log = get_logger("usp.cohort_radar")

TRANSFORM_PATH = settings.MODELS_DIR / "cohort_radar_transform.joblib"
MANIFEST_PATH = settings.MODELS_DIR / "cohort_radar_manifest.json"

#: The only relationship this layer is entitled to assert.
EDGE_LABEL = "BEHAVIORALLY_SIMILAR_TO"

#: Shown under every cohort panel and returned in every API response. Section 5
#: fixes the wording; it is not paraphrased at any call site.
DISCLAIMER = (
    "Behavioural similarity does not establish that accounts are transacting "
    "with each other, controlled by the same person, or part of the same "
    "criminal network. It is an investigation-prioritisation signal only."
)

#: Claims the radar has no evidence for and must never make. Checked against
#: rendered output rather than trusted to reviewer discipline.
FORBIDDEN_COHORT_LANGUAGE = (
    "same criminal network", "criminal network", "mule handler", "same handler",
    "connected mule ring", "mule ring", "same syndicate", "syndicate",
    "controlled by", "controlled by same person", "controlled by the same person",
    "same owner", "sent money to", "transacted with", "money flow",
    "same person", "same gang", "crime ring",
)

#: Fixed, reviewed prose the radar attaches to every response. It is exempt from
#: the language guard for the obvious reason that the disclaimer's whole job is
#: to name the claims being disclaimed - scanning it would make the guard reject
#: the one sentence in the payload that exists to prevent the misreading.
INTERPRETATION = (
    "Behaviourally similar accounts, ranked by weighted feature agreement "
    "against a frozen development-only reference. Similarity is compared "
    "against an empirical null over random account pairs; a high percentile "
    "means the pair is closer than chance would give.")

ACTION_POLICY = (
    "Cohort membership is an investigation-prioritisation signal. It does not "
    "change any account risk probability or review tier, and no neighbour is "
    "escalated by appearing here.")

SAFE_FIXED_TEXT = (DISCLAIMER, INTERPRETATION, ACTION_POLICY)

BAND_VERY_HIGH = "VERY_HIGH_SIMILARITY"
BAND_HIGH = "HIGH_SIMILARITY"
BAND_MODERATE = "MODERATE_SIMILARITY"
BAND_TYPICAL = "TYPICAL_SIMILARITY"


class CohortRadarUnavailable(RuntimeError):
    """The frozen transform has not been built yet."""


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _config() -> dict[str, Any]:
    return settings.load_config("cohort_radar")


#: Keys whose *job* is to quote the banned wording - the disclaimer that denies
#: the claims, and the denylist published as provenance. Their values are still
#: shipped; they are simply not read as assertions, because a guard that cannot
#: tell "we never say X" from "X" would force the product to stop explaining
#: itself in order to stay compliant.
_NON_WORD = re.compile(r"[^a-z0-9]+")

DECLARATIVE_KEYS = frozenset({
    "forbidden", "forbidden_language", "never_emitted", "disclaimer",
    "interpretation", "action_policy", "language_guard",
})


def assert_language_safe(payload: Any) -> None:
    """Refuse to emit a cohort claim the data cannot support.

    Applied to the rendered response, not to the code that builds it. A guard
    that only inspects intentions catches nothing; this one reads the words that
    would actually reach an analyst.

    The scan walks the structure rather than the serialised blob, so a phrase is
    judged by where it sits. Two places quote the banned wording legitimately -
    the disclaimer, whose entire purpose is to name what similarity does *not*
    establish, and the published denylist. Those are skipped by key; everything
    else, at any depth, is read as something the radar is asserting.
    """
    if isinstance(payload, str):
        offenders = [payload]
    else:
        offenders = []

        def walk(node: Any, key: str | None) -> None:
            if key is not None and key.lower() in DECLARATIVE_KEYS:
                return
            if isinstance(node, dict):
                for k, v in node.items():
                    walk(v, str(k))
                    if str(k).lower() not in DECLARATIVE_KEYS:
                        offenders.append(str(k))
            elif isinstance(node, (list, tuple, set, frozenset)):
                for v in node:
                    walk(v, key)
            elif isinstance(node, str):
                offenders.append(node)

        walk(payload, None)

    for text in offenders:
        if text in SAFE_FIXED_TEXT:
            continue
        # Separators are normalised so an enum reads the same as prose:
        # SENT_MONEY_TO, sent-money-to and "sent money to" are one claim, and
        # section 13 bans the relationship however it is spelled.
        lowered = _NON_WORD.sub(" ", text.lower())
        for phrase in FORBIDDEN_COHORT_LANGUAGE:
            if phrase in lowered:
                raise ValueError(
                    f"cohort output claims {phrase!r}; behavioural similarity "
                    f"does not establish it. Use 'behaviourally similar "
                    f"accounts'. Offending text: {text[:160]!r}")


# --------------------------------------------------------------------------
# the frozen transform
# --------------------------------------------------------------------------


@dataclass
class CohortTransform:
    """Training-only scaling statistics, categories and weights - frozen.

    Everything a similarity needs to be reproducible lives here, so a query can
    be scored years later against the same yardstick. Nothing in this object is
    derived from a label, from the locked test, or from a query.
    """

    numeric_features: list[str]
    categorical_features: list[str]
    median: np.ndarray            # (d_num,) training median, for reporting only
    iqr: np.ndarray               # (d_num,) training inter-quartile range
    scale: np.ndarray             # (d_num,) the divisor actually used
    scale_source: list[str]       # per feature: "IQR" or the documented fallback
    lower: np.ndarray             # (d_num,) training clip bound
    upper: np.ndarray             # (d_num,) training clip bound
    categories: dict[str, list[str]]
    numeric_weights: np.ndarray   # (d_num,)
    categorical_weights: np.ndarray  # (d_cat,)
    weight_source: dict[str, Any]
    null_quantile_grid: np.ndarray   # similarity at 0..100th percentile
    bands: dict[str, float]          # band name -> similarity cut-off
    band_percentiles: dict[str, float]
    n_reference_rows: int
    n_null_pairs: int
    radar_version: str
    quarantine_hash: str
    seed: int
    null_statistics: dict[str, Any] = field(default_factory=dict)
    #: Smallest fingerprint coverage observed among the reference accounts, in
    #: weight. A query supplying less than this is outside the data-availability
    #: envelope the null was built from - see :meth:`coverage`.
    min_reference_coverage: float = 0.0
    fitted_utc: str = field(default_factory=_utc)

    # -- identity ----------------------------------------------------------
    @property
    def features(self) -> list[str]:
        return list(self.numeric_features) + list(self.categorical_features)

    def feature_hash(self) -> str:
        return _sha_text("\n".join(self.features))

    def scaling_hash(self) -> str:
        parts = [repr(np.asarray(a).tolist())
                 for a in (self.median, self.iqr, self.scale, self.lower, self.upper)]
        parts.append(repr(sorted((k, sorted(v)) for k, v in self.categories.items())))
        return _sha_text("|".join(parts))

    def weights_hash(self) -> str:
        return _sha_text(repr(np.concatenate(
            [self.numeric_weights, self.categorical_weights]).tolist()))

    # -- encoding ----------------------------------------------------------
    def encode(self, frame: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Turn rows into the numeric matrix and category codes the metric uses.

        Missing values stay missing: ``NaN`` for numerics, ``-1`` for categories.
        They are never filled, because a filled value would make two accounts
        look alike exactly where the data says nothing about either.
        """
        n = frame.height
        num = np.full((n, len(self.numeric_features)), np.nan, dtype=float)
        for j, name in enumerate(self.numeric_features):
            if name not in frame.columns:
                continue
            col = frame[name].cast(pl.Float64, strict=False).to_numpy()
            num[:, j] = np.clip(col, self.lower[j], self.upper[j])

        cat = np.full((n, len(self.categorical_features)), -1, dtype=np.int32)
        for j, name in enumerate(self.categorical_features):
            if name not in frame.columns:
                continue
            lookup = {v: i for i, v in enumerate(self.categories.get(name, []))}
            values = frame[name].cast(pl.Utf8, strict=False).to_list()
            # An unseen category encodes as -1, i.e. "unknown", which the metric
            # treats as missing. Inventing a new code at query time would let a
            # query silently extend the frozen transform.
            cat[:, j] = [lookup.get(v, -1) if v is not None else -1 for v in values]
        return num, cat

    # -- the metric --------------------------------------------------------
    def _numeric_delta(self, q: np.ndarray, ref: np.ndarray) -> np.ndarray:
        """Per-feature numeric disagreement in [0,1], broadcast over ``ref``."""
        q_missing = np.isnan(q)
        r_missing = np.isnan(ref)
        with np.errstate(invalid="ignore"):
            delta = np.minimum(np.abs(ref - q) / self.scale, 1.0)
        delta = np.where(r_missing | q_missing, 1.0, delta)
        delta = np.where(r_missing & q_missing, 0.0, delta)
        return delta

    @staticmethod
    def _categorical_delta(q: np.ndarray, ref: np.ndarray) -> np.ndarray:
        """0 when the category matches, 1 when it does not.

        Two unknowns encode identically (-1) and so agree, which is the same
        rule the numeric side applies to two missing values: shared silence is
        not disagreement, but silence on one side only is.
        """
        return (ref != q).astype(float)

    def deltas(self, q_num: np.ndarray, q_cat: np.ndarray,
               ref_num: np.ndarray, ref_cat: np.ndarray
               ) -> tuple[np.ndarray, np.ndarray]:
        return (self._numeric_delta(q_num, ref_num),
                self._categorical_delta(q_cat, ref_cat))

    def similarity(self, q_num: np.ndarray, q_cat: np.ndarray,
                   ref_num: np.ndarray, ref_cat: np.ndarray) -> np.ndarray:
        """``S = 1 - Σ_j w_j δ_j``, clamped to [0,1].

        One weighted sum of per-feature disagreements. Every term is inspectable
        and attributable to a named feature, which is why the neighbour
        explanations in :func:`explain_pair` are the arithmetic itself rather
        than a story told about it.
        """
        d_num, d_cat = self.deltas(q_num, q_cat, ref_num, ref_cat)
        distance = d_num @ self.numeric_weights + d_cat @ self.categorical_weights
        return np.clip(1.0 - distance, 0.0, 1.0)

    def similarity_aligned(self, a_num: np.ndarray, a_cat: np.ndarray,
                           b_num: np.ndarray, b_cat: np.ndarray) -> np.ndarray:
        """Row-wise similarity between two equally shaped blocks of accounts.

        Used to build the null distribution, where the pairs are drawn rather
        than formed by broadcasting one query against everything.
        """
        a_missing, b_missing = np.isnan(a_num), np.isnan(b_num)
        with np.errstate(invalid="ignore"):
            d_num = np.minimum(np.abs(a_num - b_num) / self.scale, 1.0)
        d_num = np.where(a_missing | b_missing, 1.0, d_num)
        d_num = np.where(a_missing & b_missing, 0.0, d_num)
        d_cat = (a_cat != b_cat).astype(float)
        distance = d_num @ self.numeric_weights + d_cat @ self.categorical_weights
        return np.clip(1.0 - distance, 0.0, 1.0)

    # -- calibration -------------------------------------------------------
    def percentile_of(self, similarity: np.ndarray | float) -> np.ndarray:
        """Where a similarity falls in the frozen null distribution.

        This is what makes a number interpretable. 0.82 is meaningless on its
        own; "higher than 99.7% of random account pairs" is a finding.

        The scale saturates. A pair closer than every one of the sampled null
        pairs reports 100.0, which means "beyond the resolution of this null",
        not "identical" - the companion ``beyond_empirical_null`` flag on each
        neighbour marks those so the number is not read as more precise than
        the sample supports.
        """
        grid = self.null_quantile_grid
        pct = np.linspace(0.0, 100.0, len(grid))
        return np.interp(np.asarray(similarity, dtype=float), grid, pct)

    def coverage(self, q_num: np.ndarray, q_cat: np.ndarray) -> float:
        """How much of the fingerprint, by weight, a query actually supplied.

        The metric treats a missing value as full disagreement, so a query with
        almost nothing in it still produces a ranked list - of near-zero
        similarities that say nothing about anybody. Section 37 case 8 asks for
        an insufficient-data answer instead, and this is the quantity that
        decides it: the weight of the features the query can be compared on.
        """
        present_num = ~np.isnan(np.asarray(q_num, dtype=float))
        present_cat = np.asarray(q_cat) >= 0
        return float(present_num @ self.numeric_weights
                     + present_cat @ self.categorical_weights)

    def has_enough_data(self, q_num: np.ndarray, q_cat: np.ndarray) -> bool:
        """Whether a query can be placed against this reference frame at all.

        The floor is measured, not chosen: it is the least-complete account the
        transform was fitted on. A query below it is supplying less to compare
        than any account in the null, so the percentile it would receive is not
        the percentile that null describes.
        """
        got = self.coverage(q_num, q_cat)
        return got > 0.0 and got >= self.min_reference_coverage

    def band_of(self, similarity: float) -> str:
        if similarity >= self.bands[BAND_VERY_HIGH]:
            return BAND_VERY_HIGH
        if similarity >= self.bands[BAND_HIGH]:
            return BAND_HIGH
        if similarity >= self.bands[BAND_MODERATE]:
            return BAND_MODERATE
        return BAND_TYPICAL


# --------------------------------------------------------------------------
# fitting
# --------------------------------------------------------------------------


def fingerprint_features(*, registry: dict[str, Any] | None = None) -> list[str]:
    """The columns the radar is allowed to look at, re-checked every time.

    The list starts as the champion's selected features - already firewall
    admitted, by construction - and is then put through the firewall again. That
    second check is not redundant: the quarantine is a live config, and a
    feature reclassified tomorrow must break the radar rather than quietly
    remain in a frozen fingerprint.
    """
    from muleguard.features import firewall
    from muleguard.features.frame import augmented_registry
    from muleguard.models.scoring import load_bundle

    reg = registry or augmented_registry()
    selected = list(load_bundle()["feature_list_selected"])
    firewall.assert_clean(selected, context="cohort_radar_fingerprint", registry=reg)
    decision = firewall.admitted_features(selected, registry=reg)
    rejected = [c for c in selected if c not in set(decision.admitted)]
    if rejected:
        raise firewall.LeakageViolation(
            f"cohort fingerprint would include {len(rejected)} feature(s) the "
            f"firewall does not admit: {rejected[:10]}")
    return selected


def _split_kinds(frame: pl.DataFrame, features: Sequence[str]
                 ) -> tuple[list[str], list[str]]:
    numeric, categorical = [], []
    for name in features:
        dtype = frame.schema.get(name)
        if dtype is not None and dtype.is_numeric():
            numeric.append(name)
        else:
            categorical.append(name)
    return numeric, categorical


def _shap_weights(features: Sequence[str]) -> tuple[np.ndarray, dict[str, Any]]:
    """Weights from out-of-fold TreeSHAP, with the tail split by stability.

    Section 10 ranks global SHAP first. The published ranking covers the top 30
    of the 120 selected features and 47.14% of total |SHAP| mass, so it fixes
    those thirty weights *and* the size of the remaining tail. Splitting that
    tail by stability-selection gain share - source two in the same ordering -
    uses the second source for exactly the question the first cannot answer,
    and leaves the SHAP total untouched. No weight below is hand-chosen.

    If either artifact is missing the function falls back to equal weights and
    says so, because an invented weight is worse than a flat one.
    """
    from muleguard.utils import load_json

    n = len(features)
    index = {f: i for i, f in enumerate(features)}
    w = np.zeros(n, dtype=float)
    meta: dict[str, Any] = {"primary": "global_shap_importance"}

    try:
        shap = load_json(settings.METRICS_DIR / "global_shap_importance.json")
        ranking = [r for r in shap.get("ranking", []) if r["feature"] in index]
    except (FileNotFoundError, OSError, ValueError):
        ranking = []
    if not ranking:
        meta = {"primary": "equal", "reason": "global SHAP importance unavailable"}
        return np.full(n, 1.0 / n), meta

    for rec in ranking:
        w[index[rec["feature"]]] = float(rec["share_of_total_abs"])
    head_mass = float(w.sum())
    tail = [f for f in features if w[index[f]] == 0.0]
    meta.update({
        "shap_method": shap.get("attribution_method", {}).get("method"),
        "shap_out_of_fold": True,
        "n_features_weighted_by_shap": len(ranking),
        "shap_mass_covered": head_mass,
        "n_features_in_tail": len(tail),
        "tail_mass": max(0.0, 1.0 - head_mass),
    })

    if tail:
        tail_mass = max(0.0, 1.0 - head_mass)
        gains = {}
        try:
            freq = pl.read_csv(settings.ARTIFACTS_DIR / "features"
                               / "final_selection_frequency.csv")
            gains = dict(zip(freq["feature"].to_list(),
                             freq["mean_gain_share"].to_list()))
        except (FileNotFoundError, OSError):
            gains = {}
        shares = np.array([max(float(gains.get(f, 0.0)), 0.0) for f in tail])
        if shares.sum() > 0:
            meta["tail"] = "selection_frequency_gain_share"
        else:
            shares = np.ones(len(tail))
            meta["tail"] = "equal (stability-selection gain share unavailable)"
        shares = shares / shares.sum() * tail_mass
        for f, s in zip(tail, shares):
            w[index[f]] = s

    total = w.sum()
    if total <= 0:
        return np.full(n, 1.0 / n), {"primary": "equal",
                                     "reason": "weight sources summed to zero"}
    return w / total, meta


def reference_row_index() -> np.ndarray:
    """Rows the radar may use as its reference portfolio.

    The development partition, and only that. The locked test contributes no
    median, no IQR, no category and no neighbour - a held-out set that gets
    consulted by a shipped feature has stopped being held out, however
    indirectly the consultation happens.
    """
    from muleguard.models.harness import dev_split

    return np.sort(dev_split().row_index)


def fit(*, frame: pl.DataFrame | None = None,
        rows: np.ndarray | None = None) -> CohortTransform:
    """Fit and freeze the similarity transform on the development partition."""
    from muleguard.features import firewall
    from muleguard.features.frame import raw_with_meta

    cfg = _config()
    frame = raw_with_meta() if frame is None else frame
    rows = reference_row_index() if rows is None else np.asarray(rows)
    features = fingerprint_features()
    ref = frame[rows.tolist()].select(features)

    numeric, categorical = _split_kinds(ref, features)
    ordered = numeric + categorical
    weights_all, weight_meta = _shap_weights(ordered)
    d_num = len(numeric)

    lo_q, hi_q = cfg["similarity"]["clip_quantiles"]
    multiplier = float(cfg["similarity"]["numeric_scale_iqr_multiplier"])
    eps = float(cfg["similarity"]["epsilon"])

    median = np.zeros(d_num)
    iqr = np.zeros(d_num)
    lower = np.zeros(d_num)
    upper = np.zeros(d_num)
    scale = np.zeros(d_num)
    scale_source: list[str] = []
    for j, name in enumerate(numeric):
        col = ref[name].cast(pl.Float64, strict=False).to_numpy()
        finite = col[np.isfinite(col)]
        if finite.size == 0:
            # A column that is entirely missing across the reference partition
            # carries no information; a unit scale keeps the arithmetic defined
            # and every comparison on it resolves through the missingness rule.
            median[j] = iqr[j] = 0.0
            lower[j], upper[j], scale[j] = -np.inf, np.inf, 1.0
            scale_source.append("EMPTY_COLUMN_UNIT_SCALE")
            continue
        q1, q3 = np.percentile(finite, [25.0, 75.0])
        median[j] = float(np.median(finite))
        iqr[j] = float(q3 - q1)
        lower[j] = float(np.quantile(finite, lo_q))
        upper[j] = float(np.quantile(finite, hi_q))
        span = multiplier * iqr[j]
        if span > eps:
            scale[j] = span
            scale_source.append("IQR")
        else:
            # A near-constant feature has no IQR to normalise by. The documented
            # fallback is the clipped training range, and if that is degenerate
            # too the feature simply cannot separate anyone: unit scale, and
            # every pair agrees on it.
            width = float(upper[j] - lower[j])
            if width > eps:
                scale[j] = width
                scale_source.append("TRAINING_RANGE_p1_p99")
            else:
                scale[j] = 1.0
                scale_source.append("CONSTANT_UNIT_SCALE")

    categories = {}
    for name in categorical:
        values = ref[name].cast(pl.Utf8, strict=False).drop_nulls().unique().to_list()
        categories[name] = sorted(str(v) for v in values)

    quarantine = firewall.config()
    q_hash = _sha_text("\n".join(sorted(
        set(quarantine.hard_quarantine) | set(quarantine.conditional_quarantine)
        | set(quarantine.fairness_excluded))))

    transform = CohortTransform(
        numeric_features=numeric,
        categorical_features=categorical,
        median=median, iqr=iqr, scale=scale, scale_source=scale_source,
        lower=lower, upper=upper, categories=categories,
        numeric_weights=weights_all[:d_num],
        categorical_weights=weights_all[d_num:],
        weight_source=weight_meta,
        null_quantile_grid=np.zeros(1),
        bands={}, band_percentiles={},
        n_reference_rows=int(len(rows)),
        n_null_pairs=0,
        radar_version=str(cfg["radar_version"]),
        quarantine_hash=q_hash,
        seed=int(cfg["null_distribution"]["seed"]),
    )

    grid, bands, band_pct, n_pairs, null_stats = _fit_null(transform, ref, cfg)
    transform.null_quantile_grid = grid
    transform.bands = bands
    transform.band_percentiles = band_pct
    transform.n_null_pairs = n_pairs
    transform.null_statistics = null_stats

    # The data-availability floor, measured on the same rows the null came from.
    ref_num, ref_cat = transform.encode(ref)
    ref_coverage = ((~np.isnan(ref_num)) @ transform.numeric_weights
                    + (ref_cat >= 0) @ transform.categorical_weights)
    transform.min_reference_coverage = float(ref_coverage.min())
    null_stats["min_reference_coverage"] = transform.min_reference_coverage
    null_stats["median_reference_coverage"] = float(np.median(ref_coverage))
    null_stats["label_used"] = False

    log.info("cohort transform fitted: %d numeric + %d categorical features, "
             "%d reference rows, %d null pairs",
             len(numeric), len(categorical), len(rows), n_pairs)
    return transform


def _fit_null(transform: CohortTransform, ref: pl.DataFrame, cfg: dict[str, Any]
              ) -> tuple[np.ndarray, dict[str, float], dict[str, float], int, dict]:
    """The empirical null: how similar are two accounts that have nothing to do
    with each other?

    Without this number a similarity score is uninterpretable, and a hardcoded
    ``> 0.90`` is a guess dressed as a threshold. Pairs are drawn **without
    reading any label**. Section 12 suggests sampling legitimate accounts; at
    0.88% prevalence a random pair contains a positive about 1.75% of the time
    and two positives about 0.008% of the time, so label-free sampling gives the
    same null while leaving the transform provably free of ``F3924``. The
    The label-conditioned null is computed separately by the build command and
    recorded in the manifest, so the claim that the two agree is a measurement
    rather than an argument - but the frozen transform itself never sees a label.
    """
    n_pairs = int(cfg["null_distribution"]["n_pairs"])
    seed = int(cfg["null_distribution"]["seed"])
    num, cat = transform.encode(ref)
    n = num.shape[0]

    rng = np.random.default_rng(seed)
    left = rng.integers(0, n, size=n_pairs)
    right = rng.integers(0, n, size=n_pairs)
    distinct = left != right
    left, right = left[distinct], right[distinct]

    sims = np.empty(len(left), dtype=float)
    chunk = 20000
    for start in range(0, len(left), chunk):
        sl = slice(start, start + chunk)
        sims[sl] = transform.similarity_aligned(
            num[left[sl]], cat[left[sl]], num[right[sl]], cat[right[sl]])

    grid = np.percentile(sims, np.linspace(0.0, 100.0, 10001))
    grid = np.maximum.accumulate(grid)  # monotone, so np.interp is well defined
    band_pct = {
        BAND_VERY_HIGH: float(cfg["bands"]["very_high"]),
        BAND_HIGH: float(cfg["bands"]["high"]),
        BAND_MODERATE: float(cfg["bands"]["moderate"]),
    }
    bands = {name: float(np.percentile(sims, pct)) for name, pct in band_pct.items()}
    stats = {
        "sampling": "uniform random development pairs, no label consulted",
        "n_pairs_drawn": int(n_pairs),
        "n_pairs_used": int(len(left)),
        "mean": float(sims.mean()),
        "median": float(np.median(sims)),
        "std": float(sims.std()),
        "min": float(sims.min()),
        "max": float(sims.max()),
    }
    return grid, bands, band_pct, int(len(left)), stats


# --------------------------------------------------------------------------
# persistence
# --------------------------------------------------------------------------


def save(transform: CohortTransform, path=None) -> str:
    path = TRANSFORM_PATH if path is None else path
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(transform, path)
    from muleguard.utils import sha256_file

    return sha256_file(path)


_TRANSFORM_CACHE: CohortTransform | None = None


def load(path=None, *, refresh: bool = False) -> CohortTransform:
    """The frozen transform, cached per process.

    Raises :class:`CohortRadarUnavailable` rather than fitting on demand. A
    layer that silently rebuilds its own reference statistics when a file is
    missing would produce answers that differ between machines while looking
    identical, which is the worst way for this to fail.
    """
    global _TRANSFORM_CACHE
    path = TRANSFORM_PATH if path is None else path
    if _TRANSFORM_CACHE is not None and not refresh:
        return _TRANSFORM_CACHE
    if not path.exists():
        raise CohortRadarUnavailable(
            f"cohort transform not built: {path}. Run "
            f"'python -m muleguard.cli.build_cohort_radar' first.")
    _TRANSFORM_CACHE = joblib.load(path)
    return _TRANSFORM_CACHE


# --------------------------------------------------------------------------
# the reference index
# --------------------------------------------------------------------------


@dataclass
class CohortIndex:
    """The reference portfolio, encoded once and reused for every query.

    Risk probabilities are carried alongside so a neighbour can be *reported*
    with its current tier. They are read, never written: the radar cannot
    promote a neighbour it finds interesting, which is section 23's rule and the
    reason cohort membership is safe to show an analyst at all.
    """

    transform: CohortTransform
    row_index: np.ndarray
    references: list[str]
    numeric: np.ndarray
    categorical: np.ndarray
    risk: np.ndarray
    tier: list[str]
    patterns: list[frozenset[str]]
    scope: str

    def __len__(self) -> int:
        return len(self.references)

    def position_of(self, reference: str) -> int | None:
        try:
            return self.references.index(reference)
        except ValueError:
            return None


def reference_label(row_index: int) -> str:
    """The masked identifier a dataset row is shown under.

    The dataset carries no account numbers and none are invented. ``RV-<row>``
    is a pointer into the reference frame, which is all an analyst needs to pull
    the same row up again and all the data entitles anyone to show.
    """
    return f"RV-{int(row_index)}"


def with_derived_meta(values: dict[str, Any]) -> dict[str, Any]:
    """Fill in the MG_* block a caller did not send, without overwriting one it did.

    The meta-features are row-wise functions of the raw columns, so they can be
    re-derived for any submission. The catch is that deriving them from a
    *partial* row does not fail loudly - it quietly produces a different number,
    or None, for the inputs that are missing. A payload carrying 120 model
    features but not the raw columns behind ``MG_PASSTHROUGH_7D`` will
    recompute that feature as null and, if allowed to, overwrite the correct
    value the caller actually sent.

    So derivation fills gaps and never wins an argument: whatever the caller
    supplied is authoritative, and the derived block is only consulted where the
    caller supplied nothing.
    """
    try:
        derived = attach_meta(pl.DataFrame([values])).to_dicts()[0]
    except Exception as exc:  # noqa: BLE001 - partial rows are a normal input
        log.debug("meta-features not derivable: %s", exc)
        return dict(values)
    merged = dict(derived)
    merged.update({k: v for k, v in values.items() if v is not None})
    return merged


def pattern_feature_names() -> list[str]:
    """Columns the typology pattern cards read.

    Retained alongside the fingerprint so a case looked up later can be compared
    on the same pattern vocabulary as the reference frame.
    """
    from muleguard.explain.pattern_cards import PATTERN_DEFINITIONS

    return sorted({d.feature for d in PATTERN_DEFINITIONS})


def row_index_for_reference(account_reference: str) -> int | None:
    """Invert :func:`reference_label`, or None if the label is not one of ours.

    Only ``RV-<row>`` resolves. A caller-supplied account reference from a
    scoring request is not a reference-frame row and must not be silently
    matched to one.
    """
    text = str(account_reference or "").strip().upper()
    if not text.startswith("RV-"):
        return None
    try:
        row = int(text[3:])
    except ValueError:
        return None
    return row if row in set(reference_row_index().tolist()) else None


def manifest() -> dict[str, Any]:
    """The frozen transform's build record, read from disk.

    Returned by the API so a reviewer can check the radar's provenance without
    being handed a summary written by the thing being reviewed.
    """
    if not MANIFEST_PATH.exists():
        raise CohortRadarUnavailable(
            f"{MANIFEST_PATH} not found - build the cohort transform first")
    import json

    record = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    record["disclaimer"] = DISCLAIMER
    record["edge_relationship"] = EDGE_LABEL
    record["affects_model_score"] = False
    return record


_INDEX_CACHE: CohortIndex | None = None


def build_index(*, transform: CohortTransform | None = None,
                rows: np.ndarray | None = None,
                refresh: bool = False) -> CohortIndex:
    """Encode and score the reference portfolio.

    The risk column comes from the frozen scoring path - the same ``score_rows``
    the API serves - so a neighbour's tier in a cohort panel is the tier its own
    case would carry. Recomputing it here rather than storing a copy keeps the
    two from drifting apart.
    """
    global _INDEX_CACHE
    is_default_scope = rows is None
    if _INDEX_CACHE is not None and not refresh and is_default_scope:
        return _INDEX_CACHE

    from muleguard.explain.pattern_cards import match_patterns
    from muleguard.features.frame import raw_with_meta
    from muleguard.models.scoring import score_rows

    transform = load() if transform is None else transform
    # Sorted, always. The reference frame is a set of accounts, not a sequence
    # of them, so the order a caller happens to hand them in must not survive
    # into the index - and position lookups downstream assume sorted row ids.
    rows = reference_row_index() if rows is None else np.sort(np.asarray(rows))
    frame = raw_with_meta()
    subset = frame[rows.tolist()]

    num, cat = transform.encode(subset.select(transform.features))
    scored = score_rows(subset, with_explanations=False, with_counterfactual=False)
    risk = np.array([float(r["calibrated_risk"]) for r in scored])
    tier = [str(r["risk_tier"]) for r in scored]

    pattern_cols = subset.to_dicts()
    patterns = [frozenset(c["id"] for c in match_patterns(row)) for row in pattern_cols]

    index = CohortIndex(
        transform=transform, row_index=rows,
        references=[reference_label(r) for r in rows.tolist()],
        numeric=num, categorical=cat, risk=risk, tier=tier, patterns=patterns,
        scope=(f"development reference partition, {len(rows)} accounts "
               f"(locked test excluded)"))
    if is_default_scope:
        _INDEX_CACHE = index
    return index


# --------------------------------------------------------------------------
# querying
# --------------------------------------------------------------------------


def _feature_label(name: str, registry: dict[str, Any] | None = None
                   ) -> tuple[str, str, str]:
    """Name, description and family for a feature - from the data dictionary.

    Section 15 allows no invented wording. Every string an analyst reads about
    a feature originates in ``Description.xlsx``; a column the dictionary does
    not cover keeps its code and an empty description rather than acquiring a
    plausible-sounding meaning.
    """
    from muleguard.features import dictionary as fd
    from muleguard.features.frame import augmented_registry

    try:
        rec = fd.describe(name, registry or augmented_registry())
        return (str(rec.get("variable_name") or name),
                str(rec.get("description") or ""),
                str(rec.get("feature_family") or "OTHER"))
    except Exception:  # a feature absent from the registry keeps its code
        return name, "", "OTHER"


def explain_pair(index: CohortIndex, q_num: np.ndarray, q_cat: np.ndarray,
                 position: int, *, top: int = 4,
                 registry: dict[str, Any] | None = None
                 ) -> tuple[list[dict], list[dict]]:
    """Why these two accounts match, and where they do not.

    The explanation is the arithmetic, not a narrative about it: a feature's
    contribution to agreement is ``w_j * (1 - δ_j)`` and to disagreement
    ``w_j * δ_j``, the same terms that produced the score. Names come from
    ``Description.xlsx`` through the data dictionary, so nothing here is
    generated text.
    """
    t = index.transform
    d_num, d_cat = t.deltas(q_num, q_cat,
                            index.numeric[position:position + 1],
                            index.categorical[position:position + 1])
    deltas = np.concatenate([d_num[0], d_cat[0]])
    weights = np.concatenate([t.numeric_weights, t.categorical_weights])
    names = t.features

    agreement = weights * (1.0 - deltas)
    difference = weights * deltas

    def pack(order, values, kind):
        out = []
        for j in order:
            if values[j] <= 0.0:
                continue
            label, description, family = _feature_label(names[j], registry)
            out.append({
                "feature": names[j],
                "feature_name": label,
                "description": description,
                "feature_family": family,
                "weight": float(weights[j]),
                "disagreement": float(deltas[j]),
                "contribution": float(values[j]),
                "kind": kind,
            })
            if len(out) >= top:
                break
        return out

    shared = pack(np.argsort(-agreement), agreement, "SHARED")
    differs = pack(np.argsort(-difference), difference, "DIFFERS")
    return shared, differs


def _pattern_ids(row_values: dict[str, Any]) -> frozenset[str]:
    from muleguard.explain.pattern_cards import match_patterns

    return frozenset(c["id"] for c in match_patterns(row_values))


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    """Pattern agreement, reported beside behavioural similarity - never mixed in.

    Section 11 keeps these two numbers apart on purpose. Blending them would
    produce a single figure whose movement nobody could attribute, and the
    blend weight would be exactly the kind of invented constant this design
    avoids everywhere else.
    """
    if not a and not b:
        return 0.0
    union = a | b
    return float(len(a & b) / len(union)) if union else 0.0


def find_neighbors(*, index: CohortIndex, q_num: np.ndarray, q_cat: np.ndarray,
                   q_patterns: frozenset[str] = frozenset(),
                   k: int = 10, exclude_position: int | None = None,
                   with_explanations: bool = True) -> list[dict[str, Any]]:
    """The k most behaviourally similar reference accounts, ranked and explained.

    Ranking is by ``behavioral_similarity`` alone, as section 11 requires.
    Pattern agreement travels with each neighbour but never moves it up the
    list. Ties break on row index so the ordering is total and reproducible -
    an ordering that depends on sort stability is an ordering that changes when
    the reference frame is rebuilt.
    """
    t = index.transform
    sims = t.similarity(q_num, q_cat, index.numeric, index.categorical)
    if exclude_position is not None and 0 <= exclude_position < len(sims):
        sims = sims.copy()
        sims[exclude_position] = -np.inf

    k = max(0, min(int(k), len(sims)))
    if k == 0:
        return []
    order = np.lexsort((index.row_index, -sims))[:k]

    registry = None
    if with_explanations:
        from muleguard.features.frame import augmented_registry

        registry = augmented_registry()

    out: list[dict[str, Any]] = []
    for pos in order:
        pos = int(pos)
        similarity = float(sims[pos])
        if not np.isfinite(similarity):
            continue
        shared, differs = ([], [])
        if with_explanations:
            shared, differs = explain_pair(index, q_num, q_cat, pos, registry=registry)
        out.append({
            "account_reference": index.references[pos],
            "row_index": int(index.row_index[pos]),
            "behavioral_similarity": similarity,
            "similarity_percentile": float(t.percentile_of(similarity)),
            "beyond_empirical_null": bool(similarity > t.null_statistics["max"]),
            "similarity_band": t.band_of(similarity),
            "pattern_similarity": jaccard(q_patterns, index.patterns[pos]),
            "neighbor_risk_probability": float(index.risk[pos]),
            "neighbor_risk_tier": index.tier[pos],
            "shared_patterns": sorted(q_patterns & index.patterns[pos]),
            "main_shared_features": shared,
            "main_differences": differs,
        })
    return out


def mutual_edges(*, index: CohortIndex, q_num: np.ndarray, q_cat: np.ndarray,
                 neighbors: list[dict[str, Any]], k: int,
                 query_reference: str,
                 query_position: int | None = None) -> list[dict[str, Any]]:
    """Keep only neighbours that name the query back.

    One-sided nearest-neighbour lists are noisy: an isolated account's nearest
    neighbour may be nobody's nearest anything. Requiring agreement in both
    directions - A in B's top-k and B in A's top-k - removes most of that
    noise for the cost of k extra scans.

    Computed lazily, per query. Materialising the full mutual-kNN graph over the
    reference partition would be ~81M pairwise comparisons to answer a question
    about ten of them.

    Edges are BEHAVIORALLY_SIMILAR_TO. There is no code path in this module that
    emits the ownership or money-movement relationships, because the dataset
    contains no evidence that any such relationship exists.
    """
    edges: list[dict[str, Any]] = []
    for neighbor in neighbors:
        pos = index.position_of(neighbor["account_reference"])
        if pos is None:
            continue
        back = index.transform.similarity(
            index.numeric[pos], index.categorical[pos],
            index.numeric, index.categorical)
        # Exclude the neighbour from its own ranking - the row we are ranking
        # from is `pos`, not the query. Masking the query here instead would
        # make reciprocity impossible to satisfy by construction.
        back = back.copy()
        back[pos] = -np.inf
        top = np.lexsort((index.row_index, -back))[:k]
        top_refs = {index.references[int(p)] for p in top}
        if query_position is not None:
            reciprocal = query_reference in top_refs
        else:
            # A query outside the reference frame cannot be named back by
            # anyone, so reciprocity is decided on the same bar instead: the
            # query must be at least as similar as the neighbour's own k-th.
            cutoff = float(back[top[-1]]) if len(top) else 1.0
            reciprocal = neighbor["behavioral_similarity"] >= cutoff
        if reciprocal:
            edges.append({
                "source": query_reference,
                "target": neighbor["account_reference"],
                "relationship": EDGE_LABEL,
                "behavioral_similarity": neighbor["behavioral_similarity"],
                "similarity_percentile": neighbor["similarity_percentile"],
                "similarity_band": neighbor["similarity_band"],
                "mutual": True,
            })
    return edges


def cohort_summary(neighbors: list[dict[str, Any]]) -> dict[str, Any]:
    """Descriptive only. Nothing here is allowed to change anyone's tier."""
    if not neighbors:
        return {"n_neighbors": 0, "high_risk_neighbors": 0,
                "median_neighbor_risk": 0.0, "max_neighbor_risk": 0.0,
                "unusually_similar_neighbors": 0, "insufficient_data": False}
    risks = np.array([n["neighbor_risk_probability"] for n in neighbors])
    high = sum(1 for n in neighbors
               if n["neighbor_risk_tier"] in ("CRITICAL_REVIEW", "URGENT_REVIEW"))
    unusual = sum(1 for n in neighbors
                  if n["similarity_band"] in (BAND_VERY_HIGH, BAND_HIGH))
    return {
        "n_neighbors": len(neighbors),
        "high_risk_neighbors": int(high),
        "median_neighbor_risk": float(np.median(risks)),
        "max_neighbor_risk": float(risks.max()),
        "unusually_similar_neighbors": int(unusual),
        "insufficient_data": False,
    }


def cohort_for_row(row_index: int, *, k: int | None = None,
                   index: CohortIndex | None = None,
                   with_mutual: bool = True,
                   with_explanations: bool = True) -> dict[str, Any]:
    """The full cohort response for an account already in the reference frame."""
    index = build_index() if index is None else index
    position = int(np.searchsorted(index.row_index, row_index))
    if position >= len(index.row_index) or index.row_index[position] != row_index:
        raise KeyError(f"row {row_index} is not in the reference partition")
    return _assemble(
        index=index,
        q_num=index.numeric[position],
        q_cat=index.categorical[position],
        q_patterns=index.patterns[position],
        query_reference=index.references[position],
        risk=float(index.risk[position]),
        tier=index.tier[position],
        k=k, position=position, with_mutual=with_mutual,
        with_explanations=with_explanations)


def cohort_for_features(values: dict[str, Any], *, k: int | None = None,
                        index: CohortIndex | None = None,
                        query_reference: str = "QUERY",
                        risk: float | None = None, tier: str | None = None,
                        with_mutual: bool = True,
                        with_explanations: bool = True) -> dict[str, Any]:
    """The cohort response for an arbitrary scored row - an upload, say.

    The query is encoded through the frozen transform. It contributes no median,
    no category and no bound: section 19 requires that a query cannot modify the
    transform, and the only way to be sure of that is for the query path to
    contain no code that could.
    """
    index = build_index() if index is None else index
    t = index.transform

    # Pattern cards are driven mostly by the MG_* block, which a caller submits
    # raw columns for rather than computing. The reference side has that block,
    # so the query side must have it too - a Jaccard between an 11-pattern
    # vocabulary and a 3-pattern one measures the submission format, not the
    # accounts. attach_meta is row-wise and carries no fitted state, so this
    # derives the same values scoring would and consults no training data.
    enriched = with_derived_meta(values)

    frame = pl.DataFrame([{name: enriched.get(name) for name in t.features}])
    casts = [pl.col(n).cast(pl.Float64, strict=False) for n in t.numeric_features]
    casts += [pl.col(n).cast(pl.Utf8, strict=False) for n in t.categorical_features]
    frame = frame.with_columns(casts)
    q_num, q_cat = t.encode(frame)
    return _assemble(
        index=index, q_num=q_num[0], q_cat=q_cat[0],
        q_patterns=_pattern_ids(enriched), query_reference=query_reference,
        risk=risk, tier=tier, k=k, position=None, with_mutual=with_mutual,
        with_explanations=with_explanations)


def _assemble(*, index: CohortIndex, q_num, q_cat, q_patterns, query_reference,
              risk, tier, k, position, with_mutual,
              with_explanations: bool = True) -> dict[str, Any]:
    cfg = _config()
    k = int(cfg["neighbors"]["default_k"]) if k is None else int(k)
    k = max(1, min(k, int(cfg["neighbors"]["max_k"])))

    # Section 37 case 8. A query that supplies almost nothing still ranks
    # perfectly well - against a yardstick that no longer means what it says.
    # Refusing here is the controlled answer; a list of near-zero similarities
    # presented as a cohort is the uncontrolled one.
    t = index.transform
    if not t.has_enough_data(q_num, q_cat):
        payload = _envelope(index, query_reference, risk, tier, k)
        payload["cohort_summary"] = {
            "n_neighbors": 0, "high_risk_neighbors": 0,
            "median_neighbor_risk": 0.0, "max_neighbor_risk": 0.0,
            "unusually_similar_neighbors": 0, "insufficient_data": True,
        }
        payload["data_sufficiency"] = {
            "status": "INSUFFICIENT_DATA",
            "fingerprint_weight_supplied": round(t.coverage(q_num, q_cat), 6),
            "fingerprint_weight_required": round(t.min_reference_coverage, 6),
            "reason": ("The query supplies less comparable behavioural data "
                       "than the least-complete account in the reference "
                       "portfolio, so it cannot be placed against that "
                       "portfolio's similarity distribution."),
        }
        assert_language_safe(payload)
        return payload

    neighbors = find_neighbors(index=index, q_num=q_num, q_cat=q_cat,
                               q_patterns=q_patterns, k=k,
                               exclude_position=position,
                               with_explanations=with_explanations)
    edges = []
    if with_mutual:
        edges = mutual_edges(index=index, q_num=q_num, q_cat=q_cat,
                             neighbors=neighbors,
                             k=int(cfg["neighbors"]["mutual_knn_k"]),
                             query_reference=query_reference,
                             query_position=position)

    payload = _envelope(index, query_reference, risk, tier, k)
    payload["neighbors"] = neighbors
    payload["mutual_edges"] = edges
    payload["cohort_summary"] = cohort_summary(neighbors)
    assert_language_safe(payload)
    return payload


def _envelope(index: CohortIndex, query_reference: str, risk, tier,
              k: int) -> dict[str, Any]:
    """The parts of a cohort response that do not depend on finding anybody.

    Shared so the insufficient-data answer carries the same disclaimer, the same
    band table and the same provenance as a populated one. A refusal that drops
    the framing is a refusal an analyst reads as a system error.
    """
    return {
        "query_account": query_reference,
        "risk_probability": risk,
        "risk_tier": tier,
        "reference_scope": index.scope,
        "k": k,
        "neighbors": [],
        "mutual_edges": [],
        "similarity_bands": {
            name: {"similarity_at_or_above": value,
                   "null_percentile": index.transform.band_percentiles[name]}
            for name, value in index.transform.bands.items()},
        "radar_version": index.transform.radar_version,
        "interpretation": INTERPRETATION,
        "action_policy": ACTION_POLICY,
        "disclaimer": DISCLAIMER,
    }
