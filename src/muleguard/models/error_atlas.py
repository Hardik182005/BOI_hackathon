"""The Missed-Mule Error Atlas: why did the champion not surface this account?

After an out-of-fold run, every labelled mule that the served champion failed to
place inside the analyst budget is a *miss*. This module measures a fixed set of
quantities for each miss and assigns it to exactly one of six categories with a
deterministic, ordered rule list.

What this module is
-------------------
A diagnostic instrument. It reads out-of-fold predictions, the frozen scoring
bundle's lenses and the feature matrix, and writes a description. That is all.

What this module is not
-----------------------
It is not a patch mechanism. It never modifies a score, a threshold, a
prediction, a feature set or a model, and it holds no fitted state of its own.
The specification for this upgrade is explicit about the reason::

    do not manually create rules for every missed mule. Use the Error Atlas
    only to discover repeatable patterns. Any new feature must then beat the
    old model through nested CV.

So a pattern found here leaves this system as a *hypothesis for nested-CV
testing*, never as a special case for an account. There is no per-account
branch anywhere in this file, and there cannot be one: every rule reads only
the measurement fields on :class:`MissMeasurements` and every miss is passed
through the identical ordered list.

The rule constants
------------------
Fixed once in :class:`AtlasRuleConstants`, applied uniformly, and never tuned
per case. Four of the six rules have no free parameter at all - their
thresholds are definitional ("closer than", "zero", "below the median",
"outside the budget"). The two that do carry a constant reuse values the
project had already published for other purposes:

* ``missingness_z`` and ``range_violation_share`` are the deployed OOD lens's
  own thresholds from ``configs/thresholds.yaml``. The Atlas deliberately does
  not invent a second definition of "out of distribution"; it asks the lens the
  system already serves.
* ``threshold_miss_capacity_multiple`` is the one genuinely chosen number: a
  doubling of the published analyst capacity. It is stated here, applied to
  every miss identically, and reported in the artifact so a reader can redo the
  classification with a different multiple if they disagree.

Nothing here consults the locked test split. Everything is measured on the
development out-of-fold vectors.

Language contract: an account carrying a mule label is described as exactly
that. Feature-space proximity is described as feature-space proximity and never
as a relationship, transfer or connection between accounts - this dataset has
no edge table, and inventing one would be fabricated evidence.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: The six categories the specification fixes. A miss is assigned exactly one.
CATEGORIES: tuple[str, ...] = (
    "LOW_SIGNAL_MULE",
    "LOOKALIKE_MULE",
    "MODEL_DISAGREEMENT",
    "MISSING_DATA",
    "OOD_PATTERN",
    "THRESHOLD_MISS",
)

#: Columns that must never reach any evidence list, feature list or attribution
#: this module emits. The authoritative list lives in
#: ``artifacts/features/quarantined_features.json``; this frozen copy exists so
#: that :func:`assert_no_quarantined_feature` still refuses to pass a payload
#: even if that file were edited, and so the guard is unit-testable offline.
QUARANTINED_FEATURES: frozenset[str] = frozenset({
    "F2230", "F3892", "F3898", "F3899", "F3912", "F3913", "F3914", "F3915",
    "F3916", "F3917", "F3918", "F3924", "__UNNAMED__0",
})

#: How a feature-space neighbour must always be described. There is no edge
#: dataset in this project, so proximity is proximity and nothing more.
NEIGHBOUR_RELATION = "FEATURE_SPACE_NEAREST_NEIGHBOUR"
NEIGHBOUR_DISCLAIMER = (
    "Feature-space proximity in the champion's own 120-column input space, "
    "robust-scaled with the deployed OOD lens's medians and IQRs. It is a "
    "similarity of measured attributes only. It is NOT a transfer, a shared "
    "counterparty, a relationship or any other link between the two accounts: "
    "this dataset contains no edge table and none was derived."
)


@dataclass(frozen=True)
class AtlasRuleConstants:
    """Every free parameter in the classification, in one place.

    Frozen so a caller cannot mutate the rule set halfway through a run and
    produce an artifact whose rows were classified under different definitions.
    """

    #: Deployed OOD lens threshold (configs/thresholds.yaml ood.missingness_z_threshold).
    missingness_z: float = 4.0
    #: Deployed OOD lens threshold (configs/thresholds.yaml ood.range_violation_share_threshold).
    range_violation_share: float = 0.20
    #: Half of the champion's inputs absent. Definitional: below this the model
    #: still saw the majority of the evidence it was trained to read.
    missing_fraction: float = 0.50
    #: The one chosen constant - see the module docstring.
    threshold_miss_capacity_multiple: float = 2.0
    #: Neighbourhood size, reused from the deployed OOD lens (OODDetector.k).
    neighbourhood_k: int = 10

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_CONSTANTS = AtlasRuleConstants()


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MissMeasurements:
    """The measured quantities a classification is allowed to read.

    Deliberately narrow. The classifier receives this object and nothing else -
    no account identifier it could branch on, no free text, no model handle. A
    per-account special case is therefore not expressible.
    """

    #: Rank of the account under the champion's repeat-averaged OOF score,
    #: 1 = highest risk.
    champion_rank: int
    #: The analyst budget the miss was defined against.
    budget_k: int
    #: Share of the champion's own input columns that are absent for this row.
    missing_fraction: float
    #: That share expressed as a z-score against the development distribution,
    #: using the deployed OOD lens's mean and standard deviation.
    missingness_z: float
    #: Share of the champion's inputs whose value lies outside the widened
    #: development range, from the deployed OOD lens.
    range_violation_share: float
    #: Deployed OOD lens k-NN distance and its published gate.
    knn_distance: float
    knn_threshold: float
    #: How many of the OTHER model families would have placed this account
    #: inside the same budget on the same folds.
    n_peer_families_within_budget: int
    #: Labelled mule accounts among the k feature-space nearest development
    #: neighbours, excluding the account itself.
    n_mule_labels_in_neighbourhood: int
    #: Feature-space distances, self excluded.
    distance_to_nearest_known_mule: float
    distance_to_nearest_legitimate_account: float
    #: How many measured families rank this account below the development
    #: median, out of how many were measured.
    n_families_below_dev_median: int
    n_families_measured: int

    def to_dict(self) -> dict[str, Any]:
        return {k: (int(v) if isinstance(v, (int, np.integer)) and not isinstance(v, bool)
                    else float(v)) for k, v in asdict(self).items()}


# ---------------------------------------------------------------------------
# The rule list
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Rule:
    """One classification rule: a category, an exact test, and its inputs."""

    name: str
    category: str
    statement: str
    inputs: tuple[str, ...]
    rationale: str
    predicate: Callable[[MissMeasurements, AtlasRuleConstants], bool] = field(repr=False)


def _r_missing_data(m: MissMeasurements, c: AtlasRuleConstants) -> bool:
    return (m.missingness_z >= c.missingness_z
            or m.missing_fraction >= c.missing_fraction)


def _r_ood_pattern(m: MissMeasurements, c: AtlasRuleConstants) -> bool:
    return (m.knn_distance > m.knn_threshold
            or m.range_violation_share > c.range_violation_share)


def _r_threshold_miss(m: MissMeasurements, c: AtlasRuleConstants) -> bool:
    return m.champion_rank <= c.threshold_miss_capacity_multiple * m.budget_k


def _r_model_disagreement(m: MissMeasurements, c: AtlasRuleConstants) -> bool:
    return m.n_peer_families_within_budget >= 1


def _r_lookalike(m: MissMeasurements, c: AtlasRuleConstants) -> bool:
    return (m.n_mule_labels_in_neighbourhood == 0
            and m.distance_to_nearest_legitimate_account
            < m.distance_to_nearest_known_mule)


def _r_low_signal(m: MissMeasurements, c: AtlasRuleConstants) -> bool:
    return True  # residual - reached only when nothing above fired


#: Evaluated in this order; the FIRST rule that fires assigns the category.
#:
#: The order is an argument, not an accident. It runs from "the model could not
#: have known" to "the model had every chance and still did not":
#:
#:   1 MISSING_DATA        the evidence was not present to read
#:   2 OOD_PATTERN         the evidence was present but unlike anything in
#:                         training, so the model was extrapolating
#:   3 THRESHOLD_MISS      the champion DID rank the account highly; the
#:                         operating point, not the model, kept it out
#:   4 MODEL_DISAGREEMENT  the signal was recoverable on the same folds by a
#:                         peer family, so this is an estimator failure
#:   5 LOOKALIKE_MULE      no mule-labelled account sits near it and a
#:                         legitimate account sits nearer than any mule does
#:   6 LOW_SIGNAL_MULE     residual: no measured quantity distinguished it
RULES: tuple[Rule, ...] = (
    Rule(
        name="missing_data",
        category="MISSING_DATA",
        statement=("missingness_z >= {missingness_z} OR "
                   "missing_fraction >= {missing_fraction}"),
        inputs=("missingness_z", "missing_fraction"),
        rationale=("The champion could not read evidence that was not there. "
                   "Both thresholds are the deployed OOD lens's own published "
                   "values, so the Atlas does not invent a second definition of "
                   "'too much missing data'."),
        predicate=_r_missing_data,
    ),
    Rule(
        name="ood_pattern",
        category="OOD_PATTERN",
        statement=("knn_distance > knn_threshold (the lens's own dev-quantile "
                   "gate) OR range_violation_share > {range_violation_share}"),
        inputs=("knn_distance", "knn_threshold", "range_violation_share"),
        rationale=("The evidence was present but unlike the development cohort, "
                   "so the champion was extrapolating. Delegated verbatim to the "
                   "OOD lens the system already serves."),
        predicate=_r_ood_pattern,
    ),
    Rule(
        name="threshold_miss",
        category="THRESHOLD_MISS",
        statement=("champion_rank <= {threshold_miss_capacity_multiple} x "
                   "budget_k"),
        inputs=("champion_rank", "budget_k"),
        rationale=("The champion did rank the account near the top of the book; "
                   "what excluded it was the size of the review queue, not the "
                   "model's opinion. Placed above the estimator rules because "
                   "'the operating point cut it' is the more parsimonious "
                   "explanation when both apply."),
        predicate=_r_threshold_miss,
    ),
    Rule(
        name="model_disagreement",
        category="MODEL_DISAGREEMENT",
        statement="n_peer_families_within_budget >= 1",
        inputs=("n_peer_families_within_budget",),
        rationale=("At least one other model family, scored out-of-fold on the "
                   "same folds and the same labels, would have placed this "
                   "account inside the same budget. The signal was therefore "
                   "recoverable from this data; this champion did not recover "
                   "it."),
        predicate=_r_model_disagreement,
    ),
    Rule(
        name="lookalike_mule",
        category="LOOKALIKE_MULE",
        statement=("n_mule_labels_in_neighbourhood == 0 AND "
                   "distance_to_nearest_legitimate_account < "
                   "distance_to_nearest_known_mule"),
        inputs=("n_mule_labels_in_neighbourhood",
                "distance_to_nearest_legitimate_account",
                "distance_to_nearest_known_mule"),
        rationale=("In the champion's own input space this account sits inside "
                   "the legitimate book: no mule-labelled account is among its "
                   "nearest neighbours, and a legitimate account is nearer than "
                   "any mule-labelled one. Both tests are definitional - 'zero' "
                   "and 'closer than' - so there is no threshold to tune."),
        predicate=_r_lookalike,
    ),
    Rule(
        name="low_signal_mule",
        category="LOW_SIGNAL_MULE",
        statement="residual - reached only when no rule above fired",
        inputs=("n_families_below_dev_median", "n_families_measured"),
        rationale=("No measured quantity separated this account from the "
                   "legitimate book: the evidence was present and in "
                   "distribution, the champion did not rank it near the budget, "
                   "no peer family surfaced it, and its neighbourhood is not "
                   "exclusively legitimate. Recorded as an unattributed miss."),
        predicate=_r_low_signal,
    ),
)

RULE_ORDER: tuple[str, ...] = tuple(r.name for r in RULES)


def rule_book(constants: AtlasRuleConstants = DEFAULT_CONSTANTS) -> list[dict[str, Any]]:
    """The rule list as data, with the constants substituted into each test."""
    c = constants.to_dict()
    return [{
        "priority": i + 1,
        "rule": r.name,
        "category": r.category,
        "test": r.statement.format(**c),
        "reads": list(r.inputs),
        "rationale": r.rationale,
    } for i, r in enumerate(RULES)]


def classify(m: MissMeasurements,
             constants: AtlasRuleConstants = DEFAULT_CONSTANTS) -> dict[str, Any]:
    """Assign one category to one miss, showing every rule's verdict.

    Deterministic and side-effect free: the same measurements always produce the
    same category, and no state survives the call. Every rule is evaluated even
    after one has fired, so the artifact can show which explanations were
    available and which one the priority order chose.
    """
    c = constants.to_dict()
    evaluated: list[dict[str, Any]] = []
    for i, r in enumerate(RULES):
        fired = bool(r.predicate(m, constants))
        evaluated.append({
            "priority": i + 1,
            "rule": r.name,
            "category": r.category,
            "fired": fired,
            "test": r.statement.format(**c),
            "measured": {f: _num(getattr(m, f)) for f in r.inputs},
        })
    decided = next(e for e in evaluated if e["fired"])
    also = [e["rule"] for e in evaluated
            if e["fired"] and e["rule"] != decided["rule"]
            and e["rule"] != "low_signal_mule"]
    return {
        "category": decided["category"],
        "decided_by_rule": decided["rule"],
        "decided_by_test": decided["test"],
        "decided_by_measured_values": decided["measured"],
        "other_rules_that_also_fired": also,
        "rules_evaluated": evaluated,
        "rule_order": list(RULE_ORDER),
    }


def _num(v: Any) -> Any:
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    f = float(v)
    return None if not np.isfinite(f) else round(f, 6)


def rule_firing_rates(measurements: Sequence[MissMeasurements],
                      constants: AtlasRuleConstants = DEFAULT_CONSTANTS
                      ) -> dict[str, Any]:
    """How often each rule fires over a control group, e.g. the caught mules.

    A rule that fires just as often on the mules the champion DID surface is not
    explaining the misses, and an Atlas that did not measure that would be
    telling a story instead of reporting one. Reported alongside the counts so a
    reader can see the discriminating power of every rule rather than assume it.
    """
    n = len(measurements)
    out: dict[str, Any] = {"n_control_rows": n, "rates": {}}
    for r in RULES:
        if r.name == "low_signal_mule":
            continue
        fired = sum(1 for m in measurements if r.predicate(m, constants))
        out["rates"][r.name] = {
            "n_fired": fired,
            "rate": round(fired / n, 4) if n else None,
        }
    return out


# ---------------------------------------------------------------------------
# Measurement helpers (pure functions over arrays)
# ---------------------------------------------------------------------------


def dense_rank_desc(scores: np.ndarray) -> np.ndarray:
    """1-based rank, highest score first, ties broken by original order."""
    order = np.argsort(-np.asarray(scores, dtype=float), kind="stable")
    rank = np.empty(len(order), dtype=np.int64)
    rank[order] = np.arange(1, len(order) + 1)
    return rank


def robust_scale(X: np.ndarray, med: np.ndarray, iqr: np.ndarray) -> np.ndarray:
    """Median-impute then robust-scale, exactly as the deployed OOD lens does.

    Reused rather than reinvented so that "nearest neighbour" in the Atlas means
    the same thing as "nearest neighbour" in the lens the system serves.
    """
    Xi = np.where(np.isnan(X), med, X)
    return (Xi - med) / iqr


def neighbour_geometry(Z: np.ndarray, y: np.ndarray, query_rows: Sequence[int],
                       k: int = DEFAULT_CONSTANTS.neighbourhood_k
                       ) -> list[dict[str, Any]]:
    """Feature-space neighbours of each query row, with the account excluded.

    Returns, per query row: the nearest development row carrying a mule label,
    the nearest development row not carrying one, both distances, and the label
    composition of the k nearest neighbours. Self is excluded everywhere - an
    account is trivially its own nearest neighbour and including it would make
    every mule look like it sits next to a mule.
    """
    from sklearn.neighbors import NearestNeighbors

    Z = np.asarray(Z, dtype=float)
    y = np.asarray(y)
    pos_rows = np.where(y == 1)[0]
    neg_rows = np.where(y == 0)[0]

    nn_all = NearestNeighbors(n_neighbors=min(k + 1, len(Z))).fit(Z)
    nn_pos = NearestNeighbors(n_neighbors=min(2, len(pos_rows))).fit(Z[pos_rows])
    nn_neg = NearestNeighbors(n_neighbors=min(2, len(neg_rows))).fit(Z[neg_rows])

    q = np.asarray(list(query_rows), dtype=int)
    d_all, i_all = nn_all.kneighbors(Z[q])
    d_pos, i_pos = nn_pos.kneighbors(Z[q])
    d_neg, i_neg = nn_neg.kneighbors(Z[q])

    out: list[dict[str, Any]] = []
    for t, row in enumerate(q):
        keep = i_all[t] != row
        nb_idx = i_all[t][keep][:k]
        nb_d = d_all[t][keep][:k]

        pm = pos_rows[i_pos[t]] != row
        mule_j = int(pos_rows[i_pos[t]][pm][0]) if pm.any() else -1
        mule_d = float(d_pos[t][pm][0]) if pm.any() else float("inf")

        nm = neg_rows[i_neg[t]] != row
        leg_j = int(neg_rows[i_neg[t]][nm][0]) if nm.any() else -1
        leg_d = float(d_neg[t][nm][0]) if nm.any() else float("inf")

        out.append({
            "nearest_mule_labelled_row": mule_j,
            "distance_to_nearest_known_mule": mule_d,
            "nearest_legitimate_row": leg_j,
            "distance_to_nearest_legitimate_account": leg_d,
            "neighbourhood_k": int(len(nb_idx)),
            "neighbourhood_rows": [int(v) for v in nb_idx],
            "n_mule_labels_in_neighbourhood": int(y[nb_idx].sum()),
            "mean_neighbour_distance": float(nb_d.mean()) if len(nb_d) else None,
        })
    return out


def percentile_of(value: float, reference: np.ndarray) -> float:
    """Percentile of ``value`` within ``reference`` (0-100, right-inclusive)."""
    ref = np.sort(np.asarray(reference, dtype=float))
    if not len(ref):
        return float("nan")
    return float(100.0 * np.searchsorted(ref, float(value), side="right") / len(ref))


def missingness_profile(row: np.ndarray, feature_names: Sequence[str],
                        families: dict[str, str], miss_mu: float, miss_sd: float
                        ) -> dict[str, Any]:
    """Which of the champion's inputs are absent for this account, and how odd.

    ``families`` maps feature name -> feature family, taken from the project's
    feature dictionary. Nothing here is invented: a feature with no dictionary
    entry keeps whatever family the dictionary assigns it.
    """
    absent = np.isnan(np.asarray(row, dtype=float))
    frac = float(absent.mean())
    missing_names = [feature_names[j] for j in np.where(absent)[0]]
    by_family: dict[str, int] = {}
    for nm in missing_names:
        fam = families.get(nm, "OTHER")
        by_family[fam] = by_family.get(fam, 0) + 1
    return {
        "n_features": int(len(feature_names)),
        "n_missing": int(absent.sum()),
        "missing_fraction": round(frac, 6),
        "missingness_z_vs_development": round(float((frac - miss_mu) / miss_sd), 4),
        "development_mean_missing_fraction": round(float(miss_mu), 6),
        "missing_by_feature_family": dict(sorted(by_family.items(),
                                                 key=lambda kv: (-kv[1], kv[0]))),
        "missing_features": sorted(missing_names),
    }


def top_attributions(contributions: np.ndarray, feature_names: Sequence[str],
                     values: np.ndarray, top_n: int = 10) -> list[dict[str, Any]]:
    """Largest-magnitude per-feature attributions for one account.

    ``contributions`` is a (n_repeats, n_features) block: one attribution vector
    per CV repeat, from the model that actually produced that repeat's
    out-of-fold score. The mean is reported with its across-repeat standard
    deviation, because an attribution that flips sign between repeats is not
    evidence and must not be presented as though it were.
    """
    C = np.atleast_2d(np.asarray(contributions, dtype=float))
    mean = C.mean(axis=0)
    sd = C.std(axis=0)
    order = np.argsort(-np.abs(mean))[:top_n]
    out = []
    for j in order:
        name = feature_names[j]
        if name in QUARANTINED_FEATURES:  # unreachable via the firewall; belt and braces
            continue
        v = float(values[j])
        out.append({
            "feature": name,
            "value": None if not np.isfinite(v) else round(v, 6),
            "value_is_missing": bool(not np.isfinite(v)),
            "mean_contribution": round(float(mean[j]), 6),
            "contribution_std_across_repeats": round(float(sd[j]), 6),
            "direction": ("INCREASES_MODEL_SCORE" if mean[j] > 0
                          else "DECREASES_MODEL_SCORE" if mean[j] < 0
                          else "NEUTRAL"),
            "stable_sign_across_repeats": bool(np.all(np.sign(C[:, j]) == np.sign(mean[j]))
                                               and mean[j] != 0),
        })
    return out


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


def iter_strings(obj: Any) -> Iterable[str]:
    """Every string anywhere in a nested JSON-shaped structure, keys included."""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str):
                yield k
            yield from iter_strings(v)
    elif isinstance(obj, (list, tuple, set)):
        for v in obj:
            yield from iter_strings(v)


def assert_no_quarantined_feature(payload: Any, *, context: str = "error atlas") -> None:
    """Refuse to emit a payload that names a quarantined column.

    Whole-token match, so a legitimate mention of a *count* of quarantined
    columns survives while an actual feature name does not. Raised rather than
    logged: an Atlas that silently published a post-outcome column as evidence
    would be worse than one that did not run.
    """
    import re

    hits: set[str] = set()
    pattern = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
    for s in iter_strings(payload):
        for tok in pattern.findall(s):
            if tok in QUARANTINED_FEATURES:
                hits.add(tok)
    if hits:
        raise ValueError(
            f"{context}: quarantined feature(s) {sorted(hits)} appear in the "
            f"payload. These columns are post-outcome or fairness-excluded and "
            f"must never surface as evidence.")


def assert_read_only_contract(before: dict[str, Any], after: dict[str, Any]) -> None:
    """The Atlas must leave the ML core exactly as it found it.

    ``before``/``after`` are digests of the inputs the Atlas reads (see the CLI).
    A difference means something in this run wrote to a model artifact, which is
    the one thing this instrument is forbidden to do.
    """
    changed = sorted(k for k in before if before[k] != after.get(k))
    if changed:
        raise RuntimeError(
            f"error atlas modified inputs it must only read: {changed}. The "
            f"Atlas is read-only with respect to scores, thresholds, features "
            f"and models.")
