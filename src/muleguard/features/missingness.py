"""Missingness signature - treat "this field is absent" as evidence.

27.6 % of the cells in this dataset are null, and the nulls are not uniform:
3,834 of 3,925 columns carry at least one, and a naive count of nulls per row
already separates the classes a little (dev ROC-AUC 0.619). That is far too
weak to use on its own, but it is not nothing, and the *shape* of the
missingness may say more than its volume. An account with a 31-day cheque
history and no 7-day cheque history is telling a different story from an
account with neither.

This module turns that shape into columns. Four kinds:

  1. **Per-feature flags** - a binary "was this field null" for individual
     columns. Only for columns whose training missing-rate sits inside a band;
     a column that is null 0.02 % of the time or 100 % of the time carries no
     usable contrast and would only widen the pool.
  2. **Family counts and ratios** - how much of the UPI block, the CASH block,
     the CHEQUE block is absent. Families come from the data dictionary, not
     from string guessing.
  3. **Short-missing-while-long-present** - the asymmetry above, counted per
     family. Built by pairing variables that differ *only* in their window
     suffix (``..._L7D`` against ``..._L31D``), again through the dictionary.
  4. **Context blocks** - PROFILE and ALERT_CONTEXT absence, which mean
     something operationally different from a missing transaction aggregate:
     they say the bank did not have (or did not attach) the record.

Everything that involves a *decision* - which columns get a flag, which pairs
exist, which families are populated enough to be worth summarising - is fitted
on training rows only, through :meth:`MissingnessSignature.fit`. The transform
itself is row-local arithmetic, so applying it to held-out rows cannot move
information across the fold boundary. That is the whole point: the ablation
that decides whether to keep these columns has to be honest, and it cannot be
honest if the columns were designed while looking at the rows they are scored
on.

Whether any of this survives is not decided here. See
``muleguard.cli.missingness_ablation``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from muleguard.logging import get_logger

log = get_logger("features.missingness")

PREFIX = "MX_"

#: Window tokens that can end a variable name, longest first so that ``L7_14D``
#: is stripped before ``L7`` would ever be considered.
_WINDOWS = ("L7_14D", "L14_31D", "L7_31D", "L31D", "L14D", "L7D")

#: (short, long) window pairs whose asymmetry is meaningful. A short window
#: missing while the long window is present means the account has history but
#: no recent activity on that rail - dormancy, or a burst that has not landed
#: in the short aggregate yet.
_PAIRS = (("L7D", "L31D"), ("L7D", "L14D"), ("L14D", "L31D"))

#: Families whose absence is about record-keeping rather than behaviour.
_CONTEXT_FAMILIES = ("PROFILE", "ALERT_CONTEXT")

#: Families too small to summarise on their own - folded into the global stats.
_MIN_FAMILY_SIZE = 4


def _stem(variable_name: str) -> tuple[str, str] | None:
    """Split ``R_CHQ_AMT_CR_L7_14D`` into ``("R_CHQ_AMT_CR", "L7_14D")``."""
    for w in _WINDOWS:
        suffix = "_" + w
        if variable_name.endswith(suffix):
            return variable_name[: -len(suffix)], w
    return None


@dataclass
class MissingnessSignature:
    """Fitted missingness encoder. Fit on train rows, transform anything."""

    flag_idx: np.ndarray                       # column indices getting a flag
    family_idx: dict[str, np.ndarray]          # family -> column indices
    pair_idx: dict[str, np.ndarray]            # "FAMILY|L7D>L31D" -> (short, long) pairs
    context_idx: dict[str, np.ndarray]         # PROFILE / ALERT_CONTEXT
    names: list[str] = field(default_factory=list)
    source_names: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ fit
    @classmethod
    def fit(
        cls,
        X: np.ndarray,
        feature_names: Sequence[str],
        registry: Mapping[str, Mapping[str, Any]],
        *,
        min_rate: float = 0.01,
        max_rate: float = 0.99,
        max_flags: int = 200,
    ) -> "MissingnessSignature":
        """Decide which missingness columns exist, using training rows only.

        Args:
            X: training matrix, ``(n_train, n_features)``, NaN for missing.
            feature_names: column names of ``X``, in matrix order.
            registry: the augmented feature dictionary, keyed by feature name.
            min_rate, max_rate: a column earns an individual flag only if its
                training missing-rate falls inside this band. Outside it the
                indicator is near-constant and carries no contrast.
            max_flags: cap on individual flags, taken in descending order of
                indicator variance ``p(1-p)``, ties broken by column order so
                the result is deterministic.
        """
        nan = np.isnan(X)
        rate = nan.mean(axis=0)

        in_band = np.flatnonzero((rate >= min_rate) & (rate <= max_rate))
        if in_band.size > max_flags:
            var = rate[in_band] * (1.0 - rate[in_band])
            # -var for descending; in_band is already ascending, so a stable
            # sort keeps column order as the tie-break.
            keep = np.argsort(-var, kind="stable")[:max_flags]
            in_band = np.sort(in_band[keep])

        name_to_pos = {n: i for i, n in enumerate(feature_names)}

        # --- families ----------------------------------------------------
        families: dict[str, list[int]] = {}
        for name, pos in name_to_pos.items():
            entry = registry.get(name)
            fam = (entry or {}).get("feature_family")
            if fam:
                families.setdefault(str(fam), []).append(pos)
        family_idx = {
            f: np.asarray(sorted(v), dtype=np.int64)
            for f, v in families.items()
            if len(v) >= _MIN_FAMILY_SIZE
        }

        # --- window pairs -------------------------------------------------
        # stem -> {window: position}, restricted to columns actually present.
        stems: dict[str, dict[str, int]] = {}
        stem_family: dict[str, str] = {}
        for name, pos in name_to_pos.items():
            entry = registry.get(name)
            if not entry:
                continue
            var = str(entry.get("variable_name") or "")
            split = _stem(var)
            if split is None:
                continue
            stem, window = split
            stems.setdefault(stem, {})[window] = pos
            stem_family[stem] = str(entry.get("feature_family") or "UNKNOWN")

        pair_idx: dict[str, np.ndarray] = {}
        for short, long in _PAIRS:
            per_family: dict[str, list[tuple[int, int]]] = {}
            for stem, windows in stems.items():
                if short in windows and long in windows:
                    per_family.setdefault(stem_family[stem], []).append(
                        (windows[short], windows[long]))
            for fam, pairs in per_family.items():
                if len(pairs) < _MIN_FAMILY_SIZE:
                    continue
                key = f"{fam}|{short}>{long}"
                pair_idx[key] = np.asarray(sorted(pairs), dtype=np.int64)

        # --- context blocks ------------------------------------------------
        context_idx = {
            f: family_idx[f] for f in _CONTEXT_FAMILIES if f in family_idx
        }
        # PROFILE has 9 members and ALERT_CONTEXT 16, so both normally survive
        # _MIN_FAMILY_SIZE; fall back to a direct scan if a view dropped them.
        for f in _CONTEXT_FAMILIES:
            if f in context_idx:
                continue
            hits = [p for n, p in name_to_pos.items()
                    if (registry.get(n) or {}).get("feature_family") == f]
            if hits:
                context_idx[f] = np.asarray(sorted(hits), dtype=np.int64)

        sig = cls(flag_idx=in_band, family_idx=family_idx, pair_idx=pair_idx,
                  context_idx=context_idx)
        sig.names = sig._build_names(feature_names)
        sig.source_names = list(feature_names)
        log.info(
            "missingness signature: %d flags, %d families, %d window-pair groups, "
            "%d context blocks -> %d columns",
            sig.flag_idx.size, len(sig.family_idx), len(sig.pair_idx),
            len(sig.context_idx), len(sig.names))
        return sig

    # ------------------------------------------------------------- naming
    def _build_names(self, feature_names: Sequence[str]) -> list[str]:
        names = [f"{PREFIX}NULL__{feature_names[i]}" for i in self.flag_idx]
        for fam in sorted(self.family_idx):
            names.append(f"{PREFIX}FAMCNT__{fam}")
            names.append(f"{PREFIX}FAMRATIO__{fam}")
        for key in sorted(self.pair_idx):
            fam, rel = key.split("|", 1)
            short, long = rel.split(">", 1)
            names.append(f"{PREFIX}SHORTGAP__{fam}__{short}_missing_{long}_present")
            names.append(f"{PREFIX}LONGGAP__{fam}__{long}_missing_{short}_present")
        for fam in sorted(self.context_idx):
            names.append(f"{PREFIX}CTXCNT__{fam}")
            names.append(f"{PREFIX}CTXALL__{fam}")
        names.append(f"{PREFIX}TOTAL_NULL")
        names.append(f"{PREFIX}TOTAL_NULL_RATIO")
        return names

    # -------------------------------------------------------------- apply
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Return the missingness block for ``X``, shape ``(n_rows, len(names))``.

        Pure row-wise arithmetic on ``isnan(X)`` - no fitted statistic is
        consulted beyond the *choice* of which columns to summarise, which was
        made on training rows in :meth:`fit`.
        """
        nan = np.isnan(X)
        n = X.shape[0]
        blocks: list[np.ndarray] = []

        if self.flag_idx.size:
            blocks.append(nan[:, self.flag_idx].astype(np.float32))

        for fam in sorted(self.family_idx):
            idx = self.family_idx[fam]
            cnt = nan[:, idx].sum(axis=1, dtype=np.float32)
            blocks.append(cnt.reshape(n, 1))
            blocks.append((cnt / float(idx.size)).reshape(n, 1))

        for key in sorted(self.pair_idx):
            pairs = self.pair_idx[key]
            s, l = pairs[:, 0], pairs[:, 1]
            short_missing = nan[:, s]
            long_missing = nan[:, l]
            blocks.append((short_missing & ~long_missing).sum(
                axis=1, dtype=np.float32).reshape(n, 1))
            blocks.append((long_missing & ~short_missing).sum(
                axis=1, dtype=np.float32).reshape(n, 1))

        for fam in sorted(self.context_idx):
            idx = self.context_idx[fam]
            cnt = nan[:, idx].sum(axis=1, dtype=np.float32)
            blocks.append(cnt.reshape(n, 1))
            # "the whole block is absent" is a categorically different event
            # from "some of it is absent", so it gets its own column.
            blocks.append((cnt == float(idx.size)).astype(np.float32).reshape(n, 1))

        total = nan.sum(axis=1, dtype=np.float32)
        blocks.append(total.reshape(n, 1))
        blocks.append((total / float(X.shape[1])).reshape(n, 1))

        out = np.hstack(blocks).astype(np.float32, copy=False)
        if out.shape[1] != len(self.names):  # pragma: no cover - guard only
            raise RuntimeError(
                f"missingness block width {out.shape[1]} != {len(self.names)} names")
        return out

    def augment(self, X: np.ndarray) -> np.ndarray:
        """``X`` with the missingness block appended on the right."""
        return np.hstack([X, self.transform(X)])

    def augmented_names(self) -> list[str]:
        return list(self.source_names) + list(self.names)


def describe(name: str) -> str:
    """Plain-language gloss for one generated column, for the ProofGraph."""
    if not name.startswith(PREFIX):
        return name
    body = name[len(PREFIX):]
    kind, _, rest = body.partition("__")
    if kind == "NULL":
        return f"The field {rest} was absent for this account."
    if kind == "FAMCNT":
        return f"Number of absent fields in the {rest} block."
    if kind == "FAMRATIO":
        return f"Share of the {rest} block that was absent."
    if kind == "SHORTGAP":
        fam, _, rel = rest.partition("__")
        m = re.match(r"(\w+)_missing_(\w+)_present", rel)
        if m:
            return (f"{fam}: fields present over {m.group(2)} but absent over "
                    f"{m.group(1)} - history exists, recent activity does not.")
    if kind == "LONGGAP":
        fam, _, rel = rest.partition("__")
        m = re.match(r"(\w+)_missing_(\w+)_present", rel)
        if m:
            return (f"{fam}: recent {m.group(2)} activity recorded with no "
                    f"{m.group(1)} history behind it - a newly active rail.")
    if kind == "CTXCNT":
        return f"Number of absent {rest} fields."
    if kind == "CTXALL":
        return f"The entire {rest} block was absent for this account."
    if body == "TOTAL_NULL":
        return "Total number of absent fields across the whole record."
    if body == "TOTAL_NULL_RATIO":
        return "Share of the whole record that was absent."
    return name
