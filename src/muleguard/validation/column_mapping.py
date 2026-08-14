"""Resolve upload column headers to canonical F-numbers.

The organiser's validation file may arrive with human-readable variable names
(``TOTAL_CR_AMT_30D``) instead of the canonical feature ids (``F1234``), or
with case and separator drift around either. Rejecting such a file as
"schema mismatch" would be a self-inflicted failure: the data is right, only
the headers are spelled differently.

So headers are resolved against the Description.xlsx registry before the
schema check runs. Two rules keep this from becoming a guessing game:

* only **exact** matches after normalisation are accepted - no fuzzy or
  edit-distance matching, because a wrong column silently mapped into a
  required slot is far worse than an honest failure;
* any name that resolves to more than one feature is left unmapped and
  reported as ambiguous rather than arbitrarily assigned.

Every rename is returned to the caller so the Validation Lab can show exactly
what was renamed and why.
"""
from __future__ import annotations

import re
from typing import Any

import polars as pl

from ..features.dictionary import load_registry

__all__ = ["build_alias_index", "resolve_columns", "apply_column_mapping"]

_NORM = re.compile(r"[^A-Z0-9]+")


def _norm(name: str) -> str:
    """Upper-case, strip every non-alphanumeric run. ``Total Cr Amt-30d``
    and ``TOTAL_CR_AMT_30D`` normalise to the same key."""
    return _NORM.sub("", str(name).upper())


def build_alias_index(registry: dict[str, Any] | None = None
                      ) -> tuple[dict[str, set[str]], dict[str, str]]:
    """Return ``(alias -> {features}, feature -> variable_name)``.

    Aliases come from the feature id itself and from the workbook's variable
    name. Anything mapping to several features is retained with all of them so
    the caller can report the ambiguity instead of picking one.
    """
    reg = registry if registry is not None else load_registry()
    features: dict[str, Any] = reg.get("features", {})

    alias: dict[str, set[str]] = {}
    varnames: dict[str, str] = {}
    for fid, rec in features.items():
        vn = str(rec.get("variable_name") or "").strip()
        varnames[fid] = vn
        for cand in (fid, vn):
            if cand and cand.lower() != "nan":
                alias.setdefault(_norm(cand), set()).add(fid)
    return alias, varnames


def resolve_columns(columns: list[str],
                    registry: dict[str, Any] | None = None
                    ) -> dict[str, Any]:
    """Work out how each upload header maps to a canonical feature id.

    Nothing is renamed here - this only reports the plan, so a caller can show
    it to a human before any data moves.
    """
    alias, varnames = build_alias_index(registry)
    known_ids = set(varnames)

    mapping: dict[str, str] = {}
    ambiguous: list[dict[str, Any]] = []
    unmapped: list[str] = []
    already_canonical: list[str] = []

    for col in columns:
        if col in known_ids:
            already_canonical.append(col)
            continue
        hits = alias.get(_norm(col), set())
        if len(hits) == 1:
            mapping[col] = next(iter(hits))
        elif len(hits) > 1:
            ambiguous.append({"column": col, "candidates": sorted(hits)})
            unmapped.append(col)
        else:
            unmapped.append(col)

    # A rename that collides with a column already in the file would silently
    # destroy one of them. Refuse those and report them.
    collisions: list[dict[str, Any]] = []
    incoming = set(columns)
    target_counts: dict[str, list[str]] = {}
    for src, dst in mapping.items():
        target_counts.setdefault(dst, []).append(src)
    for dst, srcs in target_counts.items():
        if len(srcs) > 1 or dst in incoming:
            collisions.append({"target": dst, "sources": sorted(srcs),
                               "already_in_upload": dst in incoming})
    for c in collisions:
        for src in c["sources"]:
            mapping.pop(src, None)
            unmapped.append(src)

    return {
        "mapping": mapping,
        "n_renamed": len(mapping),
        "already_canonical": already_canonical,
        "n_already_canonical": len(already_canonical),
        "ambiguous": ambiguous,
        "collisions": collisions,
        "unmapped": sorted(set(unmapped)),
        "strategy": "exact match on normalised feature id or workbook variable name",
        "notes": [
            "no fuzzy matching: a wrongly mapped column is worse than a rejected file",
            "ambiguous and colliding names are left unmapped and reported",
        ],
    }


def apply_column_mapping(frame: pl.DataFrame,
                         registry: dict[str, Any] | None = None
                         ) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Rename what can be resolved; return the frame and the audit record."""
    plan = resolve_columns(list(frame.columns), registry)
    if plan["mapping"]:
        frame = frame.rename(plan["mapping"])
    return frame, plan
