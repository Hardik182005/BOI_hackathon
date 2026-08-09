"""Domain meta-features derived from the Description.xlsx registry.

A deliberately small set (13 columns, not hundreds of random ratios). Every one
of them:

  * is built ONLY from columns the availability firewall admits, so a
    meta-feature can never smuggle post-resolution information back in;
  * resolves its inputs by *variable name* through the registry, so the code
    reads as banking logic (``UPI_AMT_DB_L7D``) while the matrix stays keyed on
    F-numbers;
  * is computable at inference from a single account row - no group statistics,
    no target, no cross-row leakage, therefore nothing to fit inside a fold;
  * degrades to null (not zero) when its inputs are absent, so a validation file
    with a different schema produces honest missingness rather than a fake 0.

The names below are the ones an analyst sees in the ProofGraph, so each carries
a plain-language meaning that is defensible from the data dictionary alone.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from muleguard.features import dictionary as fd
from muleguard.logging import get_logger

log = get_logger("features.meta")

EPS = 1e-6
META_PREFIX = "MG_"

# Payment rails that carry a full CR/DB amount grid in the dictionary.
RAILS = [
    "UPI", "ELEC_XFER", "NET_BNKING", "MBNKING", "CASH", "ATM",
    "POS_PYMT", "BBPS", "NON_CASH_CHQ", "CHQ", "GST",
]

# Alert-description flags that make up the convergence score (all ALERT_CONTEXT).
ALERT_FLAGS = [
    "HIGH_VALUE_UPI_DB_TXNS", "MULTI_DBS_FROM_ACCOUNT", "MULTI_PG_TXNS",
    "PWD_CHANGED_LARGE_FUND_XFERS", "RCVING_FUNDS_FROM_MULITPLE_USERS",
    "RISKY_COUNTRY_TXNS", "STATUS_CHANGE_AFTER_WD", "TXN_AT_UNUSUAL_TIME",
    "MULTI_UPI_DB_TXNS", "FAILED_UPI_TXNS", "ONE_TO_MANY_UPI_PAYMENTS",
]

META_DESCRIPTIONS: dict[str, str] = {
    "MG_PASSTHROUGH_7D": (
        "Pass-through intensity (7d): total debit amount divided by total credit "
        "amount over the last 7 days. Values near or above 1 mean money leaves "
        "about as fast as it arrives - the defining behaviour of a mule account."
    ),
    "MG_PASSTHROUGH_31D": (
        "Pass-through intensity (31d): total debit amount over total credit "
        "amount across the last 31 days, the slower-moving version of the same "
        "signal."
    ),
    "MG_RETENTION_RATIO": (
        "Balance retention: average 7-day balance divided by 7-day credit "
        "amount. Low values mean incoming funds are not retained."
    ),
    "MG_BURST_7_31": (
        "Burst / dormant-activation: share of the last 31 days' transaction "
        "count that occurred in the last 7 days, relative to the 7/31 baseline "
        "of 0.226. Positive values mean recent acceleration."
    ),
    "MG_BURST_AMT_7_31": (
        "Amount burst: share of the last 31 days' transacted amount that "
        "occurred in the last 7 days, against the same time baseline."
    ),
    "MG_RAIL_FRAGMENTATION": (
        "Rail fragmentation: how many distinct payment rails (UPI, electronic "
        "transfer, net/mobile banking, cash, ATM, POS, BBPS, cheque, GST) show "
        "material debit activity in the last 7 days. Layering often spreads "
        "across rails."
    ),
    "MG_CASHOUT_PRESSURE": (
        "Cash-out pressure: combined cash and ATM debit amount over the last 7 "
        "days as a share of 7-day credit amount."
    ),
    "MG_BALANCE_DRAIN": (
        "Balance drain: 7-day debit amount relative to the average 7-day "
        "balance. High values mean throughput far exceeds what the account holds."
    ),
    "MG_NEW_ACCOUNT_ACTIVITY": (
        "New-account high activity: recent electronic + UPI debit amount scaled "
        "by how recently the account was opened. High only when a young account "
        "moves large electronic value."
    ),
    "MG_PROFILE_MISMATCH": (
        "Profile mismatch: magnitude of the account's balance deviation from its "
        "own occupation and segment peer groups (bank-supplied deviation "
        "features), so unusual behaviour is judged against comparable customers."
    ),
    "MG_ALERT_CONVERGENCE": (
        "Alert convergence: how many distinct alert-description flags fired for "
        "this account, out of 11 pre-decision flag types."
    ),
    "MG_ODD_HOUR_ALERT_RATIO": (
        "Odd-hour alert intensity: share of this account's alerts raised at "
        "night."
    ),
    "MG_MERCHANT_LEGITIMACY": (
        "Merchant legitimacy context: evidence of a genuine business - POS and "
        "GST activity, long tenure, and healthy balance retention. Used by the "
        "false-positive layer as exculpatory context, never as a whitelist."
    ),
}

# 7 days out of 31 - the share of activity a perfectly steady account would
# show in the recent window. Deviations from it are what "burst" measures.
_BURST_BASELINE = 7.0 / 31.0

# ACCT_OPN_DAYS arrives as bucket labels, per the dictionary ("buckets").
_ACCT_OPN_DAYS_BUCKETS: dict[str, float] = {
    "L7D": 7.0, "L14D": 14.0, "L31D": 31.0, "L90D": 90.0,
    "L180D": 180.0, "L365D": 365.0, "G365D": 730.0,
}
_BUCKETED_DAYS = {"ACCT_OPN_DAYS"}


class MetaFeatureBuilder:
    """Resolves variable names to F-numbers once, then builds meta columns.

    Stateless with respect to the data: constructed from the registry and a
    column list, it can be applied to a training fold, a locked holdout or an
    uploaded validation file with identical behaviour.
    """

    def __init__(self, available_columns: list[str],
                 registry: dict[str, Any] | None = None) -> None:
        reg = registry or fd.load_registry()
        available = set(available_columns)
        self._by_name: dict[str, str] = {}
        for feat, rec in reg["features"].items():
            if feat in available:
                self._by_name.setdefault(rec["variable_name"].upper(), feat)
        self.available_columns = available
        self.registry = reg
        self._resolution_log: dict[str, list[str]] = {}

    # -- column resolution ------------------------------------------------
    def col(self, variable_name: str) -> str | None:
        """F-number for a dictionary variable name, if present in the data."""
        return self._by_name.get(variable_name.upper())

    def _expr(self, variable_name: str) -> pl.Expr | None:
        """Numeric expression for a variable name.

        The cast is non-strict on purpose: a column that is textual in one
        extract and numeric in another must degrade to null, never raise, so an
        uploaded validation file with looser typing still scores.
        """
        c = self.col(variable_name)
        if c is None:
            return None
        if variable_name.upper() in _BUCKETED_DAYS:
            return self._bucket_days(c)
        return pl.col(c).cast(pl.Float64, strict=False)

    @staticmethod
    def _bucket_days(column: str) -> pl.Expr:
        """ACCT_OPN_DAYS is supplied as buckets ('L7D', 'G365D'), not a number.

        Each bucket maps to the days value it represents; the open-ended
        'greater than 365 days' bucket takes 730 so that ordering is preserved
        without pretending to know the true age.
        """
        return (
            pl.col(column)
            .cast(pl.Utf8, strict=False)
            .str.strip_chars()
            .str.to_uppercase()
            .replace_strict(_ACCT_OPN_DAYS_BUCKETS, default=None, return_dtype=pl.Float64)
        )

    def _sum(self, variable_names: list[str]) -> tuple[pl.Expr | None, list[str]]:
        """Sum of available columns; None when none of them exist."""
        parts, used = [], []
        for n in variable_names:
            e = self._expr(n)
            if e is not None:
                parts.append(e.fill_null(0.0))
                used.append(n)
        if not parts:
            return None, []
        total = parts[0]
        for p in parts[1:]:
            total = total + p
        return total, used

    @staticmethod
    def _safe_ratio(num: pl.Expr | None, den: pl.Expr | None) -> pl.Expr:
        """num / den with a null result when either side is unavailable.

        Ratios are clipped at 100: beyond that the exact value carries no extra
        information and unbounded outliers destabilise tree splits.
        """
        if num is None or den is None:
            return pl.lit(None, dtype=pl.Float64)
        return (num / (den.abs() + EPS)).clip(0.0, 100.0)

    # -- individual meta-features ----------------------------------------
    def _passthrough(self, window: str) -> pl.Expr:
        db = self._expr(f"TOT_TXNAMT_DB_L{window}D")
        cr = self._expr(f"TOT_TXNAMT_CR_L{window}D")
        return self._safe_ratio(db, cr)

    def _retention(self) -> pl.Expr:
        bal = self._expr("AVG_BAL_7DAYS")
        cr = self._expr("TOT_TXNAMT_CR_L7D")
        return self._safe_ratio(bal, cr)

    def _burst(self, short: str, long: str, kind: str) -> pl.Expr:
        s = self._expr(f"TOT_{kind}_L{short}D")
        l = self._expr(f"TOT_{kind}_L{long}D")
        if s is None or l is None:
            return pl.lit(None, dtype=pl.Float64)
        share = s / (l.abs() + EPS)
        return (share - _BURST_BASELINE).clip(-1.0, 5.0)

    def _rail_fragmentation(self) -> pl.Expr:
        indicators = []
        for rail in RAILS:
            e = self._expr(f"{rail}_AMT_DB_L7D")
            if e is None:
                e = self._expr(f"{rail}_TXNS_DB_L7D")
            if e is not None:
                indicators.append((e.fill_null(0.0) > 0).cast(pl.Float64))
        if not indicators:
            return pl.lit(None, dtype=pl.Float64)
        total = indicators[0]
        for i in indicators[1:]:
            total = total + i
        return total

    def _cashout_pressure(self) -> pl.Expr:
        cash_out, _ = self._sum(["CASH_AMT_DB_L7D", "ATM_AMT_DB_L7D"])
        cr = self._expr("TOT_TXNAMT_CR_L7D")
        return self._safe_ratio(cash_out, cr)

    def _balance_drain(self) -> pl.Expr:
        db = self._expr("TOT_TXNAMT_DB_L7D")
        bal = self._expr("AVG_BAL_7DAYS")
        return self._safe_ratio(db, bal)

    def _new_account_activity(self) -> pl.Expr:
        elec, _ = self._sum(["ELEC_XFER_AMT_DB_L7D", "UPI_AMT_DB_L7D"])
        days = self._expr("ACCT_OPN_DAYS")
        if elec is None or days is None:
            return pl.lit(None, dtype=pl.Float64)
        # Recency weight decays with account age; a 1-year-old account gets ~0.5.
        recency = 365.0 / (days.abs() + 365.0)
        return (elec.log1p() * recency).clip(0.0, 50.0)

    def _profile_mismatch(self) -> pl.Expr:
        parts = []
        for name in ("D_AVG_BAL_7DAYS_OCC", "D_AVG_BAL_14DAYS_OCC",
                     "D_AVG_BAL_31DAYS_OCC", "D_AVG_BAL_7TO31DAYS_OCC"):
            e = self._expr(name)
            if e is not None:
                parts.append(e.abs())
        if not parts:
            return pl.lit(None, dtype=pl.Float64)
        total = parts[0]
        for p in parts[1:]:
            total = total + p
        return (total / len(parts)).log1p()

    def _alert_convergence(self) -> pl.Expr:
        parts = []
        for flag in ALERT_FLAGS:
            e = self._expr(flag)
            if e is not None:
                parts.append((e.fill_null(0.0) > 0).cast(pl.Float64))
        if not parts:
            return pl.lit(None, dtype=pl.Float64)
        total = parts[0]
        for p in parts[1:]:
            total = total + p
        return total

    def _odd_hour_ratio(self) -> pl.Expr:
        night = self._expr("NIGHT_ALERTS")
        count = self._expr("COUNT_ALERTS")
        if night is None or count is None:
            return pl.lit(None, dtype=pl.Float64)
        return (night.fill_null(0.0) / pl.max_horizontal(count.fill_null(0.0), pl.lit(1.0)))

    def _merchant_legitimacy(self) -> pl.Expr:
        """Business-likeness score in [0, 4]: POS + GST + tenure + retention.

        Deliberately additive and transparent - the analyst can see which of the
        four components fired. It is exculpatory context, never a whitelist.
        """
        parts = []
        pos = self._expr("POS_PYMT_TXNS_L31D")
        if pos is not None:
            parts.append((pos.fill_null(0.0) > 0).cast(pl.Float64))
        gst = self._expr("GST_TXNS_L31D")
        if gst is not None:
            parts.append((gst.fill_null(0.0) > 0).cast(pl.Float64))
        tenure = self._expr("TENURE_AS_OF_ALERT")
        if tenure is not None:
            parts.append((tenure.fill_null(0.0) >= 3).cast(pl.Float64))
        bal = self._expr("AVG_BAL_7DAYS")
        cr = self._expr("TOT_TXNAMT_CR_L7D")
        if bal is not None and cr is not None:
            parts.append(((bal / (cr.abs() + EPS)) > 0.25).cast(pl.Float64))
        if not parts:
            return pl.lit(None, dtype=pl.Float64)
        total = parts[0]
        for p in parts[1:]:
            total = total + p
        return total

    # -- public API -------------------------------------------------------
    def build(self, df: pl.DataFrame) -> pl.DataFrame:
        """Return ``df`` with the meta-feature columns appended."""
        exprs = {
            "MG_PASSTHROUGH_7D": self._passthrough("7"),
            "MG_PASSTHROUGH_31D": self._passthrough("31"),
            "MG_RETENTION_RATIO": self._retention(),
            "MG_BURST_7_31": self._burst("7", "31", "TXNS"),
            "MG_BURST_AMT_7_31": self._burst("7", "31", "TXNAMT"),
            "MG_RAIL_FRAGMENTATION": self._rail_fragmentation(),
            "MG_CASHOUT_PRESSURE": self._cashout_pressure(),
            "MG_BALANCE_DRAIN": self._balance_drain(),
            "MG_NEW_ACCOUNT_ACTIVITY": self._new_account_activity(),
            "MG_PROFILE_MISMATCH": self._profile_mismatch(),
            "MG_ALERT_CONVERGENCE": self._alert_convergence(),
            "MG_ODD_HOUR_ALERT_RATIO": self._odd_hour_ratio(),
            "MG_MERCHANT_LEGITIMACY": self._merchant_legitimacy(),
        }
        return df.with_columns([e.alias(name) for name, e in exprs.items()])

    def names(self) -> list[str]:
        return list(META_DESCRIPTIONS.keys())

    def coverage(self, df: pl.DataFrame) -> dict[str, float]:
        """Non-null share of each meta-feature - the honest availability report."""
        built = self.build(df)
        n = max(len(built), 1)
        return {
            name: round(float(built[name].is_not_null().sum()) / n, 4)
            for name in self.names()
        }


def meta_registry_entries() -> dict[str, dict[str, Any]]:
    """Registry-shaped records so meta-features appear in the ProofGraph and UI
    with the same provenance contract as dictionary features."""
    out: dict[str, dict[str, Any]] = {}
    for name, desc in META_DESCRIPTIONS.items():
        out[name] = {
            "feature": name,
            "variable_name": name,
            "description": desc,
            "bank_finalized": False,
            "feature_family": "ENGINEERED_META",
            "transform_family": "DOMAIN_META",
            "window": "MIXED",
            "direction": "BOTH",
            "availability_class": fd.BEHAVIORAL,
            "sensitive": False,
            "sensitive_kind": None,
            "leakage_status": fd.SAFE,
            "semantic_tags": ["ENGINEERED_META", "DOMAIN_MOTIVATED"],
        }
    return out


def attach_meta_to_registry(registry: dict[str, Any]) -> dict[str, Any]:
    """Non-destructively merge meta-feature records into a loaded registry."""
    merged = dict(registry)
    features = dict(registry.get("features", {}))
    features.update(meta_registry_entries())
    merged["features"] = features
    return merged
