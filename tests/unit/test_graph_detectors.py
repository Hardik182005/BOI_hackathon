"""Structural detectors over an uploaded edge list.

Each detector is a proxy for a typology, and the tests below check both that
it fires on the shape it names and - more importantly - that it stays quiet
when the evidence is absent or too thin to support a claim.
"""
from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from muleguard.graph.adapter import TransactionGraph, load_edges

BASE = dt.datetime(2026, 1, 1)


def _graph(rows: list[dict]) -> TransactionGraph:
    edges, report = load_edges(pl.DataFrame(rows))
    return TransactionGraph(edges, report)


def _edge(a: str, b: str, amt: float = 100.0, hours: float = 0.0) -> dict:
    return {"account_from": a, "account_to": b, "amount": amt,
            "timestamp": BASE + dt.timedelta(hours=hours)}


# -- strongly connected components -----------------------------------------


def test_a_cycle_forms_a_multi_account_scc():
    g = _graph([_edge("A", "B"), _edge("B", "C"), _edge("C", "A")])
    m = g.component_metrics("A")
    assert m["scc_size"] == 3
    assert m["in_multi_account_scc"] is True
    assert "layering" in m["interpretation"]


def test_a_straight_chain_has_no_multi_account_scc():
    g = _graph([_edge("A", "B"), _edge("B", "C"), _edge("C", "D")])
    m = g.component_metrics("B")
    assert m["scc_size"] == 1
    assert m["in_multi_account_scc"] is False


def test_component_membership_separates_disjoint_rings():
    g = _graph([_edge("A", "B"), _edge("B", "A"),
                _edge("X", "Y"), _edge("Y", "X")])
    a, x = g.component_metrics("A"), g.component_metrics("X")
    assert a["component_id"] != x["component_id"]
    assert a["component_size"] == x["component_size"] == 2


def test_component_includes_accounts_reachable_only_backwards():
    """Weak connectivity ignores direction; A and C are in one cluster."""
    g = _graph([_edge("A", "B"), _edge("C", "B")])
    assert g.component_metrics("A")["component_id"] == \
        g.component_metrics("C")["component_id"]


def test_deep_chain_does_not_exhaust_the_stack():
    """Tarjan is iterative on purpose - upload size is not ours to control."""
    rows = [_edge(f"N{i}", f"N{i+1}") for i in range(3000)]
    g = _graph(rows)
    assert g.component_metrics("N0")["component_size"] == 3001


# -- smurfing proxy ---------------------------------------------------------


def test_smurfing_fires_on_many_small_in_one_large_out():
    rows = [_edge(f"S{i}", "M", 50.0, i) for i in range(8)]
    rows.append(_edge("M", "X", 400.0, 9))
    r = _graph(rows).smurfing_proxy("M")
    assert r["detected"] is True
    assert r["distinct_senders"] == 8
    assert r["inbound_to_outbound_ratio"] < 0.25


def test_smurfing_stays_quiet_with_too_few_senders():
    rows = [_edge("S0", "M", 50.0, 0), _edge("S1", "M", 50.0, 1),
            _edge("M", "X", 400.0, 2)]
    assert _graph(rows).smurfing_proxy("M")["detected"] is False


def test_smurfing_needs_both_directions():
    rows = [_edge(f"S{i}", "M", 50.0, i) for i in range(8)]
    r = _graph(rows).smurfing_proxy("M")
    assert r["detected"] is False
    assert "inbound and outbound" in r["reason"]


def test_smurfing_never_claims_structuring():
    rows = [_edge(f"S{i}", "M", 50.0, i) for i in range(8)]
    rows.append(_edge("M", "X", 400.0, 9))
    assert "not a structuring determination" in \
        _graph(rows).smurfing_proxy("M")["caveat"]


# -- shell / distributor proxy ---------------------------------------------


def test_distributor_fires_on_one_in_many_out():
    rows = [_edge("SRC", "X", 400.0, 0)]
    rows += [_edge("X", f"R{i}", 30.0, 1) for i in range(12)]
    r = _graph(rows).shell_distributor_proxy("X")
    assert r["detected"] is True
    assert r["distinct_recipients"] == 12
    assert r["distinct_senders"] == 1


def test_distributor_stays_quiet_when_many_parties_send():
    rows = [_edge(f"S{i}", "X", 40.0, i) for i in range(6)]
    rows += [_edge("X", f"R{i}", 30.0, 10) for i in range(12)]
    assert _graph(rows).shell_distributor_proxy("X")["detected"] is False


def test_distributor_caveat_admits_legitimate_payroll():
    rows = [_edge("SRC", "X", 400.0, 0)]
    rows += [_edge("X", f"R{i}", 30.0, 1) for i in range(12)]
    assert "payroll" in _graph(rows).shell_distributor_proxy("X")["caveat"]


# -- velocity anomaly -------------------------------------------------------


def test_velocity_fires_on_a_genuine_burst():
    rows = [_edge("Q", f"N{d}", 10.0, d * 24) for d in range(20)]
    rows += [_edge("Q", f"B{i}", 10.0, 21 * 24 + i / 6) for i in range(30)]
    r = _graph(rows).velocity_anomaly("Q")
    assert r["detected"] is True
    assert r["burst_multiple"] > 3.0


def test_velocity_stays_quiet_on_steady_activity():
    rows = [_edge("Q", f"N{d}", 10.0, d * 24) for d in range(30)]
    assert _graph(rows).velocity_anomaly("Q")["detected"] is False


def test_velocity_refuses_to_rate_a_span_shorter_than_the_window():
    """All activity inside one window gives nothing to compare against, so a
    ratio here would be arithmetic without meaning."""
    rows = [_edge("Q", f"R{i}", 10.0, i / 6) for i in range(13)]
    r = _graph(rows).velocity_anomaly("Q")
    assert r["detected"] is False
    assert "no baseline period" in r["reason"]
    assert "burst_multiple" not in r


def test_velocity_needs_a_minimum_number_of_edges():
    rows = [_edge("Q", "R1", 10.0, 0), _edge("Q", "R2", 10.0, 48)]
    r = _graph(rows).velocity_anomaly("Q")
    assert r["detected"] is False
    assert "fewer than four" in r["reason"]


def test_velocity_compares_against_the_accounts_own_rate():
    rows = [_edge("Q", f"N{d}", 10.0, d * 24) for d in range(30)]
    assert "own rate" in _graph(rows).velocity_anomaly("Q")["caveat"]


# -- the whole point --------------------------------------------------------


def test_detectors_are_absent_without_an_edge_file():
    """No upload means no graph claims at all - the honest default state."""
    from muleguard.graph import adapter

    assert adapter.unavailable()["status"] == "UNAVAILABLE"


@pytest.mark.parametrize("account", ["GHOST", ""])
def test_unknown_accounts_do_not_raise(account):
    g = _graph([_edge("A", "B")])
    assert g.component_metrics(account)["component_size"] == 0
    assert g.smurfing_proxy(account)["detected"] is False
    assert g.shell_distributor_proxy(account)["detected"] is False
    assert g.velocity_anomaly(account)["detected"] is False
