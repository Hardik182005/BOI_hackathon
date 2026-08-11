"""Transaction Graph Adapter (section 18, addendum UPDATE 8).

The edges below are synthetic and hand-built so that each named pattern has one
unambiguous instance. They are NOT derived from the competition data - that is
the whole point of the module, and a test that quietly manufactured edges from
F-columns would be testing the exact behaviour UPDATE 8 forbids.
"""
from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from muleguard.graph import adapter

T0 = dt.datetime(2026, 3, 1, 9, 0)


def _edges() -> pl.DataFrame:
    rows = []
    # COLLECTOR receives from 12 distinct parties within a morning ...
    for i in range(12):
        rows.append((f"PAYER{i:02d}", "COLLECTOR", 5_000.0 + i, T0 + dt.timedelta(minutes=5 * i)))
    # ... and forwards almost all of it onward two hours later (pass-through).
    rows.append(("COLLECTOR", "HOP1", 59_000.0, T0 + dt.timedelta(hours=2)))
    # A rapid layering chain of four hops.
    rows.append(("HOP1", "HOP2", 58_000.0, T0 + dt.timedelta(hours=5)))
    rows.append(("HOP2", "HOP3", 57_000.0, T0 + dt.timedelta(hours=9)))
    rows.append(("HOP3", "EXIT", 56_000.0, T0 + dt.timedelta(hours=14)))
    # A separate circular flow.
    rows.append(("RING_A", "RING_B", 1_000.0, T0))
    rows.append(("RING_B", "RING_C", 1_000.0, T0 + dt.timedelta(hours=1)))
    rows.append(("RING_C", "RING_A", 1_000.0, T0 + dt.timedelta(hours=2)))
    # A dispersing account paying 11 distinct beneficiaries.
    for i in range(11):
        rows.append(("DISPERSER", f"BENE{i:02d}", 900.0, T0 + dt.timedelta(hours=3, minutes=i)))
    rows.append(("FUNDER", "DISPERSER", 10_000.0, T0 + dt.timedelta(hours=2)))
    # Noise that must be dropped: a self-loop and a zero-value transfer.
    rows.append(("COLLECTOR", "COLLECTOR", 100.0, T0))
    rows.append(("PAYER00", "COLLECTOR", 0.0, T0))
    return pl.DataFrame(
        rows, schema=["account_from", "account_to", "amount", "timestamp"],
        orient="row")


@pytest.fixture(scope="module")
def graph() -> adapter.TransactionGraph:
    return adapter.build(_edges())


def test_default_state_is_unavailable_and_says_why():
    """No edge file means no graph - and an explanation, not an error."""
    u = adapter.unavailable()
    assert u["status"] == "UNAVAILABLE"
    assert "no sender, receiver or transaction" in u["reason"]
    assert "does not fabricate edges from feature similarity" in u["what_we_refuse_to_do"]


def test_self_loops_and_zero_amounts_are_dropped_and_counted(graph):
    assert graph.report["n_self_loops_dropped"] == 1
    assert graph.report["n_non_positive_amounts_dropped"] == 1
    assert graph.report["n_edges_used"] == graph.report["n_edges_supplied"] - 2


def test_column_aliases_are_accepted():
    """A judge exports what their system produces, not what we would prefer."""
    df = _edges().rename({"account_from": "sender", "account_to": "beneficiary",
                          "amount": "txn_amount", "timestamp": "value_date"})
    g = adapter.build(df)
    assert "COLLECTOR" in g.accounts


def test_missing_columns_raise_a_schema_error():
    df = _edges().drop("amount")
    with pytest.raises(adapter.EdgeSchemaError, match="missing"):
        adapter.build(df)


def test_fan_in_and_passthrough_detected_on_the_collector(graph):
    m = graph.account_metrics("COLLECTOR")
    assert m["fan_in_counterparties"] == 12
    assert m["fan_out_counterparties"] == 1
    assert m["passthrough_ratio"] >= 0.90
    names = {p["pattern"] for p in graph.patterns("COLLECTOR")}
    assert "FAN_IN_COLLECTION" in names
    assert "RAPID_PASSTHROUGH" in names


def test_fan_out_detected_on_the_disperser(graph):
    names = {p["pattern"] for p in graph.patterns("DISPERSER")}
    assert "FAN_OUT_DISPERSAL" in names


def test_layering_chain_follows_time_order(graph):
    chain = graph.longest_rapid_chain("COLLECTOR")
    assert chain["depth"] >= 3
    assert chain["path"][:3] == ["COLLECTOR", "HOP1", "HOP2"]


def test_circular_flow_detected(graph):
    cycle = graph.find_cycle("RING_A")
    assert cycle is not None and cycle[0] == cycle[-1] == "RING_A"
    assert "CIRCULAR_FLOW" in {p["pattern"] for p in graph.patterns("RING_A")}


def test_every_pattern_carries_the_values_that_triggered_it(graph):
    """A card that names a pattern without its evidence is an assertion."""
    for account in ("COLLECTOR", "DISPERSER", "RING_A"):
        for p in graph.patterns(account):
            assert p["observed"], f"{p['pattern']} on {account} has no evidence"
            assert p["meaning"]


def test_neighbourhood_is_bounded_and_declares_truncation(graph):
    n = graph.neighbourhood("COLLECTOR", hops=1, max_nodes=5)
    assert len(n["nodes"]) <= 5
    assert n["truncated"] is True
    assert "capped at 5 accounts" in n["truncation_note"]
    full = graph.neighbourhood("RING_A", hops=1)
    assert full["truncated"] is False and full["truncation_note"] is None


def test_contract_states_the_graph_never_feeds_the_model(graph):
    contract = " ".join(graph.summary()["contract"])
    assert "none was derived from F-columns" in contract
    assert "no graph metric is an input to the mule model" in contract
    assert "identical whether or not an edge file exists" in contract
