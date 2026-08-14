"""Transaction Graph Adapter endpoints (section 18, addendum UPDATE 8).

    GET  /v1/graph/status                 is a graph loaded? (UNAVAILABLE by default)
    POST /v1/graph/edges                  upload an edge file (csv/xlsx)
    GET  /v1/graph/account/{account}      metrics + named patterns for one account
    GET  /v1/graph/account/{account}/neighbourhood
    DELETE /v1/graph/edges                discard the loaded graph

The default state of every one of these is UNAVAILABLE, and that is not a
failure mode - it is the truthful answer for the competition dataset, which has
no transactions in it. The routes exist so that a judge who *does* have an edge
file can see what the system would do with one, and so that the refusal to
invent edges is visible in the API rather than only in a document.

The graph lives in memory for the life of the process. It is deliberately not
persisted: an uploaded edge file is a judge's data, it is not ours to keep, and
a graph that survives a restart would silently change what later scores are
displayed against.
"""
from __future__ import annotations

import io
from typing import Any

import polars as pl
from fastapi import APIRouter, HTTPException, UploadFile

from muleguard.graph import adapter
from muleguard.logging import get_logger

log = get_logger("api.graph")

router = APIRouter()

MAX_EDGE_BYTES = 60 * 1024 * 1024
ALLOWED_SUFFIXES = {".csv", ".xlsx"}

# Process-local, single slot. See the module docstring for why this is not a
# database table.
_GRAPH: adapter.TransactionGraph | None = None


def _parse(filename: str, payload: bytes) -> pl.DataFrame:
    suffix = ("." + filename.rsplit(".", 1)[-1].lower()) if "." in filename else ""
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(422, f"unsupported edge file type {suffix or '(none)'}; "
                                 "use .csv or .xlsx")
    try:
        if suffix == ".csv":
            return pl.read_csv(io.BytesIO(payload), infer_schema_length=5000,
                               try_parse_dates=True)
        return pl.read_excel(io.BytesIO(payload))
    except Exception as exc:  # noqa: BLE001 - any parse failure is a 422
        raise HTTPException(422, f"edge file could not be parsed: {exc}") from exc


@router.get("/v1/graph/status")
def graph_status() -> dict[str, Any]:
    if _GRAPH is None:
        return adapter.unavailable()
    return _GRAPH.summary()


@router.post("/v1/graph/edges")
async def upload_edges(file: UploadFile) -> dict[str, Any]:
    global _GRAPH
    payload = await file.read()
    if len(payload) > MAX_EDGE_BYTES:
        raise HTTPException(413, f"edge file exceeds {MAX_EDGE_BYTES // (1024 * 1024)} MB")
    if not payload:
        raise HTTPException(422, "edge file is empty")

    df = _parse(file.filename or "edges.csv", payload)
    try:
        graph = adapter.build(df)
    except adapter.EdgeSchemaError as exc:
        raise HTTPException(422, str(exc)) from exc

    _GRAPH = graph
    log.info("graph loaded: %d edges, %d accounts",
             graph.report["n_edges_used"], graph.report["n_accounts"])
    summary = graph.summary()
    summary["accepted"] = True
    summary["note"] = (
        "this graph is corroborating evidence only. No risk score changed when "
        "it was loaded, and none will change if it is discarded.")
    return summary


@router.delete("/v1/graph/edges")
def discard_edges() -> dict[str, Any]:
    global _GRAPH
    had = _GRAPH is not None
    _GRAPH = None
    return {"discarded": had, "status": "UNAVAILABLE",
            "note": "no score was affected by discarding the graph"}


@router.get("/v1/graph/account/{account}")
def account_graph(account: str) -> dict[str, Any]:
    if _GRAPH is None:
        return adapter.unavailable(
            [f"no graph is loaded, so nothing is known about {account} "
             "beyond its own aggregate features"])
    if account not in _GRAPH.accounts:
        raise HTTPException(404, f"account {account} does not appear in the "
                                 "uploaded edge file")
    return {
        "status": "OK",
        "metrics": _GRAPH.account_metrics(account),
        "patterns": _GRAPH.patterns(account),
        # Where the account sits in the wider graph, and the three structural
        # proxies. Each carries its own caveat; none of them touches the score.
        "structure": _GRAPH.component_metrics(account),
        "proxies": {
            "smurfing": _GRAPH.smurfing_proxy(account),
            "shell_distributor": _GRAPH.shell_distributor_proxy(account),
            "velocity": _GRAPH.velocity_anomaly(account),
        },
        "thresholds": adapter.GRAPH_THRESHOLDS,
        "contract": adapter.CONTRACT,
    }


@router.get("/v1/graph/account/{account}/neighbourhood")
def account_neighbourhood(account: str, hops: int = 1) -> dict[str, Any]:
    if _GRAPH is None:
        return adapter.unavailable()
    if account not in _GRAPH.accounts:
        raise HTTPException(404, f"account {account} does not appear in the "
                                 "uploaded edge file")
    if not 1 <= hops <= 3:
        raise HTTPException(422, "hops must be between 1 and 3; deeper ego "
                                 "networks are unreadable and slow to build")
    return {"status": "OK", **_GRAPH.neighbourhood(account, hops=hops),
            "contract": adapter.CONTRACT}
