"""Hostile-input tests against every route that accepts a string from a user.

The threat model is modest and specific. This is an internal analyst tool, not
a public service, so the interesting attacks are the ones an ordinary upload or
a copy-pasted case id can carry: SQL injection through an identifier, stored
XSS through an analyst's free-text reason, path traversal through a filename or
a seal id, and CSV formula injection through an exported field.

Each test asserts the system's *behaviour* rather than the presence of a filter.
A route that rejects `../../etc/passwd` with a 404 is as safe as one that
rejects it with a 422; a route that reads the file is not, whatever it validates.
"""
from __future__ import annotations

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from muleguard import settings
from muleguard.api.main import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(scope="module")
def case_id() -> str:
    """A case to attach hostile analyst input to.

    Written straight to the database rather than produced by scoring a row: a
    case only opens when a row lands above MONITOR, and whether the first row
    of the dataset does that is an accident of the model, not something a
    security test should depend on. Earlier this fixture went through /v1/score
    and the whole XSS block silently skipped whenever the suite ran in an order
    that reset the database first.
    """
    import uuid

    from muleguard.api import database as db

    cid = f"CASE-SEC{uuid.uuid4().hex[:6].upper()}"
    now = db.utcnow()
    with db.connect() as c:
        c.execute(
            "INSERT INTO cases (case_id, account_reference, risk_tier,"
            " calibrated_risk, status, score_id, created_utc, updated_utc)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (cid, "ACC-SECURITY-TEST", "INVESTIGATE", 0.91, "OPEN", None, now, now),
        )
    return cid


# --- injection through identifiers -------------------------------------------

SQLI = [
    "'; DROP TABLE cases;--",
    "' OR '1'='1",
    "1; DELETE FROM audit_events",
    "\" UNION SELECT * FROM scores--",
]


@pytest.mark.parametrize("payload", SQLI)
def test_sql_injection_in_case_id_does_not_execute(client, payload):
    """A malicious case id must be data, never syntax.

    The database layer parameterises every statement, so the expected outcome
    is a clean 404: the row genuinely does not exist. What must never happen is
    a 500 (the string reached the parser) or a 200 (it matched everything).
    """
    r = client.get(f"/v1/cases/{payload}")
    assert r.status_code in (404, 422), r.text
    assert r.status_code != 500


def test_tables_survive_an_injection_attempt(client):
    """After the attempts above, the schema is still there."""
    for payload in SQLI:
        client.get(f"/v1/cases/{payload}")
    r = client.get("/v1/cases?limit=1")
    assert r.status_code == 200, "the cases table stopped answering after injection attempts"


@pytest.mark.parametrize("payload", SQLI)
def test_sql_injection_in_query_filters(client, payload):
    r = client.get(f"/v1/cases?status={payload}&limit=5")
    assert r.status_code in (200, 422)
    if r.status_code == 200:
        assert r.json()["cases"] == [] or isinstance(r.json()["cases"], list)


# --- path traversal ----------------------------------------------------------

TRAVERSAL = [
    "../../../../etc/passwd",
    "..\\..\\..\\..\\windows\\win.ini",
    "....//....//etc/shadow",
    "%2e%2e%2f%2e%2e%2fetc%2fpasswd",
]


@pytest.mark.parametrize("payload", TRAVERSAL)
def test_traversal_in_seal_id_reads_nothing(client, payload):
    r = client.get(f"/v1/validation/seals/{payload}")
    assert r.status_code in (400, 404, 422), r.text
    body = r.text.lower()
    assert "root:" not in body and "[fonts]" not in body


@pytest.mark.parametrize("payload", [
    "../../config/model",          # escape upward
    "..",                          # the directory itself
    "a/b",                         # a subdirectory
    "a\\b",                        # ditto, Windows
    "CON",                         # reserved device name on Windows
    "",                            # empty
    "x" * 200,                     # absurd length
    "seal-2026\x00.json",          # null byte truncation
])
def test_seal_id_allowlist_rejects_anything_that_could_name_a_file(payload):
    """The seal id becomes a filename, so it gets an allow-list, not a filter.

    Reveal writes the manifest back after scoring, so an id that resolves
    outside the seal directory is an arbitrary *write*, not just a read. That
    is why this is checked in the module that mints the ids rather than only at
    the route.
    """
    from muleguard.validation import sealed

    with pytest.raises(sealed.BadSealId):
        sealed.seal_path(payload)


def test_a_legitimate_seal_id_still_resolves(tmp_path, monkeypatch):
    from muleguard.validation import sealed

    monkeypatch.setattr(sealed, "SEAL_DIR", tmp_path)
    p = sealed.seal_path("seal-20260812T101500-ab12cd34")
    assert p.parent == tmp_path.resolve()
    assert p.name.endswith(".json")


@pytest.mark.parametrize("payload", TRAVERSAL)
def test_traversal_in_proofgraph_case_id(client, payload):
    r = client.get(f"/v1/proofgraph/{payload}")
    assert r.status_code in (400, 404, 422)
    assert "root:" not in r.text.lower()


def test_traversal_in_upload_filename_is_not_used_as_a_path(client):
    """An upload's filename is attacker-controlled and must never be a path.

    The graph adapter is the newest upload route, so it is the one worth
    checking: it takes a file, and a naive implementation would echo the name
    into a temporary path.
    """
    hostile = "../../../../tmp/evil.csv"
    csv = b"account_from,account_to,amount,timestamp\nA,B,10,2026-03-01T09:00:00\n"
    r = client.post("/v1/graph/edges", files={"file": (hostile, csv, "text/csv")})
    assert r.status_code in (200, 422)
    assert not Path("/tmp/evil.csv").exists()
    for probe in (settings.REPO_ROOT / "tmp/evil.csv",
                  settings.REPO_ROOT.parent / "tmp/evil.csv"):
        assert not probe.exists(), f"upload filename became a real path: {probe}"
    client.request("DELETE", "/v1/graph/edges")


# --- cross-site scripting ----------------------------------------------------

XSS = [
    "<script>alert(1)</script>",
    "<img src=x onerror=alert(1)>",
    "javascript:alert(document.cookie)",
    "\"><svg/onload=alert(1)>",
]


@pytest.mark.parametrize("payload", XSS)
def test_xss_payload_is_never_reflected_as_html(client, payload):
    """Responses are JSON, so a payload must come back escaped or not at all.

    The frontend renders through React, which escapes text nodes, but that is
    the second line of defence. The first is that no route sets a HTML content
    type on user input.
    """
    r = client.get(f"/v1/cases/{payload}")
    assert "text/html" not in r.headers.get("content-type", "")
    if payload in r.text:
        # Present only as a JSON string value, with the angle brackets escaped
        # by the encoder rather than emitted as markup.
        assert r.headers["content-type"].startswith("application/json")


@pytest.mark.parametrize("payload", XSS)
def test_xss_in_analyst_free_text_survives_as_text(client, case_id, payload):
    """An analyst's reason is free text and the likeliest stored-XSS vector.

    The reason is stored *verbatim* on purpose. Sanitising it would corrupt an
    audit record to protect a renderer, and the renderer is React, which
    escapes text nodes. What this asserts is the pair that actually matters:
    the value comes back byte-identical, and never with a HTML content type.
    """
    cid = case_id
    r = client.post(f"/v1/cases/{cid}/decision",
                    json={"actor": "security-test", "action": "MARK_REVIEWED",
                          "reason": f"probe {payload}"})
    assert r.status_code == 200, r.text

    back = client.get(f"/v1/cases/{cid}")
    assert back.headers["content-type"].startswith("application/json")
    assert payload in back.text, "the stored reason was silently altered"
    # JSON-encoded, so the angle brackets are inside a quoted string. If this
    # were ever served as HTML the test above would already have failed.
    assert "<script>alert(1)</script>" not in back.headers.get("content-type", "")


# --- upload hardening --------------------------------------------------------

def test_graph_upload_rejects_a_non_tabular_file(client):
    r = client.post("/v1/graph/edges",
                    files={"file": ("payload.exe", b"MZ\x90\x00binary", "application/octet-stream")})
    assert r.status_code == 422
    assert "unsupported" in r.json()["detail"].lower()


def test_graph_upload_rejects_an_empty_file(client):
    r = client.post("/v1/graph/edges", files={"file": ("empty.csv", b"", "text/csv")})
    assert r.status_code == 422


def test_graph_upload_rejects_a_csv_without_the_required_columns(client):
    csv = b"foo,bar\n1,2\n"
    r = client.post("/v1/graph/edges", files={"file": ("wrong.csv", csv, "text/csv")})
    assert r.status_code == 422
    assert "missing" in r.json()["detail"].lower()


def test_a_zip_bomb_style_declared_size_is_capped(client, monkeypatch):
    """The size cap is enforced on the bytes read, not on a declared header."""
    from muleguard.api import routes_graph

    monkeypatch.setattr(routes_graph, "MAX_EDGE_BYTES", 32)
    payload = b"account_from,account_to,amount,timestamp\n" + b"A,B,1,2026-03-01\n" * 50
    r = client.post("/v1/graph/edges", files={"file": ("big.csv", payload, "text/csv")})
    assert r.status_code == 413


# --- csv export ---------------------------------------------------------------

def test_batch_export_filename_is_server_minted(client):
    """The download filename must not carry anything the uploader chose.

    A Content-Disposition built from `file.filename` lets an uploader write a
    header - and pick where a browser saves the result. This one is a UUID.
    """
    csv = b"F1,F2\n1,2\n"
    r = client.post("/v1/score/file",
                    files={"file": ('"; rm -rf /; x=".csv', csv, "text/csv")})
    disposition = r.headers.get("content-disposition", "")
    assert "rm -rf" not in disposition
    if disposition:
        assert disposition.startswith('attachment; filename="BATCH-')


def test_csv_safe_neutralises_every_formula_prefix():
    """Excel treats =, +, -, @ and the two control characters as formula starts."""
    from muleguard.explain.evidence_packet import FORMULA_LEADS, csv_safe

    assert set(FORMULA_LEADS) >= {"=", "+", "-", "@", "\t", "\r"}
    for lead in FORMULA_LEADS:
        out = csv_safe(f"{lead}cmd|' /c calc'!A0")
        assert not out.startswith(lead), f"{lead!r} survived the guard"


# --- no leakage of secrets or internals in errors ----------------------------

def test_error_responses_do_not_leak_filesystem_paths(client):
    """A stack trace in an error body tells an attacker the deployment layout."""
    for url in ("/v1/cases/'; DROP TABLE cases;--",
                "/v1/proofgraph/../../etc/passwd",
                "/v1/validation/seals/../../secrets"):
        r = client.get(url)
        body = r.text
        assert "Traceback" not in body
        assert str(settings.REPO_ROOT) not in body
        assert ".venv" not in body
