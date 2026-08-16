"""Milestone 6a.5, stream D -- auth, session and the audit record.

Four findings, all in `assistant/io/api/security.py` and
`assistant/io/api/app.py`:

- G3 (lens 3, F2) -- a duplicate `tenka_device` cookie is last-wins, so a
  sibling host under a shared parent domain can fix the operator's session.
- G4 (lens 3, F3) -- a blank `Origin` falls into the *absent* branch, which on
  `local` means allow and on the event socket means accept.
- G9 (lens 1, F5) -- the audit record's `path` is the caller's own string, with
  no length bound and no character class, in a ring an anonymous flood can
  flush.
- G11 (task 17's "logged, not fixed" list) -- the inbound socket re-verify sits
  behind the `type != "abort"` filter.

Each test keeps the lens's own reasoning in its docstring.
"""
from __future__ import annotations

import pytest
from starlette.websockets import WebSocketDisconnect

from assistant.io.api.security import COOKIE_NAME
from assistant.io.api.vault import Capability, TokenVault
from tests.fakes.api_client import LOCAL_PORT, build_api_client
from tests.fakes.studio_runtime import build_fake_runtime


def _client(vault: TokenVault, *, policy: str = "local"):
    return build_api_client(build_fake_runtime(), vault,
                            policies={LOCAL_PORT: policy})


# ─── G3: a second `tenka_device` cookie must not win ─────────────────────
def test_a_second_cookie_of_the_same_name_does_not_win(tmp_path):
    """Session fixation: the victim's browser is forced onto a session the
    attacker also holds.

    `connection.cookies` is Starlette's `cookie_parser`, which is last-wins on
    a duplicate name, and RFC 6265 s5.4 serialises equal-path cookies by
    creation time ascending -- so the *most recently set* duplicate is the one
    this daemon adopts. Both values below are genuine, verifiable tokens, which
    is what makes this fixation rather than a denial of service.
    """
    vault = TokenVault(tmp_path)
    victim = vault.issue("victim-laptop", frozenset(Capability))
    attacker = vault.issue("attacker-phone", frozenset(Capability))
    client = _client(vault)
    r = client.get("/v1/session",
                   headers={"Cookie": f"{COOKIE_NAME}={victim}; "
                                      f"{COOKIE_NAME}={attacker}"})
    assert r.status_code == 401, (
        f"two `{COOKIE_NAME}` cookies were answered {r.status_code} as "
        f"{r.json().get('data', {}).get('label')!r} -- the second one won")


def test_a_single_cookie_still_authenticates(tmp_path):
    """The fix must not collapse into refusing everyone."""
    vault = TokenVault(tmp_path)
    token = vault.issue("laptop", frozenset(Capability))
    client = _client(vault)
    r = client.get("/v1/session", headers={"Cookie": f"{COOKIE_NAME}={token}"})
    assert r.status_code == 200
    assert r.json()["data"]["label"] == "laptop"


def test_an_unrelated_second_cookie_is_harmless(tmp_path):
    """Only duplicates of our own name are rejected -- a page setting some
    other cookie on this host must not lock the operator out."""
    vault = TokenVault(tmp_path)
    token = vault.issue("laptop", frozenset(Capability))
    client = _client(vault)
    r = client.get("/v1/session",
                   headers={"Cookie": f"{COOKIE_NAME}={token}; "
                                      f"other=whatever; analytics=1"})
    assert r.status_code == 200
    assert r.json()["data"]["label"] == "laptop"


def test_a_duplicated_junk_cookie_does_not_authenticate_either(tmp_path):
    """The degenerate form of the same finding: a junk shadow value made a
    paired device look unknown, and `credential_from` returns the cookie
    without ever falling back to `Authorization`. Refusing the *pair* is the
    right answer either way -- what must not happen is one of the two being
    silently chosen."""
    vault = TokenVault(tmp_path)
    token = vault.issue("laptop", frozenset(Capability))
    client = _client(vault)
    r = client.get("/v1/session",
                   headers={"Cookie": f"{COOKIE_NAME}={token}; "
                                      f"{COOKIE_NAME}=junk"})
    assert r.status_code == 401


def test_the_socket_also_refuses_a_duplicated_cookie(tmp_path):
    """`cookie_credential` is the one spelling of "the credential is this
    cookie" that the HTTP gate and the socket gate share, so the socket
    inherits the fix rather than needing its own."""
    vault = TokenVault(tmp_path)
    victim = vault.issue("victim", frozenset(Capability))
    attacker = vault.issue("attacker", frozenset(Capability))
    client = _client(vault)
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(
                "/v1/events",
                headers={"Cookie": f"{COOKIE_NAME}={victim}; "
                                   f"{COOKIE_NAME}={attacker}"}) as ws:
            ws.receive_json()


# ─── G4: a blank `Origin` is malformed input, not an absent header ───────
def test_a_blank_origin_is_refused_not_ignored(tmp_path):
    """`if origin and origin.strip():` sends `Origin: "   "` down the
    *no-Origin* branch, which on `local` means allow. That is a fail-open
    response to malformed input on the CSWSH gate, on the one listener that
    also carries admin."""
    vault = TokenVault(tmp_path)
    token = vault.issue("laptop", frozenset(Capability))
    client = _client(vault)
    client.cookies.set(COOKIE_NAME, token)
    r = client.get("/v1/status", headers={"Origin": "   "})
    assert r.status_code == 403, (
        f"a whitespace-only Origin was answered {r.status_code} -- it took the "
        f"absent-Origin branch")


def test_an_empty_origin_is_refused_too(tmp_path):
    """`Origin: ""` is the same malformed shape with nothing in it at all."""
    vault = TokenVault(tmp_path)
    token = vault.issue("laptop", frozenset(Capability))
    client = _client(vault)
    client.cookies.set(COOKIE_NAME, token)
    assert client.get("/v1/status", headers={"Origin": ""}).status_code == 403


def test_an_absent_origin_is_still_allowed_on_local(tmp_path):
    """Non-browser clients send no Origin at all, and the local listener
    admits them by design. Refusing malformed input must not refuse absent
    input -- that would break every script on loopback."""
    vault = TokenVault(tmp_path)
    token = vault.issue("laptop", frozenset(Capability))
    client = _client(vault)
    client.cookies.set(COOKIE_NAME, token)
    assert client.get("/v1/status").status_code == 200


def test_a_known_origin_is_still_allowed(tmp_path):
    """The other half of the control: the gate must still say yes to this
    daemon's own front door."""
    vault = TokenVault(tmp_path)
    token = vault.issue("laptop", frozenset(Capability))
    client = _client(vault)
    client.cookies.set(COOKIE_NAME, token)
    r = client.get("/v1/status",
                   headers={"Origin": f"http://127.0.0.1:{LOCAL_PORT}"})
    assert r.status_code == 200


def test_a_blank_origin_is_refused_on_the_socket(tmp_path):
    """1008 before accept(), the same as a wrong origin. The socket's own copy
    of the guard spells it identically, so it fails open identically."""
    vault = TokenVault(tmp_path)
    token = vault.issue("laptop", frozenset(Capability))
    client = _client(vault)
    client.cookies.set(COOKIE_NAME, token)
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/v1/events",
                                      headers={"Origin": "   "}) as ws:
            ws.receive_json()


def test_an_absent_origin_still_opens_the_socket_on_local(tmp_path):
    """Control: `TestClient` and every non-browser client send no Origin."""
    vault = TokenVault(tmp_path)
    token = vault.issue("laptop", frozenset(Capability))
    client = _client(vault)
    client.cookies.set(COOKIE_NAME, token)
    with client.websocket_connect("/v1/events") as ws:
        assert ws.receive_json()["type"] == "status"


def test_pairing_refuses_a_blank_origin(tmp_path):
    """`POST /v1/pair` is the one unauthenticated write in the API, and it
    reaches the same guard through `refuse_unknown_origin`."""
    vault = TokenVault(tmp_path)
    client = _client(vault)
    r = client.post("/v1/pair", json={"code": "000000"},
                    headers={"Origin": "   "})
    assert r.status_code == 403


# ─── G9: the audit record's text is not the caller's to choose ───────────
def _audit_scope(raw_path: bytes, path: str) -> dict:
    """The ASGI scope uvicorn really builds, not the one `TestClient` builds.

    `TestClient`'s httpx layer strips control characters out of a URL before
    it ever reaches the app, so the CRLF half of this finding cannot be
    expressed through it. uvicorn instead hands the app `unquote(raw_path)`
    verbatim (`h11_impl.py:202`, `httptools_impl.py:260`), which is what this
    reproduces.
    """
    return {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": path,
        "raw_path": raw_path,
        "query_string": b"",
        "root_path": "",
        "headers": [(b"host", f"127.0.0.1:{LOCAL_PORT}".encode())],
        "client": ("127.0.0.1", 51234),
        "server": ("127.0.0.1", LOCAL_PORT),
        "state": {},
    }


async def _drive(app, scope: dict) -> list[dict]:
    sent: list[dict] = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    await app(scope, receive, send)
    return sent


def test_an_anonymous_caller_cannot_choose_the_length_of_an_audit_record(tmp_path):
    """A 3,004-character attacker-chosen `path` was stored and served verbatim
    by `GET /v1/audit`. `audit_and_tag` records every request including the
    ones that never authenticate, and the field had no length bound."""
    from assistant.io.api.security import _AUDIT_PATH_MAX

    vault = TokenVault(tmp_path)
    client = _client(vault)
    client.get("/v1/" + "a" * 3_000)
    stored = client.app.state.auth.audit.entries()
    assert stored, "sanity: the anonymous request was audited"
    longest = max(len(e.path) for e in stored)
    assert longest <= _AUDIT_PATH_MAX, (
        f"an anonymous caller wrote a {longest}-character audit record")


def test_an_ordinary_path_is_still_recorded_in_full(tmp_path):
    """The cap must not collapse into recording nothing useful -- the record
    is what an operator reads after an incident."""
    vault = TokenVault(tmp_path)
    token = vault.issue("laptop", frozenset(Capability))
    client = _client(vault)
    client.cookies.set(COOKIE_NAME, token)
    client.get("/v1/memory/knowledge")
    paths = [e.path for e in client.app.state.auth.audit.entries()]
    assert "/v1/memory/knowledge" in paths, paths


def test_an_anonymous_flood_cannot_evict_the_authenticated_record(tmp_path):
    """`AuditLog` was a single 2,000-entry `deque(maxlen=...)`, and a
    rate-limited request is still audited -- so ~2,000 anonymous requests
    flush every real entry and the limiter does not slow the flush below one
    entry per request. Evidence deletion, from an unauthenticated caller.

    Driven against a deliberately tiny ring so the shape is asserted in
    milliseconds rather than by sending two thousand requests.
    """
    from assistant.io.api.security import AuditLog

    vault = TokenVault(tmp_path)
    token = vault.issue("laptop", frozenset(Capability))
    client = _client(vault)
    client.app.state.auth.audit = AuditLog(capacity=4, anonymous_capacity=4)

    client.cookies.set(COOKIE_NAME, token)
    client.get("/v1/status")                     # the real entry
    client.cookies.clear()
    for _ in range(30):                          # the flood
        client.get("/v1/status")

    stored = client.app.state.auth.audit.entries()
    assert any(e.device_id != "-" for e in stored), (
        "30 anonymous requests flushed the one authenticated entry out of the "
        f"record: {[(e.device_id, e.outcome) for e in stored]}")


def test_the_anonymous_ring_is_itself_still_bounded():
    """Separating the rings must not turn the anonymous one into unbounded
    memory a caller can grow for free."""
    from assistant.io.api.security import AuditEntry, AuditLog

    log = AuditLog(capacity=5, anonymous_capacity=5)
    for i in range(50):
        log.record(AuditEntry(at=f"t{i:03d}", device_id="-", method="GET",
                              path="/v1/status", outcome="401"))
    assert len(log.entries()) == 5


def test_the_record_is_still_returned_oldest_first():
    """`GET /v1/audit` reverses `entries()` to show newest first, so the two
    rings have to be merged back into one chronological sequence -- not
    concatenated, which would interleave the two histories wrongly."""
    from assistant.io.api.security import AuditEntry, AuditLog

    log = AuditLog(capacity=10, anonymous_capacity=10)
    order = ["-", "dev-1", "-", "-", "dev-1", "dev-2"]
    for i, device_id in enumerate(order):
        log.record(AuditEntry(at=f"t{i:03d}", device_id=device_id,
                              method="GET", path="/v1/status", outcome="200"))
    assert [e.device_id for e in log.entries()] == order


def test_an_anonymous_caller_cannot_write_newlines_into_the_audit_trail(tmp_path):
    """Control -- this already holds, via `request.url`, which rebuilds the URL
    and re-parses it with `urllib.parse.urlsplit`, which since CPython's WHATWG
    fix strips ASCII tab, CR and LF outright.

    Pinned as a *passing* test so that a future change from `request.url.path`
    to `scope["path"]` or `raw_path` -- both of which carry the bytes through
    untouched -- flips this to failing instead of quietly becoming real CRLF
    injection into the file an operator greps after an incident.
    """
    import asyncio

    vault = TokenVault(tmp_path)
    client = _client(vault)
    scope = _audit_scope(
        raw_path=b"/v1/%0d%0aFAKE-AUDIT-LINE",
        path="/v1/\r\nFAKE-AUDIT-LINE",
    )
    asyncio.run(_drive(client.app, scope))
    stored = client.app.state.auth.audit.entries()
    assert stored, "sanity: the ASGI-level request was audited"
    # Sanity that the probe has teeth: the attacker's marker really did travel
    # all the way into the record, so the absence of the CR and LF below is
    # `urlsplit` stripping them and not the request having been refused
    # somewhere upstream of the audit middleware.
    assert any("FAKE-AUDIT-LINE" in e.path for e in stored), [e.path for e in stored]
    for entry in stored:
        assert "\n" not in entry.path and "\r" not in entry.path, repr(entry.path)


def test_a_control_character_never_reaches_the_stored_path():
    """The character class, asserted on the store, because the urlsplit control
    above only covers tab/CR/LF -- a NUL or an ANSI escape reaching a terminal
    that renders the record is the same class of forgery."""
    from assistant.io.api.security import AuditEntry, AuditLog

    log = AuditLog(capacity=5, anonymous_capacity=5)
    log.record(AuditEntry(at="t0", device_id="-", method="GET",
                          path="/v1/\x1b[2Kfake\x00", outcome="404"))
    stored = log.entries()[0].path
    assert all(0x20 <= ord(c) <= 0x7E for c in stored), repr(stored)
