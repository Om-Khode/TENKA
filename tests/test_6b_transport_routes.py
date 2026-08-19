"""`GET/POST/DELETE /v1/transports` -- managing which doors exist.

Spec `.superpowers/sdd/2026-08-16-milestone6b-transports/spec.md` §4, Task 13.

Managing which doors exist is the same class of thing as managing which
devices exist (`routes/devices.py`), so all three routes sit behind
`require_admin(Capability.SYSTEM_CONTROL)` -- loopback-and-nothing-else. That
gate is exercised the same way `test_6b_raise_routes.py` exercises it: a
`GET /v1/session` beside every refusal, so a 403 cannot be a credential that
never authenticated in the first place.

`TransportManager` (Task 9) is not exercised here at all -- it is driven by a
fake, per this task's brief, since no test in this milestone may start a real
tunnel (`cloudflared` is not installed; `tailscale` is installed and must not
be mutated).
"""
from __future__ import annotations

from assistant.io.api.app import create_app
from assistant.io.api.listeners import port_for
from assistant.io.api.policy import POLICIES
from assistant.io.api.security import COOKIE_NAME, CSRF_HEADER
from assistant.io.api.transports import transport_registry
from assistant.io.api.transports.manager import TransportError
from assistant.io.api.vault import Capability, TokenVault
from tests.fakes.api_client import LOCAL_PORT, ApiTestClient
from tests.fakes.studio_runtime import build_fake_runtime

PORTS: dict[str, int] = {name: port_for(name, LOCAL_PORT)
                         for name in ("local", "tailnet", "funnel")}
POLICY_REGISTRY: dict[int, str] = {port: name for name, port in PORTS.items()}

# A real, registered transport name -- picked off the live registry rather
# than hardcoded, so this file never special-cases a provider by name either.
A_REAL_TRANSPORT = transport_registry.names()[0]


class _FakeSession:
    def __init__(self, url: str | None) -> None:
        self.url = url


class FakeTransportManager:
    """Stands in for Task 9's `TransportManager`.

    `start_error` / `stop_error` let a test script a `TransportError` for the
    next call, without spawning anything real -- see the module docstring.
    """

    def __init__(self, running: dict[str, str] | None = None, *,
                 start_error: str | None = None,
                 stop_error: str | None = None) -> None:
        self._running: dict[str, _FakeSession] = {
            name: _FakeSession(url) for name, url in (running or {}).items()
        }
        self._start_error = start_error
        self._stop_error = stop_error
        self.start_calls: list[str] = []
        self.stop_calls: list[str] = []

    def running(self) -> dict[str, _FakeSession]:
        return dict(self._running)

    async def start(self, name: str):
        self.start_calls.append(name)
        if self._start_error is not None:
            raise TransportError(self._start_error)
        session = _FakeSession(f"https://{name}.example.ts.net")
        self._running[name] = session
        return session

    async def stop(self, name: str) -> None:
        self.stop_calls.append(name)
        if self._stop_error is not None:
            raise TransportError(self._stop_error)
        self._running.pop(name, None)


def build(vault: TokenVault, *, manager: FakeTransportManager | None = None):
    app = create_app(build_fake_runtime(), vault,
                     origins=["http://localhost:3000"],
                     listener_policies=dict(POLICY_REGISTRY))
    if manager is not None:
        app.state.transports = manager
    return app


def client_on(app, listener: str, token: str | None = None) -> ApiTestClient:
    client = ApiTestClient(app, base_url=f"http://127.0.0.1:{PORTS[listener]}")
    if token is not None:
        client.cookies.set(COOKIE_NAME, token)
    return client


# ─── the gate ────────────────────────────────────────────────────────────

def test_listing_transports_is_refused_off_the_local_listener(tmp_path):
    """`require_admin` is `policy.admin`, which is loopback alone.

    The `GET /v1/session` beside each refusal is what stops this passing
    vacuously: the credential is *good* on that listener -- 200 -- and the
    listing is still refused, so the 403 is the admin gate rather than a
    credential that never authenticated at all.
    """
    vault = TokenVault(tmp_path)
    token = vault.issue("phone", frozenset(Capability))
    app = build(vault, manager=FakeTransportManager())

    for listener in ("tailnet", "funnel"):
        client = client_on(app, listener, token)
        assert client.get("/v1/session").status_code == 200, listener
        assert client.get("/v1/transports").status_code == 403, listener


def test_starting_a_transport_is_refused_off_the_local_listener(tmp_path):
    vault = TokenVault(tmp_path)
    token = vault.issue("phone", frozenset(Capability))
    app = build(vault, manager=FakeTransportManager())

    for listener in ("tailnet", "funnel"):
        client = client_on(app, listener, token)
        assert client.get("/v1/session").status_code == 200, listener
        response = client.post(f"/v1/transports/{A_REAL_TRANSPORT}",
                               headers={CSRF_HEADER: "1"})
        assert response.status_code == 403, listener


def test_a_watching_device_on_loopback_cannot_manage_transports(tmp_path):
    """The other half of `require_admin`: SYSTEM_CONTROL, not just loopback."""
    vault = TokenVault(tmp_path)
    token = vault.issue("wall display", frozenset({Capability.OBSERVE}))
    app = build(vault, manager=FakeTransportManager())
    client = client_on(app, "local", token)

    assert client.get("/v1/transports").status_code == 403
    assert client.post(f"/v1/transports/{A_REAL_TRANSPORT}",
                       headers={CSRF_HEADER: "1"}).status_code == 403
    assert client.delete(f"/v1/transports/{A_REAL_TRANSPORT}",
                         headers={CSRF_HEADER: "1"}).status_code == 403


# ─── the listing ─────────────────────────────────────────────────────────

def test_the_listing_reports_each_transports_ceiling_and_raisable_set(tmp_path):
    """Ceiling and raisable come from `POLICIES`, never hardcoded, so Studio
    can explain *why* a control is unavailable without a second copy of the
    table."""
    vault = TokenVault(tmp_path)
    token = vault.issue("laptop", frozenset(Capability))
    app = build(vault, manager=FakeTransportManager(
        running={A_REAL_TRANSPORT: "https://foo.example.ts.net"}))
    client = client_on(app, "local", token)

    response = client.get("/v1/transports")
    assert response.status_code == 200
    rows = {row["name"]: row for row in response.json()["data"]["transports"]}

    assert set(rows) == set(transport_registry.names())
    for name, row in rows.items():
        policy = POLICIES[name]
        assert set(row["ceiling"]) == {c.value for c in policy.ceiling}
        assert set(row["raisable"]) == {c.value for c in policy.raisable}

    running_row = rows[A_REAL_TRANSPORT]
    assert running_row["running"] is True
    assert running_row["url"] == "https://foo.example.ts.net"

    other_names = [n for n in transport_registry.names() if n != A_REAL_TRANSPORT]
    for name in other_names:
        assert rows[name]["running"] is False
        assert rows[name]["url"] is None


def test_the_listing_defaults_to_nothing_running_with_no_manager_wired(tmp_path):
    """A sibling task wires `app.state.transports`. An app built before that
    lands must fail closed -- nothing running, not an assumption."""
    vault = TokenVault(tmp_path)
    token = vault.issue("laptop", frozenset(Capability))
    app = build(vault, manager=None)
    client = client_on(app, "local", token)

    response = client.get("/v1/transports")
    assert response.status_code == 200
    for row in response.json()["data"]["transports"]:
        assert row["running"] is False
        assert row["url"] is None


# ─── starting ────────────────────────────────────────────────────────────

def test_starting_an_unknown_transport_is_a_404(tmp_path):
    """Refused before the manager is ever asked.

    `manager.start_calls` is the load-bearing assertion, not the status code
    alone: an unregistered name is not in `POLICIES` either, so
    `TransportPayload`'s `POLICIES[name]` lookup would itself raise `KeyError`
    once the manager (wrongly) reported success -- which `errors.py`'s
    app-wide handler maps to a 404 all on its own. That coincidence would let
    this test pass even if the route's own membership check were deleted; the
    call-count assertion is what actually catches that regression (checked as
    part of this task's vacuity pass: deleting the route's `if name not in
    transport_registry.names()` line turns `start_calls` non-empty while the
    status code stays 404).
    """
    vault = TokenVault(tmp_path)
    token = vault.issue("laptop", frozenset(Capability))
    manager = FakeTransportManager()
    app = build(vault, manager=manager)
    client = client_on(app, "local", token)

    submitted = "ngrok-please"
    response = client.post(f"/v1/transports/{submitted}", headers={CSRF_HEADER: "1"})
    assert response.status_code == 404
    assert submitted not in response.text
    assert manager.start_calls == []


def test_starting_local_is_refused(tmp_path):
    """The KI-17 sibling: `local` is not a transport -- it is the loopback
    listener itself, with no tunnel and no adapter -- and must not be
    startable by name through this route.

    Pinned as 404, specifically, and not merely "not 200": a route that
    checked `name not in POLICIES` instead of `name not in
    transport_registry.names()` would let `"local"` (a real `POLICIES` key)
    straight through to `manager.start("local")` -- and `FakeTransportManager`
    does not itself know about KI-17, so it would happily "start" it and
    answer 200. Checked (by flipping the route to `POLICIES` and confirming
    this goes red with exactly that 200) as part of this task's vacuity pass.
    """
    assert "local" not in transport_registry.names()
    vault = TokenVault(tmp_path)
    token = vault.issue("laptop", frozenset(Capability))
    app = build(vault, manager=FakeTransportManager())
    client = client_on(app, "local", token)

    response = client.post("/v1/transports/local", headers={CSRF_HEADER: "1"})
    assert response.status_code == 404


def test_a_failed_preflight_surfaces_as_a_409_naming_nothing_secret(tmp_path):
    vault = TokenVault(tmp_path)
    token = vault.issue("laptop", frozenset(Capability))
    refusal = f"transport '{A_REAL_TRANSPORT}' refused to start: a conflicting mapping exists"
    app = build(vault, manager=FakeTransportManager(start_error=refusal))
    client = client_on(app, "local", token)

    response = client.post(f"/v1/transports/{A_REAL_TRANSPORT}",
                           headers={CSRF_HEADER: "1"})
    assert response.status_code == 409
    assert response.json()["detail"] == refusal
    assert "://" not in refusal  # the fixture's own refusal names no hostname


def test_a_missing_transport_manager_is_refused_rather_than_assumed_startable(tmp_path):
    vault = TokenVault(tmp_path)
    token = vault.issue("laptop", frozenset(Capability))
    app = build(vault, manager=None)
    client = client_on(app, "local", token)

    response = client.post(f"/v1/transports/{A_REAL_TRANSPORT}",
                           headers={CSRF_HEADER: "1"})
    assert response.status_code == 503


def test_starting_a_transport_reports_it_running_with_its_url(tmp_path):
    vault = TokenVault(tmp_path)
    token = vault.issue("laptop", frozenset(Capability))
    manager = FakeTransportManager()
    app = build(vault, manager=manager)
    client = client_on(app, "local", token)

    response = client.post(f"/v1/transports/{A_REAL_TRANSPORT}",
                           headers={CSRF_HEADER: "1"})
    assert response.status_code == 200
    row = response.json()["data"]
    assert row["name"] == A_REAL_TRANSPORT
    assert row["running"] is True
    assert row["url"] == f"https://{A_REAL_TRANSPORT}.example.ts.net"
    assert manager.start_calls == [A_REAL_TRANSPORT]


# ─── stopping ────────────────────────────────────────────────────────────

def test_stopping_a_transport_that_is_not_running_is_a_404(tmp_path):
    vault = TokenVault(tmp_path)
    token = vault.issue("laptop", frozenset(Capability))
    app = build(vault, manager=FakeTransportManager())
    client = client_on(app, "local", token)

    response = client.delete(f"/v1/transports/{A_REAL_TRANSPORT}",
                             headers={CSRF_HEADER: "1"})
    assert response.status_code == 404


def test_stopping_an_unknown_transport_is_a_404(tmp_path):
    """Same masking risk as the POST test above, and the same fix:
    `manager.stop_calls` is what actually proves the "not currently running"
    check ran, since an unregistered name's `POLICIES[name]` lookup in the
    response body would otherwise coincidentally 404 on its own even with the
    check deleted (checked as part of this task's vacuity pass)."""
    vault = TokenVault(tmp_path)
    token = vault.issue("laptop", frozenset(Capability))
    manager = FakeTransportManager()
    app = build(vault, manager=manager)
    client = client_on(app, "local", token)

    submitted = "ngrok-please"
    response = client.delete(f"/v1/transports/{submitted}", headers={CSRF_HEADER: "1"})
    assert response.status_code == 404
    assert submitted not in response.text
    assert manager.stop_calls == []


def test_stopping_a_transport_with_no_manager_wired_is_a_404(tmp_path):
    """Absent means nothing is running, so nothing here is running either."""
    vault = TokenVault(tmp_path)
    token = vault.issue("laptop", frozenset(Capability))
    app = build(vault, manager=None)
    client = client_on(app, "local", token)

    response = client.delete(f"/v1/transports/{A_REAL_TRANSPORT}",
                             headers={CSRF_HEADER: "1"})
    assert response.status_code == 404


def test_stopping_a_running_transport_succeeds_and_reports_it_stopped(tmp_path):
    vault = TokenVault(tmp_path)
    token = vault.issue("laptop", frozenset(Capability))
    manager = FakeTransportManager(running={A_REAL_TRANSPORT: "https://foo.example.ts.net"})
    app = build(vault, manager=manager)
    client = client_on(app, "local", token)

    response = client.delete(f"/v1/transports/{A_REAL_TRANSPORT}",
                             headers={CSRF_HEADER: "1"})
    assert response.status_code == 200
    row = response.json()["data"]
    assert row["running"] is False
    assert row["url"] is None
    assert manager.stop_calls == [A_REAL_TRANSPORT]

    # And a second stop, since it is no longer running, is now a 404 -- the
    # same "gone now" distinction `revoke_device` draws.
    again = client.delete(f"/v1/transports/{A_REAL_TRANSPORT}",
                          headers={CSRF_HEADER: "1"})
    assert again.status_code == 404


def test_an_unverified_stop_surfaces_as_a_409_naming_nothing_secret(tmp_path):
    """`stop` surfaces an unverified provider-side stop as a `TransportError`
    rather than raising past this route -- and its detail must never carry a
    hostname, a token or a path (manager.py's own contract)."""
    vault = TokenVault(tmp_path)
    token = vault.issue("laptop", frozenset(Capability))
    refusal = (f"transport '{A_REAL_TRANSPORT}': a mapping is STILL keyed "
               f"under public port 443 after its stop command ran")
    manager = FakeTransportManager(
        running={A_REAL_TRANSPORT: "https://foo.example.ts.net"},
        stop_error=refusal)
    app = build(vault, manager=manager)
    client = client_on(app, "local", token)

    response = client.delete(f"/v1/transports/{A_REAL_TRANSPORT}",
                             headers={CSRF_HEADER: "1"})
    assert response.status_code == 409
    assert response.json()["detail"] == refusal
