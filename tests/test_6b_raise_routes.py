"""Minting a ceiling raise, spending it, and every way it is refused.

Spec §3.4/§3.6. A raise is the one thing in this daemon that makes a device
able to do *more* than it could a moment ago, so all but one of the tests below
assert a refusal. The exception is
`test_a_raised_device_reaches_a_capability_the_ceiling_alone_refuses`, and it
is the point of the whole mechanism: seventeen refusals would all pass just as
happily against a raise route that granted nothing at all, which is exactly the
vacuous-security-test failure this milestone has already caught once.

Both routes sit behind `require_admin(SYSTEM_CONTROL)`, which is
loopback-and-nothing-else -- so a raise is minted at the keyboard, always, and
there is no remote surface here to attack.

The fixture shape is one app with all four listeners registered on their real
offsets (`listeners.port_for`), and one `TestClient` per port. That is what
lets a single test mint on the local listener and then spend the raise on the
tailnet one against the *same* `RaiseStore` -- two apps would share a vault but
not a store, and the end-to-end property would go untested.
"""
from __future__ import annotations

import logging

from dataclasses import replace

from assistant.io.api import raises as raises_module
from assistant.io.api.app import create_app
from assistant.io.api.listeners import port_for
from assistant.io.api.policy import POLICIES, ListenerPolicy
from assistant.io.api.raises import MAX_RAISE_SECONDS, RaiseStore
from assistant.io.api.security import COOKIE_NAME, CSRF_HEADER
from assistant.io.api.vault import Capability, TokenVault
from tests.fakes.api_client import LOCAL_PORT, ApiTestClient
from tests.fakes.studio_runtime import build_fake_runtime

# The three listeners at their real offsets from the loopback port, so a test
# that says "tailnet" reaches the port a tailnet listener would actually hold.
PORTS: dict[str, int] = {name: port_for(name, LOCAL_PORT)
                         for name in ("local", "tailnet", "funnel")}
POLICY_REGISTRY: dict[int, str] = {port: name for name, port in PORTS.items()}


class FakeTransports:
    """Stands in for Task 9's `TransportManager`, which this task does not own.

    Only `running()` is exercised: the raise route asks nothing else of it.
    """

    def __init__(self, running: frozenset[str] = frozenset()) -> None:
        self._running = set(running)

    def running(self) -> set[str]:
        return set(self._running)


class FakeClock:
    """A stand-in for the `time` module inside `raises.py` alone.

    Deliberately not `monkeypatch.setattr(time, "monotonic", ...)`: that patches
    the clock the rate limiter, the pair store and uvicorn's own bookkeeping all
    read. Replacing the module *reference* held by `raises.py` moves only the
    clock the raise store expires on, which is the only one any test here needs
    to move.
    """

    def __init__(self, start: float = 1_000.0) -> None:
        self.now = start

    def monotonic(self) -> float:
        return self.now


def build(vault: TokenVault, *, running: frozenset[str] = frozenset({"tailnet"}),
          store: RaiseStore | None = None):
    app = create_app(build_fake_runtime(), vault,
                     origins=["http://localhost:3000"],
                     listener_policies=dict(POLICY_REGISTRY))
    app.state.raises = store if store is not None else RaiseStore()
    app.state.transports = FakeTransports(running)
    return app


def client_on(app, listener: str, token: str | None = None) -> ApiTestClient:
    client = ApiTestClient(app, base_url=f"http://127.0.0.1:{PORTS[listener]}")
    if token is not None:
        client.cookies.set(COOKIE_NAME, token)
    return client


def body(transport: str = "tailnet", capabilities: list[str] | None = None,
         minutes: int = 30, reason: str = "fixing the build") -> dict:
    return {
        "transport": transport,
        "capabilities": ["execute"] if capabilities is None else capabilities,
        "minutes": minutes,
        "reason": reason,
    }


def post_raise(client: ApiTestClient, device_id: str, **kwargs):
    return client.post(f"/v1/devices/{device_id}/raise", json=body(**kwargs),
                       headers={CSRF_HEADER: "1"})


# ─── the gate ────────────────────────────────────────────────────────────
def test_a_raise_is_refused_on_every_listener_but_local(tmp_path):
    """`require_admin` is `policy.admin`, which is loopback alone.

    The `GET /v1/session` beside each refusal is what stops this passing
    vacuously: the credential is *good* on that listener -- 200 -- and the
    raise is still refused, so the 403 is the admin gate rather than a
    credential that never authenticated in the first place.
    """
    vault = TokenVault(tmp_path)
    token = vault.issue("phone", frozenset(Capability))
    device_id = vault.devices()[0].device_id
    app = build(vault, running=frozenset({"tailnet", "funnel"}))

    for listener in ("tailnet", "funnel"):
        client = client_on(app, listener, token)
        assert client.get("/v1/session").status_code == 200, listener
        assert post_raise(client, device_id).status_code == 403, listener
        assert client.delete(f"/v1/devices/{device_id}/raise",
                             headers={CSRF_HEADER: "1"}).status_code == 403, listener


def test_ki18_the_admin_gate_holds_even_when_the_ceiling_carries_system_control(
        tmp_path, monkeypatch):
    """KI-18. Reviewers B and C both "reproduced" this by setting `admin=True`
    on a tunnel policy and observing admin access -- that is the definition of
    the flag, not a defect. In every real `POLICIES` entry, the one ceiling
    holding SYSTEM_CONTROL (`local`'s) is also the one policy with
    `admin=True`, so `require(SYSTEM_CONTROL)` always refuses first on
    `tailnet` and `funnel` -- nothing there would fail if `admin` were
    flipped to `True` on either of them, because the capability check never
    lets a request reach `require_admin`'s own `if not policy.admin` line.

    This builds the one shape that isolates it: a fixture policy with
    `admin=False` but SYSTEM_CONTROL in its ceiling anyway, so
    `require(SYSTEM_CONTROL)` passes clean and the admin check is the only
    thing left standing between this device and `GET /v1/devices`.
    """
    probe = ListenerPolicy(
        name="probe", admin=False, allow_bearer=False, secure_cookie=False,
        ceiling=frozenset({Capability.OBSERVE, Capability.SYSTEM_CONTROL}),
        raisable=frozenset(), pairable=True,
    )
    monkeypatch.setitem(POLICIES, "probe", probe)
    port = max(PORTS.values()) + 1000  # not one of the three real listeners
    registry = dict(POLICY_REGISTRY)
    registry[port] = "probe"

    vault = TokenVault(tmp_path)
    token = vault.issue("probe-device", frozenset(Capability))
    app = create_app(build_fake_runtime(), vault,
                     origins=["http://localhost:3000"],
                     listener_policies=registry)
    app.state.raises = RaiseStore()
    app.state.transports = FakeTransports()
    client = ApiTestClient(app, base_url=f"http://127.0.0.1:{port}")
    client.cookies.set(COOKIE_NAME, token)

    assert client.get("/v1/devices").status_code == 403, (
        "admin=False must still refuse GET /v1/devices even though this "
        "policy's ceiling carries SYSTEM_CONTROL")

    # Vacuity guard: the reviewers' own experiment, run last so it cannot
    # contaminate the assertion above. Flipping only `admin` on the identical
    # policy must now let the same request through -- proving the 403 above
    # really was `require_admin`'s own gate, not an unrelated refusal (a bad
    # cookie, an Origin check, a stale device) that happened to also answer
    # 403.
    monkeypatch.setitem(POLICIES, "probe", replace(probe, admin=True))
    assert client.get("/v1/devices").status_code == 200, (
        "the vacuity guard failed: admin=True did not restore access, so the "
        "403 above was not proven to be the admin gate")


def test_a_watching_device_on_loopback_cannot_mint_a_raise(tmp_path):
    """The other half of `require_admin`: SYSTEM_CONTROL, not just loopback.

    A device paired only to watch, sitting at the keyboard, must not be able to
    lift its own ceiling -- that is the escalation arrived at from inside the
    house, and the transport check says nothing about it.
    """
    vault = TokenVault(tmp_path)
    token = vault.issue("wall display", frozenset({Capability.OBSERVE}))
    device_id = vault.devices()[0].device_id
    client = client_on(build(vault), "local", token)
    assert post_raise(client, device_id).status_code == 403


# ─── the refusal order ───────────────────────────────────────────────────
def test_an_unknown_device_is_a_404_like_revoke(tmp_path):
    """Same argument `revoke_device` already makes: "nothing happened" has to
    be distinguishable from "it is there and this is what it now holds"."""
    vault = TokenVault(tmp_path)
    token = vault.issue("laptop", frozenset(Capability))
    client = client_on(build(vault), "local", token)
    device_id = vault.devices()[0].device_id

    assert post_raise(client, "nope").status_code == 404
    assert client.delete("/v1/devices/nope/raise",
                         headers={CSRF_HEADER: "1"}).status_code == 404

    # The pair below is what stops this passing against a route that does not
    # exist at all -- which is exactly what it did before the route was
    # written, since an unrouted path is also a 404.
    assert post_raise(client, device_id, capabilities=["system_control"]).status_code == 200
    assert client.delete(f"/v1/devices/{device_id}/raise",
                         headers={CSRF_HEADER: "1"}).status_code == 200


def test_an_unknown_capability_name_is_422_and_is_never_echoed(tmp_path):
    """`mint_pair_code`'s shape, and its reasoning: a hand-built 422 must keep
    the promise `app.py`'s app-wide validation handler keeps for every
    Pydantic one -- never print the submitted string back."""
    vault = TokenVault(tmp_path)
    token = vault.issue("laptop", frozenset(Capability))
    device_id = vault.devices()[0].device_id
    client = client_on(build(vault), "local", token)

    submitted = "sudo-everything-please"
    response = post_raise(client, device_id, capabilities=[submitted])
    assert response.status_code == 422
    assert submitted not in response.text


def test_an_unknown_transport_name_is_422_and_is_never_echoed(tmp_path):
    """A transport nobody declared is a validation problem, not a 409.

    409 says "not running just now, try again", which would be a lie about a
    name that can never run -- and the operator's actual mistake is a typo, not
    a stopped tunnel. Same silence about the submitted string as the unknown
    capability above: `POLICIES`' keys are not secret, but a route that echoes
    one caller-supplied field and not another is how the next one starts
    echoing too.
    """
    vault = TokenVault(tmp_path)
    token = vault.issue("laptop", frozenset(Capability))
    device_id = vault.devices()[0].device_id
    client = client_on(build(vault), "local", token)

    submitted = "ngrok-please"
    response = post_raise(client, device_id, transport=submitted)
    assert response.status_code == 422
    assert submitted not in response.text


def test_an_unknown_device_is_refused_before_an_unknown_capability_is_parsed(tmp_path):
    """The refusal order, pinned as an order rather than as two separate facts.

    A request carrying *both* an unknown device and an unknown capability must
    answer 404, not 422. Reversed, the pair becomes an oracle: a caller could
    submit a deliberately invalid capability and read the status to learn
    whether a device id exists -- 422 for a real device, 404 for a made-up one.
    Loopback-admin-only today, so this is defence in depth rather than a live
    hole; unpinned, it is one refactor away from being neither.
    """
    vault = TokenVault(tmp_path)
    token = vault.issue("laptop", frozenset(Capability))
    device_id = vault.devices()[0].device_id
    client = client_on(build(vault), "local", token)

    both_wrong = post_raise(client, "nope", capabilities=["not-a-capability"],
                            transport="not-a-transport")
    assert both_wrong.status_code == 404

    # The same body against a device that *does* exist gets the 422 -- which is
    # what makes the 404 above an ordering fact rather than a coincidence.
    assert post_raise(client, device_id,
                      capabilities=["not-a-capability"]).status_code == 422


def test_a_capability_outside_the_transports_raisable_is_refused(tmp_path):
    """`raisable` is the static, vetted, per-transport bound a raise can never
    reach past. `tailnet` is `{execute, system_control}` and nothing else."""
    vault = TokenVault(tmp_path)
    token = vault.issue("laptop", frozenset(Capability))
    device_id = vault.devices()[0].device_id
    client = client_on(build(vault), "local", token)

    # A real capability, held by the device, simply not raisable on tailnet --
    # it is inside the ceiling already, so a raise for it is meaningless.
    assert post_raise(client, device_id, capabilities=["recall"]).status_code == 403


def test_a_raise_on_a_transport_with_no_raisable_set_is_refused(tmp_path):
    """`funnel` and `local` have `raisable=frozenset()`, so they are unraisable
    by construction rather than by a check somebody could forget. Asserted
    through the route as well, because "by construction" is only true of the
    code path that actually consults the set."""
    vault = TokenVault(tmp_path)
    token = vault.issue("laptop", frozenset(Capability))
    device_id = vault.devices()[0].device_id
    app = build(vault, running=frozenset({"tailnet", "funnel"}))
    client = client_on(app, "local", token)

    for transport in ("funnel", "local"):
        response = post_raise(client, device_id, transport=transport)
        assert response.status_code == 403, transport


def test_a_capability_the_device_was_never_issued_is_refused(tmp_path):
    """A raise lifts a transport's refusal; it cannot manufacture a grant.

    The device below holds everything tailnet's own ceiling carries, so it
    authenticates there perfectly well -- it was simply never issued EXECUTE,
    and no raise may invent one.
    """
    vault = TokenVault(tmp_path)
    vault.issue("phone", frozenset({
        Capability.OBSERVE, Capability.RECALL, Capability.CHAT_SEND,
        Capability.SCREEN, Capability.FILES,
    }))
    admin = vault.issue("laptop", frozenset(Capability))
    phone_id = [d for d in vault.devices() if d.label == "phone"][0].device_id
    client = client_on(build(vault), "local", admin)

    assert post_raise(client, phone_id, capabilities=["execute"]).status_code == 403


def test_a_transport_that_is_not_running_is_refused(tmp_path):
    """A raise scoped to a listener that is not up can never be exercised, and
    leaving one behind for a future listener of the same name is exactly what
    `RaiseStore.drop_policy` exists to prevent."""
    vault = TokenVault(tmp_path)
    token = vault.issue("laptop", frozenset(Capability))
    device_id = vault.devices()[0].device_id

    nothing_running = client_on(build(vault, running=frozenset()), "local", token)
    assert post_raise(nothing_running, device_id).status_code == 409


def test_a_missing_transport_manager_is_refused_rather_than_assumed_running(tmp_path):
    """Absent means nothing is running. The manager is a sibling task's wiring
    and an app built without it must fail closed, not mint a raise for a door
    nobody has opened."""
    vault = TokenVault(tmp_path)
    token = vault.issue("laptop", frozenset(Capability))
    device_id = vault.devices()[0].device_id
    app = build(vault)
    del app.state.transports

    assert post_raise(client_on(app, "local", token), device_id).status_code == 409


def test_minutes_are_clamped_to_the_seven_day_cap(tmp_path):
    """The cap is the safety property, not a promise the caller kept its word:
    an over-long request is clamped, never refused, so the operator gets a
    bounded raise rather than a 422 and a retry."""
    vault = TokenVault(tmp_path)
    token = vault.issue("laptop", frozenset(Capability))
    device_id = vault.devices()[0].device_id
    client = client_on(build(vault), "local", token)

    response = post_raise(client, device_id, minutes=60 * 24 * 30)   # thirty days
    assert response.status_code == 200
    # `>=` with a second of slack rather than exact equality: the payload's
    # `round()` reads the clock again after `grant()` stamped `expires_at`, so
    # exact equality quietly depends on under half a second elapsing in
    # between. Certain in practice, and a flake vector bought for nothing.
    remaining = response.json()["data"]["expiresInSeconds"]
    assert MAX_RAISE_SECONDS - 1 <= remaining <= MAX_RAISE_SECONDS


# ─── the one test that asserts the feature works ─────────────────────────
def test_a_raised_device_reaches_a_capability_the_ceiling_alone_refuses(tmp_path):
    """The whole mechanism, end to end, on a route that is genuinely refused
    without it.

    `PATCH /v1/settings` is gated on SYSTEM_CONTROL, which tailnet's fixed
    ceiling excludes -- so a device holding every grant there is still gets a
    403 over that tunnel. The raise is minted through the real route on the
    loopback listener, and the very next request over tailnet succeeds. Both
    halves are asserted: seventeen tests in this file would pass against a
    raise route that granted nothing, and this is the one that would not.
    """
    vault = TokenVault(tmp_path)
    token = vault.issue("laptop", frozenset(Capability))
    device_id = vault.devices()[0].device_id
    app = build(vault)

    remote = client_on(app, "tailnet", token)
    patch = {"changes": {"tts_speed": 1.25}}
    before = remote.patch("/v1/settings", json=patch, headers={CSRF_HEADER: "1"})
    assert before.status_code == 403, "the ceiling must refuse this without a raise"

    minted = post_raise(client_on(app, "local", token), device_id,
                        capabilities=["system_control"], minutes=30,
                        reason="changing the voice from the sofa")
    assert minted.status_code == 200

    after = remote.patch("/v1/settings", json=patch, headers={CSRF_HEADER: "1"})
    assert after.status_code == 200, "the raise must actually let this through"
    assert "tts_speed" in after.json()["data"]["saved"]

    # And it is reported as a raise, not folded into the ordinary story.
    session = remote.get("/v1/session").json()["data"]
    assert session["raised"] == ["system_control"]
    assert "system_control" in session["effective"]


# ─── scope ───────────────────────────────────────────────────────────────
def test_the_same_device_on_another_transport_is_unaffected(tmp_path):
    """The record is keyed on `(device_id, policy_name)`. A raise on tailnet
    says nothing about the same device arriving over funnel."""
    vault = TokenVault(tmp_path)
    token = vault.issue("laptop", frozenset(Capability))
    device_id = vault.devices()[0].device_id
    app = build(vault, running=frozenset({"tailnet", "funnel"}))

    assert post_raise(client_on(app, "local", token), device_id,
                      capabilities=["system_control"]).status_code == 200

    patch = {"changes": {"tts_speed": 1.25}}
    assert client_on(app, "tailnet", token).patch(
        "/v1/settings", json=patch, headers={CSRF_HEADER: "1"}).status_code == 200
    assert client_on(app, "funnel", token).patch(
        "/v1/settings", json=patch, headers={CSRF_HEADER: "1"}).status_code == 403


def test_another_device_on_the_same_transport_is_unaffected(tmp_path):
    """Nothing is lifted -- not the listener, not the ceiling. An unraised
    device on the raised transport computes exactly what it always did."""
    vault = TokenVault(tmp_path)
    admin = vault.issue("laptop", frozenset(Capability))
    other = vault.issue("phone", frozenset(Capability))
    laptop_id = [d for d in vault.devices() if d.label == "laptop"][0].device_id
    app = build(vault)

    assert post_raise(client_on(app, "local", admin), laptop_id,
                      capabilities=["system_control"]).status_code == 200

    patch = {"changes": {"tts_speed": 1.25}}
    assert client_on(app, "tailnet", other).patch(
        "/v1/settings", json=patch, headers={CSRF_HEADER: "1"}).status_code == 403


# ─── expiry and revocation ───────────────────────────────────────────────
def test_an_expired_raise_stops_working_without_any_further_call(tmp_path, monkeypatch):
    """Expiry is checked on read, never by a timer, so nothing has to run for a
    raise to stop working -- and no request, route or restart is needed to make
    the stale record harmless."""
    vault = TokenVault(tmp_path)
    token = vault.issue("laptop", frozenset(Capability))
    device_id = vault.devices()[0].device_id
    app = build(vault)

    clock = FakeClock()
    monkeypatch.setattr(raises_module, "time", clock)

    assert post_raise(client_on(app, "local", token), device_id,
                      capabilities=["system_control"], minutes=30).status_code == 200

    remote = client_on(app, "tailnet", token)
    patch = {"changes": {"tts_speed": 1.25}}
    assert remote.patch("/v1/settings", json=patch,
                        headers={CSRF_HEADER: "1"}).status_code == 200

    clock.now += 30 * 60 + 1
    assert remote.patch("/v1/settings", json=patch,
                        headers={CSRF_HEADER: "1"}).status_code == 403


def test_delete_revokes_the_raise_immediately(tmp_path):
    vault = TokenVault(tmp_path)
    token = vault.issue("laptop", frozenset(Capability))
    device_id = vault.devices()[0].device_id
    app = build(vault)
    local = client_on(app, "local", token)

    assert post_raise(local, device_id, capabilities=["system_control"]).status_code == 200

    remote = client_on(app, "tailnet", token)
    patch = {"changes": {"tts_speed": 1.25}}
    assert remote.patch("/v1/settings", json=patch,
                        headers={CSRF_HEADER: "1"}).status_code == 200

    dropped = local.delete(f"/v1/devices/{device_id}/raise", headers={CSRF_HEADER: "1"})
    assert dropped.status_code == 200
    assert dropped.json()["data"] == {"revoked": device_id}

    assert remote.patch("/v1/settings", json=patch,
                        headers={CSRF_HEADER: "1"}).status_code == 403
    # And a second DELETE says so, rather than reporting a revocation that did
    # not happen -- the same distinction `revoke_device` draws.
    assert local.delete(f"/v1/devices/{device_id}/raise",
                        headers={CSRF_HEADER: "1"}).status_code == 404


def test_revoking_the_device_drops_its_raises(tmp_path):
    """A raise outliving the device it was granted to would be absurd, and
    `revoke` is the control that has to work under duress."""
    vault = TokenVault(tmp_path)
    admin = vault.issue("laptop", frozenset(Capability))
    vault.issue("phone", frozenset(Capability))
    phone_id = [d for d in vault.devices() if d.label == "phone"][0].device_id
    store = RaiseStore()
    app = build(vault, store=store)
    local = client_on(app, "local", admin)

    assert post_raise(local, phone_id, capabilities=["system_control"]).status_code == 200
    assert store.get(phone_id, "tailnet") is not None

    assert local.delete(f"/v1/devices/{phone_id}",
                        headers={CSRF_HEADER: "1"}).status_code == 200
    assert store.get(phone_id, "tailnet") is None


def test_vault_reset_makes_every_raise_unusable(tmp_path):
    """The kill switch, observed where it counts.

    **Named for what it proves, not for what spec §3.3 asks for.** The spec
    lists `vault.reset()` among the events that *drop* a raise record; this
    test does not prove that, and cannot -- the 401 below holds whether the
    record survived or not. `TokenVault.reset()` holds no reference to the
    store (and must not: `raises.py` imports `Capability` and nothing else from
    this layer), and `server.shutdown()`, the one place that calls it, belongs
    to the task that owns the daemon lifecycle. The store-level `clear()` is
    formally carried there.

    What this pins is the load-bearing half, and it is the stronger one: after
    the switch is thrown, no request can consume a raise, because no request
    can authenticate at all. Renamed rather than deleted so the remaining gap
    stays visible in the test list instead of looking closed.
    """
    vault = TokenVault(tmp_path)
    token = vault.issue("laptop", frozenset(Capability))
    device_id = vault.devices()[0].device_id
    app = build(vault)

    assert post_raise(client_on(app, "local", token), device_id,
                      capabilities=["system_control"]).status_code == 200

    remote = client_on(app, "tailnet", token)
    patch = {"changes": {"tts_speed": 1.25}}
    assert remote.patch("/v1/settings", json=patch,
                        headers={CSRF_HEADER: "1"}).status_code == 200

    vault.reset()

    assert remote.patch("/v1/settings", json=patch,
                        headers={CSRF_HEADER: "1"}).status_code == 401


# ─── visibility (§3.6) ───────────────────────────────────────────────────
def test_using_a_raised_capability_writes_an_audit_entry(tmp_path):
    """The difference between knowing a door was unlocked and knowing it was
    walked through. The mint alone is not enough: a seven-day window has to
    leave a trail of what was actually done under it."""
    vault = TokenVault(tmp_path)
    token = vault.issue("laptop", frozenset(Capability))
    device_id = vault.devices()[0].device_id
    app = build(vault)

    assert post_raise(client_on(app, "local", token), device_id,
                      capabilities=["system_control"]).status_code == 200

    before = [e for e in app.state.auth.audit.entries() if e.outcome.startswith("raised:")]
    assert before == [], "nothing has ridden the raise yet"

    remote = client_on(app, "tailnet", token)
    assert remote.patch("/v1/settings", json={"changes": {"tts_speed": 1.25}},
                        headers={CSRF_HEADER: "1"}).status_code == 200

    used = [e for e in app.state.auth.audit.entries() if e.outcome.startswith("raised:")]
    # One, because exactly one request is made over the raised listener -- NOT
    # because the entry is filtered by whether the handler spent the raised
    # capability. It is not; `test_a_raise_in_reach_records_on_every_request_
    # not_only_the_one_that_spends_it` below is where those semantics live, and
    # a second request added here would make this 2.
    assert len(used) == 1
    assert used[0].device_id == device_id
    assert used[0].path == "/v1/settings"
    assert used[0].method == "PATCH"
    assert "system_control" in used[0].outcome


def test_a_raise_in_reach_records_on_every_request_not_only_the_one_that_spends_it(tmp_path):
    """The honest semantics of the entry above, pinned so nobody has to infer
    them from a comment.

    `applied` in `authenticate()` is a function of the raise and the policy
    alone -- it never consults what the request went on to require -- so an
    entry means "a raise put this within reach on this request", never "the
    handler spent it". A `GET /v1/session` that needs no capability at all
    records one too.

    That is over-recording, and it is deliberate. Telling reach from spending
    would mean hooking `require()` *plus* every site that checks a grant
    without it -- `POST /v1/chat` hands `device.grants` to the assistant where
    the intent gate spends EXECUTE, and `run_command` checks inline -- which is
    the enumerate-the-paths-around-the-boundary problem this milestone has
    already lost to twice. One choke point that over-records is the fail-safe
    direction.
    """
    vault = TokenVault(tmp_path)
    token = vault.issue("laptop", frozenset(Capability))
    device_id = vault.devices()[0].device_id
    app = build(vault)

    assert post_raise(client_on(app, "local", token), device_id,
                      capabilities=["system_control"]).status_code == 200

    remote = client_on(app, "tailnet", token)
    # Neither of these spends SYSTEM_CONTROL. `GET /v1/session` is gated on no
    # capability whatsoever; `GET /v1/status` is gated on OBSERVE, which
    # tailnet's own ceiling has always carried.
    assert remote.get("/v1/session").status_code == 200
    assert remote.get("/v1/status").status_code == 200

    raised = [e for e in app.state.auth.audit.entries() if e.outcome.startswith("raised:")]
    assert [(e.method, e.path) for e in raised] == [
        ("GET", "/v1/session"), ("GET", "/v1/status"),
    ]
    assert all(e.outcome == "raised:system_control" for e in raised)

    # And the mirror image: a device on the same listener with no raise of its
    # own records nothing, so the entry is about the raise rather than about
    # the listener.
    other = vault.issue("wall display", frozenset(Capability))
    unraised = client_on(app, "tailnet", other)
    assert unraised.get("/v1/session").status_code == 200
    still = [e for e in app.state.auth.audit.entries() if e.outcome.startswith("raised:")]
    assert len(still) == 2


def test_minting_a_raise_writes_a_log_line_that_names_no_secret(tmp_path, caplog):
    vault = TokenVault(tmp_path)
    token = vault.issue("laptop", frozenset(Capability))
    device_id = vault.devices()[0].device_id
    client = client_on(build(vault), "local", token)

    with caplog.at_level(logging.INFO, logger="assistant.io.api.routes.devices"):
        assert post_raise(client, device_id, capabilities=["system_control"],
                          reason="changing the voice").status_code == 200

    minted = [r.getMessage() for r in caplog.records if "raise" in r.getMessage()]
    assert minted, "minting a raise must leave a line in the log"
    line = minted[0]
    assert token not in line
    assert device_id in line and "tailnet" in line and "system_control" in line


def test_the_device_list_reports_live_raises(tmp_path):
    """`GET /v1/devices` grows a per-device summary, and no route enumerates
    raises on its own -- so nothing new becomes an oracle for which device ids
    exist."""
    vault = TokenVault(tmp_path)
    admin = vault.issue("laptop", frozenset(Capability))
    vault.issue("phone", frozenset(Capability))
    phone_id = [d for d in vault.devices() if d.label == "phone"][0].device_id
    app = build(vault)
    local = client_on(app, "local", admin)

    assert post_raise(local, phone_id, capabilities=["system_control"],
                      minutes=30, reason="printer").status_code == 200

    rows = local.get("/v1/devices").json()["data"]["devices"]
    phone = [d for d in rows if d["label"] == "phone"][0]
    laptop = [d for d in rows if d["label"] == "laptop"][0]

    assert laptop["raises"] == []
    assert len(phone["raises"]) == 1
    summary = phone["raises"][0]
    assert summary["deviceId"] == phone_id
    assert summary["transport"] == "tailnet"
    assert summary["capabilities"] == ["system_control"]
    assert summary["reason"] == "printer"
    assert 0 < summary["expiresInSeconds"] <= 30 * 60

    # There is no route that lists raises on their own.
    assert local.get("/v1/raises").status_code == 404


def test_a_raise_never_changes_admin_or_bearer_or_secure_cookie(tmp_path):
    """Invariant I5: a raise touches capabilities and nothing else about the
    listener it is scoped to."""
    vault = TokenVault(tmp_path)
    token = vault.issue("laptop", frozenset(Capability))
    device_id = vault.devices()[0].device_id
    app = build(vault)

    assert post_raise(client_on(app, "local", token), device_id,
                      capabilities=["system_control"]).status_code == 200

    tailnet = POLICIES["tailnet"]
    assert tailnet.admin is False
    assert tailnet.allow_bearer is False
    assert tailnet.secure_cookie is True

    # And observed through the wire, not only off the module data: the raised
    # device still cannot reach an admin route over the tunnel, and still
    # cannot authenticate with a bearer header there.
    remote = client_on(app, "tailnet", token)
    assert remote.get("/v1/devices").status_code == 403

    bearer = ApiTestClient(app, base_url=f"http://127.0.0.1:{PORTS['tailnet']}")
    assert bearer.get("/v1/session",
                      headers={"Authorization": f"Bearer {token}"}).status_code == 401
