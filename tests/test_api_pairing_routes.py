"""Minting a pair code, and redeeming one.

`POST /v1/pair` is the only unauthenticated write in this API, and from
Milestone 6b it is reachable from the open internet -- sitting beside a
`POST /v1/chat` that reaches `code_executor`. So the properties pinned here
are not conveniences:

- enrolling a device is loopback-only (`policy.admin`), because a compromised
  remote session must not be able to open a second, permanent door;
- grants ride on the code, server-side, so the redeeming request cannot widen
  them -- that is what makes the laptop's checkbox row a boundary;
- wrong, expired and already-used codes are one identical response, the same
  posture `verify()` holds for unknown-vs-revoked tokens;
- the attempt budget is global, and exhausting it burns the outstanding code;
- no log line ever carries a code, at any level.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

from assistant.io.api.pairing import PairCodeStore
from assistant.io.api.security import COOKIE_NAME, CSRF_HEADER, HOST_COOKIE_NAME
from assistant.io.api.vault import Capability, TokenVault
from tests.fakes.api_client import BASE_URL, LOCAL_PORT, ApiTestClient, build_api_client
from tests.fakes.studio_runtime import build_fake_runtime


def _client(vault: TokenVault, *, policies: dict[int, str],
            store: PairCodeStore | None = None) -> ApiTestClient:
    """A client on `LOCAL_PORT`, with `policies` naming what that port is.

    `store` is passed when a test needs to hold the daemon's own pair-code
    store -- to mint without going through the loopback-only route, or to
    assert afterwards that a code was burned.
    """
    return build_api_client(build_fake_runtime(), vault,
                            policies=policies, pair_store=store)


def _second_browser(client: ApiTestClient) -> ApiTestClient:
    """A fresh cookie jar against the *same* daemon.

    Not `_client(...)` a second time: that builds a second app, with its own
    pair-code store, and a code minted on one daemon is deliberately not
    redeemable on another. What a phone actually is here is a different
    browser talking to the one daemon the laptop just minted on -- same app,
    no cookies -- which is exactly what this builds.
    """
    return ApiTestClient(client.app, base_url=BASE_URL)


def _store() -> PairCodeStore:
    return PairCodeStore()


def _mint(client: ApiTestClient, label: str, grants: list[str]) -> str:
    response = client.post("/v1/pair/code", json={"label": label, "grants": grants},
                           headers={CSRF_HEADER: "1"})
    assert response.status_code == 200, response.text
    return response.json()["data"]["code"]


# ─── minting is loopback-only ────────────────────────────────────────────
def test_minting_a_code_is_loopback_only(tmp_path):
    """Device enrollment from a remote session would be privilege escalation:
    one compromised phone could mint itself a second, wider credential."""
    vault = TokenVault(tmp_path)
    token = vault.issue("phone", frozenset(Capability))
    remote = _client(vault, policies={LOCAL_PORT: "funnel"})
    remote.cookies.set(COOKIE_NAME, token)
    r = remote.post("/v1/pair/code", json={"label": "x", "grants": ["observe"]},
                    headers={CSRF_HEADER: "1"})
    assert r.status_code == 403


def test_minting_requires_authentication(tmp_path):
    client = _client(TokenVault(tmp_path), policies={LOCAL_PORT: "local"})
    assert client.post("/v1/pair/code", json={"label": "x", "grants": ["observe"]},
                       headers={CSRF_HEADER: "1"}).status_code == 401


def test_minting_needs_more_than_a_watching_grant(tmp_path):
    """`policy.admin` alone is not the whole gate. A device paired to watch,
    sitting on the loopback listener, could otherwise mint itself a code
    carrying SYSTEM_CONTROL -- the escalation the loopback rule exists to
    prevent, arrived at from inside the house instead of outside it."""
    vault = TokenVault(tmp_path)
    watcher = vault.issue("wall display", frozenset({Capability.OBSERVE}))
    client = _client(vault, policies={LOCAL_PORT: "local"})
    client.cookies.set(COOKIE_NAME, watcher)
    r = client.post("/v1/pair/code", json={"label": "x", "grants": ["system_control"]},
                    headers={CSRF_HEADER: "1"})
    assert r.status_code == 403


# ─── I-1: a minted code can never exceed the minting device's own grants ──
def test_minting_is_capped_at_what_the_device_holds(tmp_path):
    """A device issued only SYSTEM_CONTROL used to be able to mint a code
    carrying every other capability there is -- a second credential wider
    than the one it was ever granted, through the one route that exists
    specifically to prevent that escalation. `Capability`'s own docstring
    says grants are "granted per device, never implied": SYSTEM_CONTROL does
    not subsume RECALL, FILES, or CHAT_SEND, so none of the three should have
    ended up on a code this device mints.
    """
    vault = TokenVault(tmp_path)
    admin = vault.issue("laptop", frozenset({Capability.SYSTEM_CONTROL}))
    client = _client(vault, policies={LOCAL_PORT: "local"})
    client.cookies.set(COOKIE_NAME, admin)
    r = client.post("/v1/pair/code",
                    json={"label": "phone", "grants": ["recall", "files", "chat_send"]},
                    headers={CSRF_HEADER: "1"})
    assert r.status_code == 422


def test_minting_narrows_to_the_intersection_not_all_or_nothing(tmp_path):
    """The cap is an intersection, not a blanket refusal the moment any
    requested grant is missing: a device holding SYSTEM_CONTROL and OBSERVE,
    asking for {OBSERVE, RECALL}, gets a code carrying OBSERVE alone -- not a
    422, and not a code that quietly grew RECALL back in.
    """
    vault = TokenVault(tmp_path)
    admin = vault.issue("laptop", frozenset({Capability.SYSTEM_CONTROL, Capability.OBSERVE}))
    store = _store()
    client = _client(vault, policies={LOCAL_PORT: "local"}, store=store)
    client.cookies.set(COOKIE_NAME, admin)
    r = client.post("/v1/pair/code",
                    json={"label": "phone", "grants": ["observe", "recall"]},
                    headers={CSRF_HEADER: "1"})
    assert r.status_code == 200
    assert store.current().grants == frozenset({Capability.OBSERVE})


def test_a_device_holding_everything_can_still_mint_everything(tmp_path):
    """The intersection must not become a hidden ceiling for the ordinary
    case: the usual admin device (issued every capability, per `main.py`'s
    bootstrap) still mints exactly what it asks for.
    """
    vault = TokenVault(tmp_path)
    admin = vault.issue("laptop", frozenset(Capability))
    store = _store()
    client = _client(vault, policies={LOCAL_PORT: "local"}, store=store)
    client.cookies.set(COOKIE_NAME, admin)
    r = client.post("/v1/pair/code",
                    json={"label": "phone", "grants": ["recall", "files", "chat_send"]},
                    headers={CSRF_HEADER: "1"})
    assert r.status_code == 200
    assert store.current().grants == frozenset(
        {Capability.RECALL, Capability.FILES, Capability.CHAT_SEND})


def test_an_unknown_capability_is_422(tmp_path):
    vault = TokenVault(tmp_path)
    owner = vault.issue("laptop", frozenset(Capability))
    client = _client(vault, policies={LOCAL_PORT: "local"})
    client.cookies.set(COOKIE_NAME, owner)
    r = client.post("/v1/pair/code", json={"label": "x", "grants": ["chat"]},
                    headers={CSRF_HEADER: "1"})
    assert r.status_code == 422


def test_a_minted_code_carries_a_qr_and_the_loopback_endpoint(tmp_path):
    vault = TokenVault(tmp_path)
    owner = vault.issue("laptop", frozenset(Capability))
    client = _client(vault, policies={LOCAL_PORT: "local"})
    client.cookies.set(COOKIE_NAME, owner)
    data = client.post("/v1/pair/code",
                       json={"label": "Pixel 8", "grants": ["observe"]},
                       headers={CSRF_HEADER: "1"}).json()["data"]
    assert set(data) == {"code", "expiresAt", "endpoints", "qrSvg"}
    assert data["endpoints"] == [f"http://127.0.0.1:{LOCAL_PORT}"]
    assert "<svg" in data["qrSvg"]


def test_the_qr_never_carries_the_code_as_readable_text(tmp_path):
    """The code rides in the URL *fragment*, which is never sent to a server
    and so never lands in an access log -- and the SVG is paths, not text, so
    the code cannot be recovered by grepping a saved page either."""
    vault = TokenVault(tmp_path)
    owner = vault.issue("laptop", frozenset(Capability))
    client = _client(vault, policies={LOCAL_PORT: "local"})
    client.cookies.set(COOKIE_NAME, owner)
    data = client.post("/v1/pair/code", json={"label": "p", "grants": ["observe"]},
                       headers={CSRF_HEADER: "1"}).json()["data"]
    assert data["code"] not in data["qrSvg"]


# ─── redeeming a code ────────────────────────────────────────────────────
def test_a_minted_code_pairs_a_device_and_sets_the_cookie(tmp_path):
    vault = TokenVault(tmp_path)
    owner = vault.issue("laptop", frozenset(Capability))
    client = _client(vault, policies={LOCAL_PORT: "local"})
    client.cookies.set(COOKIE_NAME, owner)
    code = _mint(client, "Pixel 8", ["observe", "chat_send"])
    fresh = _second_browser(client)
    r = fresh.post("/v1/pair", json={"code": code})
    assert r.status_code == 204
    assert COOKIE_NAME in r.cookies
    assert {d.label for d in vault.devices()} == {"laptop", "Pixel 8"}


def test_the_paired_device_can_immediately_use_its_cookie(tmp_path):
    """204 with no body is only a credential hand-off if the cookie works."""
    vault = TokenVault(tmp_path)
    owner = vault.issue("laptop", frozenset(Capability))
    client = _client(vault, policies={LOCAL_PORT: "local"})
    client.cookies.set(COOKIE_NAME, owner)
    code = _mint(client, "Pixel 8", ["observe"])
    fresh = _second_browser(client)
    fresh.post("/v1/pair", json={"code": code})
    assert fresh.get("/v1/status").status_code == 200


def test_the_pairing_request_cannot_widen_its_grants(tmp_path):
    """Grants come from the code. This is the boundary the checkbox row buys.

    The extra `grants` key is ignored rather than rejected on purpose: the
    property worth pinning is that a client which *tries* to widen still ends
    up with exactly what the laptop authorised, not that it gets a 422 it
    could route around by dropping the key.
    """
    vault = TokenVault(tmp_path)
    owner = vault.issue("laptop", frozenset(Capability))
    client = _client(vault, policies={LOCAL_PORT: "local"})
    client.cookies.set(COOKIE_NAME, owner)
    code = _mint(client, "p", ["observe"])
    fresh = _second_browser(client)
    fresh.post("/v1/pair", json={"code": code, "grants": ["system_control"]})
    paired = [d for d in vault.devices() if d.label == "p"][0]
    assert paired.grants == frozenset({Capability.OBSERVE})


def test_the_pairing_request_cannot_rename_the_device(tmp_path):
    """The label is what the user typed on the laptop while choosing grants.
    A device that could name itself could name itself `laptop` -- and the
    revoke list is where she decides which row to kill."""
    vault = TokenVault(tmp_path)
    owner = vault.issue("laptop", frozenset(Capability))
    client = _client(vault, policies={LOCAL_PORT: "local"})
    client.cookies.set(COOKIE_NAME, owner)
    code = _mint(client, "Pixel 8", ["observe"])
    fresh = _second_browser(client)
    fresh.post("/v1/pair", json={"code": code, "label": "laptop"})
    assert sorted(d.label for d in vault.devices()) == ["Pixel 8", "laptop"]


# ─── I-2: grants are also capped by the redeeming listener ────────────────
def test_pairing_over_quick_narrows_grants_to_the_listeners_ceiling(tmp_path):
    """A code minted with every capability, redeemed over Cloudflare's `quick`
    tunnel (ceiling: OBSERVE alone), must mint a device that can only ever
    watch -- exactly `effective(pair_code.grants, policy)`, never the code's
    own, unnarrowed set.
    """
    vault = TokenVault(tmp_path)
    store = _store()
    code = store.mint("phone", frozenset(Capability)).code
    client = _client(vault, policies={LOCAL_PORT: "quick"}, store=store)
    r = client.post("/v1/pair", json={"code": code})
    assert r.status_code == 204
    paired = [d for d in vault.devices() if d.label == "phone"][0]
    assert paired.grants == frozenset({Capability.OBSERVE})


def test_pairing_over_quick_permanently_limits_the_device_even_on_a_wider_listener(tmp_path):
    """The narrowing at issue time is not a per-request ceiling that lifts on
    a better connection later -- it changes what the vault actually stored.
    A device that paired over `quick` and ended up with OBSERVE alone stays
    OBSERVE-only when it later connects over `funnel`, because
    `GET /v1/session` reports the issued grants the vault has on file, and
    `quick` never let anything but OBSERVE reach `issue()` in the first
    place. This is the surprising consequence the route's docstring calls
    out explicitly.
    """
    vault = TokenVault(tmp_path)
    store = _store()
    code = store.mint("phone", frozenset(Capability)).code
    quick_client = _client(vault, policies={LOCAL_PORT: "quick"}, store=store)
    r = quick_client.post("/v1/pair", json={"code": code})
    assert r.status_code == 204
    # `quick`, `tailnet` and `funnel` all set `secure_cookie`, so since
    # `fix/6a5-api-review` they write the `__Host-` prefixed name -- a browser
    # will not store that one unless it is host-only, which is what stops a
    # sibling under `*.trycloudflare.com` planting a `tenka_device` inward.
    # Only `local` still uses the unprefixed name. Both are read.
    cookie_value = r.cookies[HOST_COOKIE_NAME]

    funnel_client = _client(vault, policies={LOCAL_PORT: "funnel"})
    funnel_client.cookies.set(HOST_COOKIE_NAME, cookie_value)
    data = funnel_client.get("/v1/session").json()["data"]
    assert data["grants"] == ["observe"]


def test_pairing_with_nothing_the_listener_can_carry_is_refused(tmp_path):
    """A code minted without OBSERVE, redeemed over `quick` (ceiling: OBSERVE
    alone), has nothing left after the intersection. Refused exactly like a
    wrong code -- the alternative is `TokenVault.issue()` raising on an empty
    grant set, which must never reach a caller as a raw exception. The code
    is still burned (single-use): this is not a retryable failure.
    """
    vault = TokenVault(tmp_path)
    store = _store()
    code = store.mint("phone", frozenset({Capability.FILES, Capability.SYSTEM_CONTROL})).code
    client = _client(vault, policies={LOCAL_PORT: "quick"}, store=store)
    r = client.post("/v1/pair", json={"code": code})
    assert r.status_code == 401
    assert vault.devices() == []


def test_pair_needs_no_authentication(tmp_path):
    """It is how a device gets a credential -- and therefore the only
    unauthenticated write in the API."""
    vault = TokenVault(tmp_path)
    store = _store()
    code = store.mint("p", frozenset({Capability.CHAT_SEND})).code
    client = _client(vault, policies={LOCAL_PORT: "local"}, store=store)
    assert client.post("/v1/pair", json={"code": code}).status_code == 204


def test_a_code_is_single_use(tmp_path):
    vault = TokenVault(tmp_path)
    store = _store()
    code = store.mint("p", frozenset({Capability.CHAT_SEND})).code
    client = _client(vault, policies={LOCAL_PORT: "local"}, store=store)
    assert client.post("/v1/pair", json={"code": code}).status_code == 204
    assert client.post("/v1/pair", json={"code": code}).status_code == 401
    assert len(vault.devices()) == 1


def test_wrong_expired_and_reused_codes_are_indistinguishable(tmp_path):
    """Four failure modes, one response: reused, expired, wrong, and
    malformed. The property was previously pinned for three of the four --
    no expired code was ever actually tried against the route, so the fourth
    was proved by reasoning about `consume()`, not by execution.
    """
    vault = TokenVault(tmp_path)
    store = _store()
    code = store.mint("p", frozenset({Capability.CHAT_SEND})).code
    client = _client(vault, policies={LOCAL_PORT: "local"}, store=store)
    client.post("/v1/pair", json={"code": code})           # consume it

    # Minted with `now` already past its own TTL, so it is born expired --
    # deterministic, no sleep, no flake. Minting also replaces whatever code
    # was live, but `code` above is already consumed and gone from the store
    # regardless, so this does not disturb it.
    expired = store.mint("q", frozenset({Capability.CHAT_SEND}),
                         now=time.monotonic() - 1000).code

    bodies, statuses = set(), set()
    for attempt in (code, expired, "AAAA-AAAA", "garbage"):
        r = client.post("/v1/pair", json={"code": attempt})
        statuses.add(r.status_code)
        bodies.add(r.text)
    assert statuses == {401} and len(bodies) == 1


# ─── the budget ──────────────────────────────────────────────────────────
def test_repeated_wrong_codes_lock_out(tmp_path):
    vault = TokenVault(tmp_path)
    client = _client(vault, policies={LOCAL_PORT: "local"})
    codes = [f"AAAA-AAA{i}" for i in range(12)]
    statuses = [client.post("/v1/pair", json={"code": c}).status_code for c in codes]
    assert 429 in statuses


def test_a_couple_of_typos_do_not_lock_anyone_out(tmp_path):
    """The other half of the budget's job: it must not fire on ordinary use.
    Someone reading a code off a screen mistypes once or twice, not ten times
    -- and getting it right clears the slate."""
    vault = TokenVault(tmp_path)
    store = _store()
    code = store.mint("p", frozenset({Capability.CHAT_SEND})).code
    client = _client(vault, policies={LOCAL_PORT: "local"}, store=store)
    for typo in ("AAAA-AAAA", "BBBB-BBBB"):
        assert client.post("/v1/pair", json={"code": typo}).status_code == 401
    assert store.current() is not None, "two typos must not burn a live code"
    assert client.post("/v1/pair", json={"code": code}).status_code == 204


def test_the_pair_budget_is_global_not_per_client(tmp_path):
    """Per-IP keying is worthless on this route. cloudflared and tailscale
    serve both connect from 127.0.0.1, so every caller on earth already shares
    one key -- and any forwarded-IP header is the attacker's own input, so
    keying on that would hand out a fresh budget per spoofed value."""
    vault = TokenVault(tmp_path)
    store = _store()
    store.mint("p", frozenset({Capability.CHAT_SEND}))
    client = _client(vault, policies={LOCAL_PORT: "local"}, store=store)
    statuses = [
        client.post("/v1/pair", json={"code": f"AAAA-AAA{i}"},
                    headers={"X-Forwarded-For": f"10.0.0.{i}"}).status_code
        for i in range(12)
    ]
    assert 429 in statuses


def test_exhausting_the_budget_burns_the_live_code(tmp_path):
    """Failing the budget destroys the outstanding code rather than pausing
    guessing at it. An attacker must then beat a NEW 180-second window that
    only the laptop can open. Being locked out of pairing is an acceptable
    cost; leaving a code standing under sustained guessing is not -- and this
    is the one denial-of-service we accept ON PURPOSE, because the alternative
    trades a security property for an availability one."""
    vault = TokenVault(tmp_path)
    store = _store()
    real = store.mint("p", frozenset({Capability.CHAT_SEND})).code
    client = _client(vault, policies={LOCAL_PORT: "local"}, store=store)
    for i in range(12):
        client.post("/v1/pair", json={"code": f"AAAA-AAA{i}"})
    assert store.current() is None
    assert client.post("/v1/pair", json={"code": real}).status_code in (401, 429)


def test_minting_a_fresh_code_reopens_the_pairing_window(tmp_path):
    """The remedy for the accepted denial of service, and the reason it is
    acceptable. Minting is loopback + admin + SYSTEM_CONTROL, so only someone
    already at the machine can clear the lockout -- and that is precisely the
    person the attacker was denying."""
    vault = TokenVault(tmp_path)
    store = _store()
    owner = vault.issue("laptop", frozenset(Capability))
    client = _client(vault, policies={LOCAL_PORT: "local"}, store=store)
    statuses = [client.post("/v1/pair", json={"code": f"AAAA-AAA{i}"}).status_code
                for i in range(12)]
    assert statuses[-1] == 429, "sanity: the budget has to be spent to be reopened"

    client.cookies.set(COOKIE_NAME, owner)
    code = _mint(client, "Pixel 8", ["observe"])
    fresh = _second_browser(client)
    assert fresh.post("/v1/pair", json={"code": code}).status_code == 204


# ─── the code never reaches a log ────────────────────────────────────────
def test_no_log_line_ever_contains_a_code(tmp_path, caplog):
    vault = TokenVault(tmp_path)
    owner = vault.issue("laptop", frozenset(Capability))
    client = _client(vault, policies={LOCAL_PORT: "local"})
    client.cookies.set(COOKIE_NAME, owner)
    with caplog.at_level(logging.DEBUG):
        code = _mint(client, "p", ["observe"])
        client.post("/v1/pair", json={"code": code})
    assert code not in caplog.text


def test_a_failed_attempt_does_not_log_what_was_tried(tmp_path, caplog):
    """A wrong code is still somebody's near-miss at a live one."""
    client = _client(TokenVault(tmp_path), policies={LOCAL_PORT: "local"})
    with caplog.at_level(logging.DEBUG):
        client.post("/v1/pair", json={"code": "ZZZZ-ZZZZ"})
    assert "ZZZZ-ZZZZ" not in caplog.text


def test_the_audit_trail_records_the_attempt_without_the_code(tmp_path):
    """The one place a pairing attempt *is* recorded: method, path, outcome."""
    vault = TokenVault(tmp_path)
    client = _client(vault, policies={LOCAL_PORT: "local"})
    client.post("/v1/pair", json={"code": "ZZZZ-ZZZZ"})
    entries = client.app.state.auth.audit.entries()
    assert ("POST", "/v1/pair", "401") in {(e.method, e.path, e.outcome) for e in entries}
    assert all("ZZZZ-ZZZZ" not in e.path for e in entries)


# ─── I-3: a read failure during issue() must not destroy other devices ────
def test_pairing_answers_503_and_survives_when_the_vault_cannot_be_read(tmp_path, monkeypatch):
    """The code is real and gets consumed -- `TokenVault.issue()` refusing to
    write from a devices.json it could not read must not surface as a raw
    500, and must not destroy the device that was already there. Proved
    under real Windows contention by the review (`PermissionError
    [WinError 5]` from concurrent readers/writers); simulated here
    deterministically by breaking reads of exactly this vault's
    `devices.json`, nothing else.
    """
    vault = TokenVault(tmp_path)
    vault.issue("existing", frozenset({Capability.OBSERVE}))  # must survive
    store = _store()
    code = store.mint("phone", frozenset({Capability.OBSERVE})).code
    client = _client(vault, policies={LOCAL_PORT: "local"}, store=store)

    devices_json = tmp_path / "devices.json"
    original_read_text = Path.read_text

    def broken(self, *args, **kwargs):
        if self == devices_json:
            raise PermissionError("simulated lock")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", broken)
    r = client.post("/v1/pair", json={"code": code})
    monkeypatch.setattr(Path, "read_text", original_read_text)

    assert r.status_code == 503
    assert {d.label for d in vault.devices()} == {"existing"}


def test_pairing_answers_503_not_403_when_the_vault_cannot_be_written(tmp_path, monkeypatch):
    """The other half of the same lock. `_save()`'s underlying write raises
    `PermissionError` under the identical Windows contention that breaks
    reads -- and an uncaught `PermissionError` is mapped by `errors.py` to
    403 "protected path", which the review proved this route actually
    answered before `VaultWriteError` existed: a caller presenting a
    perfectly good, single-use code would be told it was not allowed,
    when the truth was "the vault could not be saved to right now".
    """
    from assistant.io.api import vault as vault_module

    vault = TokenVault(tmp_path)
    vault.issue("existing", frozenset({Capability.OBSERVE}))  # must survive
    store = _store()
    code = store.mint("phone", frozenset({Capability.OBSERVE})).code
    client = _client(vault, policies={LOCAL_PORT: "local"}, store=store)

    original_atomic_write = vault_module._atomic_write

    def broken(path, content):
        if path.name == "devices.json":
            raise PermissionError("simulated lock")
        return original_atomic_write(path, content)

    monkeypatch.setattr(vault_module, "_atomic_write", broken)
    r = client.post("/v1/pair", json={"code": code})
    monkeypatch.setattr(vault_module, "_atomic_write", original_atomic_write)

    assert r.status_code == 503
    assert {d.label for d in vault.devices()} == {"existing"}
