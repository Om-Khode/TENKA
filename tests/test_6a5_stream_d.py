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
