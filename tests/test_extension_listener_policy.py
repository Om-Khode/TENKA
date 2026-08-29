"""
test_extension_listener_policy.py — Latch Task 10: the fourth listener.

The browser extension connects over a loopback WebSocket. Rather than
hand-rolling a socket, it gets a `ListenerPolicy` and a port in the registry,
because `policy.py` already keys authority on the accepting local port and
`listeners.py` explains at length why that is the only discriminator a client
cannot forge.

What this file pins:

  - the policy grants **nothing**. `effective()` narrows a device holding every
    capability there is down to the empty set. The extension is a *target*, not
    a principal — TENKA drives it; it never asks TENKA to run an intent — so
    there is no capability for a ceiling to permit. What follows is the part
    worth a test: a stolen extension token carries zero intent authority.
  - it mints nothing and administers nothing.
  - the port is registered, and an unregistered port still resolves to `None`.
    Fail-closed does not stop being the rule because a fourth entry arrived.

The reachability half — that every HTTP route on this port refuses — lives in
`test_6b_listener_matrix.py`, which now includes this listener in its table.
Both halves are needed: this file says the ceiling is empty, that one says the
app actually behaves that way.

Run: py -3.11 -m pytest tests/test_extension_listener_policy.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest  # noqa: E402

import assistant.config as config  # noqa: E402
from assistant.io.api.listeners import (  # noqa: E402
    LISTENER_OFFSETS,
    policy_name_for_port,
    port_for,
)
from assistant.io.api.policy import POLICIES, effective  # noqa: E402
from assistant.io.api.vault import Capability  # noqa: E402

POLICY_NAME = "extension"
ALL_CAPABILITIES = frozenset(Capability)


@pytest.fixture()
def policy():
    assert POLICY_NAME in POLICIES, "the extension policy is not registered"
    return POLICIES[POLICY_NAME]


# ─── It grants nothing ───────────────────────────────────────────────────


def test_a_maximal_device_gets_nothing_on_this_listener(policy):
    assert effective(ALL_CAPABILITIES, policy) == frozenset(), (
        "the extension listener carries a capability. It is a target, not a "
        "principal — it never asks for an intent to run — so anything it "
        "carries is authority nobody needed and a stolen token could spend."
    )


def test_the_ceiling_is_empty_rather_than_merely_small(policy):
    assert policy.ceiling == frozenset()
    assert policy.raisable == frozenset()


def test_no_raise_can_widen_it(policy):
    # A raise widens only the transport side, and only within `raisable`.
    # With `raisable` empty there is nothing for a raise to reach, so a
    # compromised raise endpoint cannot turn this listener into a live one.
    assert effective(ALL_CAPABILITIES, policy, ALL_CAPABILITIES) == frozenset()


def test_it_administers_and_mints_nothing(policy):
    assert policy.admin is False, "the extension listener may not manage devices"
    assert policy.pairable is False, (
        "the extension listener offers pairing. Its credential is minted out of "
        "band by browser_extension_setup and pasted into the popup, so a pairing "
        "route here is a credential-minting surface nobody needs."
    )
    assert policy.allow_bearer is False, (
        "the extension listener accepts a bearer header. The WebSocket endpoint "
        "checks the protocol's own token in the hello frame; consulting the API's "
        "bearer machinery as well means two auth systems half-sharing a door."
    )


def test_every_field_is_stated_explicitly(policy):
    # The existing policies argue at length that omission must never mean
    # inheritance. Asserted rather than trusted: a field left to a default is a
    # decision nobody made.
    for field_name in ("name", "admin", "allow_bearer", "secure_cookie",
                       "ceiling", "raisable", "pairable"):
        assert hasattr(policy, field_name), f"{field_name} is missing"
    assert policy.name == POLICY_NAME


# ─── The port ────────────────────────────────────────────────────────────


def test_the_port_is_registered_at_the_slot_quick_vacated():
    assert LISTENER_OFFSETS[POLICY_NAME] == 3
    base = config.STUDIO_API_PORT
    assert policy_name_for_port(port_for(POLICY_NAME, base), base) == POLICY_NAME


def test_an_unregistered_port_still_resolves_to_nothing():
    # The fail-closed rule does not weaken because a fourth listener arrived.
    base = config.STUDIO_API_PORT
    unused = max(LISTENER_OFFSETS.values()) + 1
    assert policy_name_for_port(base + unused, base) is None


def test_every_policy_has_a_port_and_every_port_a_policy():
    # `listeners.py` declares the map as a literal precisely so a policy added
    # without a port fails loudly instead of inheriting one.
    assert set(LISTENER_OFFSETS) == set(POLICIES), (
        "POLICIES and LISTENER_OFFSETS disagree. A policy with no port is a "
        "listener nobody gave a socket; a port with no policy grants nothing "
        "and will refuse every request for a reason nobody wrote down."
    )


def test_no_two_listeners_share_a_port():
    offsets = list(LISTENER_OFFSETS.values())
    assert len(offsets) == len(set(offsets)), (
        f"two listeners share an offset: {LISTENER_OFFSETS}. Policy is resolved "
        f"from the accepting port, so a shared port means one listener silently "
        f"inherits the other's authority — which is KI-17 exactly."
    )


def test_the_extension_port_is_not_the_local_port():
    # KI-17's shape: anything that resolves to `local` gets admin, bearer, and a
    # ceiling holding EXECUTE and SYSTEM_CONTROL.
    base = config.STUDIO_API_PORT
    assert port_for(POLICY_NAME, base) != port_for("local", base)


# ─── The listener is actually bound ──────────────────────────────────────
#
# Everything above this line passed while nothing listened on the port.
#
# `serve()` bound only `local` and built the app with
# `listener_policies={port: "local"}`. The `/latch` route existed, the policy
# existed, the offset existed, and every unit test constructed its own app with
# `{8790: "extension"}` by hand -- so the one thing nobody asserted was that the
# daemon does the same. The extension sat in "Disconnected. Retrying shortly."
# for a reason no test could see.


def test_serve_registers_the_extension_port_in_listener_policies():
    """A port absent from `listener_policies` grants nothing and refuses
    everything, so registering it matters as much as binding the socket.

    Read out of the `listener_policies=` argument itself, not by searching the
    function for the word "extension". The first version of this test did the
    latter and passed against the bug it was written for, because
    `port_for("extension", port)` contains that word too. A structural test that
    matches a substring matches the wrong thing eventually.
    """
    import ast
    import inspect

    from assistant.io.api import server

    tree = ast.parse(inspect.getsource(server.serve).lstrip())

    mappings = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and getattr(node.func, "id", None) == "create_app"):
            continue
        for kw in node.keywords:
            if kw.arg == "listener_policies" and isinstance(kw.value, ast.Dict):
                mappings.append([
                    v.value for v in kw.value.values if isinstance(v, ast.Constant)
                ])

    assert mappings, "serve() does not pass a literal listener_policies dict"
    names = mappings[0]
    assert "local" in names, f"the local listener is missing: {names}"
    assert "extension" in names, (
        f"listener_policies maps {names}. The route, the policy and the port "
        f"offset can all be correct while the daemon registers only `local` -- "
        f"which is exactly what shipped, and the extension sat on 'Retrying "
        f"shortly' with nothing in any log."
    )


def test_serve_binds_a_second_socket():
    import ast
    import inspect

    from assistant.io.api import server

    tree = ast.parse(inspect.getsource(server.serve).lstrip())
    binds = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "bind_listener"
    ]
    assert len(binds) >= 2, (
        f"serve() calls bind_listener {len(binds)} time(s). The extension "
        f"listener needs its own socket; the policy alone binds nothing."
    )


def test_the_extension_listener_is_not_allowed_to_take_the_daemon_down():
    """A browser driver that cannot bind must not stop TENKA answering.

    Asserted on the source rather than by standing up a daemon with the port
    already taken -- that is a slow test, and a slow test is one that stops
    being run.
    """
    import inspect

    from assistant.io.api import server

    source = inspect.getsource(server.serve)
    marker = "extension_sock = bind_listener"
    assert marker in source, "the extension bind moved; this guard needs updating"

    before = source[: source.index(marker)]
    after = source[source.index(marker) :]

    assert before.rstrip().endswith("try:"), (
        "the extension bind is not the first statement of a try block. A port "
        "collision there would propagate and stop the Studio daemon starting "
        "at all -- so a browser driver failing would take the assistant with it."
    )
    assert "logger.warning" in after, (
        "a failed extension bind is swallowed without a word. The symptom is a "
        "browser extension stuck on 'Retrying shortly' and nothing in the log."
    )
