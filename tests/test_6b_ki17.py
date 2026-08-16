"""KI-17 — a 6b tunnel pointed at the existing port inherits `local`.

The handoff's §6 says it plainly: 6b's load-bearing assumption is 'policy is
keyed on the accepting port, so a tunnel cannot choose its own policy.' That
assumption is true, and KI-17 is exactly how it fails anyway. These tests are
written before the feature for that reason.
"""
import pytest

from assistant.io.api.listeners import (
    LISTENER_OFFSETS, local_port, policy_name_for_port, port_for,
)
from assistant.io.api.policy import POLICIES

BASE = 8787


def test_every_policy_has_a_port_and_no_two_share_one():
    ports = {name: port_for(name, BASE) for name in POLICIES}
    assert set(ports) == set(POLICIES)
    assert len(set(ports.values())) == len(ports), ports


def test_no_transport_shares_the_port_local_holds():
    for name in POLICIES:
        if name == "local":
            continue
        assert port_for(name, BASE) != local_port(BASE), name


def test_the_port_map_covers_exactly_the_declared_policies():
    """A policy added without a port, or a port without a policy, is a
    listener nobody declared -- which `policy_for_port` answers 401 to, but
    only after somebody has already bound the socket."""
    assert set(LISTENER_OFFSETS) == set(POLICIES)


def test_a_port_resolves_back_to_its_own_policy_name():
    for name in POLICIES:
        assert policy_name_for_port(port_for(name, BASE), BASE) == name


def test_an_unmapped_port_resolves_to_nothing():
    assert policy_name_for_port(BASE + 99, BASE) is None
    assert policy_name_for_port(BASE - 1, BASE) is None
