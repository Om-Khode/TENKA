# tests/test_api_pairing_store.py
"""Tests for the in-memory pair-code store (assistant/io/api/pairing.py).

A pair code is a short-lived, single-use credential that carries the
capabilities a new device will receive when it scans the pairing QR. These
tests pin down the load-bearing properties: unambiguous alphabet, exactly-once
consumption, monotonic-clock expiry, single live code, and no raising on
untrusted input.
"""
from __future__ import annotations

from assistant.io.api.pairing import CODE_TTL_SECONDS, PairCodeStore, _ALPHABET
from assistant.io.api.vault import Capability


def test_a_minted_code_is_typeable_and_unambiguous():
    code = PairCodeStore().mint("phone", frozenset({Capability.CHAT})).code
    assert len(code) == 9 and code[4] == "-"
    assert all(c in _ALPHABET for c in code.replace("-", ""))
    assert not (set("ILOU") & set(code))     # misread as 1, 1, 0, V


def test_consume_works_exactly_once():
    store = PairCodeStore()
    code = store.mint("phone", frozenset({Capability.CHAT})).code
    assert store.consume(code) is not None
    assert store.consume(code) is None       # replay is indistinguishable from wrong


def test_an_expired_code_is_refused():
    store = PairCodeStore()
    code = store.mint("phone", frozenset({Capability.CHAT}), now=0.0).code
    assert store.consume(code, now=CODE_TTL_SECONDS + 0.01) is None


def test_a_code_is_alive_right_up_to_its_ttl():
    store = PairCodeStore()
    code = store.mint("phone", frozenset({Capability.CHAT}), now=0.0).code
    assert store.consume(code, now=CODE_TTL_SECONDS - 0.01) is not None


def test_minting_invalidates_the_previous_code():
    """At most one live code. Otherwise a forgotten QR screen from an hour ago
    is still a working credential path."""
    store = PairCodeStore()
    first = store.mint("phone", frozenset({Capability.CHAT})).code
    store.mint("laptop", frozenset({Capability.CHAT}))
    assert store.consume(first) is None


def test_grants_travel_with_the_code():
    store = PairCodeStore()
    grants = frozenset({Capability.CHAT, Capability.FILES})
    code = store.mint("phone", grants).code
    assert store.consume(code).grants == grants


def test_wrong_code_returns_none_without_raising():
    store = PairCodeStore()
    store.mint("phone", frozenset({Capability.CHAT}))
    for bad in ("", "   ", "AAAA-AAAA", "nope", "7K2M9QX4", "\ud800"):
        assert store.consume(bad) is None


def test_current_reports_nothing_once_expired():
    store = PairCodeStore()
    store.mint("phone", frozenset({Capability.CHAT}), now=0.0)
    assert store.current(now=0.0) is not None
    assert store.current(now=CODE_TTL_SECONDS + 1) is None


def test_codes_do_not_repeat():
    store = PairCodeStore()
    seen = {store.mint("p", frozenset({Capability.CHAT})).code for _ in range(200)}
    assert len(seen) == 200
