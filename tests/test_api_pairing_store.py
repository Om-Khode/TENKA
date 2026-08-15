# tests/test_api_pairing_store.py
"""Tests for the in-memory pair-code store (assistant/io/api/pairing.py).

A pair code is a short-lived, single-use credential that carries the
capabilities a new device will receive when it scans the pairing QR. These
tests pin down the load-bearing properties: unambiguous alphabet, exactly-once
consumption, monotonic-clock expiry, single live code, and no raising on
untrusted input.
"""
from __future__ import annotations

import hmac
import threading

import pytest

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


# ─── Fix round 1 additions ───────────────────────────────────────────────

def test_repr_never_reveals_the_code():
    """A dataclass's default __repr__ prints every field. Without
    `field(repr=False)` on `code`, an f-string, an uncaught exception, or
    pytest's own assertion-rewrite on a failing `==` would put a live pair
    code straight into a log or terminal."""
    pair_code = PairCodeStore().mint("phone", frozenset({Capability.CHAT}))
    rendered = repr(pair_code)
    assert pair_code.code not in rendered
    assert "phone" in rendered                # non-secret fields still show


def test_mint_refuses_an_empty_grant_set():
    """Mirrors TokenVault.issue(): a zero-grant credential can still
    authenticate, turning any route gated by authentication alone into an
    oracle. Refused at the source, not surfaced from inside issue() later."""
    with pytest.raises(ValueError):
        PairCodeStore().mint("phone", frozenset())


def test_the_critical_section_is_one_unbroken_lock_hold(monkeypatch):
    """A barrier only synchronises when threads *start* -- CPython's GIL
    serialises consume()'s few bytecodes so tightly that a plain N-thread
    race passes whether or not the lock is actually held across the whole
    read-compare-clear sequence. This test forces the interleaving instead
    of hoping for it, by making the deepest point in that sequence --
    `hmac.compare_digest` -- pause and wait for a second concurrent caller
    to reach the same point.

    If `consume()` holds one lock across the entire read-compare-clear
    sequence (the correct implementation), a second concurrent call cannot
    get anywhere near `compare_digest` until the first has already cleared
    the slot and released the lock -- so the wait below times out, and
    exactly one call succeeds. If the critical section is split into
    separate `with self._lock:` blocks around the read, the compare, and
    the clear (the regression this guards against), both threads reach
    `compare_digest` while the slot is still populated, and both succeed.

    Verified this actually catches that regression: temporarily splitting
    `consume()`'s single `with self._lock:` into three separate
    acquire/release blocks (read, compare, clear) makes this test fail with
    2 successes instead of 1; restoring the single block makes it pass
    again. See the fix-round-2 entry in task-4-report.md for the transcript.
    """
    store = PairCodeStore()
    code = store.mint("phone", frozenset({Capability.CHAT})).code

    order_lock = threading.Lock()
    calls: list[int] = []
    second_arrived = threading.Event()
    real_compare_digest = hmac.compare_digest

    def racing_compare_digest(a: bytes, b: bytes) -> bool:
        with order_lock:
            is_first = not calls
            calls.append(1)
        if is_first:
            # Bounded, not indefinite: a correct implementation never lets a
            # second caller arrive here at all (it blocks on the lock
            # instead), so timing out is the expected outcome for the
            # correct code, not a flake.
            second_arrived.wait(timeout=0.3)
        else:
            second_arrived.set()
        return real_compare_digest(a, b)

    monkeypatch.setattr(hmac, "compare_digest", racing_compare_digest)

    results: list[object] = [None, None]
    barrier = threading.Barrier(2)

    def attempt(i: int) -> None:
        barrier.wait()
        results[i] = store.consume(code)

    threads = [threading.Thread(target=attempt, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    successes = [r for r in results if r is not None]
    assert len(successes) == 1


def test_a_wrong_guess_does_not_burn_the_live_code():
    """Only a matching code may clear the slot. Otherwise one garbage guess
    -- accidental or an attacker fishing -- would kill a legitimate pairing
    session before its owner ever gets to redeem it."""
    store = PairCodeStore()
    code = store.mint("phone", frozenset({Capability.CHAT})).code
    assert store.consume("WRONG-CODE") is None
    assert store.consume(code) is not None


def test_consume_rejects_non_str_and_oversized_input_without_raising():
    """The brief's untrusted-input test only covers malformed strings.
    `consume` also has to survive a caller passing the wrong type entirely,
    and a wire value with no realistic relation to a 9-character code."""
    store = PairCodeStore()
    store.mint("phone", frozenset({Capability.CHAT}))
    for bad in (None, b"7K2M-9QX4", 12345, "X" * 10_000):
        assert store.consume(bad) is None
