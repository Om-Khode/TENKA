"""`_refuse`'s three sentences, and the property that must never move under
them: `capability_refusal()`'s pass/fail decision is a function of
`current_grants` alone, never of `current_raise_context`. The live-test defect
this milestone fixes was a wrong *sentence*, not a wrong *decision* -- a
tailnet-paired device correctly refused EXECUTE, then told to re-pair, which
cannot fix it, instead of told to raise, which can.

Three states, told apart by what the device was issued and what this
transport's `raisable` literal holds -- neither of which lives on
`current_grants`, which is already the narrowed, post-ceiling set:

1. Never issued the capability at all. The original sentence, byte-identical.
2. Issued, narrowed away here, but raisable on this transport: must mention
   raising and must not say the device lacks it.
3. Issued, but this transport can never carry it: must not promise a raise.
"""
from __future__ import annotations

import pytest

from assistant import actions
from assistant.core.capabilities import Capability

_ORIGINAL_SENTENCE = (
    "That needs the execute permission, which this device doesn't have."
)


def _grants_token(granted):
    return actions.set_grants(frozenset(granted))


def _raise_token(context):
    return actions.set_raise_context(context)


# ─── State 1: never issued ────────────────────────────────────────────────
def test_a_never_issued_capability_gets_the_original_sentence():
    """No RaiseContext installed at all -- the call sites that predate this
    fix, or a source nobody has wired it for. Must degrade to exactly the old
    sentence, not a new one."""
    gtoken = _grants_token({Capability.CHAT_SEND})
    try:
        result = actions._refuse(Capability.EXECUTE)
    finally:
        actions.current_grants.reset(gtoken)
    assert result == _ORIGINAL_SENTENCE


def test_a_never_issued_capability_gets_the_original_sentence_even_with_context():
    """A RaiseContext *is* installed, but EXECUTE isn't in `issued` -- the
    device was simply never granted it, on any transport, ever. Still the
    original sentence: raisable must not matter when issued doesn't hold."""
    gtoken = _grants_token({Capability.CHAT_SEND})
    rtoken = _raise_token(actions.RaiseContext(
        issued=frozenset({Capability.CHAT_SEND}),
        raisable=frozenset({Capability.EXECUTE}),
        ceiling=frozenset({Capability.CHAT_SEND}),
    ))
    try:
        result = actions._refuse(Capability.EXECUTE)
    finally:
        actions.current_raise_context.reset(rtoken)
        actions.current_grants.reset(gtoken)
    assert result == _ORIGINAL_SENTENCE


# ─── State 2: issued, ceilinged here, raisable here ───────────────────────
def test_an_issued_ceilinged_raisable_capability_points_at_a_raise():
    gtoken = _grants_token({Capability.CHAT_SEND})
    rtoken = _raise_token(actions.RaiseContext(
        issued=frozenset({Capability.CHAT_SEND, Capability.EXECUTE}),
        raisable=frozenset({Capability.EXECUTE, Capability.SYSTEM_CONTROL}),
        ceiling=frozenset({Capability.CHAT_SEND}),
    ))
    try:
        result = actions._refuse(Capability.EXECUTE)
    finally:
        actions.current_raise_context.reset(rtoken)
        actions.current_grants.reset(gtoken)
    assert "raise" in result.lower()
    # Must not claim the device lacks it -- it does hold it, just not here.
    assert "doesn't have" not in result
    assert result != _ORIGINAL_SENTENCE


# ─── State 3: issued, never raisable on this transport ────────────────────
def test_an_issued_never_raisable_capability_does_not_promise_a_raise():
    gtoken = _grants_token({Capability.CHAT_SEND})
    rtoken = _raise_token(actions.RaiseContext(
        issued=frozenset({Capability.CHAT_SEND, Capability.EXECUTE}),
        raisable=frozenset(),  # nothing raisable on this transport at all
        ceiling=frozenset({Capability.CHAT_SEND}),
    ))
    try:
        result = actions._refuse(Capability.EXECUTE)
    finally:
        actions.current_raise_context.reset(rtoken)
        actions.current_grants.reset(gtoken)
    assert "raise" not in result.lower()
    assert "doesn't have" not in result
    assert result != _ORIGINAL_SENTENCE


# ─── Shape: every sentence stays speakable ────────────────────────────────
@pytest.mark.parametrize("context", [
    None,
    # `ceiling` is required rather than defaulted: it is what
    # `durable_capability_refusal` reads, and a default would let a call site
    # that forgot it answer "holds nothing durably" and refuse the operator.
    # These two carry CHAT_SEND only, matching the tunnel ceilings that make
    # the second and third refusal sentences reachable at all.
    actions.RaiseContext(issued=frozenset({Capability.CHAT_SEND, Capability.SYSTEM_CONTROL}),
                         raisable=frozenset({Capability.SYSTEM_CONTROL}),
                         ceiling=frozenset({Capability.CHAT_SEND})),
    actions.RaiseContext(issued=frozenset({Capability.CHAT_SEND, Capability.SYSTEM_CONTROL}),
                         raisable=frozenset(),
                         ceiling=frozenset({Capability.CHAT_SEND})),
])
def test_every_sentence_stays_speakable(context):
    """Under 120 chars, no path separator, no digits-as-error-code -- it may
    reach tts.speak(). Checked against SYSTEM_CONTROL, the longest capability
    name, so the length bound is checked at its worst case."""
    gtoken = _grants_token({Capability.CHAT_SEND})
    rtoken = _raise_token(context)
    try:
        result = actions._refuse(Capability.SYSTEM_CONTROL)
    finally:
        actions.current_raise_context.reset(rtoken)
        actions.current_grants.reset(gtoken)
    assert len(result) < 120, result
    assert "\\" not in result and "/" not in result, result
    assert not any(ch.isdigit() for ch in result), result


# ─── The property that must not move: capability_refusal()'s decision ─────
@pytest.mark.parametrize("context", [
    None,
    actions.RaiseContext(issued=frozenset({Capability.CHAT_SEND}), raisable=frozenset(),
                         ceiling=frozenset({Capability.CHAT_SEND})),
    actions.RaiseContext(issued=frozenset({Capability.CHAT_SEND, Capability.EXECUTE}),
                         raisable=frozenset({Capability.EXECUTE}),
                         ceiling=frozenset({Capability.CHAT_SEND})),
    actions.RaiseContext(issued=frozenset({Capability.CHAT_SEND, Capability.EXECUTE}),
                         raisable=frozenset(),
                         ceiling=frozenset({Capability.CHAT_SEND})),
])
def test_capability_refusal_still_refuses_regardless_of_raise_context(context):
    """The decision is a function of current_grants alone. Four different
    raise contexts, same missing grant: all four must still refuse."""
    gtoken = _grants_token({Capability.CHAT_SEND})
    rtoken = _raise_token(context)
    try:
        result = actions.capability_refusal(Capability.EXECUTE)
    finally:
        actions.current_raise_context.reset(rtoken)
        actions.current_grants.reset(gtoken)
    assert result is not None


@pytest.mark.parametrize("context", [
    None,
    actions.RaiseContext(issued=frozenset({Capability.CHAT_SEND, Capability.EXECUTE}),
                         raisable=frozenset(),
                         ceiling=frozenset({Capability.CHAT_SEND})),
])
def test_capability_refusal_still_allows_regardless_of_raise_context(context):
    """The mirror image: when the capability IS in current_grants, no raise
    context content can turn that into a refusal."""
    gtoken = _grants_token({Capability.CHAT_SEND, Capability.EXECUTE})
    rtoken = _raise_token(context)
    try:
        result = actions.capability_refusal(Capability.EXECUTE)
    finally:
        actions.current_raise_context.reset(rtoken)
        actions.current_grants.reset(gtoken)
    assert result is None


# ─── Every set_grants() call site installs the raise context too ──────────
def test_local_sources_get_a_local_raise_context():
    """A local caller's raise context: issued everything, nothing left to
    raise. `_grants_for_item`'s local branch and this one must agree that a
    local source never actually reaches the raisable/never-raisable
    branches -- LOCAL_GRANTS already holds everything."""
    assert actions.LOCAL_RAISE_CONTEXT.issued == actions.LOCAL_GRANTS
    assert actions.LOCAL_RAISE_CONTEXT.raisable == frozenset()


def test_a_studio_queue_item_without_a_fifth_slot_degrades_to_none():
    """main._raise_context_for_item must not crash or invent a context for a
    'studio' item enqueued by a caller that never supplied issued/raisable --
    it degrades to None, which _refuse then reads as 'no context installed'."""
    from assistant import main as main_mod
    assert main_mod._raise_context_for_item(("studio", "hi", frozenset(), "device:x")) is None


def test_a_studio_queue_item_carries_its_raise_context():
    from assistant import main as main_mod
    context = actions.RaiseContext(issued=frozenset({Capability.EXECUTE}),
                                   raisable=frozenset({Capability.EXECUTE}),
                                   ceiling=frozenset({Capability.EXECUTE}))
    item = ("studio", "hi", frozenset(), "device:x", context)
    assert main_mod._raise_context_for_item(item) is context


def test_a_local_queued_item_gets_the_local_raise_context():
    from assistant import main as main_mod
    assert main_mod._raise_context_for_item(("stt", "hi", 100)) is actions.LOCAL_RAISE_CONTEXT
    assert main_mod._raise_context_for_item(("chat", "hi")) is actions.LOCAL_RAISE_CONTEXT
