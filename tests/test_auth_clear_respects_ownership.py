"""The planner's auth-failure teardown cannot discard someone else's state.

KI-27, closed. `planner/executor.py` cleared a pending state through a
loop-local — `state = pending_registry.get(name); state.clear()` — and both
pending arm/clear AST sweeps match on the **receiver's name**, so this one site
was invisible to them while every other clear in the tree was covered. It was
reasoned safe (same-request teardown of a state the same call armed), but the
sweeps exist to promise that a new unguarded clear cannot ship silently, and
this was the one shape where that promise did not hold.

Two halves, tested in two places:

- the **sweep** now follows a name back to `pending_registry.get(...)`, so the
  shape is visible whatever the variable is called — pinned in
  `tests/test_6b_principal.py` against synthetic source, because once this fix
  lands the tree contains no example of it to walk.
- the **site** now goes through `pending.try_clear`, so a state owned by
  another principal survives — pinned here, on the real function.

Why the site test matters separately: the sweep only proves the *shape* would
be noticed. It says nothing about whether this particular clear honours
ownership. A fix that satisfied the sweep by renaming the variable to
`pending_thing` would pass every assertion over there.

Run with:  py -3.11 -m pytest tests/test_auth_clear_respects_ownership.py -v
"""
import contextlib
import pathlib
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from assistant.actions.planner.executor import _clear_auth_pending_states  # noqa: E402
from assistant.pending import PendingState, pending_registry  # noqa: E402


_NAME = "pending_test_auth_clear"


@pytest.fixture()
def state():
    """A real registered PendingState, removed again afterwards.

    Registered rather than mocked: `_clear_auth_pending_states` looks the state
    up through `pending_registry.get`, and the registry is a process-wide
    singleton — a mock would not be found, and the test would pass by walking
    nothing.
    """
    st = PendingState(_NAME, timeout=300.0)
    pending_registry.register(st)
    try:
        yield st
    finally:
        st.clear()
        pending_registry._states.pop(_NAME, None)


def _armed_by(state, principal):
    """Arm `state` on behalf of `principal`, bypassing the ambient default.

    `try_arm`'s principal argument defaults to the ambient one; passing it
    explicitly is what lets these tests set up a state owned by somebody who is
    not the caller doing the clearing.
    """
    from assistant.pending import try_arm
    assert try_arm(state, {"goal": "x"}, principal=principal), (
        "arming the fixture state failed -- the test would prove nothing"
    )


@contextlib.contextmanager
def _as(principal):
    """Run the block as `principal`.

    `_clear_auth_pending_states` takes no principal argument -- it reads the
    ambient one, exactly as the real turn does. So *who is clearing* is set
    here, and it is the whole variable under test. The first draft of the
    owner-case test omitted this and failed: the state was armed by
    `device:phone` while the ambient principal was still unset, so `try_clear`
    refused. Correct behaviour, wrong setup -- worth recording, because a test
    that had asserted only the refusals would have called that a pass.
    """
    import assistant.core.principal as principal_mod
    token = principal_mod.current_principal.set(principal)
    try:
        yield
    finally:
        principal_mod.current_principal.reset(token)


# ─── the permitted path ──────────────────────────────────────────────────────

def test_the_owner_can_clear_its_own_auth_state(state):
    """**The answer, not the refusal**, and first for that reason. A teardown
    that refused everything would satisfy the ownership test below while
    leaving an OAuth prompt armed after every failed auth step — so the next
    unrelated utterance gets read as an answer to it."""
    _armed_by(state, "device:phone")

    with _as("device:phone"):
        cleared = _clear_auth_pending_states({_NAME: False})
    assert cleared == [_NAME], f"the owner's own state was not cleared: {cleared}"
    assert not state.active


def test_a_state_already_armed_before_the_step_is_left_alone(state):
    """Only states this step armed are torn down. Something already active
    belonged to a conversation the planner is not part of, and clearing it is
    the denial of service `try_clear` exists to stop — reached here through the
    snapshot rather than through ownership."""
    _armed_by(state, "device:phone")

    with _as("device:phone"):          # the OWNER, so only the snapshot refuses
        cleared = _clear_auth_pending_states({_NAME: True})
    assert cleared == []
    assert state.active, "a pre-existing pending state was torn down"


# ─── the refusal ─────────────────────────────────────────────────────────────

def test_a_foreign_principals_state_survives(state, monkeypatch):
    """KI-27's actual hazard. The old bare `.clear()` discarded whatever it
    found; the operator's open confirmation would vanish with nothing to
    explain why."""
    _armed_by(state, "device:phone")
    with _as("local"):                 # the clearing caller is somebody else
        cleared = _clear_auth_pending_states({_NAME: False})

    assert cleared == [], f"a foreign principal's state was cleared: {cleared}"
    assert state.active, (
        "the state was torn down by a principal that does not own it -- this is "
        "the discard KI-27 was filed for"
    )


def test_the_refused_attempt_is_parked_for_the_owner(state):
    """Refusing silently would leave the owner with an unexplained delay.
    `try_clear` parks the attempt the same way a foreign answer or a foreign
    arm is parked, and the owner reads it on her next answer."""
    _armed_by(state, "device:phone")
    with _as("local"):
        _clear_auth_pending_states({_NAME: False})

    assert state.take_foreign_attempts() > 0, (
        "the refused clear left no trace, so the owner is never told that "
        "something else reached for her open question"
    )


def test_an_unset_principal_clears_nothing(state):
    """Fail closed. An unset principal owns nothing, in both directions —
    the same default that makes `current_grants=None` refuse everything."""
    _armed_by(state, "device:phone")
    with _as(None):
        assert _clear_auth_pending_states({_NAME: False}) == []
    assert state.active


# ─── the site keeps using the mechanism ──────────────────────────────────────

def test_the_executor_does_not_reintroduce_a_bare_clear():
    """Source-level, and narrow on purpose.

    `tests/test_6b_principal.py` sweeps the whole tree for this and would catch
    it too. This is here so the failure lands in the file that explains why:
    the sweep reports a location, and the argument for `try_clear` at this
    particular location is the docstring next to it.
    """
    import ast

    src = (_ROOT / "assistant" / "actions" / "planner" / "executor.py").read_text(
        encoding="utf-8")
    assert "try_clear" in src, "the auth teardown no longer uses try_clear"

    # AST, not a line scan. The first draft grepped for the literal
    # `state.clear()` and matched the sentence in `_clear_auth_pending_states`'
    # own docstring explaining what it replaced -- a test that fails on its own
    # explanation is a test nobody keeps.
    offenders = [
        f"line {node.lineno}"
        for node in ast.walk(ast.parse(src))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "clear"
        and isinstance(node.func.value, ast.Name)
    ]
    assert not offenders, (
        f"a bare pending-state clear is back in the planner executor: "
        f"{offenders}. Use `pending.try_clear(state)` -- see KI-27."
    )
