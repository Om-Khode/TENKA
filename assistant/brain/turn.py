"""The one place a turn's authority is installed.

Four sites install the three authority contextvars today -- `main.py` and three
background runners -- and each one repeats the sequence by hand. Repetition is
not the problem; **disagreement** is. `main.py` installs grants *last*, with an
argument written into the source:

    Set LAST, immediately before the `try` whose `finally` resets it [...] An
    adversarial review found this three statements higher up: a raise anywhere
    in that window skipped the reset entirely and left the grant set installed
    in the queue consumer's context after the turn ended. The documented
    fail-closed property inverts there -- whatever ran next inherited the last
    turn's grants instead of none.

`scheduler.py` installs grants **first**. Same three calls, opposite order, so
it has the arrangement `main.py` was fixed for: if `set_principal` or
`set_raise_context` raised, the grant set would outlive the task with no reset.
Neither call does anything that plausibly raises, which is exactly why it
survived review twice -- and is the same argument that was rejected in
`main.py`.

So this is not a refactor for tidiness. It is one implementation of an ordering
that had two, one of which was wrong.

**The order, and why each position is what it is.**

    principal       first: a raise here strands an identity, and an identity
                    with no grants can do nothing. The reverse would strand
                    privilege.
    raise context   second: it has no fail-closed property of its own -- an
                    absent one only ever makes a refusal sentence vaguer.
    grants          LAST, with nothing between it and the `try`.

Resets run in reverse, grants first, for the same reason in mirror: the window
where grants are installed and unguarded is as short as it can be made.

**What this module deliberately does not do.** It does not dispatch, classify,
plan or decide anything. It installs authority, runs what it was handed, and
resets. `brain/authority.py` and `brain/task.py` are not imported here, and
`brain/__init__.py` imports neither -- so a background runner that imports
`run_turn` gains no path to the resume machinery. That is load-bearing; see
`tests/test_brain_authority.py`'s A5 test, whose enforcement narrowed from "the
background runners import nothing from brain" to "they import nothing from
brain that can construct or resume a Task" when this module arrived.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Optional

if TYPE_CHECKING:
    from ..core.capabilities import Capability

logger = logging.getLogger("brain.turn")


async def run_turn(
    *,
    grants: "frozenset[Capability]",
    principal: Optional[str],
    raise_context: Any,
    work: Callable[[], Awaitable[Any]],
    label: str = "turn",
) -> Any:
    """Install the turn's authority, run `work()`, and always reset.

    Keyword-only throughout. Three of the four arguments are security inputs
    and two of them are `frozenset`s and strings that would read identically in
    either position -- a positional call site that swapped principal and label
    would type-check, run, and attribute the turn to the wrong caller.

    `work` is a zero-argument callable rather than a coroutine, so it is created
    *inside* the installed context. Passing an already-created coroutine would
    work by accident here (a coroutine body does not begin until awaited) and
    break the moment a caller wrapped it in a task -- which copies the context
    at creation, not at await.

    Returns whatever `work()` returns. Exceptions propagate unchanged: this
    swallows nothing, because a background runner deciding for itself that a
    failed turn is fine is how a silent failure gets its silence.
    """
    from ..actions import (
        current_grants, current_principal, current_raise_context,
        set_grants, set_principal, set_raise_context,
    )

    # Order is the contract. See the module docstring for why each position is
    # where it is, and `main.py`'s own comment for the review that found it.
    principal_token = set_principal(principal)
    raise_token = set_raise_context(raise_context)
    # Grants LAST. Nothing may be added between this line and the `try`.
    grants_token = set_grants(grants)

    try:
        return await work()
    finally:
        # Reverse order, grants first: the window in which grants are installed
        # is closed before anything else is touched.
        current_grants.reset(grants_token)
        current_raise_context.reset(raise_token)
        current_principal.reset(principal_token)
        logger.debug(f"[BRAIN] {label}: authority reset")
