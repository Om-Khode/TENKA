"""Who is driving the turn in flight.

`core/capabilities.py` answers *what* a caller may do. This answers *who* the
caller is, and they are genuinely different questions: 6a.5 asked only the
first, and that is precisely why a paired device legitimately holding `FILES`
could answer a file confirmation the person at the keyboard had armed. The
device was allowed to delete files. It just was not the one that had been
asked. See KI-13.

Lives in `core/` rather than beside `current_grants` in `actions/__init__.py`
for one structural reason: `pending.py` has to read it at arm time, and domain
code importing `actions/` would invert the layering the whole tree is built on.
`actions/__init__.py` re-exports all three names, so the two contextvars still
read as the pair they are at every call site that installs them.
"""
from __future__ import annotations

import contextvars

current_principal: contextvars.ContextVar["str | None"] = \
    contextvars.ContextVar("tenka_current_principal", default=None)
"""Who the caller driving this turn *is*.

A contextvar rather than a parameter for exactly the reason `current_grants`
is one: `execute()` re-enters itself through `planner/executor.py`, pending
states are armed several frames below the turn that authorised them, and
threading an identity through every handler signature would mean every future
arming site is one forgotten argument away from arming a confirmation that
belongs to nobody. A contextvar is inherited by nested tasks automatically.

**The default is `None`, and `None` owns nothing.** Not "any owner" and not
"the local one" -- the absence of a decision is not a decision to allow, the
same way an unset `current_grants` refuses rather than permits. A turn that
reaches a pending answer site without anyone setting this cannot answer
anything, and that is the safe behaviour for that bug.
"""

LOCAL_PRINCIPAL: str = "local"
"""The identity of a caller physically at this machine.

Voice, the console, and the background automation runners (the scheduler and
the event bus) all state it explicitly, the same way they state `LOCAL_GRANTS`.
Device principals are spelled `f"device:{device_id}"` at the route that knows
which device authenticated, so the two namespaces cannot collide and no device
can name itself `"local"` -- the string is built from a prefix this side owns,
never from anything a caller sends.

It is never what an unset contextvar falls back to; see `current_principal`.
"""


def set_principal(principal: "str | None") -> "contextvars.Token":
    """Declare who the turn about to run *is*. Reset the token when it ends.

    Deliberately a plain setter with no default, matching `set_grants`: every
    call site has to name a principal, so "which caller is this?" is answered
    once, visibly, at the place that knows. `None` is a legal argument and
    means "nobody said" -- it owns nothing, which is the fail-closed answer.
    """
    return current_principal.set(principal)
