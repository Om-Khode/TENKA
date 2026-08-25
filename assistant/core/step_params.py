"""How a step's fields become an adapter's parameters. One implementation.

P5 wanted `Executor.run(step) -> Verdict` in `brain/`, and named
`actions/planner/executor.py` as a file that would use it. Those two cannot
both happen: `brain` sits *above* `actions`, there is not one `actions -> brain`
import in the tree, and adding one would be the fifth layer inversion -- the
same shape P4b removed from `automation/event_bus.py` one phase earlier.

The part that actually needs sharing is small and pure, so it lives here, below
both. `brain/executor.py` calls it; the planner's own step runner calls it; and
the rule about pinned values has one implementation rather than one per layer,
which is the whole reason the rule keeps being broken.

**The rule.** Structure decides, constraints are hard, the string follows:

    parameters   what the step says to do, as fields
    constraints  what the user pinned, applied last and verbatim
    goal         the sentence, for adapters that still parse one -- and it
                 never overwrites a structured field of the same name

The middle line is `CLAUDE.md`'s gotcha. "mobile as 99999" is a HARD
constraint: not rounded, not normalised, not replaced by a value some adapter
found more plausible. Applying constraints last is what makes that true
whatever `parameters` holds, and copying rather than referencing is what stops
an adapter that normalises in place from rewriting the pin for every later step
in the same plan.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional


def build_step_params(
    parameters: Optional[Mapping[str, Any]] = None,
    constraints: Optional[Mapping[str, Any]] = None,
    goal: str = "",
    *,
    extra_constraints: Optional[Mapping[str, Any]] = None,
    goal_key: str,
) -> dict:
    """Merge one step's fields into the dict an adapter receives.

    `extra_constraints` are the enclosing Task's, which apply to every step in
    it without each step having to repeat them. They are applied after the
    step's own for the same reason the step's are applied after `parameters`:
    the more specific pin should be the one that is hardest to lose, and a
    caller that pinned something for the whole task meant it for this step too.

    `goal_key` exists because the planner's tool manifest already chooses which
    parameter carries the instruction -- `url` for `open_browser`, `query` for
    `web_search` -- and that choice is the adapter's contract, not something to
    re-derive here. `setdefault` rather than assignment: a step that states the
    field explicitly has said something more precise than the sentence did, and
    the sentence must not overwrite it.

    It is **required**, with no default. It had one -- `"goal"` -- and a
    mutation that changed it to nonsense turned nothing red, because every
    caller states it and the default was reachable only by a caller who had not
    thought about it. That is precisely the caller this rule exists for: an
    unstated goal key is how the sentence became the universal meaning in the
    first place, and a silent fallback here would quietly restore it for one
    adapter at a time.
    """
    params: dict = dict(parameters or {})

    for source in (constraints or {}, extra_constraints or {}):
        for key, value in source.items():
            params[key] = value

    if goal:
        params.setdefault(goal_key, goal)
    return params
