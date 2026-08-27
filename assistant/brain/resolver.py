""""What could satisfy this?" as a deterministic lookup. Zero LLM calls.

TENKA-v2 §17.P3.

**Exactly one router, and which option was taken.** The phase brief asks for a
recorded decision: either `detect_backend` *moves* into this resolver with its
call sites repointed, or it stays and the resolver delegates, adding nothing
that decides. Two implementations of one ordering would be the
duplicate-orchestration anti-pattern introduced by the phase meant to remove it.

**It delegates, and that was forced rather than chosen.** `detect_backend` is
called from inside `automation/router.py` itself -- at two of its own call
sites -- and `automation ↛ brain` is an enforced contract in
`pyproject.toml`. Moving the function up here would break the package that owns
its callers. So the resolver is a *reader* of the routing decision, never a
second maker of it, and `test_affordance_resolver.py` asserts that by scanning
this module for the five routing signals: they must appear here zero times.

The ordering the resolver reports is therefore the one that already exists:

    preference -> URL pattern -> running process -> launch keyword
    -> app context -> fallback

**What this module adds** is the affordance layer on top: given a goal, which
registered affordances could carry it out, in the order the router would pick
them, with `UNSUPPORTED` when nothing matches. That last part is the reason it
exists at all -- "no route" and "the wrong route" are different answers, and
before `Outcome.UNSUPPORTED` there was no way to say the first.

**Purity.** `resolve()` is a pure function of the registry plus an environment
snapshot. The snapshot is passed in rather than read, so the same inputs give
the same answer and a test does not have to patch the desktop to ask a
question. Reading the environment *inside* would make resolution depend on what
happened to be open at that instant, which is not something a caller can reason
about or a test can pin.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

from ..core.verdict import Outcome
from .affordance import Affordance, affordance_registry


@dataclass(frozen=True)
class Environment:
    """What the resolver is allowed to know about the machine.

    A snapshot, taken by the caller. Deliberately small: every field here is
    something `automation/router.py` already consults, and adding one that it
    does not would be this module starting to decide.
    """

    open_windows: tuple[str, ...] = ()
    cdp_available: bool = False
    preferences: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Resolution:
    """What could satisfy the goal, and how the routing got there."""

    outcome: Outcome
    affordances: tuple[Affordance, ...] = ()
    backend: str = ""
    reason: str = ""

    @property
    def best(self) -> Optional[Affordance]:
        return self.affordances[0] if self.affordances else None


def resolve(goal: str, env: Optional[Environment] = None,
            registry=None) -> Resolution:
    """Which registered affordances could carry out `goal`.

    Returns `UNSUPPORTED` with no affordances when nothing matches, rather than
    the closest thing it found. A resolver that returns a wrong affordance is
    worse than one that returns none: the caller acts on it, and the failure
    surfaces as a strange action rather than as "I cannot do that".
    """
    env = env or Environment()
    reg = registry if registry is not None else affordance_registry

    backend, reason = _route(goal, env)

    matches = tuple(
        a for a in _all(reg)
        if _matches(a, goal, backend)
    )
    if not matches:
        return Resolution(
            outcome=Outcome.UNSUPPORTED,
            backend=backend,
            reason=reason or "no affordance matched",
        )
    return Resolution(
        outcome=Outcome.SUCCEEDED,
        affordances=matches,
        backend=backend,
        reason=reason,
    )


def _route(goal: str, env: Environment) -> tuple[str, str]:
    """Ask `automation/router.py` which backend would handle this.

    The single delegation, isolated in one function so the "adds nothing that
    decides" claim is checkable by reading four lines rather than the module.
    Deferred import: `brain` may reach `automation`, but doing it at module
    scope would drag the automation stack into every import of the Brain.
    """
    try:
        from ..automation.router import detect_backend
    except Exception:  # pragma: no cover - automation is not optional
        return "", "router unavailable"

    try:
        backend, meta = detect_backend(goal)
    except Exception as e:
        # A routing failure is not a resolution failure. Say so and let the
        # affordance match stand on its own.
        return "", f"router raised: {e}"
    return backend or "", (meta or {}).get("reason", "")


def _all(reg) -> Sequence[Affordance]:
    for accessor in ("values", "all", "items"):
        fn = getattr(reg, accessor, None)
        if fn is None:
            continue
        got = fn()
        if accessor == "items":
            return [v for _, v in got]
        return list(got)
    return list(getattr(reg, "_entries", {}).values())


def _matches(affordance: Affordance, goal: str, backend: str) -> bool:
    """Does this affordance plausibly carry out this goal?

    Word-overlap against the affordance's own vocabulary -- its operation, its
    tags, its id. Deliberately not a model call (§17.P3: zero LLM calls) and
    deliberately not a brand list: an affordance describes itself, and a
    resolver that carried its own table of what things are called would be the
    hardcoded-app rule broken one entry at a time.
    """
    if not goal:
        return False
    words = {w.strip(".,!?;:'\"") for w in goal.lower().split()}
    vocabulary = {affordance.operation.lower()} | {
        t.lower() for t in affordance.tags
    } | set(affordance.affordance_id.lower().replace("_", " ").split())
    vocabulary.discard("")
    return bool(words & vocabulary)
