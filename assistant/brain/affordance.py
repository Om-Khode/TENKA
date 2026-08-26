"""What TENKA can *do*, as data. Not what she is *allowed* to do.

TENKA-v2 §7.3. An affordance is a thing the assistant can accomplish -- open an
application, fill a form, play a track. A **capability** is `core/capabilities.py`'s
security enum and nothing else. The source documents used one word for both,
and that collision is what made an earlier plan look implementable while it
quietly proposed re-keying the only working security control in the tree. This
package keeps them apart by vocabulary, and `brain/__init__.py`'s note is
pinned by a test.

**AF1 and AF2 are enforced at registration, not at dispatch**, and that is the
whole point of doing this in a registry rather than a dict:

- **AF1** -- an affordance names an intent in `config.INTENTS`. A typo becomes
  an ImportError at startup rather than a `tool_registry.get()` miss on the day
  someone asks for it.
- **AF2** -- an affordance's required capability *equals* its intent's, read
  from `core/intent_capabilities.py`. It may not declare something weaker. An
  affordance claiming `OBSERVE` for an intent that costs `EXECUTE` would be a
  second, quieter answer to "what does this cost", and the enforcement point in
  `actions/__init__.py` would still refuse it -- so the declaration would be a
  lie that only shows up as a confusing refusal.

The registry does not store the capability. `Affordance.requires()` reads the
intent table live, for the reason `Task.requires()` does: a stored answer goes
stale the moment a classification changes, and the one thing a security-adjacent
number must never be is out of date with the table it came from.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..core.registry import RegistryBase

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..core.capabilities import Capability


class AffordanceError(ValueError):
    """A registration that would be a lie. Raised at import, deliberately."""


@dataclass(frozen=True)
class Affordance:
    """One thing TENKA can accomplish, and the intent that carries it out.

    `intent` is the execution ABI -- it must be in `config.INTENTS`, because
    that is the only vocabulary `actions.execute()` understands.

    `operation` is what this affordance *does* in the abstract ("open",
    "fill", "play"). It is the field a planner reasons with, and it is
    deliberately not the intent: several affordances share `computer_task` and
    differ only here.
    """

    affordance_id: str
    intent: str
    operation: str = ""
    description: str = ""
    # What the caller must supply. Names only -- the schema lives with the
    # handler, and duplicating it here would be a second thing to keep in sync.
    parameters: tuple[str, ...] = ()
    tags: frozenset[str] = field(default_factory=frozenset)

    def requires(self) -> "Capability":
        """What this affordance costs, read from the intent table.

        Live rather than stored, exactly as `Task.requires()` is. An unlisted
        intent costs `EXECUTE` -- the same fail-closed default dispatch uses,
        so an affordance for a new intent is expensive until somebody says
        otherwise rather than free until somebody notices.
        """
        from ..core.intent_capabilities import DEFAULT_REQUIRED, REQUIRED_CAPABILITY
        return REQUIRED_CAPABILITY.get(self.intent, DEFAULT_REQUIRED)


class AffordanceRegistry(RegistryBase[Affordance]):
    """`RegistryBase`, extended -- not a second registry implementation.

    The one thing added is the check at `register()`. `RegistryBase` is
    deliberately import-free of the rest of `assistant/`, so it cannot know
    what an intent is; this subclass can, and validating here is what makes
    AF1 and AF2 import-time properties.
    """

    def __init__(self) -> None:
        super().__init__("affordance")

    def register(self, key: str, obj: Affordance) -> Affordance:
        self._validate(key, obj)
        return super().register(key, obj)

    @staticmethod
    def _validate(key: str, obj: Affordance) -> None:
        from ..config import INTENTS
        from ..core.intent_capabilities import (
            DEFAULT_REQUIRED, REQUIRED_CAPABILITY,
        )

        if not isinstance(obj, Affordance):
            raise AffordanceError(
                f"{key!r} is not an Affordance: {type(obj).__name__}")

        if key != obj.affordance_id:
            raise AffordanceError(
                f"registered under {key!r} but calls itself "
                f"{obj.affordance_id!r}; two names for one thing is how a "
                f"lookup misses")

        # AF1.
        if obj.intent not in INTENTS:
            raise AffordanceError(
                f"affordance {key!r} names intent {obj.intent!r}, which is not "
                f"in config.INTENTS -- `actions.execute()` would refuse it, so "
                f"this fails now instead of on the day someone asks for it")

        # AF2, and it took two mutations to get this right.
        #
        # The first version compared `obj.requires()` to the table that
        # `requires()` reads, and disabling it turned nothing red -- so it read
        # as tautological. It is not: `Affordance` is a plain dataclass, anyone
        # may subclass it and override `requires()` to answer something
        # cheaper, and the registry would accept that while
        # `actions/__init__.py` still refused the dispatch. The declaration
        # would be a lie surfacing as a confusing refusal rather than as an
        # error here. It went green only because no test had a subclass.
        #
        # The second version added a separate "did you override `requires()`"
        # check, and *that* went green too -- because this comparison already
        # caught the same case. Two checks for one property means neither can
        # be mutated red, which is how a control gets deleted later on the
        # grounds that "the tests still pass". One check, and it is this one:
        # it catches any divergence however produced, not just an override.
        expected = REQUIRED_CAPABILITY.get(obj.intent, DEFAULT_REQUIRED)
        actual = obj.requires()
        if actual is not expected:
            raise AffordanceError(
                f"affordance {key!r} requires {actual}, but intent "
                f"{obj.intent!r} costs {expected} -- what an intent costs is "
                f"read from `core/intent_capabilities.py` and is not an "
                f"affordance's to answer")


# The one registry. Components self-register into it; nothing else constructs
# an `AffordanceRegistry`, for the same reason there is one `tool_registry`.
affordance_registry: AffordanceRegistry = AffordanceRegistry()


def seed_from_handlers() -> int:
    """Register one affordance per intent that actually has a handler.

    §17.P3 asks for self-registration by existing components, and this is the
    honest first form of it: `tool_registry` already knows which intents have a
    handler, because each one registered itself with a decorator. Mirroring
    that is a *true* statement about what TENKA can do -- unlike reading
    `config.INTENTS`, which lists what she can be asked for, including things
    with no handler behind them.

    It is a floor, not the finished shape. Several affordances will eventually
    share one intent and differ by `operation`; this gives one per intent, so
    "what can you do" answers from live state instead of from nothing. A
    handler-less intent is deliberately absent: claiming it would be exactly
    the invented capability §13's K1 exists to prevent.

    Idempotent, because `main.py` may call it after a reload and a duplicate
    registration would otherwise raise.
    """
    from ..config import INTENTS
    from .. import actions  # noqa: F401  -- importing registers the handlers
    from ..actions.registry import tool_registry

    added = 0
    for intent in sorted(INTENTS):
        if not tool_registry.has(intent):
            continue
        key = f"intent:{intent}"
        if affordance_registry.has(key):
            continue
        affordance_registry.register(key, Affordance(
            affordance_id=key,
            intent=intent,
            operation=intent,
            description=f"Carried out by the {intent} handler.",
        ))
        added += 1
    return added
