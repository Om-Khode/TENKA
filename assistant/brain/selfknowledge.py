"""What TENKA knows about herself, read from live state and gated like anything else.

TENKA-v2 §13. Nothing existed before this: a grep for a self-knowledge path
returned nothing, and `TENKA_Capabilities.md` was referenced by no code.

**K1 — the model explains facts; it never supplies them.** Every answer here
traces to something the running process can be asked: the affordance registry,
`llm/router.py`'s resolved chain, Task state. Never to a document, never to the
model's own recollection of what TENKA is. That is not a style preference --
a model asked "what can you do" will produce a confident, plausible, and
partly-invented list, and the invented parts are indistinguishable from the
real ones.

**K2 — an unavailable fact is said to be unavailable.** One fixed sentence,
`UNAVAILABLE`, rather than a hedge composed per call. A guess about her own
implementation is the worst kind she can make, because the person asking has no
way to check it.

**K3 — read-only.** Nothing here decides anything or changes anything.

**K4 — gated on the fact class, not on a detail level**, and this is where the
obvious design is wrong in a specific way. A three-level `public/technical/
developer` label is something a *caller asks for*; a capability is something a
caller either holds or does not. Mapping levels onto capabilities would put
"technical" behind `OBSERVE` -- and `OBSERVE` is in **every** ceiling including
`funnel`, so a publicly reachable URL would get her current task and her
resolved model chain.

So facts are classified by what they *are*, and each class names the capability
that already governs the same information elsewhere in the tree:

    ARCHITECTURE   nothing beyond the route's own -- it is a public repository
    CONFIGURATION  OBSERVE          `GET /v1/settings` already sits there
    ACTIVITY       RECALL           a read of what she is doing and was told
    TRANSPORT      SYSTEM_CONTROL   `GET /v1/transports` is
                                    `require_admin(SYSTEM_CONTROL)`,
                                    loopback-only, for exactly this reason

Self-Knowledge must not become a second, ungated route to a fact an existing
route gates. `test_self_knowledge.py` asserts that per class, against the
routes themselves.

**K5 — live state, never a cached document.** A cache would be a second source
of truth, which §19 forbids, and a stale answer about her own configuration is
worse than no answer.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Callable, Optional

from ..core.capabilities import Capability

# K2's sentence. One literal, so it cannot drift into a hedge that sounds like
# a soft yes.
UNAVAILABLE = (
    "I don't have reliable information about that part of my current "
    "implementation."
)


class FactClass(str, enum.Enum):
    """What a fact *is*. Not how detailed it is."""

    ARCHITECTURE = "architecture"
    CONFIGURATION = "configuration"
    ACTIVITY = "activity"
    TRANSPORT = "transport"


# K4's table. The comment on each row is the route that already governs the
# same information; the test compares against those routes rather than against
# this dict, so agreeing with itself is not enough.
REQUIRED_CAPABILITY: dict[FactClass, Optional[Capability]] = {
    # In a public repository. Gating it would be theatre.
    FactClass.ARCHITECTURE: None,
    # `io/api/routes/settings.py` -> require(Capability.OBSERVE)
    FactClass.CONFIGURATION: Capability.OBSERVE,
    # A read of her own memory and current work.
    FactClass.ACTIVITY: Capability.RECALL,
    # `io/api/routes/transports.py` -> require_admin(Capability.SYSTEM_CONTROL)
    FactClass.TRANSPORT: Capability.SYSTEM_CONTROL,
}


@dataclass(frozen=True)
class Fact:
    """One answerable question about TENKA, and where the answer comes from.

    `read` is a zero-argument callable rather than a value: K5. Evaluating at
    definition time would cache, and a cached answer about her own live
    configuration is exactly the second source of truth §19 forbids.
    """

    key: str
    fact_class: FactClass
    description: str
    read: Callable[[], object]

    def requires(self) -> Optional[Capability]:
        return REQUIRED_CAPABILITY[self.fact_class]


class SelfKnowledge:
    """The registry of what she can be asked about herself.

    Deliberately not a `RegistryBase`: that one raises on a duplicate key,
    which is right for handlers and wrong here -- two modules describing the
    same fact should be a startup error, and it is, but the read path also
    needs to answer "I cannot tell you" rather than raise, and mixing those two
    failure modes in one type made both harder to read. This is a thin dict
    with a gate.
    """

    def __init__(self) -> None:
        self._facts: dict[str, Fact] = {}

    def register(self, fact: Fact) -> Fact:
        if fact.key in self._facts:
            raise ValueError(f"self-knowledge fact {fact.key!r} already registered")
        self._facts[fact.key] = fact
        return fact

    def keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._facts))

    def get(self, key: str) -> Optional[Fact]:
        return self._facts.get(key)

    # ── the read path ──

    def answer(self, key: str, granted: "frozenset[Capability] | None" = None):
        """The fact, or `UNAVAILABLE`. Never a guess, never a raise.

        `granted` is the turn's grant set. `None` means "nobody said", and is
        treated as holding nothing -- the same fail-closed reading
        `actions.execute()` gives it. A caller that does not state its
        authority does not get a gated fact.
        """
        fact = self._facts.get(key)
        if fact is None:
            return UNAVAILABLE

        required = fact.requires()
        if required is not None:
            if not granted or required not in granted:
                # Deliberately the same sentence as an unknown fact. "You may
                # not ask that" tells a caller the fact exists, which is itself
                # the thing `GET /v1/transports` is loopback-only to avoid.
                return UNAVAILABLE

        try:
            value = fact.read()
        except Exception:
            # K2. A read that fails is an unavailable fact, not an opportunity
            # to describe what it would have said.
            return UNAVAILABLE
        return UNAVAILABLE if value in (None, "", (), [], {}) else value


self_knowledge: SelfKnowledge = SelfKnowledge()


# ─── the facts, each reading live state ─────────────────────────────────────

def _affordances() -> tuple:
    from .affordance import affordance_registry
    entries = getattr(affordance_registry, "_entries", {})
    return tuple(sorted(entries))


def _intents() -> tuple:
    from ..config import INTENTS
    return tuple(sorted(INTENTS))


def _model_for(task_type: str = "default") -> str:
    """The resolved chain for a task type, from the router's own table.

    §13.1: "which model are you using" is answered from `llm/router.py`'s
    resolved chain, never from an assumption. Read at call time, so a config
    change is reflected without a restart.
    """
    from ..llm.router import TASK_MODEL_MAP
    chain = TASK_MODEL_MAP.get(task_type) or TASK_MODEL_MAP.get("default") or []
    return ", ".join(
        f"{entry.get('provider', '?')}/{entry.get('model', '?')}"
        if isinstance(entry, dict) else str(entry)
        for entry in chain
    )


def _personality() -> str:
    """Which personality is loaded, from the personalities package's own
    accessor rather than from the database row it derives from -- one source of
    truth, and it is the one the running process actually consults."""
    from ..personalities import get_active_personality_id
    return get_active_personality_id()


for _fact in (
    Fact("affordances", FactClass.ARCHITECTURE,
         "What TENKA can accomplish, from the affordance registry.",
         _affordances),
    Fact("intents", FactClass.ARCHITECTURE,
         "The execution vocabulary, from config.INTENTS.",
         _intents),
    Fact("model_chain", FactClass.CONFIGURATION,
         "The resolved provider/model chain for the default task.",
         _model_for),
    Fact("personality", FactClass.CONFIGURATION,
         "The personality currently loaded.",
         _personality),
):
    self_knowledge.register(_fact)
