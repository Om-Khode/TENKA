"""The one place model input is assembled. Six profiles, each a whitelist.

TENKA-v2 §12.1. A subsystem receives what its profile names and nothing else,
so "context is minimized" becomes a property of a data structure rather than a
claim about a prompt string.

**Whitelists, not blacklists**, and the direction is the whole design. A
blacklist is a list of things someone remembered to exclude; every field added
later is included by default, and the failure is silent — a new field reaches a
model and nothing says so. A whitelist fails the other way: a field nobody
listed simply does not arrive, which is visible the first time someone looks
for it and harmless when they do not.

**It lives in `core/`, not `brain/`**, and that is a correction to the plan
rather than a preference. §17.P10 puts the Context Builder in `brain/context.py`
— but `actions/planner/planner.py` and `code_executor/orchestrator.py` also
assemble model input, and both sit *below* `brain/` with no `actions -> brain`
import in the tree. A Builder they cannot import is a Builder they will route
around, which is how the second implementation gets written. This is the third
phase of the plan to hit the same wall (P5's merge rule and P8's `TaskStep`
were the others) and it gets the same answer: the shared, pure part goes below
everyone.

**Three things happen here, in this order, and the order is load-bearing:**

    1. whitelist   a field not on the profile is dropped
    2. fence       a field whose provenance is not TENKA gets a labelled,
                   nonce-delimited block (C1), applied here rather than by each
                   caller (C2)
    3. redact      `redact_secrets_strict`, after fencing, so a block's
                   provenance label survives while its secret-shaped contents
                   do not (§12.3)

Redaction last is not an implementation detail. Redacting first would leave the
fence to be built around already-scrubbed text, which is fine — but a fence
built *after* redaction would have to be re-scanned, and the two orders differ
the moment a fence's own boilerplate contains something the redactor dislikes.
One order, stated, tested.

**C3, and this module will not overstate itself: fencing raises the cost of
injection, it does not close it.** The model is told which bytes are data;
nothing here forces it to care. KI-14, KI-15 and KI-16 are *mitigated* by this
boundary, not fixed, and anything claiming closure needs an adversarial live
test that §22 puts out of scope.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from .fence import render_untrusted_block
from .redact import redact_secrets_strict

# ─── the profiles ────────────────────────────────────────────────────────────
#
# Straight from §12.1. Each value is the complete set of field names that
# profile may carry; anything else handed to `build()` is dropped.

PROFILES: dict[str, frozenset[str]] = {
    "interpretation": frozenset({
        "current_message", "recent_conversation", "minimal_state",
    }),
    "planning": frozenset({
        "task", "constraints", "resolved_affordances",
        "environment_state", "relevant_memory", "relevant_observations",
    }),
    "execution": frozenset({
        "task_step", "affordance", "parameters", "preconditions",
    }),
    "verification": frozenset({
        "intended_operation", "expected_outcome", "observation",
        "required_state",
    }),
    "response": frozenset({
        "relevant_conversation", "task_verdict", "personality_state",
    }),
    "self_knowledge": frozenset({
        "metadata",
    }),
}

# Fields whose *content* TENKA's own code did not author, per profile-agnostic
# name. C1: these are fenced with a provenance label wherever they appear.
#
# The complement is the interesting half, so it is worth saying which fields are
# deliberately *not* here and why. `constraints` and `parameters` are values the
# user pinned or the planner structured -- they are short, they are consumed as
# data by an adapter rather than read as prose by a model, and fencing them
# would put a hundred-token notice around "seat: 14A". `task_verdict`,
# `preconditions` and `metadata` are TENKA's own words about her own state.
UNTRUSTED_FIELDS: frozenset[str] = frozenset({
    "current_message",
    "recent_conversation",
    "relevant_conversation",
    "relevant_memory",
    "relevant_observations",
    "observation",
    "environment_state",
})


class UnknownProfile(KeyError):
    """A profile nobody defined. Raised rather than defaulted: falling back to
    a permissive profile is how a whitelist becomes a blacklist by accident."""


@dataclass(frozen=True)
class ContextBundle:
    """What a subsystem receives, and how big it was.

    `fields` is the built context. `dropped` names what was handed in and left
    out — kept rather than discarded because a silently-dropped field is
    indistinguishable from one nobody passed, and the first question anyone
    asks when a prompt looks wrong is "did my field get through".
    """

    profile: str
    fields: Mapping[str, Any] = field(default_factory=dict)
    dropped: tuple[str, ...] = ()
    fenced: tuple[str, ...] = ()

    @property
    def size_bytes(self) -> int:
        """UTF-8 bytes of everything this bundle would put in front of a model.

        §15's `context_bytes_by_profile`, and §12's O3: without a number, "the
        context is minimized" is an assertion nobody can check. Measured on the
        built object rather than on a rendered prompt so it stays honest about
        what the Builder produced, not about what a caller did afterwards.
        """
        return sum(
            len(str(k).encode("utf-8")) + len(str(v).encode("utf-8"))
            for k, v in self.fields.items()
        )


def build(profile: str, **fields: Any) -> ContextBundle:
    """Assemble the context for `profile`. Unlisted fields do not arrive."""
    try:
        allowed = PROFILES[profile]
    except KeyError:
        raise UnknownProfile(
            f"{profile!r} is not a context profile; known: "
            f"{sorted(PROFILES)}"
        ) from None

    built: dict[str, Any] = {}
    dropped: list[str] = []
    fenced: list[str] = []

    for name, value in fields.items():
        if name not in allowed:
            dropped.append(name)
            continue
        if value is None or value == "":
            # An empty field is not a field. Carrying it produces an empty
            # labelled block, which costs tokens and reads to a model as a
            # section that exists and says nothing.
            continue

        if name in UNTRUSTED_FIELDS:
            rendered = render_untrusted_block(_as_text(value), label=name)
            fenced.append(name)
        else:
            rendered = value

        built[name] = (redact_secrets_strict(rendered)
                       if isinstance(rendered, str) else rendered)

    return ContextBundle(
        profile=profile,
        fields=built,
        dropped=tuple(sorted(dropped)),
        fenced=tuple(sorted(fenced)),
    )


def _as_text(value: Any) -> str:
    """Render a value for fencing.

    A list becomes one item per line rather than a Python repr: the fence is
    read by a model, and `['a', 'b']` spends tokens on quoting and brackets
    that carry no meaning to it.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return "\n".join(f"{k}: {v}" for k, v in value.items())
    if isinstance(value, Iterable):
        return "\n".join(str(v) for v in value)
    return str(value)


def bytes_by_profile(bundles: "Iterable[ContextBundle]") -> dict[str, int]:
    """`{profile: total bytes}` across bundles, for §15's telemetry field.

    Summed rather than replaced, because a turn can build the same profile more
    than once — a planner that replans builds `planning` again, and the cost
    the operator is asking about is the total.
    """
    totals: dict[str, int] = {}
    for bundle in bundles:
        totals[bundle.profile] = totals.get(bundle.profile, 0) + bundle.size_bytes
    return totals
