"""How a durable fact was obtained, and who is allowed to write one.

TENKA-v2 §10. **Durable state** is anything that survives the process *and* can
influence a later turn. Both clauses: a log file is durable and influences
nothing; a preference row is durable and steers routing.

Eleven such stores exist. Between them they record provenance as a free string,
and the strings in the tree today are `user`, `explicit`, `correction`,
`confirmed`, `reflection`, `conversation`, `user_msg`, `tenka_resp`, `studio`,
`code`, `ocr`, `llm`, `regex`, `procedure`, `shortcut` and three more. Some name
who said it, some name which subsystem wrote it, and nothing distinguishes
"the user told me this" from "a model guessed it once at 2am".

That distinction is the whole point, because these values do not sit still.
A preference steers backend routing at priority 1, and a preference value
reaches a code-generation prompt. So the question a consumer needs to ask is
not *how confident is this* -- a model will happily assert 0.9 about a guess --
but *how was this obtained*.

**D1 — provenance is required on write.** The enum below, not a free string.

**D2 — provenance is consulted on read.** A consumer that acts on durable state
declares the minimum tier it accepts. `at_least()` is that comparison.

**D3 — a single inference never becomes an unattended behaviour change.**
Promotion to `REPEATED_INFERENCE` is counted by TENKA from data it already
stores, never asserted by the model that produced the candidate. The reflection
prompt already says "minimum 3 occurrences" and nothing checks; a rule that
lets the writer choose its own tier is a rename, not a control.

**Unknown strings are treated as the least trusted thing here, not the most.**
`memory.save_fact` used to default `source="user"` -- the *highest* tier -- so a
forgotten argument manufactured an explicit user statement. That default is
gone, and this module must not reintroduce its shape by being generous about
spellings it does not recognise.
"""
from __future__ import annotations

import logging
from enum import Enum

logger = logging.getLogger("provenance")


class Provenance(Enum):
    """How a durable value was obtained. §10.3's D1 list, unchanged."""

    EXPLICIT_USER_STATEMENT = "explicit_user_statement"
    USER_CORRECTION = "user_correction"
    VERIFIED_OBSERVATION = "verified_observation"
    REPEATED_INFERENCE = "repeated_inference"
    SINGLE_INFERENCE = "single_inference"
    EXTERNAL_CONTENT = "external_content"
    SYSTEM = "system"


# The ladder, as a total order over *belief about the user*.
#
# `SYSTEM` is deliberately absent: TENKA's own bookkeeping -- a backup run, a
# schedule's installed_by -- is not a claim about the person, and ranking it
# against "the user said so" would invite a consumer to accept one where it
# meant the other. `minimum_provenance` handles it explicitly instead.
_RANK: dict[Provenance, int] = {
    Provenance.EXPLICIT_USER_STATEMENT: 5,
    Provenance.USER_CORRECTION: 5,
    Provenance.VERIFIED_OBSERVATION: 4,
    Provenance.REPEATED_INFERENCE: 3,
    Provenance.SINGLE_INFERENCE: 2,
    Provenance.EXTERNAL_CONTENT: 1,
}

# `USER_CORRECTION` ties with `EXPLICIT_USER_STATEMENT` rather than exceeding
# it: both are the user speaking. What makes a correction special is not that
# it is trusted *more*, but that it **supersedes** -- §10.3's fourth property.
# That is a write-side rule about replacing the previous row, not a read-side
# rule about outranking it, and conflating them is how a store accumulates two
# contradictory facts both marked authoritative.


# ─── Reading the strings the tree already writes ─────────────────────────────
#
# Every spelling in `assistant/` on 2026-08-28, mapped once. A writer that
# passes a `Provenance` is unaffected; this exists for the sixteen call sites
# that still pass text, and it is the reason adopting the enum can be a
# separate commit from defining it.

_KNOWN: dict[str, Provenance] = {
    # the user, in their own words
    "user": Provenance.EXPLICIT_USER_STATEMENT,
    "explicit": Provenance.EXPLICIT_USER_STATEMENT,
    "user_msg": Provenance.EXPLICIT_USER_STATEMENT,
    "confirmed": Provenance.EXPLICIT_USER_STATEMENT,
    "correction": Provenance.USER_CORRECTION,
    # TENKA watched it happen
    "verified": Provenance.VERIFIED_OBSERVATION,
    "observation": Provenance.VERIFIED_OBSERVATION,
    "procedure": Provenance.VERIFIED_OBSERVATION,
    "shortcut": Provenance.VERIFIED_OBSERVATION,
    # a model said so -- once, unless TENKA counted otherwise
    "reflection": Provenance.SINGLE_INFERENCE,
    "llm": Provenance.SINGLE_INFERENCE,
    "code": Provenance.SINGLE_INFERENCE,
    "gemini_bbox": Provenance.SINGLE_INFERENCE,
    # read off the world; nobody vouches for it
    "conversation": Provenance.EXTERNAL_CONTENT,
    "tenka_resp": Provenance.EXTERNAL_CONTENT,
    "ocr": Provenance.EXTERNAL_CONTENT,
    "studio": Provenance.EXTERNAL_CONTENT,
    # TENKA's own bookkeeping
    "system": Provenance.SYSTEM,
    "regex": Provenance.SYSTEM,
    "backup_run": Provenance.SYSTEM,
    "backup_onboarding": Provenance.SYSTEM,
}


def classify(raw: "str | Provenance") -> Provenance:
    """Read a recorded provenance. An unrecognised one is the least trusted.

    Not the most. `save_fact(source="user")` defaulted to the top of the ladder
    and a forgotten argument manufactured an explicit user statement; being
    generous about unknown spellings here would rebuild that hole with extra
    steps.
    """
    if isinstance(raw, Provenance):
        return raw
    key = (raw or "").strip().lower()
    known = _KNOWN.get(key)
    if known is not None:
        return known
    logger.warning(
        "[PROVENANCE] unrecognised source %r — treating as external content",
        raw,
    )
    return Provenance.EXTERNAL_CONTENT


def at_least(actual: "str | Provenance",
             minimum: Provenance) -> bool:
    """D2. Does `actual` meet the bar a consumer declared?

    `SYSTEM` satisfies only a `SYSTEM` minimum. It is off the belief ladder in
    both directions: TENKA's own bookkeeping is not evidence about the user,
    and a consumer asking for user-stated input must not be handed it.
    """
    got = classify(actual)
    if got is Provenance.SYSTEM or minimum is Provenance.SYSTEM:
        return got is minimum
    return _RANK[got] >= _RANK[minimum]


# ─── Who may write a durable store ───────────────────────────────────────────
#
# §10.6's allow-list, enforced by an AST sweep in
# `tests/test_durable_write_gate.py` -- the same mechanism that guards the
# pending-state arming sites, and for the same reason: the check is easy and
# the paths around it are what bite.
#
# Keyed by module path relative to `assistant/`. The value is the set of writer
# functions that module is permitted to call. `storage/repos/` is absent
# deliberately: those modules *are* the stores, and listing them would make the
# sweep assert that the implementation may call itself.

WRITER_ALLOW_LIST: dict[str, frozenset[str]] = {
    # the facades -- the intended front door for each store
    "memory.py": frozenset({"save_turn", "save_fact", "save_typed_fact"}),
    "preferences.py": frozenset({"set_preference"}),
    "personality.py": frozenset({"update_traits"}),
    "knowledge_graph.py": frozenset({"add_fact"}),
    # `knowledge.py` is absent on purpose: it *defines* `add_works_entry` and
    # `add_never_entry` rather than calling them. Its own dead-entry test found
    # that, which is the point of having one -- a permission nobody exercises
    # is a permission nobody will notice is wrong.

    # the turn itself: conversation rows and the facts extracted from a turn
    "main.py": frozenset({"save_turn", "save_typed_fact"}),

    # a user answering a question TENKA asked
    "actions/memory_search.py": frozenset({"save_typed_fact"}),
    "actions/pending_handlers.py": frozenset({"add_works_entry"}),

    # observed outcomes, not beliefs about the user
    "automation/router.py": frozenset({"set_preference"}),
    "code_executor/retry.py": frozenset({"add_never_entry"}),

    # the nightly cycle. On the list because it legitimately writes -- and
    # constrained on the *read* side instead, where its output is capped
    # (`repos/preference.py:USER_STATED_SOURCES`) and cannot reach a
    # code-generation prompt. D3: it proposes, it does not choose its own tier.
    "reflection.py": frozenset({"set_preference", "update_traits"}),
}
