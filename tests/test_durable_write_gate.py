"""Who may write durable state, and how a reader asks where it came from.

TENKA-v2 §10.6. Durable state is anything that survives the process *and* can
influence a later turn — eleven stores, and between them they recorded
provenance as a free string with eighteen spellings. Some name who said it,
some name which subsystem wrote it, and none of them separated "the user told
me" from "a model guessed once at 2am". That mattered because a preference
steers backend routing at priority 1 and can reach a code-generation prompt.

Two mechanisms here, and they fail in different ways:

**The ladder** is ordinary code and its tests are ordinary tests.

**The allow-list** is an AST sweep, the same shape as the pending-arming sweeps,
and it is the one that can pass while measuring nothing. Every sweep below
asserts its walk was non-empty, because a sweep over zero call sites is green
forever and reads exactly like a clean codebase.

Run with:  py -3.11 -m pytest tests/test_durable_write_gate.py -v
"""
import ast
import pathlib
import sys
import warnings

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from assistant.core.provenance import (  # noqa: E402
    Provenance, WRITER_ALLOW_LIST, at_least, classify,
)

_PKG = _ROOT / "assistant"

# The durable writers. Names, because the calls are made through several
# facades and matching on the callee's module would miss `from x import y`.
_WRITERS = frozenset({
    "save_turn", "save_fact", "save_typed_fact", "set_preference",
    "add_fact", "update_traits", "add_works_entry", "add_never_entry",
})


def _write_sites() -> "dict[str, set[str]]":
    """{module path relative to assistant/: {writer names it calls}}."""
    found: dict[str, set[str]] = {}
    for path in sorted(_PKG.rglob("*.py")):
        rel = path.relative_to(_PKG).as_posix()
        # `storage/repos/` *is* the store. A sweep that included it would be
        # asserting that the implementation is allowed to be itself.
        if rel.startswith("storage/"):
            continue
        try:
            with warnings.catch_warnings():
                # Some modules in the tree carry invalid escape sequences in
                # string literals. Real, unrelated, and not this sweep's
                # business -- without this every assertion here reports eight
                # warnings about files it merely read.
                warnings.simplefilter("ignore", DeprecationWarning)
                tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = (getattr(node.func, "attr", None)
                    or getattr(node.func, "id", None))
            if name in _WRITERS:
                # A definition is not a call site. `memory.py` defines
                # `save_fact` *and* calls the repo's -- only the call counts.
                found.setdefault(rel, set()).add(name)
    return found


# ─── the allow-list ──────────────────────────────────────────────────────────

def test_the_sweep_walks_something():
    """First, because every assertion below is vacuous without it. A sweep over
    an empty set passes forever and looks like a clean tree."""
    sites = _write_sites()
    assert sites, "walked nothing -- the writers were renamed or moved"
    assert sum(len(v) for v in sites.values()) >= 8, (
        f"only {sum(len(v) for v in sites.values())} write sites found; the "
        f"tree had more than that when this was written")


def test_no_module_outside_the_allow_list_writes_durable_state():
    """§10.6's gate. A store that anything may write has no provenance rule,
    only a provenance *habit*."""
    offenders = {}
    for module, writers in _write_sites().items():
        permitted = WRITER_ALLOW_LIST.get(module)
        if permitted is None:
            offenders[module] = sorted(writers)
    assert not offenders, (
        f"these modules write durable state and are not on the allow-list: "
        f"{offenders}. Add them to core/provenance.py with the stores they "
        f"may write, or route the write through a facade that is already "
        f"listed."
    )


def test_an_allow_listed_module_may_only_write_what_it_declared():
    """Being on the list is not a blanket permission. `code_executor/retry.py`
    may record a failed approach; it may not write a preference that steers
    routing."""
    over = {}
    for module, writers in _write_sites().items():
        permitted = WRITER_ALLOW_LIST.get(module)
        if permitted is None:
            continue
        extra = writers - permitted
        if extra:
            over[module] = sorted(extra)
    assert not over, (
        f"these modules write stores they never declared: {over}")


def test_the_allow_list_has_no_dead_entries():
    """A stale entry is a permission nobody is using and nobody will notice is
    wrong. It is also how the list stops describing the tree."""
    sites = _write_sites()
    dead = sorted(m for m in WRITER_ALLOW_LIST if m not in sites)
    assert not dead, (
        f"allow-list entries with no write site: {dead} -- delete them, or "
        f"the list is describing a tree that no longer exists")


# ─── the ladder ──────────────────────────────────────────────────────────────

def test_an_unknown_source_is_the_least_trusted_thing_here():
    """The shape of the bug this project already fixed once.

    `save_fact` defaulted `source="user"` -- the *top* of the ladder -- so a
    forgotten argument manufactured an explicit user statement. Being generous
    about unrecognised spellings would rebuild that hole with extra steps.
    """
    assert classify("something-nobody-defined") is Provenance.EXTERNAL_CONTENT
    assert classify("") is Provenance.EXTERNAL_CONTENT
    assert classify(None) is Provenance.EXTERNAL_CONTENT
    assert not at_least("something-nobody-defined",
                        Provenance.SINGLE_INFERENCE)


def test_the_strings_the_tree_actually_writes_are_all_mapped():
    """The mapping is only useful while it covers what is really written. An
    unmapped spelling silently becomes `EXTERNAL_CONTENT`, which is safe but
    wrong -- a user's own statement would be demoted to a stranger's."""
    import re

    written = set()
    for path in _PKG.rglob("*.py"):
        try:
            src = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):  # pragma: no cover
            continue
        written.update(re.findall(r'source\s*=\s*"([a-z_]+)"', src))

    assert written, "walked nothing -- no source= literals found"
    unmapped = sorted(
        s for s in written
        if classify(s) is Provenance.EXTERNAL_CONTENT
        and classify(s) is not _expected_external(s)
    )
    assert not unmapped, (
        f"these provenance strings are written by the tree and not mapped in "
        f"core/provenance.py: {unmapped}")


def _expected_external(s: str) -> Provenance:
    """The spellings that genuinely *are* external content, so the test above
    does not flag them as unmapped."""
    return (Provenance.EXTERNAL_CONTENT
            if s in {"tenka_resp", "ocr", "studio"}
            else Provenance.SYSTEM)


@pytest.mark.parametrize("higher,lower", [
    (Provenance.EXPLICIT_USER_STATEMENT, Provenance.VERIFIED_OBSERVATION),
    (Provenance.VERIFIED_OBSERVATION, Provenance.REPEATED_INFERENCE),
    (Provenance.REPEATED_INFERENCE, Provenance.SINGLE_INFERENCE),
    (Provenance.SINGLE_INFERENCE, Provenance.EXTERNAL_CONTENT),
])
def test_the_ladder_is_ordered(higher, lower):
    assert at_least(higher, lower)
    assert not at_least(lower, higher)


def test_a_correction_ties_with_a_statement_rather_than_outranking_it():
    """Both are the user speaking. What makes a correction special is that it
    **supersedes** the previous row -- a write-side rule about replacing, not a
    read-side rule about outranking. Conflating the two is how a store ends up
    holding two contradictory facts both marked authoritative."""
    assert at_least(Provenance.USER_CORRECTION,
                    Provenance.EXPLICIT_USER_STATEMENT)
    assert at_least(Provenance.EXPLICIT_USER_STATEMENT,
                    Provenance.USER_CORRECTION)


def test_system_bookkeeping_is_off_the_belief_ladder():
    """A backup run is not a claim about the person. A consumer asking for
    user-stated input must not be handed it, and a consumer asking for
    bookkeeping must not be handed a guess about the user."""
    assert at_least(Provenance.SYSTEM, Provenance.SYSTEM)
    assert not at_least(Provenance.SYSTEM,
                        Provenance.EXPLICIT_USER_STATEMENT)
    assert not at_least(Provenance.SYSTEM, Provenance.EXTERNAL_CONTENT), (
        "system bookkeeping satisfied a belief-about-the-user minimum")
    assert not at_least(Provenance.EXPLICIT_USER_STATEMENT,
                        Provenance.SYSTEM)


def test_a_model_cannot_write_itself_to_the_top_of_the_ladder():
    """D3, at the only point this module can enforce it. `reflection` is the
    nightly LLM cycle; whatever it claims, its output is a single inference
    until TENKA has counted the repetitions herself."""
    assert classify("reflection") is Provenance.SINGLE_INFERENCE
    assert not at_least("reflection", Provenance.REPEATED_INFERENCE)
    assert not at_least("llm", Provenance.REPEATED_INFERENCE)


def test_the_enums_own_values_classify_back_to_themselves():
    """A value this module wrote must read back as what it wrote.

    Obvious once stated, and it was missing: the first writer to store
    `Provenance.SINGLE_INFERENCE.value` would have had it treated as an
    unrecognised string and demoted to `EXTERNAL_CONTENT`, with a warning
    blaming the caller. `main.py`'s fact extraction is that first writer.
    """
    for member in Provenance:
        assert classify(member.value) is member, (
            f"{member.value!r} does not round-trip -- a writer using the enum "
            f"would be demoted to external content")


def test_fact_extraction_is_not_recorded_as_the_user_speaking():
    """D3. The user said something; a *model* decided which part was a fact,
    what to call it, and what the value was. That is an inference over their
    words, not the words themselves -- and it must not outrank one."""
    assert classify("conversation") is Provenance.SINGLE_INFERENCE
    assert not at_least("conversation", Provenance.REPEATED_INFERENCE)
    assert not at_least("conversation", Provenance.EXPLICIT_USER_STATEMENT)


# ─── one ladder, not two ─────────────────────────────────────────────────────

def test_the_preference_store_does_not_keep_its_own_ladder():
    """The duplication this reconciles.

    `storage/repos/preference.py` classified `source` before this module
    existed -- `USER_STATED_SOURCES` and a `_MODEL_PROPOSED_SOURCES` written
    out by hand. Two ladders answering the same question, and they had already
    drifted: `assistant` and `inference` were model-proposed there and
    unrecognised here, which resolves to `EXTERNAL_CONTENT`. One string, two
    answers, nothing to notice it.

    Asserted at source, because the failure is invisible at runtime: both sets
    would keep working, separately, forever.
    """
    src = (_ROOT / "assistant" / "storage" / "repos"
           / "preference.py").read_text(encoding="utf-8")
    assert "spellings_at_least" in src, (
        "the preference store no longer derives its provenance set from "
        "core/provenance.py -- there are two ladders again")
    assert 'frozenset({\n    "user"' not in src, (
        "USER_STATED_SOURCES is a hand-written literal again")


def test_the_derived_set_still_holds_every_spelling_it_used_to():
    """A derived set that quietly narrowed would demote real user statements
    to guesses, and the clamp would start capping deliberate corrections."""
    from assistant.storage.repos.preference import USER_STATED_SOURCES

    for spelling in ("user", "explicit", "correction", "confirmed"):
        assert spelling in USER_STATED_SOURCES, (
            f"{spelling!r} was user-stated before the sets were derived and "
            f"is not now")


def test_the_model_proposed_spellings_are_not_user_stated():
    """The other direction. The nightly cycle's own spellings, and the two the
    preference store knew about that this ladder did not."""
    from assistant.storage.repos.preference import USER_STATED_SOURCES

    for spelling in ("reflection", "inference", "assistant", "llm"):
        assert spelling not in USER_STATED_SOURCES, (
            f"{spelling!r} counts as the user speaking")
        assert classify(spelling) is Provenance.SINGLE_INFERENCE
