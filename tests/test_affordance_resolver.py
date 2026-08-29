"""What TENKA can do, as data, and one router rather than two.

TENKA-v2 §17.P3. Seven properties, each its own test, because an aggregate
would pass while any one regressed.

**The "exactly one router" decision, recorded.** The brief asks for either
`detect_backend` to move into the resolver, or for it to stay with the resolver
delegating and adding nothing that decides. It delegates -- and that was forced
rather than chosen: `detect_backend` is called from inside
`automation/router.py` at two of its own call sites, and `automation ↛ brain`
is an enforced contract. Moving it would break the package that owns its
callers. `test_the_resolver_implements_no_routing_signal` is what stops the
delegation quietly growing into a second implementation.

Run with:  py -3.11 -m pytest tests/test_affordance_resolver.py -v
"""
import ast
import pathlib
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from assistant.brain.affordance import (  # noqa: E402
    Affordance, AffordanceError, AffordanceRegistry,
)
from assistant.brain.resolver import Environment, resolve  # noqa: E402
from assistant.core.capabilities import Capability  # noqa: E402
from assistant.core.verdict import Outcome  # noqa: E402

_BRAIN = _ROOT / "assistant" / "brain"
_RESOLVER = _BRAIN / "resolver.py"


@pytest.fixture
def registry():
    reg = AffordanceRegistry()
    reg.register("open_app", Affordance(
        "open_app", "computer_task", operation="open",
        tags=frozenset({"launch", "start"})))
    reg.register("play_track", Affordance(
        "play_track", "code_executor", operation="play",
        tags=frozenset({"music"})))
    return reg


# ─── AF1: every affordance names a real intent ───────────────────────────────

def test_an_affordance_naming_an_unknown_intent_is_rejected(registry):
    """At registration, not at dispatch. A typo becomes a startup error rather
    than a `tool_registry.get()` miss on the day someone asks for it."""
    with pytest.raises(AffordanceError, match="config.INTENTS"):
        registry.register("bad", Affordance("bad", "not_an_intent"))


def test_a_real_intent_is_accepted(registry):
    """The control. A registry that rejected everything would pass the test
    above and hold nothing."""
    registry.register("note_it", Affordance("note_it", "create_note"))
    assert registry.get("note_it") is not None


def test_registering_under_a_different_name_is_rejected(registry):
    """Two names for one thing is how a lookup misses."""
    with pytest.raises(AffordanceError, match="calls itself"):
        registry.register("one", Affordance("another", "create_note"))


# ─── AF2: the capability equals the intent's ─────────────────────────────────

def test_the_required_capability_comes_from_the_intent_table(registry):
    from assistant.core.intent_capabilities import REQUIRED_CAPABILITY

    got = registry.get("open_app").requires()
    assert got is REQUIRED_CAPABILITY["computer_task"]


def test_an_unlisted_intent_costs_execute():
    """The fail-closed default dispatch already uses. An affordance for a new
    intent is expensive until somebody says otherwise, rather than free until
    somebody notices."""
    from assistant.core.intent_capabilities import (
        DEFAULT_REQUIRED, REQUIRED_CAPABILITY,
    )

    unlisted = [i for i in ("small_talk", "unknown", "get_time")
                if i not in REQUIRED_CAPABILITY]
    assert DEFAULT_REQUIRED is Capability.EXECUTE, (
        "the fail-closed default changed; AF2's argument depends on it")
    if unlisted:
        assert Affordance("x", unlisted[0]).requires() is Capability.EXECUTE


def test_an_affordance_cannot_carry_its_own_capability():
    """AF2's real teeth: there is no field to declare a weaker one with. A
    stored answer would be a second, quieter reply to "what does this cost",
    and the enforcement point would still refuse it -- so the declaration could
    only ever be a lie that surfaces as a confusing refusal."""
    import dataclasses

    fields = {f.name for f in dataclasses.fields(Affordance)}
    assert not any("capabilit" in f or f == "requires" for f in fields), (
        f"Affordance stores a capability: {sorted(fields)}")


def test_a_subclass_may_not_answer_the_capability_question(registry):
    """**From a green mutant.** The base `Affordance` has no field to declare a
    weaker capability with, so comparing `requires()` to the table it reads was
    tautological -- disabling that check turned nothing red.

    A subclass can lie, though. It would be accepted here while
    `actions/__init__.py` still refused the dispatch, so the declaration would
    surface as a confusing refusal rather than as an error at registration.
    """
    class Cheaper(Affordance):
        def requires(self):
            return Capability.OBSERVE

    with pytest.raises(AffordanceError, match="requires"):
        registry.register("cheap", Cheaper("cheap", "computer_task"))


def test_a_plain_affordance_is_still_accepted(registry):
    """The control for the check above: rejecting every registration would
    satisfy it and leave the registry empty."""
    registry.register("ok", Affordance("ok", "computer_task"))
    assert registry.get("ok").requires() is Capability.EXECUTE


def test_the_capability_is_read_live_not_cached(monkeypatch):
    """Same reason `Task.requires()` reads live: a stored answer goes stale the
    moment a classification changes."""
    from assistant.core import intent_capabilities as ic

    aff = Affordance("probe", "create_note")
    before = aff.requires()
    monkeypatch.setitem(ic.REQUIRED_CAPABILITY, "create_note",
                        Capability.SYSTEM_CONTROL)
    assert aff.requires() is Capability.SYSTEM_CONTROL
    assert before is not Capability.SYSTEM_CONTROL, "the probe proved nothing"


# ─── UNSUPPORTED rather than a wrong answer ──────────────────────────────────

def test_nothing_matching_is_unsupported(registry):
    """A resolver that returns a wrong affordance is worse than one that
    returns none: the caller acts on it, and the failure surfaces as a strange
    action rather than as "I cannot do that"."""
    res = resolve("xyzzy quux", Environment(), registry=registry)

    assert res.outcome is Outcome.UNSUPPORTED
    assert res.affordances == ()
    assert res.best is None


def test_a_match_is_returned(registry):
    """The control. Always answering UNSUPPORTED passes the test above."""
    res = resolve("open notepad", Environment(), registry=registry)

    assert res.outcome is Outcome.SUCCEEDED
    assert res.best.affordance_id == "open_app"


def test_an_empty_goal_resolves_to_nothing(registry):
    assert resolve("", Environment(), registry=registry).outcome \
        is Outcome.UNSUPPORTED


# ─── purity ──────────────────────────────────────────────────────────────────

def test_resolution_is_a_pure_function_of_registry_and_environment(registry):
    """Same inputs, same answer. The environment is passed in rather than read,
    so resolution does not depend on what happened to be open at that instant
    -- which is neither reasonable for a caller nor pinnable by a test."""
    env = Environment(open_windows=("Notepad",), browser_driver_available=True)

    first = resolve("open notepad", env, registry=registry)
    second = resolve("open notepad", env, registry=registry)

    assert first == second


def test_the_environment_is_a_parameter_not_a_read():
    """Asserted structurally: a resolver that read the desktop itself would
    still pass the equality test above on a quiet machine."""
    src = _RESOLVER.read_text(encoding="utf-8")
    tree = ast.parse(src)
    called = {
        (getattr(n.func, "attr", None) or getattr(n.func, "id", None))
        for n in ast.walk(tree) if isinstance(n, ast.Call)
    }
    for reader in ("get_open_windows", "screenshot", "list_ocr_blocks",
                   "latch_state_snapshot"):
        assert reader not in called, (
            f"the resolver reads the environment itself via {reader!r}")


# ─── exactly one router ──────────────────────────────────────────────────────

# The five signals `automation/router.py:detect_backend` orders. Each must be
# implemented in exactly one module, and that module is not this one.
_ROUTING_SIGNALS = {
    "preference lookup": ("_check_routing_preference", "get_preference"),
    "URL pattern": ("_URL_PATTERN",),
    "running process": ("_detect_running_app", "get_open_windows"),
    "launch keyword": ("_LAUNCH_RE", "run_app_match"),
    "app context": ("_extract_target_app",),
}


@pytest.mark.parametrize("signal", sorted(_ROUTING_SIGNALS))
def test_the_resolver_implements_no_routing_signal(signal):
    """**The decision, enforced.** The resolver reads the routing answer; it
    must never compute one. Two implementations of one ordering is the
    duplicate-orchestration anti-pattern this phase exists to remove."""
    src = _RESOLVER.read_text(encoding="utf-8")
    body = "\n".join(line for line in src.splitlines()
                     if not line.lstrip().startswith("#"))

    for token in _ROUTING_SIGNALS[signal]:
        assert token not in body, (
            f"the resolver implements the {signal} signal itself ({token!r}); "
            f"it must delegate to automation/router.py")


def test_the_resolver_delegates_through_exactly_one_function():
    """Isolated in `_route` so "adds nothing that decides" is checkable by
    reading four lines rather than the whole module."""
    tree = ast.parse(_RESOLVER.read_text(encoding="utf-8"))
    callers = [
        n.name for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        and "detect_backend" in ast.unparse(n)
    ]
    assert callers == ["_route"], (
        f"detect_backend is reached from {callers}; it must have one call site")


def test_the_router_still_owns_the_signals():
    """Anti-vacuity for the scan above. If the signals had moved out of
    `automation/router.py` entirely, every assertion there would pass while
    naming nothing."""
    src = (_ROOT / "assistant" / "automation" / "router.py").read_text(
        encoding="utf-8")
    for signal, tokens in _ROUTING_SIGNALS.items():
        assert any(t in src for t in tokens), (
            f"the {signal} signal is no longer in automation/router.py -- "
            f"the resolver's delegation target moved")


# ─── the brain's vocabulary ──────────────────────────────────────────────────

def test_no_brand_name_appears_in_brain():
    """§17.P3: a resolver built on top of a brand-name regex has not fixed
    anything. The canvas products moved to `core/known_apps.py` in an earlier
    commit; this stops them coming back one module up."""
    brands = ("figma", "miro", "excalidraw", "tldraw", "spotify", "chrome",
              "firefox", "notepad", "whatsapp", "gmail", "youtube")
    offenders = []
    for path in sorted(_BRAIN.glob("*.py")):
        for brand in brands:
            for where in _code_strings(path) | _code_names(path):
                if brand in where.lower():
                    offenders.append(f"{path.name}: {brand} in {where!r}")
    assert not offenders, f"brand names in brain/: {offenders}"


def _code_strings(path: pathlib.Path) -> set:
    """Every string literal in `path` that is not a docstring.

    By AST, and only after the first version failed on `brain/executor.py`'s
    module docstring -- which uses "open notepad" as the example of a goal that
    can be re-read into something else. That is prose explaining why structured
    parameters exist, and forbidding it would forbid explaining the rule.

    A comment-stripper is not enough: it removes `#` lines and leaves
    docstrings, which is exactly where the illustrative examples live. Sixth
    time this project has had a sweep match its own prose.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))

    docstrings = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)):
            continue
        body = getattr(node, "body", None)
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            docstrings.add(id(body[0].value))

    return {
        n.value for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
        and id(n) not in docstrings
    }


def _code_names(path: pathlib.Path) -> set:
    """Identifiers and attribute names -- a `_SPOTIFY_RE` names a brand too."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name):
            names.add(n.id)
        elif isinstance(n, ast.Attribute):
            names.add(n.attr)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef,
                            ast.ClassDef)):
            names.add(n.name)
    return names


def test_the_brand_scan_can_see_a_brand_in_code():
    """Positive control. The test above asserts an absence, and a scan that
    reads nothing finds no brands either -- there is deliberately none left in
    `brain/` to prove it against."""
    import tempfile

    src = "\n".join([
        '"""A docstring mentioning notepad, which is allowed."""',
        'PATTERN = "spotify|figma"',
        'def open_chrome():',
        '    pass',
    ])
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                     encoding="utf-8") as fh:
        fh.write(src)
        tmp = pathlib.Path(fh.name)
    try:
        strings = _code_strings(tmp)
        names = _code_names(tmp)
    finally:
        tmp.unlink()

    assert any("spotify" in s for s in strings), "missed a brand in a literal"
    assert any("chrome" in n for n in names), "missed a brand in an identifier"
    assert not any("notepad" in s for s in strings), (
        "the docstring was scanned; prose explaining the rule is not a "
        "violation of it")


def test_the_word_capability_means_only_the_security_enum_in_brain():
    """§6's vocabulary rule. What TENKA can *do* is an affordance; what she is
    *allowed* to do is a capability. The source documents used one word for
    both, and that collision is what made an earlier plan look implementable
    while it proposed re-keying the only working security control in the tree.
    """
    # `selfknowledge.py` is on the list because K4 is *about* capabilities:
    # each fact class names the one that already governs the same information
    # elsewhere. That is the enum, used for what the enum is for -- not the
    # collision this rule exists to prevent.
    allowed_files = {"authority.py", "task.py", "affordance.py", "turn.py",
                     "executor.py", "selfknowledge.py", "__init__.py"}

    # Code, not prose. This scan stripped `#` comments and read docstrings, so
    # `development.py` failed it for a module docstring that says git must
    # never affect "capability availability" -- which is the rule being
    # explained, not a violation of it. Forbidding that would forbid writing
    # the rule down. Seventh time a sweep in this project has matched its own
    # commentary; the brand scan above already learned it.
    for path in sorted(_BRAIN.glob("*.py")):
        used = _code_strings(path) | _code_names(path)
        if not any("capabilit" in u.lower() for u in used):
            continue
        assert path.name in allowed_files, (
            f"{path.name} uses the word 'capability' in code; in this package "
            f"it means `core/capabilities.py`'s enum and nothing else -- what "
            f"TENKA can do is an affordance")


def test_the_resolver_never_mentions_capabilities_at_all():
    """The resolver answers "what could do this", never "may it". Mixing the
    two in one module is how the two words collapse back together."""
    used = _code_strings(_RESOLVER) | _code_names(_RESOLVER)
    assert not any("capabilit" in u.lower() for u in used), (
        "the resolver answers \"what could do this\", never \"may it\" -- "
        "mixing the two in one module is how the two words collapse back "
        "together")


def test_the_registry_extends_the_shared_primitive():
    """§17.P3 says extend `RegistryBase`, not duplicate it. A second registry
    implementation is a second set of thread-safety bugs."""
    from assistant.core.registry import RegistryBase

    assert issubclass(AffordanceRegistry, RegistryBase)
