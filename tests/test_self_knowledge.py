"""She answers about herself from live state, or says she cannot.

TENKA-v2 §13. Net-new: before this, a grep for a self-knowledge path returned
nothing and `TENKA_Capabilities.md` was referenced by no code.

The rule that shapes every test here is **K4**, and it is subtler than it
looks. The obvious design gates on a *detail level* -- `public` / `technical` /
`developer` -- and maps those to capabilities. That gets the boundary wrong in
a specific way: `OBSERVE` is in **every** ceiling including `funnel`, so
"technical requires OBSERVE" would hand a publicly reachable URL her current
task and her resolved model chain. A level is something a caller *asks for*; a
capability is something a caller *holds*.

So facts are classified by what they are, and each class names the capability
that already governs the same information elsewhere. The tests below compare
against those routes, not against the table -- a table agreeing with itself
proves nothing.

Run with:  py -3.11 -m pytest tests/test_self_knowledge.py -v
"""
import pathlib
import re
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from assistant.brain.selfknowledge import (  # noqa: E402
    UNAVAILABLE, Fact, FactClass, SelfKnowledge, self_knowledge,
)
from assistant.core.capabilities import Capability  # noqa: E402

_ALL = frozenset(Capability)
_NOTHING = frozenset()


# ─── K4: gated on the fact class, matching the route that already gates it ───

def test_architecture_needs_nothing_beyond_the_route():
    """It is all in a public repository. Gating it would be theatre, and
    theatre that makes her unable to describe herself to her own operator."""
    assert self_knowledge.get("intents").requires() is None


def test_configuration_matches_the_settings_route():
    """`GET /v1/settings` is `require(Capability.OBSERVE)`. Self-Knowledge must
    not become a second, cheaper door to the same fact."""
    src = (_ROOT / "assistant" / "io" / "api" / "routes"
           / "settings.py").read_text(encoding="utf-8")
    assert "require(Capability.OBSERVE)" in src, (
        "the settings route no longer requires OBSERVE; re-derive the "
        "CONFIGURATION class against whatever now publishes this fact")

    assert self_knowledge.get("model_chain").requires() is Capability.OBSERVE
    assert self_knowledge.get("personality").requires() is Capability.OBSERVE


def test_transport_matches_the_transports_route():
    """`GET /v1/transports` is `require_admin(SYSTEM_CONTROL)` and
    loopback-only, for exactly this reason."""
    from assistant.brain.selfknowledge import REQUIRED_CAPABILITY

    src = (_ROOT / "assistant" / "io" / "api" / "routes"
           / "transports.py").read_text(encoding="utf-8")
    assert "require_admin(Capability.SYSTEM_CONTROL)" in src

    assert REQUIRED_CAPABILITY[FactClass.TRANSPORT] \
        is Capability.SYSTEM_CONTROL


def test_no_fact_class_is_gated_on_observe_by_default():
    """The trap this design avoids. `OBSERVE` is in every ceiling including
    `funnel`, so a scheme that put everything technical behind it would publish
    her current task to a public URL."""
    from assistant.brain.selfknowledge import REQUIRED_CAPABILITY
    from assistant.io.api.policy import POLICIES

    funnel = frozenset(POLICIES["funnel"].ceiling)
    assert Capability.OBSERVE in funnel, (
        "OBSERVE left the funnel ceiling; K4's argument needs rechecking")

    assert REQUIRED_CAPABILITY[FactClass.ACTIVITY] not in (None,
                                                          Capability.OBSERVE)
    assert REQUIRED_CAPABILITY[FactClass.TRANSPORT] not in (None,
                                                            Capability.OBSERVE)


@pytest.mark.parametrize("key", ["model_chain", "personality"])
def test_a_gated_fact_is_withheld_without_the_capability(key):
    assert self_knowledge.answer(key, _NOTHING) == UNAVAILABLE
    assert self_knowledge.answer(key, None) == UNAVAILABLE


@pytest.mark.parametrize("key", ["model_chain", "personality"])
def test_a_gated_fact_is_given_with_it(key):
    """The control. Withholding everything satisfies the test above and makes
    the feature useless."""
    answer = self_knowledge.answer(key, _ALL)
    assert answer != UNAVAILABLE
    assert answer


def test_an_ungated_fact_needs_no_grants():
    answer = self_knowledge.answer("intents", _NOTHING)
    assert answer != UNAVAILABLE
    assert "small_talk" in answer


def test_a_withheld_fact_is_indistinguishable_from_an_unknown_one():
    """"You may not ask that" tells a caller the fact exists, which is the
    thing `GET /v1/transports` is loopback-only to avoid."""
    assert (self_knowledge.answer("model_chain", _NOTHING)
            == self_knowledge.answer("no_such_fact", _ALL))


# ─── K1 / K2: facts, or an admission. Never a guess. ─────────────────────────

def test_an_unknown_question_gets_the_fixed_sentence():
    assert self_knowledge.answer("what_colour_is_my_car", _ALL) == UNAVAILABLE


def test_a_failing_read_is_unavailable_not_an_excuse():
    """A read that raises is an unavailable fact, not an opportunity to
    describe what it would have said."""
    reg = SelfKnowledge()

    def _boom():
        raise RuntimeError("the registry is on fire")

    reg.register(Fact("broken", FactClass.ARCHITECTURE, "x", _boom))
    assert reg.answer("broken", _ALL) == UNAVAILABLE


def test_an_empty_read_is_unavailable():
    """An empty affordance registry means she has nothing to report, and
    reporting "" would read as an answer."""
    reg = SelfKnowledge()
    reg.register(Fact("empty", FactClass.ARCHITECTURE, "x", lambda: ()))
    assert reg.answer("empty", _ALL) == UNAVAILABLE


def test_the_unavailable_sentence_is_one_literal():
    """K2 asks for a fixed sentence rather than a hedge composed per call, so
    it cannot drift into something that sounds like a soft yes."""
    assert "don't have reliable information" in UNAVAILABLE
    assert "?" not in UNAVAILABLE


def test_the_handler_and_the_brain_agree_on_the_sentence():
    """The handler cannot import `brain/` -- `actions` sits below it -- so the
    literal is duplicated. Two copies is acceptable; two copies that drift
    apart silently is not."""
    from assistant.actions.self_knowledge import UNAVAILABLE as handler_copy

    assert handler_copy == UNAVAILABLE


# ─── K5: live state, never a cache ───────────────────────────────────────────

def test_a_fact_is_read_at_call_time(monkeypatch):
    """A cached answer about her own configuration is exactly the second source
    of truth §19 forbids, and a stale one is worse than none."""
    from assistant import personalities

    first = self_knowledge.answer("personality", _ALL)
    monkeypatch.setattr(personalities, "get_active_personality_id",
                        lambda: "something_else")
    second = self_knowledge.answer("personality", _ALL)

    assert first != second, "the personality answer was cached"
    assert second == "something_else"


def test_the_fact_stores_a_callable_not_a_value():
    """Structurally, so a future fact registered with a computed value fails
    here rather than going stale in production."""
    for key in self_knowledge.keys():
        assert callable(self_knowledge.get(key).read), key


# ─── K3: read-only ───────────────────────────────────────────────────────────

def test_self_knowledge_writes_nothing():
    """It never decides and never changes anything. Asserted by source scan:
    the module must not reach a writer."""
    import ast

    src = (_ROOT / "assistant" / "brain" / "selfknowledge.py").read_text(
        encoding="utf-8")
    called = {
        (getattr(n.func, "attr", None) or getattr(n.func, "id", None))
        for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Call)
    }
    for writer in ("save_turn", "execute", "set_preference", "write_text",
                   "commit", "save_typed_fact", "set_grants"):
        assert writer not in called, f"self-knowledge calls {writer!r}"


# ─── the intent reached all four places ──────────────────────────────────────

def test_the_intent_exists():
    from assistant.config import INTENTS
    assert "self_knowledge" in INTENTS


def test_the_intent_has_a_capability():
    """The fail-closed default would make an unlisted intent cost EXECUTE, so
    an unclassified `self_knowledge` would be refused over every transport --
    a feature that silently never works."""
    from assistant.core.intent_capabilities import REQUIRED_CAPABILITY

    assert REQUIRED_CAPABILITY["self_knowledge"] is Capability.CHAT_SEND


def test_the_intent_has_a_handler():
    from assistant.actions.registry import tool_registry
    import assistant.actions  # noqa: F401

    assert tool_registry.has("self_knowledge")


def test_the_intent_is_in_the_classifier_catalogue():
    from assistant import config
    assert "self_knowledge" in config.INTENT_SYSTEM_PROMPT


def test_the_capabilities_doc_documents_it():
    """The fourth sync point `.claude/rules/llm-and-intents.md` names."""
    doc = (_ROOT / "TENKA_Capabilities.md").read_text(encoding="utf-8")
    assert "`self_knowledge`" in doc


def test_the_capabilities_doc_agrees_with_the_intent_list():
    """P12's exit: the doc is *verified against* the registry rather than
    hand-maintained in parallel, which would be a second source of truth.

    Verified rather than generated, deliberately: each row carries prose a
    generator would flatten, and the prose is the reason anyone reads it.
    """
    from assistant.config import INTENTS

    doc = (_ROOT / "TENKA_Capabilities.md").read_text(encoding="utf-8")
    # Only the intent tables. The transport table further down uses the same
    # row shape for `local` / `tailnet` / `funnel`, which are not intents.
    body = doc.split("## Remote access")[0]
    documented = set(re.findall(r"^\| `([a-z_]+)`", body, re.M))

    # `manifest_dispatch` is synthetic -- fired only by `regex_router`, with
    # params a user cannot ask for -- and is deliberately absent, exactly as it
    # is from the classifier catalogue.
    undocumented = set(INTENTS) - documented - {"manifest_dispatch"}
    assert not undocumented, (
        f"intents with no row in TENKA_Capabilities.md: {sorted(undocumented)}")

    # `browser_action` / `app_action` are registered handlers but internal
    # routing targets, correctly absent from INTENTS.
    invented = documented - set(INTENTS) - {"browser_action", "app_action"}
    assert not invented, (
        f"TENKA_Capabilities.md describes capabilities that do not exist: "
        f"{sorted(invented)} -- a page that lies about her")


# ─── the affordance list is live, not the intent list ────────────────────────

def test_what_she_can_do_comes_from_handlers_not_from_the_intent_list():
    """`config.INTENTS` lists what she can be *asked* for, including anything
    with no handler behind it. Claiming those would be the invented capability
    K1 exists to prevent."""
    from assistant.brain.affordance import affordance_registry, seed_from_handlers
    from assistant.actions.registry import tool_registry
    import assistant.actions  # noqa: F401

    seed_from_handlers()
    entries = getattr(affordance_registry, "_entries", {})
    assert entries, "nothing seeded -- 'what can you do' would report nothing"

    for key, aff in entries.items():
        assert tool_registry.has(aff.intent), (
            f"affordance {key} names {aff.intent}, which has no handler")


def test_seeding_twice_is_safe():
    """`main.py` may call it after a reload, and a duplicate registration would
    otherwise raise at startup."""
    from assistant.brain.affordance import seed_from_handlers

    seed_from_handlers()
    assert seed_from_handlers() == 0
