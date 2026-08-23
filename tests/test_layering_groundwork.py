"""The layer inversions P1 removed stay removed.

`pyproject.toml` had sixteen `ignore_imports` entries and no positive `layers`
contract, so the arrow order in `CLAUDE.md` was convention rather than
enforcement. Two inversions had no exemption at all and were only found by
running a candidate contract:

* `telemetry -> automation.manifest_runtime` — a domain module reaching into
  automation to bump a selector's failure counter.
* `config -> llm.prompts` — via `LLM_SYSTEM_PROMPT = _get_llm_system_prompt()`
  evaluated at module scope, which pulled every domain facade behind it into
  `config`'s import graph. That single line is where fifteen of the sixteen
  exemptions came from.

`lint-imports` covers the contracts. These cover the mechanisms, because a
contract can be satisfied by code that reintroduces the coupling in a shape
the contract does not name — an inversion moved one module along, say.

Run with:  py -3.11 -m pytest tests/test_layering_groundwork.py -v
"""
import pathlib
import re
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ─── telemetry publishes, it does not reach ──────────────────────────────────

def test_telemetry_does_not_import_automation():
    """Source-level, not import-graph: `lint-imports` sees the graph, this sees
    the intent. A deferred import inside a function is still the inversion."""
    src = (_ROOT / "assistant" / "telemetry.py").read_text(encoding="utf-8")
    offenders = [
        line.strip() for line in src.splitlines()
        if re.search(r"^\s*(?:from\s+\.automation|from\s+\.\.automation|"
                     r"import\s+assistant\.automation)", line)
    ]
    assert not offenders, (
        f"telemetry imports automation: {offenders}. Publish the signal and let "
        f"the consumer subscribe -- `automation -> telemetry` is the legal "
        f"direction, not the reverse."
    )


def test_a_correction_reaches_a_registered_observer():
    """The replacement mechanism actually delivers. An observer registry that
    nothing ever calls would satisfy the test above and lose the feature."""
    from assistant import telemetry

    seen = []
    telemetry.register_correction_observer(seen.append)
    try:
        telemetry._notify_correction_observers("manifest_dispatch")
        assert seen == ["manifest_dispatch"], (
            f"the observer was not called: {seen}. The selector-demotion "
            f"feedback is silently dead."
        )
    finally:
        telemetry._correction_observers.remove(seen.append) \
            if seen.append in telemetry._correction_observers else None


def test_registering_the_same_observer_twice_does_not_double_count():
    """A re-init -- personality switch, dispatcher rebuild -- must not make one
    correction count twice into a demote-after-3 counter."""
    from assistant import telemetry

    calls = []

    def obs(intent):
        calls.append(intent)

    telemetry.register_correction_observer(obs)
    telemetry.register_correction_observer(obs)
    try:
        telemetry._notify_correction_observers("manifest_dispatch")
        assert len(calls) == 1, (
            f"one correction produced {len(calls)} notifications; a rebuild "
            f"would inflate the failure counter"
        )
    finally:
        while obs in telemetry._correction_observers:
            telemetry._correction_observers.remove(obs)


def test_a_raising_observer_does_not_break_the_correction_record():
    """The correction row is already written by this point. An observer is
    never load-bearing, so a broken one must not cost the record."""
    from assistant import telemetry

    def boom(_intent):
        raise RuntimeError("observer is broken")

    ok = []
    telemetry.register_correction_observer(boom)
    telemetry.register_correction_observer(ok.append)
    try:
        telemetry._notify_correction_observers("manifest_dispatch")
        assert ok == ["manifest_dispatch"], (
            "a raising observer stopped the others from being notified"
        )
    finally:
        for cb in (boom, ok.append):
            while cb in telemetry._correction_observers:
                telemetry._correction_observers.remove(cb)


def test_the_dispatcher_subscribes_when_it_is_initialised():
    """The other end. `init_dispatcher` is the wiring point, so a correction
    for a manifest dispatch has to reach `record_last_dispatch_correction`."""
    from assistant import telemetry
    from assistant.automation import manifest_runtime

    hits = []

    class _Disp:
        def record_last_dispatch_correction(self):
            hits.append(1)

    manifest_runtime.init_dispatcher(_Disp())
    try:
        assert manifest_runtime._on_correction in telemetry._correction_observers, (
            "init_dispatcher registered no observer -- the feedback path is "
            "not wired at either end"
        )
        telemetry._notify_correction_observers("manifest_dispatch")
        assert hits, "the dispatcher was not notified"

        hits.clear()
        telemetry._notify_correction_observers("small_talk")
        assert not hits, (
            "the dispatcher was notified for an intent it did not dispatch; "
            "filtering is the subscriber's job and it is not filtering"
        )
    finally:
        manifest_runtime.reset_for_test()


# ─── config carries no prompt builders ───────────────────────────────────────

@pytest.mark.parametrize("name", [
    "LLM_SYSTEM_PROMPT", "build_personality_prompt", "build_intent_prompt",
])
def test_config_no_longer_re_exports_prompt_builders(name):
    """`LLM_SYSTEM_PROMPT` was the load-bearing one: assigned at module scope,
    so importing `config` imported `llm/prompts` and every domain facade behind
    it. Re-adding any of the three re-opens fifteen exemptions."""
    from assistant import config
    assert not hasattr(config, name), (
        f"config.{name} is back. It is what made `config -> llm` and, through "
        f"it, `config -> storage` -- call llm.prompts directly."
    )


def test_the_prompt_builders_still_exist_where_they_belong():
    """Removing a re-export must not remove the function."""
    from assistant.llm import prompts
    for fn in ("get_system_prompt", "build_personality_prompt",
               "build_intent_prompt"):
        assert callable(getattr(prompts, fn, None)), f"llm.prompts.{fn} missing"


# ─── the exemption list only shrinks ─────────────────────────────────────────

def test_the_ignore_imports_list_did_not_grow():
    """`CLAUDE.md` rule 12: never add to `ignore_imports`. If a new import
    needs an exemption, the layering is wrong. Pinned numerically because a
    rule in a document is not a check.

    Two remain, both the same root cause: `core/runtime_config` reaching the
    settings repo. Deliberately not fixed in P1 -- `io/api` reads it too, and
    `io.api` may not reach past `core`+`config`, so relocating it to a domain
    module trades two `config -> storage` exemptions for a transitive
    `io.api -> storage` violation. The honest fix is injection at the Brain
    boundary.
    """
    text = (_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    entries = re.findall(r'"assistant\.[\w.]+ -> assistant\.[\w.]+"', text)
    assert len(entries) <= 2, (
        f"{len(entries)} ignore_imports entries: {entries}. This list went "
        f"from sixteen to two; it does not grow again. A new import that needs "
        f"an exemption means the layering is wrong."
    )
    assert all("runtime_config" in e for e in entries), (
        f"an exemption appeared that is not the known runtime_config path: "
        f"{entries}"
    )

def test_re_initialising_the_dispatcher_does_not_multiply_notifications():
    """The bug this replaced. `init_dispatcher` first built a fresh closure per
    call, so the dedupe never matched and every re-init added an observer --
    each reading the same module global, so one correction bumped the
    selector's failure counter once per init.

    A demote-after-3 counter that reaches three on the first correction after
    three rebuilds is worse than no feedback at all. Found by
    `test_telemetry.py::TestMe1CorrectionFeedback` failing only in company: it
    passed alone, which is exactly the shape a per-file baseline cannot see.
    """
    from assistant import telemetry
    from assistant.automation import manifest_runtime

    hits = []

    class _Disp:
        def record_last_dispatch_correction(self):
            hits.append(1)

    try:
        for _ in range(3):
            manifest_runtime.init_dispatcher(_Disp())
        telemetry._notify_correction_observers("manifest_dispatch")
        assert len(hits) == 1, (
            f"three inits produced {len(hits)} notifications for one "
            f"correction. The failure counter would advance {len(hits)}x per "
            f"correction."
        )
    finally:
        manifest_runtime.reset_for_test()
