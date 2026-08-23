"""The turn pipeline installs its authority through one implementation.

P4c. `process_text_from_queue` was the fourth site installing the three
authority contextvars by hand, and the only one that had the order right --
which is why `brain/turn.py` exists and why the other three were changed to
match it rather than the reverse. This is the last one.

The function is now two: a nine-line `process_text_from_queue` that installs
the turn's authority through `run_turn`, and `_turn_pipeline` holding everything
that was there before, unchanged. Split rather than nested in a closure for one
concrete reason -- the two AST sweeps that walk the pre-dispatch region look for
a `try:` body, and a closure would put that body one level deeper than they
look. That is KI-32's vacuity exactly, and it is asserted against in
`test_6a5_predispatch_gate.py` rather than trusted here.

What this file pins is the seam: the pipeline installs nothing itself, there is
exactly one install site in `main.py`, an unstated grant set becomes the empty
one rather than the local one, and the authority is gone by the time control
returns to whatever queued the turn -- on the success path, the raise path, and
the early-return path that most of the pre-dispatch branches take.

Run with:  py -3.11 -m pytest tests/test_brain_turn_pipeline.py -v
"""
import ast
import contextvars
import pathlib
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import assistant.actions as A  # noqa: E402
from assistant.core.capabilities import Capability  # noqa: E402

_MAIN_PY = _ROOT / "assistant" / "main.py"

_INSTALL_CALLS = frozenset({"set_grants", "set_principal", "set_raise_context"})
_GRANTS = frozenset({Capability.CHAT_SEND, Capability.FILES})


def _fn(name: str) -> ast.AsyncFunctionDef:
    tree = ast.parse(_MAIN_PY.read_text(encoding="utf-8"))
    matches = [n for n in ast.walk(tree)
               if isinstance(n, ast.AsyncFunctionDef) and n.name == name]
    assert len(matches) == 1, (
        f"expected exactly one `{name}` in main.py, found {len(matches)}")
    return matches[0]


def _install_sites(node) -> list:
    return [(n.lineno, ast.unparse(n.func))
            for n in ast.walk(node)
            if isinstance(n, ast.Call)
            and (getattr(n.func, "attr", None) in _INSTALL_CALLS
                 or getattr(n.func, "id", None) in _INSTALL_CALLS)]


# ─── one install site ────────────────────────────────────────────────────────

def test_the_install_detector_can_see_an_install():
    """The positive control for the two tests below, which both assert that a
    walk found *nothing*. A broken walk finds nothing too, so without this they
    would pass on a detector that matched no call at all -- and there is no
    example left in the tree to prove it against, which is the point of the
    phase. The shape is fed in directly instead.
    """
    src = "\n".join([
        "async def f(grants):",
        "    tok = _actions_module.set_grants(grants)",
        "    other = set_principal(None)",
    ])
    sites = _install_sites(ast.parse(src))
    assert len(sites) == 2, f"the detector missed an install: {sites}"
    assert {name for _, name in sites} == {
        "_actions_module.set_grants", "set_principal"}



def test_the_pipeline_installs_no_authority_of_its_own():
    """**The one that would undo the phase.** A second install inside the
    region takes a second token, and the outer reset then restores the *inner*
    value -- a turn's grants left live in the queue consumer's context after
    the turn ended. That is the failure the ordering was fixed for, arriving
    from the other end."""
    sites = _install_sites(_fn("_turn_pipeline"))
    assert not sites, (
        f"`_turn_pipeline` installs authority itself at {sites}. It runs "
        "*inside* `run_turn`, which installed it already."
    )


def test_main_has_exactly_one_authority_install_site():
    """Tree-wide over the file, not just the two functions: a helper anywhere
    in `main.py` that installs a grant set is the same defect with a different
    address."""
    tree = ast.parse(_MAIN_PY.read_text(encoding="utf-8"))
    sites = _install_sites(tree)
    assert sites == [], (
        f"main.py installs authority directly at {sites}; the only install "
        "site for any turn is `brain/turn.py:run_turn`."
    )


def test_the_wrapper_routes_through_run_turn():
    fn = _fn("process_text_from_queue")
    calls = [n for n in ast.walk(fn)
             if isinstance(n, ast.Call)
             and (getattr(n.func, "id", None) in ("run_turn", "_run_turn")
                  or getattr(n.func, "attr", None) == "run_turn")]
    assert calls, "the turn pipeline no longer goes through run_turn"

    kwargs = {k.arg for k in calls[0].keywords}
    assert {"grants", "principal", "raise_context", "work"} <= kwargs, (
        f"run_turn called without the full authority triple: {sorted(kwargs)}")


def test_the_work_is_passed_as_a_callable_not_a_coroutine():
    """`work` has to be created *inside* the installed context. An
    already-created coroutine happens to work here -- a coroutine body does not
    begin until awaited -- and breaks the moment a caller wraps the call in a
    task, because a task copies the context at creation."""
    fn = _fn("process_text_from_queue")
    call = next(n for n in ast.walk(fn)
                if isinstance(n, ast.Call)
                and getattr(n.func, "id", None) in ("run_turn", "_run_turn"))
    work = next(k.value for k in call.keywords if k.arg == "work")
    assert isinstance(work, (ast.Lambda, ast.Name)), (
        f"`work=` is {type(work).__name__} -- pass a callable, not the result "
        f"of calling one: {ast.unparse(work)}")


def test_the_label_names_the_source():
    """The only thing distinguishing one turn from another in the log, on the
    day one of them misbehaves."""
    fn = _fn("process_text_from_queue")
    call = next(n for n in ast.walk(fn)
                if isinstance(n, ast.Call)
                and getattr(n.func, "id", None) in ("run_turn", "_run_turn"))
    label = next((k.value for k in call.keywords if k.arg == "label"), None)
    assert label is not None, "the pipeline's turns are unlabelled in the log"
    assert "source" in ast.unparse(label), (
        f"the label does not name the source: {ast.unparse(label)}")


# ─── behaviour at the seam ───────────────────────────────────────────────────

@pytest.fixture
def pipeline(monkeypatch):
    """Replace `_turn_pipeline` with a probe. Everything above it -- the
    wrapper, `run_turn`, the contextvars -- is the real code."""
    from assistant import main as main_mod

    seen: dict = {}

    async def _probe(source, transcription, bridge, stt_ms=None):
        seen["args"] = (source, transcription, bridge, stt_ms)
        seen["grants"] = A.current_grants.get()
        seen["principal"] = A.current_principal.get()
        seen["raise_context"] = A.current_raise_context.get()
        if seen.get("raise_it"):
            raise RuntimeError("pipeline blew up")
        if seen.get("return_early"):
            return None
        return "ran"

    monkeypatch.setattr(main_mod, "_turn_pipeline", _probe)
    return main_mod, seen


@pytest.fixture(autouse=True)
def clean_context():
    g = A.current_grants.set(None)
    p = A.current_principal.set(None)
    r = A.current_raise_context.set(None)
    yield
    A.current_raise_context.reset(r)
    A.current_principal.reset(p)
    A.current_grants.reset(g)


@pytest.mark.asyncio
async def test_the_pipeline_sees_what_the_caller_was_issued(pipeline):
    main_mod, seen = pipeline
    await main_mod.process_text_from_queue(
        "studio", "hello", object(), grants=_GRANTS, principal="dev-7")
    assert seen["grants"] == _GRANTS
    assert seen["principal"] == "dev-7"


@pytest.mark.asyncio
async def test_an_unstated_grant_set_becomes_the_empty_set(pipeline):
    """`None` means "nobody said", and it is not a synonym for the local full
    set. `run_turn` would install `None` verbatim, so the conversion has to
    happen at this call site -- and `capability_refusal` treating an unset
    contextvar as a refusal is the backstop, not the rule."""
    main_mod, seen = pipeline
    await main_mod.process_text_from_queue("studio", "hello", object())
    assert seen["grants"] == frozenset(), (
        f"an unstated grant set became {seen['grants']!r}")
    assert Capability.EXECUTE not in seen["grants"]


@pytest.mark.asyncio
async def test_the_pipeline_arguments_survive_the_hop(pipeline):
    main_mod, seen = pipeline
    bridge = object()
    await main_mod.process_text_from_queue(
        "stt", "what time is it", bridge, stt_ms=412, grants=_GRANTS)
    assert seen["args"] == ("stt", "what time is it", bridge, 412)


@pytest.mark.asyncio
async def test_authority_is_gone_when_the_turn_returns(pipeline):
    main_mod, _ = pipeline
    await main_mod.process_text_from_queue(
        "studio", "hello", object(), grants=_GRANTS, principal="dev-7")
    assert A.current_grants.get() is None, "grants leaked past the turn"
    assert A.current_principal.get() is None
    assert A.current_raise_context.get() is None


@pytest.mark.asyncio
async def test_authority_is_gone_when_the_turn_raises(pipeline):
    """A leaked grant set is not a window. It is a standing grant for whatever
    the queue consumer runs next."""
    main_mod, seen = pipeline
    seen["raise_it"] = True
    with pytest.raises(RuntimeError):
        await main_mod.process_text_from_queue(
            "studio", "hello", object(), grants=_GRANTS)
    assert A.current_grants.get() is None, "grants leaked past a failed turn"
    assert A.current_principal.get() is None


@pytest.mark.asyncio
async def test_authority_is_gone_when_a_branch_returns_early(pipeline):
    """The path most of the pre-dispatch branches take: refuse, or skip, and
    `return` long before dispatch. Nine of the twelve returning branches in
    that region do this, so it is the common case rather than the edge."""
    main_mod, seen = pipeline
    seen["return_early"] = True
    await main_mod.process_text_from_queue(
        "funnel", "shut down", object(), grants=frozenset())
    assert A.current_grants.get() is None
    assert A.current_principal.get() is None


@pytest.mark.asyncio
async def test_a_second_turn_does_not_inherit_the_first(pipeline):
    """The failure the ordering argument is about, stated as behaviour rather
    than as source structure: two turns in the same context, the second
    carrying nothing."""
    main_mod, seen = pipeline
    await main_mod.process_text_from_queue(
        "stt", "one", object(), grants=A.LOCAL_GRANTS,
        principal=A.LOCAL_PRINCIPAL)
    assert seen["grants"] == A.LOCAL_GRANTS

    await main_mod.process_text_from_queue("funnel", "two", object())
    assert seen["grants"] == frozenset(), (
        "the second turn inherited the first turn's grants")
    assert seen["principal"] is None


@pytest.mark.asyncio
async def test_the_turn_runs_in_a_task_without_losing_its_authority(pipeline):
    """Why `work` is a callable. A task copies the context at creation, so a
    pre-created coroutine would be created outside the installed context and
    see nothing -- and this is how the real consumer runs turns."""
    import asyncio

    main_mod, seen = pipeline
    await asyncio.create_task(main_mod.process_text_from_queue(
        "stt", "hello", object(), grants=A.LOCAL_GRANTS,
        principal=A.LOCAL_PRINCIPAL))
    assert seen["grants"] == A.LOCAL_GRANTS, (
        "the turn ran in a task and saw no grants")


# ─── LOCAL_GRANTS is installed only where it is meant to be ──────────────────

def test_local_grants_is_installed_at_no_new_site():
    """§2.3 lists where local authority may be claimed: the two scheduler
    branches, the event bus, and the queue consumer deriving it from the item's
    source. A `LOCAL_GRANTS` passed to `run_turn` from anywhere else is a
    remote-reachable path claiming local authority."""
    hits = []
    for path in sorted((_ROOT / "assistant").rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = (getattr(node.func, "id", None)
                    or getattr(node.func, "attr", None))
            if name not in ("run_turn", "_run_turn"):
                continue
            for kw in node.keywords:
                if kw.arg == "grants" and "LOCAL_GRANTS" in ast.unparse(kw.value):
                    hits.append(str(path.relative_to(_ROOT)).replace("\\", "/"))

    assert hits, "walked nothing -- no run_turn call passes grants at all"
    assert set(hits) <= {
        "assistant/scheduler.py",
        "assistant/brain/turn.py",
    }, (
        f"LOCAL_GRANTS reaches run_turn from an unlisted file: {sorted(set(hits))}"
    )


def test_the_queue_consumer_derives_grants_from_the_source():
    """The fourth entry point does not hardcode local authority: it asks
    `_grants_for_item`, which is what makes a `funnel` item carry a ceiling
    rather than the operator's own set.

    Found by *who calls the wrapper* rather than by a function name, so it does
    not go quietly green when the consumer is renamed."""
    tree = ast.parse(_MAIN_PY.read_text(encoding="utf-8"))
    callers = [n for n in ast.walk(tree)
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
               and any(isinstance(c, ast.Call)
                       and getattr(c.func, "id", None)
                       == "process_text_from_queue"
                       for c in ast.walk(n))]
    assert callers, "nothing in main.py calls process_text_from_queue"

    for fn in callers:
        src = ast.unparse(fn)
        assert "_grants_for_item" in src, (
            f"`{fn.name}` starts a turn without deriving its grants from the "
            "item's source"
        )
        assert "_principal_for_item" in src, (
            f"`{fn.name}` starts a turn without a principal -- the turn owns "
            "no pending state and can answer no confirmation"
        )
