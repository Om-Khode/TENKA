"""One install order, and it always unwinds.

P4a. Four sites installed the three authority contextvars by hand, and two of
them disagreed about the order. `main.py` installs grants **last**, with the
reason in its own source:

    An adversarial review found this three statements higher up: a raise
    anywhere in that window skipped the reset entirely and left the grant set
    installed in the queue consumer's context after the turn ended. The
    documented fail-closed property inverts there -- whatever ran next
    inherited the last turn's grants instead of none.

`scheduler.py` installed grants **first** -- the arrangement `main.py` was fixed
for. Neither setter plausibly raises, which is why it survived two reviews, and
is precisely the argument that was rejected over there. So this is not a tidying
commit: it is one implementation of an ordering that had two, one of them wrong.

The leak tests below matter more than the ordering test. An ordering mistake is
a window; a missing reset is a *permanent* grant set in whatever runs next, and
`current_grants` defaulting to `None` is the fail-closed property the whole
security model rests on.

Run with:  py -3.11 -m pytest tests/test_brain_turn.py -v
"""
import ast
import pathlib
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import assistant.actions as A  # noqa: E402
from assistant.brain.turn import run_turn  # noqa: E402
from assistant.core.capabilities import Capability  # noqa: E402

_GRANTS = frozenset({Capability.CHAT_SEND, Capability.FILES})


class _Ctx:
    def __init__(self):
        self.issued = frozenset(Capability)
        self.raisable = frozenset()
        self.ceiling = frozenset(Capability)


@pytest.fixture(autouse=True)
def clean_context():
    """Every test asserts on the contextvars, so a leak from one would be
    indistinguishable from the defect under test in the next."""
    g = A.current_grants.set(None)
    p = A.current_principal.set(None)
    r = A.current_raise_context.set(None)
    yield
    A.current_raise_context.reset(r)
    A.current_principal.reset(p)
    A.current_grants.reset(g)


async def _noop():
    return "done"


# ─── the authority is actually installed ─────────────────────────────────────

@pytest.mark.asyncio
async def test_all_three_are_visible_to_the_work():
    """The point of the function. `work` is a callable rather than a coroutine
    so it is *created* inside the installed context -- an already-created
    coroutine works here by accident and breaks the moment a caller wraps it in
    a task, since a task copies the context at creation, not at await."""
    seen = {}

    async def _work():
        seen["grants"] = A.current_grants.get()
        seen["principal"] = A.current_principal.get()
        seen["raise"] = A.current_raise_context.get()
        return "ok"

    ctx = _Ctx()
    out = await run_turn(grants=_GRANTS, principal="local",
                         raise_context=ctx, work=_work)
    assert out == "ok"
    assert seen["grants"] == _GRANTS
    assert seen["principal"] == "local"
    assert seen["raise"] is ctx


@pytest.mark.asyncio
async def test_the_return_value_passes_through():
    assert await run_turn(grants=_GRANTS, principal="local",
                          raise_context=None, work=_noop) == "done"


# ─── it always unwinds ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_everything_is_reset_on_the_success_path():
    await run_turn(grants=_GRANTS, principal="local",
                   raise_context=_Ctx(), work=_noop)
    assert A.current_grants.get() is None, "grants leaked past a successful turn"
    assert A.current_principal.get() is None
    assert A.current_raise_context.get() is None


@pytest.mark.asyncio
async def test_everything_is_reset_when_the_work_raises():
    """**The one that matters.** A leaked grant set is not a window, it is a
    permanent grant for whatever runs next in that context -- and `None`
    refusing everything is the property the security model rests on."""
    async def _boom():
        raise RuntimeError("work failed")

    with pytest.raises(RuntimeError):
        await run_turn(grants=_GRANTS, principal="local",
                       raise_context=_Ctx(), work=_boom)

    assert A.current_grants.get() is None, (
        "grants survived an exception -- whatever runs next inherits them"
    )
    assert A.current_principal.get() is None
    assert A.current_raise_context.get() is None


@pytest.mark.asyncio
async def test_everything_is_reset_on_the_abort_path():
    """`core/abort.py` fires at loop boundaries, so cancellation is a normal
    exit here, not an exotic one. `CancelledError` derives from
    `BaseException`, so a `finally` catches it and an `except Exception` would
    not -- which is why the reset is in a `finally`."""
    import asyncio

    async def _cancelled():
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await run_turn(grants=_GRANTS, principal="local",
                       raise_context=_Ctx(), work=_cancelled)

    assert A.current_grants.get() is None, "grants survived an abort"
    assert A.current_principal.get() is None
    assert A.current_raise_context.get() is None


@pytest.mark.asyncio
async def test_an_exception_is_not_swallowed():
    """A background runner deciding for itself that a failed turn is fine is
    how a silent failure gets its silence."""
    class _Marker(Exception):
        pass

    async def _boom():
        raise _Marker("this must reach the caller")

    with pytest.raises(_Marker):
        await run_turn(grants=_GRANTS, principal="local",
                       raise_context=None, work=_boom)


# ─── the order is the contract ───────────────────────────────────────────────

def _install_sequence(func_src: str) -> "list[str]":
    """The three setter calls in source order, by name."""
    wanted = {"set_principal", "set_raise_context", "set_grants"}
    order = []
    for node in ast.walk(ast.parse(func_src)):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in wanted):
            order.append((node.lineno, node.func.id))
    return [name for _, name in sorted(order)]


def test_grants_are_installed_last():
    """Read off the source, because the *order* is what was wrong in one of the
    two sites this replaces, and order is not observable from the outside once
    all three are installed."""
    import inspect

    from assistant.brain import turn as turn_mod
    src = inspect.getsource(turn_mod.run_turn)
    assert _install_sequence(src) == [
        "set_principal", "set_raise_context", "set_grants"
    ], (
        f"install order is {_install_sequence(src)}. Grants go last: a raise "
        f"between the first install and the `try` leaves them installed with "
        f"no reset, which inverts the fail-closed default."
    )


def test_nothing_sits_between_the_last_install_and_the_try():
    """`main.py` carries this rule as a comment and it is the same rule here.
    Anything inserted in that window is code that can raise while grants are
    installed and unguarded."""
    import inspect

    from assistant.brain import turn as turn_mod
    lines = inspect.getsource(turn_mod.run_turn).splitlines()

    grants_at = next(i for i, l in enumerate(lines) if "set_grants(" in l)
    try_at = next(i for i, l in enumerate(lines)
                  if l.strip() == "try:" and i > grants_at)

    between = [l.strip() for l in lines[grants_at + 1:try_at]
               if l.strip() and not l.strip().startswith("#")]
    assert not between, (
        f"statements sit between the grants install and the try: {between}"
    )


def test_grants_are_reset_first():
    """Mirror of the install order: the window in which grants are installed is
    closed before anything else is touched."""
    import inspect

    from assistant.brain import turn as turn_mod
    src = inspect.getsource(turn_mod.run_turn)
    finally_body = src[src.index("finally:"):]

    resets = [name for name in ("current_grants", "current_raise_context",
                                "current_principal")
              if f"{name}.reset(" in finally_body]
    order = sorted(resets, key=lambda n: finally_body.index(f"{n}.reset("))
    assert order[0] == "current_grants", (
        f"reset order is {order}; grants are reset first"
    )
    assert len(order) == 3, f"only {len(order)} of the three are reset: {order}"


# ─── the scheduler uses it, and installs nothing itself ──────────────────────

def test_the_scheduler_installs_no_authority_of_its_own():
    """Two sites became zero. A second install order in the tree is how the two
    disagreed in the first place, so the check is that the scheduler has none
    rather than that it has the right one."""
    src = (_ROOT / "assistant" / "scheduler.py").read_text(encoding="utf-8")
    code = "\n".join(l.split("#", 1)[0] for l in src.splitlines())
    for setter in ("set_grants(", "set_principal(", "set_raise_context("):
        assert setter not in code, (
            f"scheduler.py still installs authority itself ({setter}) -- there "
            f"are two orders again"
        )


def test_both_scheduler_branches_go_through_run_turn():
    """`web_search` and `procedure`. A migration that converted one and left the
    other would pass every test above."""
    src = (_ROOT / "assistant" / "scheduler.py").read_text(encoding="utf-8")
    calls = [n for n in ast.walk(ast.parse(src))
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == "run_turn"]
    assert len(calls) == 2, (
        f"scheduler.py calls run_turn {len(calls)} time(s); the web_search and "
        f"procedure branches each need it"
    )
    labels = {kw.value.value for c in calls for kw in c.keywords
              if kw.arg == "label" and isinstance(kw.value, ast.Constant)}
    assert labels == {"schedule:web_search", "schedule:procedure"}, (
        f"labels are {labels} -- a mislabelled turn is a misattributed log line"
    )


@pytest.mark.asyncio
async def test_a_scheduled_procedure_still_meets_the_execute_backstop(monkeypatch):
    """`procedure_executor.run_procedure` checks EXECUTE itself, and that
    backstop is the reason the scheduler installs `LOCAL_GRANTS` at all. If
    `run_turn` installed a narrower set, every scheduled procedure would start
    being refused -- a silent, total feature loss."""
    seen = {}

    async def _fake_run_procedure(proc, goal):
        seen["grants"] = A.current_grants.get()
        return "ran"

    import assistant.procedure_executor as pe
    monkeypatch.setattr(pe, "run_procedure", _fake_run_procedure)

    import assistant.procedures as procs
    monkeypatch.setattr(procs, "find_by_name_or_trigger",
                        lambda goal: {"id": 1, "name": goal, "steps": []})

    from assistant.scheduler import _async_run_handler
    out = await _async_run_handler({"task_type": "procedure",
                                   "task_goal": "morning", "name": "t"})

    assert out == "ran"
    assert seen["grants"] == A.LOCAL_GRANTS, (
        f"a scheduled procedure ran with {seen['grants']} -- "
        f"`run_procedure`'s own EXECUTE check refuses anything narrower"
    )
    assert A.current_grants.get() is None, "grants leaked out of the scheduler"
