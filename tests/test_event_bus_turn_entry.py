"""A fired monitor is a source of a turn, not an orchestrator of one.

P4b. `automation/event_bus.py` was the single place in `automation/` reaching
forward into `actions/` -- for `execute`, and for the three local-authority
constants that live there. That was §8.3's fourth layer inversion, and it was
held up by a comment rather than a contract.

It also installed the three authority contextvars by hand, **grants first** --
the same arrangement `scheduler.py` had and the one `main.py` was fixed for: a
raise between the first install and the `try` leaves the grant set installed
with no reset. `brain/turn.py:run_local_intent` installs them in one order, in
one place, and its import is what lets `automation ↛ actions` be asserted.

The behavioural test that matters is the grant one. A fired monitor runs with
`LOCAL_GRANTS` on the argument that installing one required `EXECUTE` durably;
if `run_local_intent` handed it anything narrower, every monitor would begin
being refused -- silently, since `_on_action_complete` logs a failure and moves
on.

Run with:  py -3.11 -m pytest tests/test_event_bus_turn_entry.py -v
"""
import ast
import pathlib
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import assistant.actions as A  # noqa: E402

_BUS = _ROOT / "assistant" / "automation" / "event_bus.py"


@pytest.fixture(autouse=True)
def clean_context():
    g = A.current_grants.set(None)
    p = A.current_principal.set(None)
    r = A.current_raise_context.set(None)
    yield
    A.current_raise_context.reset(r)
    A.current_principal.reset(p)
    A.current_grants.reset(g)


def _bus_with_dispatcher():
    """A bus with the turn dispatcher injected, as `main.py` does at startup.

    Constructed properly rather than via `__new__`: `EventBus.__init__` only
    sets fields (threads start in `start()`), and the dispatcher slot is one of
    those fields. The earlier `__new__` shortcut skipped it, so these tests
    started failing the moment the dependency was inverted -- correctly, since a
    bus with no dispatcher is now a bus that refuses to run anything.
    """
    from assistant.automation.event_bus import EventBus
    from assistant.brain.turn import run_local_intent

    bus = EventBus()
    bus.set_turn_dispatcher(run_local_intent)
    return bus


# ─── the layer inversion is gone, and stays gone ─────────────────────────────

def test_the_event_bus_does_not_import_actions():
    """Asserted by AST rather than by the `lint-imports` summary, so the reason
    lands in a test file next to the argument. The contract in
    `pyproject.toml` is the tree-wide version of the same claim."""
    src = _BUS.read_text(encoding="utf-8")
    reaching = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.ImportFrom) and node.module:
            if "actions" in node.module.split("."):
                reaching.append(node.module)
        elif isinstance(node, ast.Import):
            reaching += [a.name for a in node.names
                         if "actions" in a.name.split(".")]
    assert not reaching, (
        f"event_bus.py imports {reaching}. A source of turns must not reach "
        f"forward into the package that dispatches them -- that is §8.3's "
        f"fourth layer inversion."
    )


def test_the_event_bus_installs_no_authority_of_its_own():
    """Three sites installed the contextvars by hand; this was the third. A
    second install order in the tree is how two of them came to disagree, so the
    check is that there is none here rather than that the one here is right."""
    src = _BUS.read_text(encoding="utf-8")
    code = "\n".join(l.split("#", 1)[0] for l in src.splitlines())
    for setter in ("set_grants(", "set_principal(", "set_raise_context("):
        assert setter not in code, (
            f"event_bus.py still installs authority itself ({setter})"
        )


def test_the_forbidden_contract_is_declared():
    """The AST check above covers this file; the contract covers the package.
    Without it, the next module in `automation/` can reach `actions/` freely and
    nothing says otherwise -- which is the state this was in for the whole of
    6a.5 and 6b."""
    toml = (_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "Automation never imports actions" in toml, (
        "the boundary is back to being a comment"
    )


# ─── a fired monitor still gets local authority ──────────────────────────────

@pytest.mark.asyncio
async def test_a_fired_monitor_runs_with_local_grants(monkeypatch):
    """The grant is the point. Installing a monitor requires `EXECUTE` durably,
    so a fired one spends the installer's grant -- and `execute()` refuses an
    unset grant set by design, so anything narrower means every monitor starts
    failing into a log line."""
    seen = {}

    async def _fake_execute(intent, params, llm_response="", **kw):
        seen["intent"] = intent
        seen["params"] = params
        seen["grants"] = A.current_grants.get()
        seen["principal"] = A.current_principal.get()
        seen["raise"] = A.current_raise_context.get()
        return "ran"

    monkeypatch.setattr(A, "execute", _fake_execute)

    bus = _bus_with_dispatcher()
    out = await bus._run_code_executor("check the disk", "device:phone")

    assert out == "ran"
    assert seen["intent"] == "code_executor"
    assert seen["params"] == {"goal": "check the disk"}
    assert seen["grants"] == A.LOCAL_GRANTS, (
        f"a fired monitor ran with {seen['grants']} -- execute() refuses an "
        f"unset or narrower set and the failure is only a log line"
    )
    assert seen["principal"] == A.LOCAL_PRINCIPAL, (
        "a state armed by the fired action would be owned by nobody"
    )
    assert seen["raise"] is A.LOCAL_RAISE_CONTEXT


@pytest.mark.asyncio
async def test_authority_does_not_leak_out_of_a_fired_monitor(monkeypatch):
    """Both exit paths. A leaked grant set is permanent for whatever runs next
    in that context, and the event bus fires on a background loop that outlives
    the action."""
    async def _ok(*a, **kw):
        return "ran"

    async def _boom(*a, **kw):
        raise RuntimeError("action failed")

    bus = _bus_with_dispatcher()

    monkeypatch.setattr(A, "execute", _ok)
    await bus._run_code_executor("x", "local")
    assert A.current_grants.get() is None, "grants leaked after a fired monitor"

    monkeypatch.setattr(A, "execute", _boom)
    with pytest.raises(RuntimeError):
        await bus._run_code_executor("x", "local")
    assert A.current_grants.get() is None, (
        "grants leaked after a fired monitor raised"
    )
    assert A.current_principal.get() is None
    assert A.current_raise_context.get() is None


# ─── installed_by is recorded at fire ────────────────────────────────────────

@pytest.mark.asyncio
async def test_the_installer_is_logged_at_fire(monkeypatch, caplog):
    """`installed_by` (schema v21, KI-30) was recorded at install and never read
    at fire. The moment it matters is the moment it runs: a monitor installed
    months ago by a device since revoked still fires with `LOCAL_GRANTS`, which
    is exactly the argument KI-30 examined. So the log has to say whose grant is
    being spent."""
    async def _ok(*a, **kw):
        return "ran"

    monkeypatch.setattr(A, "execute", _ok)

    bus = _bus_with_dispatcher()

    with caplog.at_level("INFO", logger="event_bus"):
        await bus._run_code_executor("x", "device:phone")

    assert any("device:phone" in r.message for r in caplog.records), (
        f"the installer was not logged at fire: "
        f"{[r.message for r in caplog.records]}"
    )


@pytest.mark.asyncio
async def test_a_monitor_with_no_recorded_installer_still_fires(monkeypatch, caplog):
    """Rows predating schema v21 have no `installed_by`. Refusing to fire them
    would break every monitor installed before that migration; saying "unknown"
    is honest and keeps them working."""
    async def _ok(*a, **kw):
        return "ran"

    monkeypatch.setattr(A, "execute", _ok)

    bus = _bus_with_dispatcher()

    with caplog.at_level("INFO", logger="event_bus"):
        out = await bus._run_code_executor("x", "")
    assert out == "ran", "a pre-v21 monitor stopped firing"
    assert any("unknown" in r.message for r in caplog.records)


def test_the_fire_path_passes_the_installer_through():
    """`_fire_action` has the monitor row in scope and is the only caller. A
    coroutine that accepts `installed_by` and a caller that never supplies it
    would satisfy every test above while logging "unknown" forever."""
    src = _BUS.read_text(encoding="utf-8")
    call = src[src.index("self._run_code_executor("):]
    call = call[:call.index(")") + 1]
    assert "installed_by" in call, (
        f"_fire_action does not pass the installer through: {call!r}"
    )


# ─── the dependency is inverted, not relocated ───────────────────────────────

def test_the_event_bus_imports_neither_actions_nor_brain():
    """P4b removed `automation -> actions` by importing `brain.turn` instead,
    and that was the same inversion moved one layer up: the order is
    `... -> automation -> actions -> brain -> main`, so `brain` is above
    `automation` too. Honest accounting, and now fixed properly -- `main.py`
    owns both sides and hands the callable down."""
    src = _BUS.read_text(encoding="utf-8")
    reaching = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.ImportFrom) and node.module:
            parts = node.module.split(".")
            if "actions" in parts or "brain" in parts:
                reaching.append(node.module)
        elif isinstance(node, ast.Import):
            reaching += [a.name for a in node.names
                         if {"actions", "brain"} & set(a.name.split("."))]
    assert not reaching, (
        f"event_bus.py imports {reaching}; the dispatcher is injected so it "
        f"needs neither"
    )


def test_main_injects_the_dispatcher():
    """An injection point nothing calls leaves every monitor inert -- and the
    refusal below would make that look like a deliberate safety feature."""
    src = (_ROOT / "assistant" / "main.py").read_text(encoding="utf-8")
    assert "set_turn_dispatcher" in src, (
        "main.py never installs the dispatcher, so no monitor can run"
    )
    install = src.index("set_turn_dispatcher")
    start = src.index("_event_bus.start(")
    assert install < start, (
        "the dispatcher is installed after the bus starts, so a monitor firing "
        "in between is refused"
    )


@pytest.mark.asyncio
async def test_a_bus_with_no_dispatcher_refuses_to_run(monkeypatch):
    """**Fail closed.** A fired monitor with no dispatcher installed must not
    fall back to inventing its own authority -- that is the entire thing this
    indirection exists to prevent. It says so instead, which is also honest
    about not having run."""
    called = []

    async def _spy(*a, **kw):
        called.append(a)
        return "ran"

    monkeypatch.setattr(A, "execute", _spy)

    from assistant.automation.event_bus import EventBus
    bus = EventBus()                       # no dispatcher installed

    out = await bus._run_code_executor("do something", "local")
    assert not called, "a monitor ran with no dispatcher installed"
    assert "could not run" in out.lower(), (
        f"the refusal does not say it did not run: {out!r}"
    )
