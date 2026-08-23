"""The real runtimes accept everything their protocols and callers pass.

These are `Protocol`s, so nothing enforces them at runtime and `isinstance`
against a `runtime_checkable` Protocol checks only that the *methods exist* --
never their signatures. A concrete implementation can therefore drift from its
protocol, and from the routes that call it, and every unit test still passes,
because the tests inject a fake that was updated in the same commit as the
protocol.

That is not hypothetical. KI-30's first attempt added `ceiling` to
`ChatRuntime.send`, to `tests/fakes/studio_runtime.py`, and to the route that
calls it -- and not to `LiveChatRuntime.send`, the only implementation that
runs in production. 201 tests passed. `POST /v1/chat` then 500'd on **every**
request:

    TypeError: LiveChatRuntime.send() got an unexpected keyword argument 'ceiling'

The fake and the real diverged, and nothing but starting the app composed them.
This file composes them without starting the app.

Signature-level, not call-level: a call-level test needs a live turn, and the
whole point is to catch the mismatch in the suite rather than in the log.

Run with:  py -3.11 -m pytest tests/test_runtime_signature_conformance.py -v
"""
import inspect
import pathlib
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _accepted_kwargs(fn) -> set[str]:
    """Keyword names `fn` will accept, or `{"**"}` if it takes **kwargs."""
    sig = inspect.signature(fn)
    names = set()
    for name, p in sig.parameters.items():
        if name == "self":
            continue
        if p.kind is inspect.Parameter.VAR_KEYWORD:
            return {"**"}
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                      inspect.Parameter.KEYWORD_ONLY):
            names.add(name)
    return names


def _assert_conforms(impl, proto, method: str):
    want = _accepted_kwargs(getattr(proto, method))
    got = _accepted_kwargs(getattr(impl, method))
    if "**" in got:
        return
    missing = want - got
    assert not missing, (
        f"{impl.__name__}.{method}() does not accept {sorted(missing)}, which "
        f"{proto.__name__}.{method}() declares. A caller passing it gets a "
        f"TypeError at runtime -- Protocols are not enforced, and "
        f"isinstance() against a runtime_checkable Protocol checks method "
        f"names only, never signatures."
    )


# ─── the pair that broke ─────────────────────────────────────────────────────

def test_live_chat_runtime_conforms_to_the_chat_runtime_protocol():
    from assistant.actions.studio_runtime import LiveChatRuntime
    from assistant.io.api.runtime import ChatRuntime

    _assert_conforms(LiveChatRuntime, ChatRuntime, "send")


def test_the_studio_dispatch_conforms_to_the_chat_dispatch_protocol():
    """The other half of the same chain: `LiveChatRuntime.send` forwards to
    `ChatDispatch.submit`, implemented by `main.py`'s `_StudioDispatch`."""
    from assistant.actions.studio_runtime import ChatDispatch
    from assistant.main import _StudioDispatch

    _assert_conforms(_StudioDispatch, ChatDispatch, "submit")


def test_the_route_passes_nothing_the_runtime_cannot_take():
    """Closes the third side of the triangle. The protocol and the
    implementation can agree with each other and still both be behind the
    route, which is what actually calls them."""
    import ast

    from assistant.actions.studio_runtime import LiveChatRuntime

    src = (pathlib.Path(_ROOT) / "assistant" / "io" / "api" / "routes"
           / "chat.py").read_text(encoding="utf-8")
    tree = ast.parse(src)

    calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and getattr(n.func, "attr", None) == "send"
    ]
    assert calls, (
        "no `.send(...)` call found in routes/chat.py -- the walk found "
        "nothing and this test would pass forever"
    )

    accepted = _accepted_kwargs(LiveChatRuntime.send)
    for call in calls:
        passed = {kw.arg for kw in call.keywords if kw.arg}
        unknown = passed - accepted
        assert not unknown, (
            f"routes/chat.py calls .send() with {sorted(unknown)}, which "
            f"LiveChatRuntime.send() does not accept. This is the exact shape "
            f"that 500'd every POST /v1/chat."
        )


# ─── and the fakes, so they cannot drift the other way ──────────────────────

def test_the_test_fake_accepts_everything_the_real_runtime_does():
    """A fake that accepts *less* than the real one hides a working feature;
    a fake that accepts *more* hides a broken one. The second is what happened:
    the fake grew `ceiling` and the real implementation did not, so the suite
    proved the route worked against something production never runs."""
    from assistant.actions.studio_runtime import LiveChatRuntime
    from tests.fakes.studio_runtime import FakeChatRuntime

    real = _accepted_kwargs(LiveChatRuntime.send)
    fake = _accepted_kwargs(FakeChatRuntime.send)
    if "**" in fake:
        return
    assert real <= fake or fake >= real - {"**"}, (
        f"fake accepts {sorted(fake)}, real accepts {sorted(real)}"
    )
    drifted = fake - real
    assert not drifted, (
        f"the fake accepts {sorted(drifted)} and the real runtime does not. "
        f"Every test using this fake is passing against a signature that does "
        f"not exist in production."
    )


@pytest.mark.parametrize("proto_name,impl_path,method", [
    ("ChatRuntime", "assistant.actions.studio_runtime:LiveChatRuntime", "send"),
    ("ChatDispatch", "assistant.main:_StudioDispatch", "submit"),
])
def test_the_conformance_check_itself_is_not_vacuous(proto_name, impl_path,
                                                     method):
    """If `_accepted_kwargs` ever returned an empty set for a protocol, every
    assertion above would hold trivially."""
    import importlib

    from assistant.actions.studio_runtime import ChatDispatch
    from assistant.io.api.runtime import ChatRuntime

    proto = {"ChatRuntime": ChatRuntime, "ChatDispatch": ChatDispatch}[proto_name]
    mod, _, cls = impl_path.partition(":")
    impl = getattr(importlib.import_module(mod), cls)

    assert len(_accepted_kwargs(getattr(proto, method))) >= 3, (
        f"{proto_name}.{method} appears to take almost no arguments -- the "
        f"introspection is broken and the conformance tests measure nothing"
    )
    assert len(_accepted_kwargs(getattr(impl, method))) >= 3
