"""slash_commands._resolve_running_main() — finding the module the daemon
is actually running, not a second, disconnected import of it.

Background (see slash_commands.py's docstring on the helper for the full
story): `start_assistant.bat` launches the daemon as `python -m
assistant.main`. Running a module with `-m` executes it as `__main__` but
does NOT register it under its dotted name (`"assistant.main"`) in
`sys.modules`. So a later plain `import assistant.main` -- which is what
every call site here used to do -- finds nothing registered and imports the
file a *second* time, producing a fresh module object whose globals sit at
their initial values, forever disconnected from the running daemon. That is
the exact bug `/studio pair` hit live: it always reported "the Studio
daemon is not running" because it was reading `_studio_pair_store` off the
wrong module object.

A normal test process can never reproduce this by accident -- every test
here imports `assistant.main` normally, which registers it under its
dotted name on first import, so there is only ever one module object. The
tests below drive the resolver directly with a fabricated `__main__` to
prove the discriminator logic in isolation; test_resolve_running_main
_subprocess.py separately proves it against a real `-m` launch, which is
the scenario a normal test process structurally cannot produce.
"""
from __future__ import annotations

import sys
import types
from types import SimpleNamespace

from assistant import slash_commands


def test_resolves_to_running_main_when_dash_m_spec_name_matches(monkeypatch):
    """The fabricated `__main__` mimics `python -m assistant.main`: a
    `__spec__` whose `.name` is `"assistant.main"`. The resolver must return
    this exact object -- the one carrying the daemon's real globals -- not
    the result of a fresh import.
    """
    fake_main = types.ModuleType("__main__")
    fake_main.__spec__ = SimpleNamespace(name="assistant.main")
    fake_main._studio_pair_store = "running-daemon-store"
    monkeypatch.setitem(sys.modules, "__main__", fake_main)

    resolved = slash_commands._resolve_running_main()

    assert resolved is fake_main
    assert resolved._studio_pair_store == "running-daemon-store"


def test_falls_back_to_import_when_main_belongs_to_another_program(monkeypatch):
    """A `__main__` whose spec name is something else entirely means TENKA
    is embedded as a library inside some other entry point -- that
    `__main__` must not be mistaken for the assistant daemon.
    """
    fake_main = types.ModuleType("__main__")
    fake_main.__spec__ = SimpleNamespace(name="some_other_program.cli")
    monkeypatch.setitem(sys.modules, "__main__", fake_main)

    import assistant.main as real_main

    resolved = slash_commands._resolve_running_main()

    assert resolved is real_main
    assert resolved is not fake_main


def test_falls_back_to_import_when_spec_is_none(monkeypatch):
    """An interactive interpreter session sets `__main__.__spec__ = None` --
    the resolver must not raise on the attribute access, and must fall back
    to a plain import.
    """
    fake_main = types.ModuleType("__main__")
    fake_main.__spec__ = None
    monkeypatch.setitem(sys.modules, "__main__", fake_main)

    import assistant.main as real_main

    resolved = slash_commands._resolve_running_main()

    assert resolved is real_main


def test_falls_back_to_import_when_spec_attribute_is_absent(monkeypatch):
    """A frozen build or an unusually constructed `__main__` may lack a
    `__spec__` attribute entirely (not merely set it to None) -- the
    `getattr(..., None)` guard must handle that too.
    """
    fake_main = types.ModuleType("__main__")
    delattr(fake_main, "__spec__")  # types.ModuleType always starts with one
    monkeypatch.setitem(sys.modules, "__main__", fake_main)

    import assistant.main as real_main

    resolved = slash_commands._resolve_running_main()

    assert resolved is real_main


def test_falls_back_to_import_when_main_is_entirely_absent(monkeypatch):
    """`sys.modules["__main__"]` not existing at all is not a realistic
    CPython state, but the resolver reads via `.get()` specifically so this
    can never KeyError -- prove it doesn't.
    """
    monkeypatch.delitem(sys.modules, "__main__", raising=False)

    import assistant.main as real_main

    resolved = slash_commands._resolve_running_main()

    assert resolved is real_main


def test_pytest_process_matches_the_fallback_path():
    """Sanity check on the premise: under pytest, `__main__` is pytest's own
    entry point, never `assistant.main` -- so an unpatched call must already
    take the fallback branch and return the real, singly-imported module.
    """
    import assistant.main as real_main

    resolved = slash_commands._resolve_running_main()

    assert resolved is real_main
