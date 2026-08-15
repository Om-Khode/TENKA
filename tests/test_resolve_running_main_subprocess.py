"""Subprocess reproduction of the `-m` module-identity bug.

This is the test that would have caught the original bug: every other test
in this suite imports `assistant.main` normally (`import assistant.main`),
which registers it in `sys.modules` under its dotted name on first import --
so a test *process* only ever sees one module object, and the split this
bug depends on cannot appear no matter how the test is written. Reproducing
it for real requires an actual `python -m ...` launch, in a genuinely
separate process.

The real `assistant/main.py` cannot be launched directly here: its
`if __name__ == "__main__":` block starts the full daemon -- audio device
init, Playwright driver warmup, the wake-word listener, a blocking chat
input thread -- none of which is safe, fast, or deterministic inside a test
process (and this project's own test rules ban driving real hardware from
tests). So this test launches a MINIMAL stand-in `assistant/main.py` under
`-m`, in a scratch package whose `__init__.py` extends its own `__path__`
to fall back to the REAL assistant package on disk. That means every
module actually under test -- `assistant.slash_commands` (unmodified,
verbatim production code) and everything it imports at module scope
(`config`, `settings`, `storage.db`, `storage.repos.settings`) -- is the
genuine production code. Only the entry-point file itself (the part that
would otherwise start hardware) is swapped for a lightweight stand-in that
reproduces just the property the bug depends on: being executed as
`__main__` with `__spec__.name == "assistant.main"`, without Python ever
registering it under that dotted name.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import assistant  # only to locate the real package directory on disk

REAL_ASSISTANT_DIR = Path(assistant.__file__).resolve().parent


_INIT_TEMPLATE = (
    "# Stand-in for assistant/__init__.py -- extends this package's search\n"
    "# path to fall back to the REAL assistant package on disk, so every\n"
    "# submodule except main.py is the genuine, unmodified production code.\n"
    "__path__.append(r\"{real_dir}\")\n"
)

_MAIN_STANDIN = '''"""Lightweight stand-in for assistant/main.py's entry point.

See test_resolve_running_main_subprocess.py for why the real main.py
cannot be launched here. This file reproduces only the property the bug
and its fix depend on: it is executed as `-m assistant.main`, i.e. as
`__main__` with `__spec__.name == "assistant.main"`, without Python
registering it under that dotted name in sys.modules.
"""

# Mirrors assistant.main's real `_studio_pair_store` global: a plain,
# module-level default that only the "running daemon" ever overwrites.
# Distinguishing on __name__ lets the second, plain import below (which
# re-executes this same file under the name "assistant.main" instead of
# "__main__") observe the static default instead.
if __name__ == "__main__":
    _studio_pair_store = "SENTINEL_RUNNING_STORE"
else:
    _studio_pair_store = None

if __name__ == "__main__":
    import sys as _sys

    from assistant.slash_commands import _resolve_running_main

    resolved = _resolve_running_main()
    print("RESOLVED_IS_DUNDER_MAIN=" + str(resolved is _sys.modules["__main__"]))
    print("RESOLVED_STORE=" + str(getattr(resolved, "_studio_pair_store", "MISSING")))

    # Prove the bug's precondition still holds: a naive plain
    # `import assistant.main` -- what every call site used to do -- really
    # does produce a SECOND, disconnected module object whose globals sit
    # at their static defaults. That is the exact "daemon is not running"
    # symptom hit live, reached through a stale/wrong module instead of a
    # missing store.
    import assistant.main as second_import
    print("SECOND_IMPORT_IS_DIFFERENT_OBJECT=" + str(second_import is not resolved))
    print("SECOND_IMPORT_STORE=" + str(getattr(second_import, "_studio_pair_store", "MISSING")))
'''


def test_resolver_finds_the_running_main_across_a_real_dash_m_launch(tmp_path):
    fake_pkg = tmp_path / "assistant"
    fake_pkg.mkdir()

    (fake_pkg / "__init__.py").write_text(
        _INIT_TEMPLATE.format(real_dir=str(REAL_ASSISTANT_DIR)),
        encoding="utf-8",
    )
    (fake_pkg / "main.py").write_text(_MAIN_STANDIN, encoding="utf-8")

    env = dict(os.environ)
    env["PYTHONPATH"] = str(tmp_path)

    result = subprocess.run(
        [sys.executable, "-m", "assistant.main"],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, (
        f"child process failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    out = result.stdout

    # The fix: the resolver must find the *running* __main__, carrying the
    # sentinel only the running daemon would have set.
    assert "RESOLVED_IS_DUNDER_MAIN=True" in out, out
    assert "RESOLVED_STORE=SENTINEL_RUNNING_STORE" in out, out

    # The bug's precondition, still real: a naive plain import lands on a
    # different object entirely, with the store back at its static default.
    assert "SECOND_IMPORT_IS_DIFFERENT_OBJECT=True" in out, out
    assert "SECOND_IMPORT_STORE=None" in out, out
