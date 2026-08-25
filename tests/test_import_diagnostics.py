"""A blocked native extension is not a missing package.

`assistant/core/import_diagnostics.py` exists because the log said otherwise.
On 2026-08-25 a start produced these three lines, four seconds apart, in one
process:

    21:07:35 [tts]    ERROR:   Kokoro initialization failed: kokoro not installed
    21:07:39 [faiss]  INFO:    Successfully loaded faiss.
    21:07:43 [memory] WARNING: Vector search dependencies (faiss,
                               sentence-transformers) not found.

faiss loaded and was then reported missing. Every one of `kokoro`, `torch`,
`faiss`, `sentence_transformers`, `spacy` and `speechbrain` imported fine at a
prompt immediately afterwards. Windows Smart App Control had blocked
`torch\\_C.pyd` for that run -- wheel extensions are unsigned, so SAC decides on
cloud reputation and the same file can be admitted on one start and refused on
the next. Both handlers caught `ImportError`, discarded it, and asserted a
cause neither of them had checked.

The cost is not cosmetic: "not installed" tells the operator to run pip, pip
reports it is already satisfied, the next start works because the reputation
lookup went the other way, and the actual cause is never found.

Run with:  py -3.11 -m pytest tests/test_import_diagnostics.py -v
"""
import ast
import pathlib
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from assistant.core.import_diagnostics import (  # noqa: E402
    _installed_version, describe_import_failure,
)

# `pytest` is installed wherever this runs, and is not one of the optional
# dependencies the module was written for -- so it stands in for "present"
# without making the test depend on TENKA's own optional stack being healthy.
_PRESENT = "pytest"
_ABSENT = "tenka-package-that-does-not-exist"

_BLOCKED = ImportError(
    "DLL load failed while importing _C: A dynamic link library (DLL) "
    "initialization routine failed."
)


# ─── the distinction the module exists to make ───────────────────────────────

def test_a_missing_package_is_reported_as_missing():
    msg = describe_import_failure(ImportError("No module named x"), _ABSENT)
    assert "is not installed" in msg
    assert f"pip install {_ABSENT}" in msg


def test_an_installed_package_is_never_called_missing():
    """**The whole point.** The old handlers said "not installed" whatever the
    exception was, and that sentence is what sent the operator to pip."""
    msg = describe_import_failure(_BLOCKED, _PRESENT)
    assert "not installed" not in msg, msg
    assert "is installed but the import failed" in msg
    assert "not a missing package" in msg


def test_the_loader_s_own_words_survive():
    """The handler discarded the exception entirely. Whatever the diagnosis
    above concludes, the raw text is the only thing that can identify *which*
    extension was refused -- `_C.pyd` in this case."""
    msg = describe_import_failure(_BLOCKED, _PRESENT)
    assert "DLL load failed while importing _C" in msg
    assert "ImportError" in msg


def test_a_mixed_result_names_only_what_is_actually_absent():
    msg = describe_import_failure(ImportError("boom"), _PRESENT, _ABSENT)
    assert f"pip install {_ABSENT}" in msg
    assert f"pip install {_PRESENT}" not in msg, (
        "the message tells the operator to install something already present")


def test_the_installed_version_is_named():
    """So a version mismatch is visible in the same line, without a second
    round trip asking the operator to run `pip show`."""
    version = _installed_version(_PRESENT)
    assert version, "the probe cannot see a package that is certainly installed"
    msg = describe_import_failure(_BLOCKED, _PRESENT)
    assert version in msg


def test_a_package_that_is_not_there_probes_as_none():
    assert _installed_version(_ABSENT) is None


# ─── the Windows hint, only when the evidence is there ───────────────────────

@pytest.mark.parametrize("text", [
    "DLL load failed while importing _C",
    "%1 is not a valid Win32 application",
    "Access is denied",
    "initialization routine failed",
])
def test_a_loader_level_failure_gets_the_unsigned_extension_hint(text):
    msg = describe_import_failure(ImportError(text), _PRESENT)
    assert "Smart App Control" in msg, msg
    assert "unsigned" in msg


def test_an_ordinary_import_error_does_not_get_the_hint():
    """**The direction that makes the hint worthless.** Appending it to every
    failure turns it into boilerplate nobody reads, and would point at a
    security policy for a plain circular import."""
    msg = describe_import_failure(
        ImportError("cannot import name 'foo' from partially initialized "
                    "module 'bar' (most likely due to a circular import)"),
        _PRESENT,
    )
    assert "Smart App Control" not in msg, msg
    assert "circular import" in msg


def test_the_probe_does_not_reimport_the_thing_that_just_failed():
    """Re-attempting the import here would either fail identically or -- worse
    -- succeed on the retry and make the message contradict the event it is
    describing. Metadata only."""
    import inspect

    from assistant.core import import_diagnostics

    src = inspect.getsource(import_diagnostics._installed_version)
    tree = ast.parse(src.lstrip())
    imported = {
        n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
    } | {
        a.name for n in ast.walk(tree) if isinstance(n, ast.Import)
        for a in n.names
    }
    assert imported == {"importlib.metadata"}, (
        f"the probe reaches for more than packaging metadata: {imported}")
    calls = {n.func.id for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "__import__" not in calls and "import_module" not in calls


# ─── the two call sites stopped guessing ─────────────────────────────────────

def test_no_handler_still_hardcodes_not_installed():
    """The sentence that caused this. Both handlers asserted it from inside an
    `except ImportError`, where the one thing they could not know was whether
    the package was there."""
    offenders = []
    for rel in ("assistant/io/audio/tts.py",
                "assistant/storage/repos/memory.py"):
        path = _ROOT / rel
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler):
                continue
            body = ast.unparse(node)
            if "describe_import_failure" in body:
                continue
            if "not installed" in body or "not found" in body:
                offenders.append((rel, node.lineno))
    assert not offenders, (
        f"an except handler still asserts a package is absent without "
        f"checking: {offenders}. Use `describe_import_failure(exc, ...)`.")


@pytest.mark.parametrize("rel,expected", [
    ("assistant/io/audio/tts.py", {"kokoro", "torch"}),
    ("assistant/storage/repos/memory.py",
     {"faiss-cpu", "sentence-transformers", "torch"}),
])
def test_each_call_site_names_the_distributions_it_depends_on(rel, expected):
    """`torch` is in both lists deliberately -- it is what was actually blocked,
    and neither handler imports it directly, so neither would have named it
    without being told to."""
    tree = ast.parse((_ROOT / rel).read_text(encoding="utf-8"))
    named = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and getattr(node.func, "id", None) in (
                    "describe_import_failure", "_describe")):
            continue
        for arg in node.args[1:]:
            if isinstance(arg, ast.Constant):
                named.add(arg.value)
    assert named == expected, f"{rel} declares {named}, expected {expected}"
