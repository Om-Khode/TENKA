"""Say why an optional import failed, instead of guessing.

Two subsystems degrade quietly when an optional dependency will not import,
and both said the same thing about it: *not installed*. On 2026-08-25 that was
false in a way that cost real debugging time. The log read:

    21:07:35 [tts]   ERROR:   Kokoro initialization failed: kokoro not installed
    21:07:39 [faiss] INFO:    Successfully loaded faiss.
    21:07:43 [memory] WARNING: Vector search dependencies (faiss,
                               sentence-transformers) not found.

faiss loaded, and four seconds later the same process reported it missing.
Nothing was missing. Windows Smart App Control had blocked `torch\\_C.pyd` --
a `.pyd` out of a PyPI wheel is unsigned, so it is admitted or refused on cloud
reputation, which is why it happens on some starts and not others. `torch`
failing takes Kokoro and `sentence_transformers` down with it, and both
handlers reported the symptom as an absence.

An operator reading "not installed" installs it. It is already there, the next
start works because the reputation lookup went the other way, and the real
cause is never found.

So: keep the graceful degradation, and stop asserting a cause the handler does
not know. This asks the packaging metadata whether the distribution is present
and phrases the failure accordingly -- missing, or present-but-unloadable with
the loader's own words attached.

Deliberately generic. It knows nothing about which package it is describing,
and takes the distribution names from its caller.
"""
from __future__ import annotations

_WINDOWS_LOADER_HINTS = (
    "dll load failed",
    "is not a valid win32 application",
    "access is denied",
    "initialization routine failed",
)


def _installed_version(distribution: str) -> str | None:
    """The installed version of `distribution`, or None if it is not there.

    Metadata, not an import: importing is the thing that just failed, and
    re-attempting it here would either fail the same way or -- worse -- succeed
    on a second try and make the message disagree with the event it describes.
    """
    try:
        from importlib.metadata import PackageNotFoundError, version
    except Exception:  # pragma: no cover - stdlib since 3.8
        return None
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None
    except Exception:
        # A damaged dist-info is not the question being asked.
        return None


def describe_import_failure(exc: BaseException, *distributions: str) -> str:
    """One sentence naming what actually happened, for a log line.

    `distributions` are pip names, in the order they should be reported. The
    caller knows them; this does not try to derive them from the exception,
    because an import name and a distribution name are frequently different
    (`faiss` / `faiss-cpu`, `sentence_transformers` / `sentence-transformers`)
    and guessing wrong is how the message becomes misleading again.
    """
    detail = f"{type(exc).__name__}: {exc}".strip()

    present = [(d, v) for d in distributions
               if (v := _installed_version(d)) is not None]
    absent = [d for d in distributions if _installed_version(d) is None]

    if absent and not present:
        return (f"{', '.join(absent)} is not installed "
                f"-- pip install {' '.join(absent)} ({detail})")

    installed = ", ".join(f"{d} {v}" for d, v in present)
    is_are = "is" if len(present) == 1 else "are"

    if absent:
        return (f"{installed} {is_are} installed but {', '.join(absent)} is "
                f"not -- pip install {' '.join(absent)} ({detail})")

    message = (f"{installed} {is_are} installed but the import failed -- this "
               f"is not a missing package ({detail})")

    if any(hint in detail.lower() for hint in _WINDOWS_LOADER_HINTS):
        message += (
            ". A native extension was refused by the loader. On Windows the "
            "usual cause is a security policy blocking an unsigned .pyd: "
            "Smart App Control admits binaries on publisher reputation, and "
            "wheel extensions are unsigned, so the same file can load on one "
            "start and be blocked on the next. Windows Security > App & "
            "browser control will say if it blocked something"
        )

    return message
