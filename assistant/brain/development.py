"""What changed recently, read from git. It answers; it never decides.

TENKA-v2 §17.P16, and §13.1's last row: *"What changed recently?"* is answered
from git development history, and *never* from a runtime capability claim.

**The direction of that sentence is the whole phase.** Git tells her what was
*worked on*. It must never tell her what she *can do*, what she is *allowed* to
do, or what is *true*. Those come from the affordance registry, the capability
enum and the fact store respectively, and a commit message is none of them --
it is prose somebody typed, and in a repository with more than one contributor
it is prose somebody *else* typed.

Five separate things it must not touch, asserted as five separate tests
(`test_development_history.py`), because one "git is read-only" test would pass
while any single one of them leaked:

    a Brain decision          resolution, planning, dispatch
    capability availability   what she can do
    authorization             what she may do
    Task execution            what actually runs
    memory truth              what she believes

**Commit subjects are untrusted text.** They reach a model, so they are fenced
by `core/fence.py` on the way out (C1) -- a commit message reading "ignore
previous instructions" is exactly the shape the fence exists for, and this repo
happens to contain commit subjects written in TENKA's own voice, which is worse
rather than better.

**The subprocess is bounded in four ways**, none of them optional: an explicit
argument list (never a shell), a timeout, a commit cap, and a working directory
pinned to the package's own repository. A self-knowledge read that can hang is
a turn that can hang.
"""
from __future__ import annotations

import logging
import pathlib
import subprocess
import sys

logger = logging.getLogger("brain.development")

# Bounds. A read of her own history is a nicety; it may not cost a turn.
_TIMEOUT_SECONDS = 3.0
_MAX_COMMITS = 10
_MAX_SUBJECT = 120

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent


def recent_changes(limit: int = 5) -> str:
    """Recent commit subjects, fenced, or "" when git cannot answer.

    Returns "" rather than raising or explaining: `SelfKnowledge.answer()`
    turns an empty read into the standard admission, and a tree with no git --
    a released copy, an export, a zip download -- is a normal state rather than
    an error worth narrating.
    """
    lines = _subjects(limit)
    if not lines:
        return ""

    from ..core.fence import render_untrusted_block

    # Fenced because it reaches a model and nobody here wrote it. In this
    # repository the subjects are partly TENKA's own voice, which makes an
    # unfenced replay *more* confusing rather than less: she would be reading
    # her own past sentences as though they were instructions now.
    return render_untrusted_block("\n".join(lines), label="git_history")


def _subjects(limit: int) -> "list[str]":
    """`<date> <subject>` for the last `limit` commits."""
    count = max(1, min(int(limit or 1), _MAX_COMMITS))
    try:
        proc = subprocess.run(
            # An explicit argument list: no shell, so nothing here can be
            # made to interpret a metacharacter.
            ["git", "log", f"-{count}", "--format=%cs %s"],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_SECONDS,
            # Windows: keep a console window from flashing on every ask.
            creationflags=(subprocess.CREATE_NO_WINDOW
                           if sys.platform == "win32" else 0),
        )
    except Exception as e:
        logger.debug(f"[DEV] git unavailable: {e}")
        return []

    if proc.returncode != 0:
        # Not a repository, or git is not installed. Both are ordinary.
        logger.debug(f"[DEV] git exited {proc.returncode}")
        return []

    return [
        line.strip()[:_MAX_SUBJECT]
        for line in (proc.stdout or "").splitlines()
        if line.strip()
    ][:count]
