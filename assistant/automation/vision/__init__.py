"""Vision-based desktop automation (agent loop, verifier, TODO classifier).

Formerly a single ``vision.py`` module. Callers that do
``from ..automation import vision`` continue to work — attribute access
is proxied to the ``agent`` submodule via ``__getattr__``.
"""

from .agent import run_computer_task  # noqa: F401 — public API


def __getattr__(name):
    from . import agent as _agent
    return getattr(_agent, name)
