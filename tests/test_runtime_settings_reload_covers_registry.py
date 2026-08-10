"""Every registered runtime setting must actually be refreshed at startup.

config.py resolves each `_runtime_setting()` at import, long before
storage.db.init_db() runs -- so at import time `_get_db_value()` returns None
for everything and every constant takes its hardcoded default. main.py fixes
that by calling `config.reload_runtime_settings()` after init_db(), which
re-resolves the whole registry and reassigns the module globals.

That reassignment is a hand-written list, and a hand-written mirror of a
registry drifts. It did: the three `studio_api_*` settings landed in
Milestone 5a and were never added, so `/set studio_api_enabled true` wrote
the DB row, `/config` displayed it (it reads the DB live), and the daemon
never started -- config.STUDIO_API_ENABLED was still the import-time False.
Diagnosed live on 2026-08-10 after the setting appeared to have no effect.

This test fails on the omission itself rather than on a symptom, so the next
setting added cannot repeat it.
"""
import inspect
import re

from assistant import config
from assistant.core import runtime_config


# Registered, but deliberately not a config constant. `personality` is
# resolved through the personality module's own store, never read off
# `config`, so there is nothing for reload to reassign -- confirmed by
# `hasattr(config, "PERSONALITY")` being False. Named here rather than
# skipped silently: an exemption nobody can see is how the omission below
# survived in the first place.
NO_MODULE_CONSTANT = {"personality"}


def _is_reassigned(key: str, source: str) -> bool:
    """Whitespace-tolerant: a long assignment wrapped after `get(` is the same
    assignment. Matching a one-line spelling only would fail on formatting
    rather than on substance, and a check that fails for the wrong reason is
    one people learn to edit around."""
    return re.search(rf'new_values\.get\(\s*"{re.escape(key)}"', source) is not None


def test_every_registered_setting_is_reassigned_by_reload():
    source = inspect.getsource(config.reload_runtime_settings)

    missing = [
        key for key in sorted(runtime_config.REGISTRY)
        if key not in NO_MODULE_CONSTANT and not _is_reassigned(key, source)
    ]

    assert not missing, (
        "reload_runtime_settings() re-resolves these from the DB but never "
        "assigns them to a module global, so they keep their import-time "
        "defaults for the life of the process: " + ", ".join(missing)
    )


def test_every_reassigned_setting_is_declared_global():
    """An assignment without a matching `global` silently writes a function
    local instead of the module constant -- the same end state as the bug
    above, with no error anywhere."""
    source = inspect.getsource(config.reload_runtime_settings)

    declared: set[str] = set()
    for line in re.findall(r"^\s*global (.+)$", source, flags=re.MULTILINE):
        declared.update(name.strip() for name in line.split(","))

    assigned = set(re.findall(r"^\s*([A-Z][A-Z0-9_]*) = new_values\.get", source,
                              flags=re.MULTILINE))

    assert assigned <= declared, (
        "assigned in reload_runtime_settings() without a `global` declaration, "
        "so the write lands on a local and the constant never changes: "
        + ", ".join(sorted(assigned - declared))
    )


def test_exemptions_are_real():
    """An exemption must be justified by there genuinely being no constant to
    refresh. Without this, NO_MODULE_CONSTANT becomes a place to hide a real
    omission -- the same failure as an allowlist nobody rechecks."""
    for key in NO_MODULE_CONSTANT:
        assert key in runtime_config.REGISTRY, f"{key} is exempt but not registered at all"
        assert not hasattr(config, key.upper()), (
            f"config.{key.upper()} exists, so {key} has a constant that reload must "
            "refresh -- remove it from NO_MODULE_CONSTANT"
        )


def test_the_studio_keys_specifically_are_covered():
    """The three that actually broke. Named explicitly so a future refactor
    of the generic checks above cannot quietly stop covering them."""
    source = inspect.getsource(config.reload_runtime_settings)
    for key in ("studio_api_enabled", "studio_api_port", "studio_api_origins"):
        assert _is_reassigned(key, source), key
