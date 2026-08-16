"""Milestone 6a.5, stream A: the EXECUTE capability boundary.

Spec: `.superpowers/sdd/2026-08-16-milestone6a5-security/spec.md` §5.1.

Each test keeps the reasoning that motivated it. The boundary this file pins
is new, so every default in it is fail-closed: an unset grant set refuses, an
unclassified intent refuses, and a ceiling is a literal rather than something
derived from the enum.
"""
import ast
import pathlib

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent


# ─── A1: Capability lives in core/ ───────────────────────────────────────
def test_capability_lives_in_core_and_imports_nothing():
    """core/ is the layer that imports nothing. The enum has to live there for
    actions/ to read it without an import-linter exemption."""
    src = (_ROOT / "assistant" / "core" / "capabilities.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
    non_stdlib = [n for n in imports
                  if not (isinstance(n, ast.Import) and
                          all(a.name in ("enum",) for a in n.names))]
    assert not non_stdlib, f"core/capabilities.py must import only enum: {non_stdlib}"


def test_vault_still_exports_capability():
    """Every existing `from .vault import Capability` must keep working."""
    from assistant.io.api.vault import Capability as FromVault
    from assistant.core.capabilities import Capability as FromCore
    assert FromVault is FromCore


# ─── A2: EXECUTE, and ceilings as explicit literals ──────────────────────
def test_execute_exists():
    from assistant.core.capabilities import Capability
    assert Capability.EXECUTE.value == "execute"


@pytest.mark.parametrize("name", ["tailnet", "funnel", "quick"])
def test_no_transport_carries_execute(name):
    """funnel is the open internet and CHAT_SEND reaches every intent. The
    ceiling is what stops a pair code becoming code execution on this machine."""
    from assistant.core.capabilities import Capability
    from assistant.io.api.policy import POLICIES
    assert Capability.EXECUTE not in POLICIES[name].ceiling


@pytest.mark.parametrize("name", ["tailnet", "funnel", "quick"])
def test_no_transport_carries_system_control(name):
    """PATCH /v1/settings turns the camera on and speaker verification off."""
    from assistant.core.capabilities import Capability
    from assistant.io.api.policy import POLICIES
    assert Capability.SYSTEM_CONTROL not in POLICIES[name].ceiling


def test_local_carries_everything():
    """The operator at the keyboard keeps full power; this milestone is not a
    downgrade of the local path."""
    from assistant.core.capabilities import Capability
    from assistant.io.api.policy import POLICIES
    assert POLICIES["local"].ceiling == frozenset(Capability)


def test_no_ceiling_is_derived_from_the_enum():
    """A ceiling spelled `frozenset(Capability)` grants every future capability
    automatically, over exactly the listeners that must never get one for free.
    Only `local` may be enum-derived, and it is asserted by value above."""
    src = (_ROOT / "assistant" / "io" / "api" / "policy.py").read_text(encoding="utf-8")
    assert "_ALL_CAPABILITIES" not in src


def test_effective_can_only_narrow():
    """policy.py's central invariant. Pinned so a future raise mechanism (6b)
    cannot quietly turn the intersection into a union."""
    from assistant.core.capabilities import Capability
    from assistant.io.api.policy import POLICIES, effective
    every = frozenset(Capability)
    for policy in POLICIES.values():
        assert effective(every, policy) <= policy.ceiling
        assert effective(frozenset(), policy) == frozenset()
