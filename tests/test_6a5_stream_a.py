"""Milestone 6a.5, stream A: the EXECUTE capability boundary.

Spec: `.superpowers/sdd/2026-08-16-milestone6a5-security/spec.md` §5.1.

Each test keeps the reasoning that motivated it. The boundary this file pins
is new, so every default in it is fail-closed: an unset grant set refuses, an
unclassified intent refuses, and a ceiling is a literal rather than something
derived from the enum.
"""
import ast
import pathlib

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
