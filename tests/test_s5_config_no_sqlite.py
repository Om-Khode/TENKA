"""Verify config.py no longer imports or accesses SQLite (S5 layering rule)."""

import ast
from pathlib import Path

import pytest

CONFIG_PATH = Path(__file__).parent.parent / "assistant" / "config.py"


def _read_source() -> str:
    """Read config.py source, stripping BOM if present."""
    source = CONFIG_PATH.read_text(encoding="utf-8-sig")
    return source


def test_config_no_sqlite_imports_at_module_level():
    """config.py source must not import personality, preferences,
    settings, or memory at module level."""
    source = _read_source()
    tree = ast.parse(source)

    banned_modules = {"settings", "personality", "preferences", "memory"}

    for stmt in tree.body:
        if isinstance(stmt, ast.ImportFrom):
            if stmt.module:
                for banned in banned_modules:
                    assert banned not in stmt.module, (
                        f"config.py has module-level import of '{banned}' — "
                        f"violates S5 layering rule"
                    )
        if isinstance(stmt, ast.Import):
            for alias in stmt.names:
                for banned in banned_modules:
                    assert banned not in alias.name, (
                        f"config.py has module-level import of '{banned}'"
                    )


def test_config_no_eager_init_settings_db():
    """config.py must not call init_settings_db() eagerly at module level."""
    source = _read_source()
    tree = ast.parse(source)

    for stmt in tree.body:
        if isinstance(stmt, ast.Try):
            for sub in ast.walk(stmt):
                if isinstance(sub, ast.Call):
                    func = sub.func
                    if isinstance(func, ast.Attribute) and func.attr == "init_settings_db":
                        pytest.fail("config.py still eagerly calls init_settings_db()")
                    if isinstance(func, ast.Name) and func.id == "init_settings_db":
                        pytest.fail("config.py still eagerly calls init_settings_db()")


def test_config_imports_runtime_config_from_core():
    """config.py should import setting resolution from core.runtime_config."""
    source = _read_source()
    assert "from .core.runtime_config import" in source


def test_config_no_direct_settings_usage():
    """config.py must not directly call settings.get() at module level."""
    source = _read_source()
    tree = ast.parse(source)

    for stmt in tree.body:
        if isinstance(stmt, (ast.Expr, ast.Assign)):
            for node in ast.walk(stmt):
                if isinstance(node, ast.Attribute):
                    if (isinstance(node.value, ast.Name) and
                            node.value.id == "settings" and
                            node.attr == "get"):
                        pytest.fail("config.py directly calls settings.get()")
