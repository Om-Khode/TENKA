"""Tests for core/runtime_config.py — setting resolution."""

import os
from unittest.mock import patch

import pytest


def test_setting_returns_default_when_no_db_no_env():
    """With no DB initialized and no env var, return the hardcoded default."""
    from assistant.core import runtime_config
    runtime_config.REGISTRY.clear()

    with patch("assistant.core.runtime_config._get_db_value", return_value=None):
        result = runtime_config.setting(
            "test_key", "default_val", cast=str, description="test"
        )
    assert result == "default_val"


def test_setting_env_overrides_default():
    """Env var (uppercase key) overrides the hardcoded default."""
    from assistant.core import runtime_config
    runtime_config.REGISTRY.clear()

    with patch("assistant.core.runtime_config._get_db_value", return_value=None):
        with patch.dict(os.environ, {"TEST_KEY_2": "from_env"}):
            result = runtime_config.setting(
                "test_key_2", "default_val", cast=str, description="test"
            )
    assert result == "from_env"


def test_setting_db_overrides_env():
    """DB value takes precedence over env var."""
    from assistant.core import runtime_config
    runtime_config.REGISTRY.clear()

    with patch("assistant.core.runtime_config._get_db_value", return_value="from_db"):
        with patch.dict(os.environ, {"TEST_KEY_3": "from_env"}):
            result = runtime_config.setting(
                "test_key_3", "default_val", cast=str, description="test"
            )
    assert result == "from_db"


def test_setting_bool_cast_from_env():
    """Bool cast handles string 'true'/'false' correctly from env."""
    from assistant.core import runtime_config
    runtime_config.REGISTRY.clear()

    with patch("assistant.core.runtime_config._get_db_value", return_value=None):
        with patch.dict(os.environ, {"BOOL_KEY": "true"}):
            result = runtime_config.setting("bool_key", False, cast=bool, description="")
    assert result is True


def test_setting_bool_cast_from_db():
    """Bool cast from DB value (already truthy)."""
    from assistant.core import runtime_config
    runtime_config.REGISTRY.clear()

    with patch("assistant.core.runtime_config._get_db_value", return_value="yes"):
        result = runtime_config.setting("bool_db_key", False, cast=bool, description="")
    assert result is True


def test_setting_registers_metadata():
    """Each call registers key metadata in REGISTRY."""
    from assistant.core import runtime_config
    runtime_config.REGISTRY.clear()

    with patch("assistant.core.runtime_config._get_db_value", return_value=None):
        runtime_config.setting(
            "meta_key", 42, cast=int,
            description="test desc", needs_restart=True
        )
    assert "meta_key" in runtime_config.REGISTRY
    assert runtime_config.REGISTRY["meta_key"]["default"] == 42
    assert runtime_config.REGISTRY["meta_key"]["cast"] is int
    assert runtime_config.REGISTRY["meta_key"]["needs_restart"] is True


def test_reload_all_re_reads_settings():
    """reload_all() re-resolves every registered key."""
    from assistant.core import runtime_config
    runtime_config.REGISTRY.clear()

    # Register a setting with default
    with patch("assistant.core.runtime_config._get_db_value", return_value=None):
        runtime_config.setting("reload_key", "old", cast=str, description="")

    # Simulate DB now returning a new value
    with patch("assistant.core.runtime_config._get_db_value", return_value="new"):
        results = runtime_config.reload_all()

    assert results["reload_key"] == "new"


def test_get_db_value_returns_none_when_no_db():
    """_get_db_value returns None when DB hasn't been initialized."""
    from assistant.core import runtime_config

    with patch("assistant.storage.db.get_db", return_value=None):
        result = runtime_config._get_db_value("any_key")
    assert result is None
