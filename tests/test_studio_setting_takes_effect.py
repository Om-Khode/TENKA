"""The end-to-end shape of the bug that kept the daemon off.

`/set studio_api_enabled true` wrote the DB row and `/config` displayed it --
that command reads the DB live -- but main.py's `if config.STUDIO_API_ENABLED`
was still the import-time default, so the daemon never started and nothing was
logged either way. Diagnosed live on 2026-08-10.

The unit above (test_runtime_settings_reload_covers_registry) catches the
omission statically. This one proves the behaviour against a real SQLite file,
because that is the failure the user actually hit.
"""
import importlib

import pytest


@pytest.fixture
def db_at(tmp_path, monkeypatch):
    """A real DB in a tmp dir. Integration tests here hit real SQLite on
    purpose -- mocked DBs have masked migration failures in this repo before."""
    from assistant.storage import db as db_mod

    path = tmp_path / "memory" / "tenka.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    db_mod.init_db(path)
    yield db_mod
    db_mod.close_for_restore()


def test_studio_api_enabled_survives_the_import_time_default(db_at, monkeypatch):
    from assistant import config
    from assistant.storage.repos.settings import SettingsRepo

    # The daemon is off by default; that is the state the user starts from.
    monkeypatch.setattr(config, "STUDIO_API_ENABLED", False, raising=False)

    SettingsRepo(db_at.get_db()).set("studio_api_enabled", True, source="user")

    config.reload_runtime_settings()

    assert config.STUDIO_API_ENABLED is True, (
        "the DB says the Studio daemon is enabled; main.py reads this constant "
        "to decide whether to start it, so a stale False here is the daemon "
        "silently never starting"
    )


def test_a_browser_setting_survives_too(db_at, monkeypatch):
    """Same omission, different family -- ten browser/automation settings had
    it as well, so this is not a Studio-only guard."""
    from assistant import config
    from assistant.storage.repos.settings import SettingsRepo

    monkeypatch.setattr(config, "BROWSER_CDP_PORT", 9222, raising=False)
    SettingsRepo(db_at.get_db()).set("browser_cdp_port", 9333, source="user")

    config.reload_runtime_settings()

    assert config.BROWSER_CDP_PORT == 9333
