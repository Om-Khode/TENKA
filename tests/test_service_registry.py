"""Tests for service_registry.py — JSON-backed service configuration."""

import json
from pathlib import Path

import pytest


_SERVICES_PATH = Path(__file__).resolve().parent.parent / "assistant" / "services.json"


class TestServicesJsonStructure:
    """Validate the raw JSON file before any Python loading."""

    def test_json_is_parseable(self):
        data = json.loads(_SERVICES_PATH.read_text(encoding="utf-8"))
        assert isinstance(data, dict)

    def test_schema_version_present(self):
        data = json.loads(_SERVICES_PATH.read_text(encoding="utf-8"))
        assert data["schema_version"] == 1

    def test_every_service_has_required_fields(self):
        data = json.loads(_SERVICES_PATH.read_text(encoding="utf-8"))
        required = {
            "auth_type", "packages", "min_scopes", "auth_extras",
            "banned_auth_classes", "constructor_names", "blocked_actions",
            "env_aliases", "developer_url", "router_hint",
        }
        for name, svc in data["services"].items():
            missing = required - set(svc.keys())
            assert not missing, f"Service '{name}' missing fields: {missing}"

    def test_auth_type_is_valid(self):
        data = json.loads(_SERVICES_PATH.read_text(encoding="utf-8"))
        valid_types = {"oauth", "device", "none"}
        for name, svc in data["services"].items():
            assert svc["auth_type"] in valid_types, \
                f"Service '{name}' has invalid auth_type '{svc['auth_type']}'"

    def test_blocked_actions_are_pairs(self):
        data = json.loads(_SERVICES_PATH.read_text(encoding="utf-8"))
        for name, svc in data["services"].items():
            for i, entry in enumerate(svc["blocked_actions"]):
                assert isinstance(entry, list) and len(entry) == 2, \
                    f"Service '{name}' blocked_actions[{i}] must be [pattern, reason]"


class TestDerivedConstants:
    """Verify the derived public constants match expected shapes and content."""

    def test_oauth_package_map_contains_spotipy(self):
        from assistant.service_registry import OAUTH_PACKAGE_MAP
        assert OAUTH_PACKAGE_MAP["spotipy"] == "spotify"

    def test_oauth_package_map_contains_google(self):
        from assistant.service_registry import OAUTH_PACKAGE_MAP
        assert OAUTH_PACKAGE_MAP["google-api-python-client"] == "gmail"
        assert OAUTH_PACKAGE_MAP["google-auth-oauthlib"] == "gmail"

    def test_device_auth_package_map_contains_neonize(self):
        from assistant.service_registry import DEVICE_AUTH_PACKAGE_MAP
        assert DEVICE_AUTH_PACKAGE_MAP["neonize"] == "whatsapp"

    def test_oauth_auth_extras_gmail(self):
        from assistant.service_registry import OAUTH_AUTH_EXTRAS
        assert OAUTH_AUTH_EXTRAS["gmail"]["access_type"] == "offline"
        assert OAUTH_AUTH_EXTRAS["gmail"]["prompt"] == "consent"

    def test_oauth_auth_extras_excludes_empty(self):
        from assistant.service_registry import OAUTH_AUTH_EXTRAS
        assert "spotify" not in OAUTH_AUTH_EXTRAS
        assert "whatsapp" not in OAUTH_AUTH_EXTRAS

    def test_oauth_min_scopes_gmail(self):
        from assistant.service_registry import OAUTH_MIN_SCOPES
        assert "https://www.googleapis.com/auth/gmail.readonly" in OAUTH_MIN_SCOPES["gmail"]

    def test_oauth_min_scopes_spotify(self):
        from assistant.service_registry import OAUTH_MIN_SCOPES
        assert "user-read-playback-state" in OAUTH_MIN_SCOPES["spotify"]

    def test_service_blocked_actions_gmail(self):
        from assistant.service_registry import SERVICE_BLOCKED_ACTIONS
        patterns = [p for p, _ in SERVICE_BLOCKED_ACTIONS["gmail"]]
        assert "messages().send" in patterns
        assert "messages().delete" in patterns

    def test_service_blocked_actions_excludes_clean_services(self):
        from assistant.service_registry import SERVICE_BLOCKED_ACTIONS
        assert "spotify" not in SERVICE_BLOCKED_ACTIONS
        assert "whatsapp" not in SERVICE_BLOCKED_ACTIONS

    def test_package_env_aliases_spotipy(self):
        from assistant.service_registry import PACKAGE_ENV_ALIASES
        aliases = PACKAGE_ENV_ALIASES["spotipy"]
        assert aliases["SPOTIPY_CLIENT_ID"] == "$SPOTIFY_CLIENT_ID"
        assert aliases["SPOTIPY_REDIRECT_URI"] == "http://localhost:8888/callback"

    def test_all_service_names(self):
        from assistant.service_registry import ALL_SERVICE_NAMES
        data = json.loads(_SERVICES_PATH.read_text(encoding="utf-8"))
        assert isinstance(ALL_SERVICE_NAMES, frozenset)
        assert ALL_SERVICE_NAMES == frozenset(data["services"].keys())

    def test_all_banned_auth_classes(self):
        from assistant.service_registry import ALL_BANNED_AUTH_CLASSES
        assert "SpotifyOAuth" in ALL_BANNED_AUTH_CLASSES
        assert "InstalledAppFlow" in ALL_BANNED_AUTH_CLASSES
        assert "Flow" in ALL_BANNED_AUTH_CLASSES

    def test_all_constructor_names_includes_generic(self):
        from assistant.service_registry import ALL_CONSTRUCTOR_NAMES
        assert "Client" in ALL_CONSTRUCTOR_NAMES

    def test_all_constructor_names_includes_per_service(self):
        from assistant.service_registry import ALL_CONSTRUCTOR_NAMES
        assert "Spotify" in ALL_CONSTRUCTOR_NAMES
        assert "build" in ALL_CONSTRUCTOR_NAMES
        assert "Credentials" in ALL_CONSTRUCTOR_NAMES

    def test_constructor_skip_pattern_is_valid_regex(self):
        import re
        from assistant.service_registry import CONSTRUCTOR_SKIP_PATTERN
        re.compile(CONSTRUCTOR_SKIP_PATTERN)

    def test_developer_urls(self):
        from assistant.service_registry import DEVELOPER_URLS
        assert "spotify" in DEVELOPER_URLS
        assert "gmail" in DEVELOPER_URLS
        assert all(v.startswith("https://") for v in DEVELOPER_URLS.values())

    def test_developer_urls_excludes_null(self):
        from assistant.service_registry import DEVELOPER_URLS
        assert "whatsapp" not in DEVELOPER_URLS


class TestRouterPromptFragments:
    """Verify ROUTER_HINTS is assembled from per-service + generic hints."""

    def test_router_hints_contains_service_hints(self):
        from assistant.service_registry import ROUTER_HINTS
        assert "spotipy" in ROUTER_HINTS
        assert "google-api-python-client" in ROUTER_HINTS
        assert "neonize" in ROUTER_HINTS

    def test_router_hints_contains_generic_hints(self):
        from assistant.service_registry import ROUTER_HINTS
        assert "pycaw" in ROUTER_HINTS
        assert "opencv-python" in ROUTER_HINTS

    def test_router_hints_is_formatted_as_rules(self):
        from assistant.service_registry import ROUTER_HINTS
        assert ROUTER_HINTS.startswith("Package rules:")


class TestSchemaVersionEnforcement:
    """Verify the loader rejects bad schema versions."""

    def test_load_rejects_wrong_version(self, tmp_path):
        bad = {"schema_version": 999, "services": {}}
        p = tmp_path / "bad.json"
        p.write_text(json.dumps(bad), encoding="utf-8")
        from assistant.service_registry import _load_services
        with pytest.raises(ValueError, match="schema_version=999"):
            _load_services(p)

    def test_load_rejects_missing_version(self, tmp_path):
        bad = {"services": {}}
        p = tmp_path / "bad.json"
        p.write_text(json.dumps(bad), encoding="utf-8")
        from assistant.service_registry import _load_services
        with pytest.raises(ValueError, match="schema_version=0"):
            _load_services(p)

    def test_load_rejects_missing_file(self, tmp_path):
        from assistant.service_registry import _load_services
        with pytest.raises(FileNotFoundError, match="not found"):
            _load_services(tmp_path / "nonexistent.json")

    def test_load_rejects_corrupt_json(self, tmp_path):
        p = tmp_path / "corrupt.json"
        p.write_text("{not valid json", encoding="utf-8")
        from assistant.service_registry import _load_services
        with pytest.raises(ValueError, match="not valid JSON"):
            _load_services(p)
