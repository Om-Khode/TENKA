"""Tests for core.known_apps — shared app registry (T7)."""

import pytest

from assistant.core.known_apps import (
    KNOWN_APPS,
    AppEntry,
    get_apps_by_category,
    get_category,
    resolve_app,
)


# --- resolve_app ---


class TestResolveApp:
    def test_canonical_name(self):
        assert resolve_app("spotify") == ("spotify", "music_app")

    def test_alias(self):
        assert resolve_app("yt music") == ("youtube music", "music_app")

    def test_case_insensitive(self):
        assert resolve_app("Spotify") == ("spotify", "music_app")
        assert resolve_app("CHROME") == ("chrome", "browser")

    def test_whitespace_stripped(self):
        assert resolve_app("  telegram  ") == ("telegram", "messaging_default")

    def test_unknown_returns_none(self):
        assert resolve_app("vlc") is None
        assert resolve_app("") is None

    def test_underscore_variant(self):
        assert resolve_app("youtube_music") == ("youtube music", "music_app")
        assert resolve_app("apple_music") == ("apple music", "music_app")

    def test_no_space_variant(self):
        assert resolve_app("youtubemusic") == ("youtube music", "music_app")

    def test_short_aliases(self):
        assert resolve_app("wa") == ("whatsapp", "messaging_default")
        assert resolve_app("tg") == ("telegram", "messaging_default")

    def test_youtube_alias(self):
        assert resolve_app("youtube") == ("youtube music", "music_app")


# --- get_category ---


class TestGetCategory:
    def test_known(self):
        assert get_category("spotify") == "music_app"
        assert get_category("whatsapp") == "messaging_default"
        assert get_category("gmail") == "email_app"
        assert get_category("chrome") == "browser"

    def test_alias(self):
        assert get_category("wa") == "messaging_default"

    def test_unknown(self):
        assert get_category("unknown_app") is None


# --- get_apps_by_category ---


class TestGetAppsByCategory:
    def test_music(self):
        music = get_apps_by_category("music_app")
        assert "spotify" in music
        assert "youtube music" in music
        assert "apple music" in music
        assert "soundcloud" in music

    def test_messaging(self):
        messaging = get_apps_by_category("messaging_default")
        assert "whatsapp" in messaging
        assert "telegram" in messaging
        assert len(messaging) == 5

    def test_browser(self):
        browsers = get_apps_by_category("browser")
        assert set(browsers) == {"chrome", "firefox", "edge", "brave", "opera", "safari", "vivaldi"}

    def test_empty_category(self):
        assert get_apps_by_category("nonexistent") == []


# --- Text editor category ---


class TestTextEditorCategory:
    def test_notepad_is_text_editor(self):
        assert get_category("notepad") == "text_editor"

    def test_notepad_plus_plus_is_text_editor(self):
        assert get_category("notepad++") == "text_editor"

    def test_sublime_text_alias_resolves(self):
        assert resolve_app("sublime text") == ("sublime", "text_editor")

    def test_vscode_alias_resolves(self):
        assert resolve_app("vscode") == ("code", "text_editor")

    def test_text_editor_count(self):
        editors = get_apps_by_category("text_editor")
        assert len(editors) == 8


# --- Extra browsers ---


class TestExtraBrowsers:
    def test_opera_is_browser(self):
        assert get_category("opera") == "browser"

    def test_vivaldi_is_browser(self):
        assert get_category("vivaldi") == "browser"

    def test_browser_total_count(self):
        browsers = get_apps_by_category("browser")
        assert len(browsers) == 7


# --- Data integrity ---


class TestDataIntegrity:
    def test_all_entries_have_category(self):
        for name, entry in KNOWN_APPS.items():
            assert entry.category, f"{name} has empty category"

    def test_no_alias_collides_with_canonical(self):
        canonicals = set(KNOWN_APPS.keys())
        for name, entry in KNOWN_APPS.items():
            for alias in entry.aliases:
                assert alias not in canonicals, (
                    f"alias '{alias}' of '{name}' collides with a canonical name"
                )

    def test_no_duplicate_aliases(self):
        seen: dict[str, str] = {}
        for name, entry in KNOWN_APPS.items():
            for alias in entry.aliases:
                assert alias not in seen, (
                    f"alias '{alias}' claimed by both '{seen[alias]}' and '{name}'"
                )
                seen[alias] = name

    def test_categories_are_valid(self):
        valid = {"music_app", "messaging_default", "email_app", "browser", "text_editor"}
        for name, entry in KNOWN_APPS.items():
            assert entry.category in valid, (
                f"{name} has unknown category '{entry.category}'"
            )

    def test_resolve_covers_all_canonical_and_aliases(self):
        for name, entry in KNOWN_APPS.items():
            assert resolve_app(name) is not None, f"canonical '{name}' not resolvable"
            for alias in entry.aliases:
                result = resolve_app(alias)
                assert result is not None, f"alias '{alias}' of '{name}' not resolvable"
                assert result[0] == name, (
                    f"alias '{alias}' resolves to '{result[0]}', expected '{name}'"
                )


# --- Integration: preferences uses the shared registry ---


class TestPreferenceCorrectionIntegration:
    """Verify preferences._normalize_app_name still works after migration."""

    def test_normalize_canonical(self):
        from assistant.preferences import _normalize_app_name
        assert _normalize_app_name("Spotify") == "spotify"

    def test_normalize_alias(self):
        from assistant.preferences import _normalize_app_name
        assert _normalize_app_name("yt music") == "youtube music"

    def test_normalize_with_suffix(self):
        from assistant.preferences import _normalize_app_name
        assert _normalize_app_name("Spotify app") == "spotify"
        assert _normalize_app_name("Chrome application") == "chrome"

    def test_normalize_unknown_short(self):
        from assistant.preferences import _normalize_app_name
        assert _normalize_app_name("vlc") == "vlc"

    def test_normalize_unknown_long(self):
        from assistant.preferences import _normalize_app_name
        assert _normalize_app_name("this is not an app name at all") == ""
