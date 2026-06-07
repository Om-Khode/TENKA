"""
Shared app registry — the single source of truth for known application names,
categories, and aliases. Any module that needs to identify or classify an app
name imports from here.

Adding a new app requires one row in KNOWN_APPS. No code changes elsewhere.
"""

from typing import NamedTuple, Optional


class AppEntry(NamedTuple):
    category: str
    aliases: list[str]


KNOWN_APPS: dict[str, AppEntry] = {
    # --- Music ---
    "spotify": AppEntry("music_app", []),
    "youtube music": AppEntry("music_app", ["youtube_music", "youtubemusic", "yt music", "youtube"]),
    "apple music": AppEntry("music_app", ["apple_music"]),
    "soundcloud": AppEntry("music_app", []),
    # --- Messaging ---
    "whatsapp": AppEntry("messaging_default", ["wa"]),
    "telegram": AppEntry("messaging_default", ["tg"]),
    "discord": AppEntry("messaging_default", []),
    "signal": AppEntry("messaging_default", []),
    "slack": AppEntry("messaging_default", []),
    # --- Email ---
    "gmail": AppEntry("email_app", ["email"]),
    "outlook": AppEntry("email_app", []),
    # --- Browsers ---
    "chrome": AppEntry("browser", ["google chrome", "chromium"]),
    "firefox": AppEntry("browser", ["mozilla firefox"]),
    "edge": AppEntry("browser", ["microsoft edge"]),
    "brave": AppEntry("browser", ["brave browser"]),
    "opera": AppEntry("browser", []),
    "safari": AppEntry("browser", []),
    "vivaldi": AppEntry("browser", []),
    # --- Text Editors ---
    "notepad": AppEntry("text_editor", []),
    "wordpad": AppEntry("text_editor", []),
    "notepad++": AppEntry("text_editor", ["notepad plus plus", "npp"]),
    "sublime": AppEntry("text_editor", ["sublime text"]),
    "code": AppEntry("text_editor", ["vscode", "visual studio code", "vs code"]),
    "vim": AppEntry("text_editor", []),
    "nano": AppEntry("text_editor", []),
    "gedit": AppEntry("text_editor", []),
}

# --- Derived lookup (built once at import) ---

_APP_LOOKUP: dict[str, tuple[str, str]] = {}
for _canonical, _entry in KNOWN_APPS.items():
    _APP_LOOKUP[_canonical] = (_canonical, _entry.category)
    for _alias in _entry.aliases:
        _APP_LOOKUP[_alias] = (_canonical, _entry.category)


def resolve_app(name: str) -> Optional[tuple[str, str]]:
    """Resolve any app name or alias to (canonical_name, category), or None."""
    return _APP_LOOKUP.get(name.lower().strip())


def get_category(name: str) -> Optional[str]:
    """Return the category for an app name/alias, or None if unknown."""
    result = resolve_app(name)
    return result[1] if result else None


def get_apps_by_category(category: str) -> list[str]:
    """Return canonical names of all apps in the given category."""
    return [name for name, entry in KNOWN_APPS.items() if entry.category == category]
