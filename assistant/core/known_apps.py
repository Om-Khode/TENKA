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
    # --- Canvas / drawing ---
    # Grouped by what they cost the automation stack, not by what they are for.
    # An app whose document is a canvas exposes no accessible tree worth
    # reading: the DOM or UIA layer sees one big surface and nothing inside it,
    # so a step aimed at such an app has to go to the vision tier or it will
    # confidently click on nothing.
    #
    # These were nine brand names inside a regex in `automation/router.py`,
    # which `CLAUDE.md` forbids outright -- "a regex that mentions a brand
    # name" is the second bullet under THE rule. The behaviour was right and
    # the location was wrong: this is a fact *about an app*, and facts about
    # apps live in this table. Adding the tenth canvas app is now a row here
    # rather than an edit to a routing expression.
    "figma": AppEntry("canvas_app", []),
    "miro": AppEntry("canvas_app", []),
    "excalidraw": AppEntry("canvas_app", []),
    "tldraw": AppEntry("canvas_app", []),
    "sketch": AppEntry("canvas_app", []),
    "google slides": AppEntry("canvas_app", ["gslides", "slides"]),
    "google drawings": AppEntry("canvas_app", ["gdrawings"]),
    "flutter": AppEntry("canvas_app", []),
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

# --- Derived lookup, rebuilt when the table it derives from changes ---
#
# This was built once at import, into a module-level dict, and that quietly
# contradicted the promise at the top of this file: *adding a new app requires
# one row in KNOWN_APPS, no code changes elsewhere*. True at edit time, false at
# runtime -- a row added after import was invisible to `resolve_app`, and
# `resolve_app` is how every caller asks. THE rule says apps are "discovered,
# learned, or taught at runtime"; a lookup frozen at import is the one thing
# that cannot support that.
#
# Rebuilt on a size change rather than on every call. `KNOWN_APPS` is small and
# read constantly (the router walks it per goal), so recomputing each time is
# waste -- but a stale answer is a wrong route, and the cost of noticing is one
# `len()`.
#
# A size check catches an addition or a removal. It does not catch a row being
# *replaced* with a different category under the same name, which is not
# something any caller does today and would need a real invalidation hook if it
# ever were. Said out loud rather than left as a silent limit.

_APP_LOOKUP: "dict[str, tuple[str, str]]" = {}
_LOOKUP_SIZE = -1


def _lookup() -> "dict[str, tuple[str, str]]":
    global _LOOKUP_SIZE
    if _LOOKUP_SIZE != len(KNOWN_APPS):
        _APP_LOOKUP.clear()
        for canonical, entry in KNOWN_APPS.items():
            _APP_LOOKUP[canonical] = (canonical, entry.category)
            for alias in entry.aliases:
                _APP_LOOKUP[alias] = (canonical, entry.category)
        _LOOKUP_SIZE = len(KNOWN_APPS)
    return _APP_LOOKUP


def resolve_app(name: str) -> Optional[tuple[str, str]]:
    """Resolve any app name or alias to (canonical_name, category), or None."""
    return _lookup().get(name.lower().strip())


def get_category(name: str) -> Optional[str]:
    """Return the category for an app name/alias, or None if unknown."""
    result = resolve_app(name)
    return result[1] if result else None


def get_apps_by_category(category: str) -> list[str]:
    """Return canonical names of all apps in the given category."""
    return [name for name, entry in KNOWN_APPS.items() if entry.category == category]
