"""Goal verification for the vision agent.

Provides fast heuristic checks (window title, SMTC media query) and a
full LLM-based screen verification pass. Extracted from the monolithic
``vision.py`` to isolate verification logic from the agent loop.
"""

import logging
import re
from typing import Callable

from ._parsing import _parse_plan
from ...core.known_apps import KNOWN_APPS

logger = logging.getLogger("computer_agent")

_MUSIC_STOPWORDS = frozenset(
    {"open", "play", "playing", "listen", "song", "music", "track",
     "from", "with", "and", "the", "a", "on", "in", "by"}
) | frozenset(word for name in KNOWN_APPS for word in name.split())

# Playback-intent verbs. SMTC media-state verification only makes sense for
# goals that control playback — NOT for app-management goals that merely
# mention a media app (e.g. "close spotify"). Gating on app names (KI-7) made
# the verifier declare "wrong song" on a successful close and loop forever.
# Keyed on verbs, never app names, so it stays generic across new apps.
_PLAYBACK_VERB_RE = re.compile(
    r"\b(play|playing|pause|resume|unpause|listen|skip|shuffle|repeat|queue|next|previous)\b"
)


def _is_music_playback_goal(goal: str) -> bool:
    """True only when the goal is about controlling playback (play/pause/skip…),
    not merely opening, closing, or switching an app that plays media."""
    return bool(_PLAYBACK_VERB_RE.search(goal.lower()))


# --- Verification System Prompt ---

VERIFICATION_SYSTEM_PROMPT = """\
You are a goal verification agent. You are given a user's goal and the CURRENT state of their \
screen (captured AFTER actions were executed). Your job is to determine whether the goal has \
been achieved based on what you can see.

IMPORTANT WINDOW TITLE CONVENTIONS:
- Music apps show the currently playing song as the window title
  e.g. "Ed Sheeran - Perfect" means music is playing and "Perfect" by Ed Sheeran is the track
- Browsers show page title as window title
- If active window is "{Artist} - {Song}" format, music IS playing

When verifying music playback goals, if the active window title contains \
the song name or artist name from the goal, mark achieved=true immediately.

VERIFICATION RULES:
- ACTION VERBS REQUIRE OBSERVABLE COMPLETION, NOT PREPARATION. When the goal uses a \
transitive action verb (fill, submit, send, post, save, sign in / log in, register, schedule, \
install, delete, share, play, open, etc.), the screen must show evidence the action ACTUALLY \
COMPLETED — not just that the user is set up to do it. Examples of preparation vs completion:
    • "fill the form"     — preparation: fields populated. completion: form submitted \
(success page, thank-you, redirect, form cleared, confirmation dialog).
    • "send a message"    — preparation: text in compose box. completion: message visible in \
the thread with a sent indicator (timestamp, checkmark, "sent").
    • "save the file"     — preparation: typed content. completion: title bar no longer shows \
the unsaved marker (asterisk, "modified", "unsaved").
    • "sign in to X"      — preparation: credentials typed. completion: logged-in UI visible \
(avatar, dashboard, account menu, "welcome" greeting).
    • "play <song>"       — preparation: song appears in search results. completion: bottom \
player bar shows the title with a moving progress bar / pause button visible. Window title \
matching the song is also evidence of playback.
  If the screen only shows preparation, set achieved=false and put the missing finalization \
step in `remaining` (e.g. "click Submit", "press Send", "save the file", "click Sign in").
- For input fields: a field is FILLED only when it shows solid user-typed text. Grey \
placeholder text ("Enter your name", "What would you like to tell us?") means the field is \
STILL EMPTY.
- For window-title evidence (browsers, music apps, editors): if the title clearly reflects \
the completed state (e.g. song name in a music app means it's playing, page title in browser \
means navigation succeeded), that counts as observable completion.
- EXCEPTION — narrow scope: if the user explicitly limited the task ("just fill the fields, \
don't submit"; "draft a message to John but don't send it"; "open Notepad"), then \
preparation IS the completion. Respect the user's stated scope.
- Be strict: only set achieved=true when the screen clearly shows the goal is complete. \
Do NOT assume success — verify it from the screen content.

Respond with ONLY a JSON object:
{
  "achieved": true or false,
  "result": "If achieved, a concise answer or summary to report to the user. If not achieved, leave empty.",
  "remaining": "If NOT achieved, describe what still needs to be done. If achieved, leave empty."
}
"""


def _is_yes_answer(answer: str) -> bool:
    """Tolerant YES/NO parser. Accepts YES, Yes., 'Yes — visible', Y, etc."""
    if not isinstance(answer, str):
        return False
    a = answer.strip().upper()
    if not a:
        return False
    while a and not a[0].isalpha():
        a = a[1:]
    if not a:
        return False
    if a.startswith("YES"):
        return True
    if a == "Y":
        return True
    return False


def _quick_verify_from_window_title(goal: str, active_window: str) -> bool | None:
    """
    Fast verification without LLM by checking window title.
    Returns True if verified, False if definitely not done, None if unsure.
    """
    goal_lower = goal.lower()
    window_lower = active_window.lower()

    music_keywords = ["play", "playing", "listen"]
    if any(kw in goal_lower for kw in music_keywords):
        stopwords = _MUSIC_STOPWORDS
        goal_words = [w for w in goal_lower.split() if w not in stopwords and len(w) > 2]
        if any(word in window_lower for word in goal_words):
            return True

    return None


def _get_now_playing() -> dict | None:
    """
    Query Windows System Media Transport Controls (SMTC) to get the
    currently playing media track. Returns dict with 'title', 'artist',
    'app' keys, or None if nothing is playing or query fails.
    """
    try:
        import winrt.windows.media.control as wmc
        import asyncio

        async def _query():
            sessions = await wmc.GlobalSystemMediaTransportControlsSessionManager.request_async()
            session = sessions.get_current_session()
            if session is None:
                return None
            info = await session.try_get_media_properties_async()
            if info is None:
                return None
            status = session.get_playback_info()
            is_playing = str(status.playback_status) == "PlaybackStatus.PLAYING"
            return {
                "title": info.title or "",
                "artist": info.artist or "",
                "app": session.source_app_user_model_id or "",
                "is_playing": is_playing,
            }

        import nest_asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                nest_asyncio.apply()
                result = loop.run_until_complete(_query())
            else:
                result = loop.run_until_complete(_query())
        except RuntimeError:
            loop = asyncio.new_event_loop()
            result = loop.run_until_complete(_query())
            loop.close()
        if result:
            logger.info(f"[AGENT] SMTC now playing: {result['artist']} - {result['title']} (playing={result['is_playing']})")
        return result
    except Exception as e:
        logger.warning(f"[AGENT] SMTC query failed (winrt not installed?): {e}")
        return None


async def _verify_goal(goal: str, llm_func: Callable) -> dict:
    """
    Re-capture the screen and ask the LLM whether the goal has been achieved.

    Returns:
        A dict with keys: achieved (bool), result (str), remaining (str).
        On failure, returns {"achieved": False, "result": "", "remaining": "Verification failed"}.
    """
    from ...io import screen
    from ... import llm
    from ...llm.contracts import ask_for_agent_verify

    from ...io import screen as _screen_ver
    active_ver = _screen_ver.get_active_window()
    if _is_music_playback_goal(goal):
        goal_words_ver = [w for w in goal.lower().split()
                          if len(w) > 3 and w not in _MUSIC_STOPWORDS]
        if goal_words_ver and " - " in active_ver:
            title_lower = active_ver.lower()
            matched_ver = sum(1 for w in goal_words_ver if w in title_lower)
            if matched_ver >= 1:
                result_text = f"{active_ver} is now playing"
                logger.info(f"[AGENT] [OK] Verifier: window title confirms playback: '{active_ver}'")
                return {"achieved": True, "result": result_text, "remaining": ""}

    if _is_music_playback_goal(goal):
        now_playing = _get_now_playing()
        if now_playing and now_playing.get("is_playing"):
            title = now_playing["title"].lower()
            artist = now_playing["artist"].lower()
            goal_lower = goal.lower()
            goal_words = [w for w in goal_lower.split() if len(w) > 3 and w not in _MUSIC_STOPWORDS]
            matched = sum(1 for w in goal_words if w in title or w in artist)
            if matched >= 1:
                result_text = f"{now_playing['artist']} - {now_playing['title']} is now playing"
                logger.info(f"[AGENT] [OK] SMTC confirmed playing: {result_text}")
                return {"achieved": True, "result": result_text, "remaining": ""}
            else:
                logger.info(f"[AGENT] SMTC says playing '{now_playing['title']}' by '{now_playing['artist']}' — does not match goal, not achieved")
                return {"achieved": False, "result": "", "remaining": f"Wrong track playing: '{now_playing['title']}' by '{now_playing['artist']}'. Need to find and play the correct song."}
        elif now_playing and not now_playing.get("is_playing"):
            title = now_playing.get("title", "").lower()
            artist = now_playing.get("artist", "").lower()
            goal_words = [w for w in goal.lower().split()
                          if len(w) > 3 and w not in _MUSIC_STOPWORDS]
            matched = sum(1 for w in goal_words if w in title or w in artist)
            if matched >= 1:
                result_text = f"{now_playing['artist']} - {now_playing['title']} is now playing"
                logger.info(f"[AGENT] [OK] SMTC: correct song loaded (may be briefly paused): {result_text}")
                return {"achieved": True, "result": result_text, "remaining": ""}
            logger.info(f"[AGENT] SMTC says wrong song paused: {now_playing['title']} by {now_playing['artist']}")
            return {"achieved": False, "result": "", "remaining": f"Wrong song paused: '{now_playing['title']}'. Need to find and play the correct song."}

    import asyncio as _asyncio
    await _asyncio.sleep(1.2)
    logger.info("[AGENT] Verification: re-capturing screen...")
    screenshot_b64 = screen.capture_screenshot_base64()
    active_window = screen.get_active_window()

    verify_prompt = (
        f"GOAL: {goal}\n"
        f"ACTIVE WINDOW: \"{active_window}\"\n\n"
        f"Has the goal been achieved? Examine the screenshot carefully."
    )

    if screenshot_b64:
        raw = (await llm.get_vision_response(
            image_base64=screenshot_b64,
            prompt=verify_prompt,
            system_prompt=VERIFICATION_SYSTEM_PROMPT,
            json_mode=True,
        )).text
    else:
        screen_desc = screen.describe_screen_for_llm()
        raw = await ask_for_agent_verify(
            verify_prompt + f"\n\nSCREEN TEXT:\n{screen_desc}",
            system_prompt=VERIFICATION_SYSTEM_PROMPT,
            json_mode=True,
        )

    if raw == "__LLM_UNAVAILABLE__":
        logger.warning("[AGENT] Verification LLM unavailable — assuming not achieved")
        return {"achieved": False, "result": "", "remaining": "LLM unavailable for verification"}

    try:
        parsed = _parse_plan(raw)
        if parsed and "achieved" in parsed:
            logger.info(f"[AGENT] Verification result: achieved={parsed.get('achieved')}")
            return parsed
    except Exception as e:
        logger.error(f"[AGENT] Verification parse error: {e}")

    return {"achieved": False, "result": "", "remaining": "Verification response could not be parsed"}
