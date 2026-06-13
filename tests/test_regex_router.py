"""
test_regex_router.py — Unit tests for the regex pre-router.

Run: python -m pytest test_regex_router.py -v
"""

import sys
import os
import types
from dataclasses import dataclass, field

# ─── Stub assistant.intent so heavy deps (torch, llm, etc.) aren't loaded ────

@dataclass
class IntentResult:
    intent: str = "unknown"
    response: str = ""
    params: dict = field(default_factory=dict)

    def get_param(self, key, default=""):
        return self.params.get(key, default)

_orig_intent = sys.modules.get("assistant.intent")
_orig_config = sys.modules.get("assistant.config")

_intent_mod = types.ModuleType("assistant.intent")
_intent_mod.IntentResult = IntentResult
sys.modules["assistant.intent"] = _intent_mod

# Stub assistant.config with the subset regex_router needs (avoids .env / sqlite side-effects)
_we_stubbed_config = False
if "assistant.config" not in sys.modules:
    _config_mod = types.ModuleType("assistant.config")
    _config_mod.BROWSER_NAMES = frozenset({
        "chrome", "firefox", "edge", "brave", "opera", "safari", "vivaldi", "browser",
    })
    # automation.router (imported transitively via manifest registry)
    # references config.ASSISTANT_NAME_LOWER; supply it here so the stub
    # is complete and the 38 unrelated tests downstream don't trip on it.
    _config_mod.ASSISTANT_NAME_LOWER = "tenka"
    sys.modules["assistant.config"] = _config_mod
    _we_stubbed_config = True

import assistant.regex_router as rr

# Restore sys.modules so sibling test files import the real assistant.intent /
# assistant.config. rr's own bindings were resolved at the import above and
# stay pointed at the stubs, so tests in this file are unaffected.
if _orig_intent is None:
    sys.modules.pop("assistant.intent", None)
else:
    sys.modules["assistant.intent"] = _orig_intent
if _we_stubbed_config:
    if _orig_config is None:
        sys.modules.pop("assistant.config", None)
    else:
        sys.modules["assistant.config"] = _orig_config

pre_route = rr.pre_route

# ─── Helpers ─────────────────────────────────────────────────────────────────

def matches(text, expected_intent):
    result = pre_route(text)
    assert result is not None, f"Expected match for {text!r}, got None"
    assert result.intent == expected_intent, (
        f"{text!r}: expected {expected_intent!r}, got {result.intent!r}"
    )
    return result


def no_match(text):
    result = pre_route(text)
    assert result is None, f"Expected no match for {text!r}, got {result.intent!r}"


# ─── get_time ────────────────────────────────────────────────────────────────

def test_time_phrases():
    matches("what time is it", "get_time")
    matches("what's the time", "get_time")
    matches("whats the time", "get_time")
    matches("current time", "get_time")
    matches("time please", "get_time")
    matches("What Time Is It", "get_time")


# ─── read_screen ─────────────────────────────────────────────────────────────

def test_screen_phrases():
    matches("take a screenshot", "read_screen")
    matches("screenshot", "read_screen")
    matches("what's on my screen", "read_screen")
    matches("look at my screen", "read_screen")


# ─── hide / show avatar ──────────────────────────────────────────────────────

def test_hide_phrases():
    matches("hide", "hide_avatar")
    matches("go away", "hide_avatar")
    matches("hide yourself", "hide_avatar")
    matches("disappear", "hide_avatar")


def test_show_phrases():
    matches("come back", "show_avatar")
    matches("show yourself", "show_avatar")
    matches("reappear", "show_avatar")


# ─── browse_url ──────────────────────────────────────────────────────────────

def test_goto_url():
    r = matches("go to youtube.com", "open_browser")
    assert r.params["url"] == "youtube.com"

    r = matches("visit https://google.com", "open_browser")
    assert r.params["url"] == "https://google.com"

    r = matches("navigate to www.github.com", "open_browser")
    assert r.params["url"] == "www.github.com"


def test_bare_url():
    r = matches("youtube.com", "open_browser")
    assert r.params["url"] == "youtube.com"

    r = matches("https://example.com", "open_browser")
    assert "example.com" in r.params["url"]


def test_open_url_routes_to_browse():
    r = matches("open youtube.com", "open_browser")
    assert r.params["url"] == "youtube.com"

    r = matches("open https://github.com", "open_browser")
    assert "github.com" in r.params["url"]


def test_goto_compound_action_falls_through():
    """Multi-step 'go to X and search/click/type Y' must NOT match open_browser."""
    no_match("go to wikipedia and search for Python programming")
    no_match("go to google and search for weather")
    no_match("visit youtube and click on trending")
    no_match("navigate to github and find my repositories")
    no_match("go to amazon and search for headphones")
    no_match("go to gmail and type a new email")


def test_goto_simple_still_matches():
    """Simple navigation without compound actions must still work."""
    r = matches("go to wikipedia", "open_browser")
    assert r.params["url"] == "wikipedia"

    r = matches("go to https://en.wikipedia.org", "open_browser")
    assert "wikipedia.org" in r.params["url"]


# ─── computer_task ───────────────────────────────────────────────────────────

def test_open_app():
    matches("open settings", "computer_task")
    matches("open notepad", "computer_task")
    matches("launch chrome", "computer_task")
    matches("start task manager", "computer_task")
    matches("open spotify", "computer_task")
    matches("open calculator", "computer_task")


def test_open_non_app_words_fall_through():
    no_match("open file")
    no_match("open the document")
    no_match("open a folder")


def test_open_app_case_insensitive():
    matches("Open Settings", "computer_task")
    matches("LAUNCH CHROME", "computer_task")


# ─── code_executor (music) ───────────────────────────────────────────────────

def test_play_falls_through_to_llm():
    # "play {X}" is intentionally NOT shortcut-routed. The LLM + intent.Guard 3
    # are needed to distinguish "play X on spotify" (code_executor) from
    # "play X on browser" / "play X on youtube" (computer_task → browser).
    no_match("play my liked songs on spotify")
    no_match("play some music")
    no_match("play Bohemian Rhapsody")
    no_match("start playing music")
    no_match("play cat video on youtube")
    no_match("play something on browser")


def test_music_controls():
    matches("pause", "code_executor")
    matches("resume", "code_executor")
    matches("next song", "code_executor")
    matches("previous song", "code_executor")
    matches("skip track", "code_executor")
    matches("stop the music", "code_executor")
    matches("volume up", "code_executor")
    matches("turn it down", "code_executor")


def test_music_case_insensitive():
    # "PLAY some music" falls through to the LLM (no _PLAY_RE shortcut).
    # Music transport controls remain case-insensitive on the fast path.
    no_match("PLAY some music")
    matches("Next Song", "code_executor")


# ─── set_reminder ────────────────────────────────────────────────────────────

def test_set_reminder():
    r = matches("remind me in 5 minutes to drink water", "set_reminder")
    assert r.params["goal"] == "remind me in 5 minutes to drink water"

    matches("remind me to call mom at 3pm", "set_reminder")
    matches("remind in 10 minutes", "set_reminder")


def test_set_reminder_requires_time_anchor_or_to_verb():
    """C-Q1: tighten the reminder regex so statements like
    'remind me Aanya works at X' do not get classified as reminders.
    Must still match a time anchor (today/tomorrow/at X/in N/on day) or
    a 'to <verb>' clause."""
    from assistant.regex_router import _REMINDER_RE

    # Positives — time anchors of various shapes
    assert _REMINDER_RE.match("remind me tomorrow about the meeting")
    assert _REMINDER_RE.match("remind me at 5pm")
    assert _REMINDER_RE.match("remind me on Monday")
    assert _REMINDER_RE.match("remind me in 2 hours")
    assert _REMINDER_RE.match("remind me today")
    # Positives — to-verb form
    assert _REMINDER_RE.match("remind me to call the dentist")
    assert _REMINDER_RE.match("remind me to send the report")

    # Negatives — no time anchor and no 'to <verb>' clause
    assert _REMINDER_RE.match("remind me Aanya works at Razorpay") is None
    assert _REMINDER_RE.match("remind me of the joke") is None
    assert _REMINDER_RE.match("remind me what happened") is None


# ─── cancel_reminder ─────────────────────────────────────────────────────────

def test_cancel_reminder():
    matches("cancel reminder", "cancel_reminder")
    matches("stop my reminders", "cancel_reminder")
    matches("delete the reminder", "cancel_reminder")
    matches("remove all reminders", "cancel_reminder")


def test_cancel_not_set():
    r = matches("cancel my reminder for tomorrow", "cancel_reminder")
    assert r.intent == "cancel_reminder"


# ─── web_search ──────────────────────────────────────────────────────────────

def test_web_search():
    r = matches("search for mechanical keyboards", "web_search")
    assert r.params["query"] == "mechanical keyboards"

    r = matches("google best python tutorials", "web_search")
    assert r.params["query"] == "best python tutorials"

    r = matches("look up weather in Mumbai", "web_search")
    assert r.params["query"] == "weather in Mumbai"

    r = matches("find anime recommendations", "web_search")
    assert r.params["query"] == "anime recommendations"


def test_find_file_falls_through():
    """'find todo.txt' should NOT route to web_search (has file extension)."""
    no_match("find todo.txt")
    no_match("find report.pdf")
    no_match("find notes.docx")
    no_match("search for config.json")


def test_find_file_keywords_fall_through():
    """'find' with file-related keywords should fall through to LLM."""
    no_match("find file named todo")
    no_match("find files on desktop")
    no_match("find my downloads folder")
    no_match("search for documents")


def test_search_with_browser_falls_through():
    """'search X on chrome' should fall through to LLM so computer_task opens the browser."""
    no_match("search weather in Berlin on chrome")
    no_match("search for recipes on firefox")
    no_match("google news on edge")
    no_match("find tutorials on brave")
    no_match("look up scores on browser")


def test_web_search_still_works_after_file_guard():
    """Normal web searches must still route to web_search."""
    matches("search for python tutorials", "web_search")
    matches("google best restaurants", "web_search")
    matches("look up weather", "web_search")
    matches("find anime recommendations", "web_search")


# ─── store_memory ────────────────────────────────────────────────────────────

def test_store_memory():
    r = matches("remember that I need to buy milk", "store_memory")
    assert r.params["content"] == "I need to buy milk"

    r = matches("remember that my birthday is on 1st Aug", "store_memory")
    assert "birthday" in r.params["content"]

    r = matches("keep in mind that I'm allergic to peanuts", "store_memory")
    assert r.params["content"] == "I'm allergic to peanuts"

    r = matches("don't forget that the meeting is at 3pm", "store_memory")
    assert r.params["content"] == "the meeting is at 3pm"

    r = matches("dont forget that I like biryani", "store_memory")
    assert r.params["content"] == "I like biryani"


def test_store_memory_without_that():
    """'remember X' without 'that' should still route to store_memory."""
    r = matches("remember my favorite browser is Firefox", "store_memory")
    assert r.params["content"] == "my favorite browser is Firefox"

    r = matches("remember my name is Alex", "store_memory")
    assert r.params["content"] == "my name is Alex"

    r = matches("remember the wifi password is abc1234", "store_memory")
    assert r.params["content"] == "the wifi password is abc1234"

    r = matches("Remember Spotify keeps crashing on my machine", "store_memory")
    assert r.params["content"] == "Spotify keeps crashing on my machine"

    r = matches("keep in mind I prefer dark mode", "store_memory")
    assert r.params["content"] == "I prefer dark mode"

    r = matches("don't forget I'm vegetarian", "store_memory")
    assert r.params["content"] == "I'm vegetarian"


def test_remember_to_routes_store_memory():
    """'remember to X' should route to store_memory (how_to type)."""
    r = matches("remember to fix the printer restart the spooler", "store_memory")
    assert "fix the printer" in r.params["content"]

    r = matches("remember to check the router first", "store_memory")
    assert "check the router" in r.params["content"]


def test_remember_face_voice_falls_through():
    """'remember my face/voice' are camera/mic actions, not facts."""
    result = pre_route("remember my face")
    assert result is None or result.intent != "store_memory"

    result = pre_route("remember my voice")
    assert result is None or result.intent != "store_memory"

    result = pre_route("remember this face")
    assert result is None or result.intent != "store_memory"


# ─── create_note ─────────────────────────────────────────────────────────────

def test_create_note():
    r = matches("note that the meeting is at 3pm", "create_note")
    assert r.params["content"] == "the meeting is at 3pm"

    r = matches("write down that password is abc123", "create_note")
    assert "password" in r.params["content"]

    r = matches("make a note that project is due Friday", "create_note")
    assert r.params["content"] == "project is due Friday"

    r = matches("note down pick up groceries tomorrow", "create_note")
    assert "groceries" in r.params["content"]


# ─── memory_query ────────────────────────────────────────────────────────────

def test_memory_query():
    r = matches("what do you know about my preferences", "memory_query")
    assert r.params["query"] == "my preferences"

    r = matches("what do you remember about my sister", "memory_query")
    assert r.params["query"] == "my sister"

    r = matches("do you remember my birthday", "memory_query")
    assert r.params["query"] == "my birthday"

    r = matches("do you know my favorite color", "memory_query")
    assert "favorite color" in r.params["query"]

    r = matches("recall the recipe I mentioned", "memory_query")
    assert r.params["query"] == "the recipe I mentioned"


def test_memory_query_personal_questions_pass_through_to_llm():
    """'what is my X' / 'what's my X' style questions DO NOT match the
    regex router and fall through to the LLM intent classifier.

    History: an earlier revision routed these to memory_query via regex, but
    the pattern was too broad (e.g. "what's my IP address" should route to
    code_executor, not memory_query). CT-1/I6 narrowed _MEMORY_RE to only
    unambiguous recall verbs ("recall X", "do you remember X", "what do you
    know about X"). The ambiguous "my X" cases now go to the LLM + Guard 5.
    This test pins the narrowing so it does not regress.
    """
    no_match("what is my favorite pokemon")
    no_match("what's my birthday")
    no_match("what my favorite pokemon?")
    no_match("who is my best friend")

    # Other "my X" interrogatives also fall through to the LLM.
    no_match("when is my birthday")
    no_match("when's my anniversary")
    no_match("what was my previous address")
    no_match("who's my doctor")

    # The unambiguous recall verbs still match via the regex fast-path.
    r = matches("recall my favorite pokemon", "memory_query")
    assert "favorite pokemon" in r.params["query"]
    r = matches("do you remember my birthday", "memory_query")
    assert "birthday" in r.params["query"]


def test_commitment_recall_first_person():
    """livetest follow-up: 'what did I commit to / promise / owe'
    fast-paths to memory_query so the OPEN PROMISES surfacing fires.
    Passes the FULL text as the query (the commitment-shape detector in
    memory_search greps for the verb, not the captured tail)."""
    for q in [
        "what did I commit to this week",
        "what did I promise",
        "what have I committed to",
        "what do I owe",
        "what am I supposed to do today",
        "what did I agree to",
        "what had I pledged",
    ]:
        r = matches(q, "memory_query")
        assert r.params["query"] == q, (
            f"{q!r}: full text must be the query so commitment detection "
            f"fires, got {r.params['query']!r}"
        )


def test_commitment_recall_third_party_does_not_match():
    """Third-party recall is NOT covered by this fast-path — those need
    the LLM classifier to learn about KG entities (v1.1)."""
    for q in [
        "what does Aanya owe me",
        "what did Karan promise",
        "what is Priya supposed to do",
    ]:
        # Should fall through (None) — the LLM classifier handles these.
        # If a future LLM fix routes them to memory_query that's fine; this
        # test only asserts the regex itself doesn't pre-empt with a wrong
        # answer.
        result = pre_route(q)
        if result is not None:
            assert result.intent != "memory_query" or result.params.get("query") == q, (
                f"{q!r}: third-party recall should not be hijacked here"
            )


# ─── recording ──────────────────────────────────────────────────────────────

def test_start_recording():
    matches("start recording", "start_recording")
    matches("Start Recording", "start_recording")
    matches("begin recording", "start_recording")
    matches("start a recording", "start_recording")


def test_stop_recording():
    matches("stop recording", "stop_recording")
    matches("Stop Recording", "stop_recording")
    matches("end recording", "stop_recording")
    matches("finish recording", "stop_recording")
    matches("stop the recording", "stop_recording")


def test_recording_not_app_launch():
    """'start recording' must NOT route to computer_task."""
    r = pre_route("start recording")
    assert r is not None
    assert r.intent != "computer_task"


# ─── procedure delete ───────────────────────────────────────────────────────

def test_proc_delete_forget_and_forgot():
    """Both 'forget' and 'forgot' should match procedure delete regex."""
    r = rr.match_procedure_command("forget procedure search on youtube")
    assert r is not None
    assert r.params["action"] == "delete"
    assert r.params["name"] == "search on youtube"

    r = rr.match_procedure_command("forgot procedure search on youtube")
    assert r is not None
    assert r.params["action"] == "delete"
    assert r.params["name"] == "search on youtube"


def test_proc_delete_variants():
    r = rr.match_procedure_command("delete procedure morning routine")
    assert r is not None and r.params["action"] == "delete"

    r = rr.match_procedure_command("remove the procedure morning routine")
    assert r is not None and r.params["action"] == "delete"

    r = rr.match_procedure_command("drop procedure test thing")
    assert r is not None and r.params["action"] == "delete"


# ─── Fall-through (must NOT match) ───────────────────────────────────────────

def test_compound_requests_still_match_first_intent():
    # Compound requests DO match the pre-router for the first intent.
    # The planner override in main.py (needs_planning check) then overrides
    # the result to "planner" for multi-step execution. The pre-router
    # doesn't need to detect compound requests itself.
    matches("search for keyboards and then open amazon", "web_search")
    matches("open settings and change the volume", "computer_task")
    # "play music and remind me in 10 minutes" no longer has a regex shortcut —
    # falls through to the LLM (planner). Asserted as no_match here.
    no_match("play music and remind me in 10 minutes")


def test_ambiguous_fall_through():
    no_match("can you play that song I was talking about earlier")
    no_match("what do you think about music")
    no_match("tell me a joke")
    no_match("how are you doing")


# ─── Event monitors ──────────────────────────────────────────────────

def test_monitor_list():
    matches("show my monitors", "manage_monitor")
    matches("list monitors", "manage_monitor")
    matches("show all monitors", "manage_monitor")
    matches("list my event monitors", "manage_monitor")
    matches("Show My Monitors", "manage_monitor")


def test_monitor_crud():
    matches("pause the Spotify monitor", "manage_monitor")
    matches("resume the Discord monitor", "manage_monitor")
    matches("delete all monitors", "manage_monitor")
    matches("disable the focus monitor", "manage_monitor")
    matches("enable the song monitor", "manage_monitor")
    matches("remove the window monitor", "manage_monitor")


def test_monitor_not_schedule():
    """'show my monitors' must NOT route to manage_schedule."""
    r = pre_route("show my monitors")
    assert r is not None
    assert r.intent == "manage_monitor"


# ─── Scheduled tasks ─────────────────────────────────────────────────

def test_schedule_list():
    matches("show my schedules", "manage_schedule")
    matches("list schedules", "manage_schedule")
    matches("list my schedule", "manage_schedule")


def test_schedule_crud():
    matches("schedule a daily weather check", "manage_schedule")
    matches("cancel the morning schedule", "manage_schedule")
    matches("pause the daily schedule", "manage_schedule")
    matches("resume the weekly schedule", "manage_schedule")


# ─── Fall-through (must NOT match) ───────────────────────────────────────────

def test_no_match_misc():
    no_match("   ")
    no_match("summarize my recordings")
    no_match("send a message to Arjun")


def test_shutdown_phrases():
    matches("shut down", "shutdown")
    matches("shutdown", "shutdown")
    matches("exit", "shutdown")
    matches("quit", "shutdown")
    matches("exit program", "shutdown")
    matches("turn off", "shutdown")
    matches("stop the assistant", "shutdown")


def test_shutdown_not_conversational():
    no_match("goodbye")
    no_match("good night")
    no_match("bye bye")
    no_match("see you later")


def test_exact_phrase_sets_tolerate_trailing_punctuation():
    """STT (Whisper) routinely appends terminal punctuation to short
    utterances. "Exit." used to bypass _SHUTDOWN_PHRASES and get
    LLM-classified as computer_task → vision-loop runaway. tl_norm
    strips trailing punctuation once and feeds all five exact-phrase
    sets, so every short command is now punctuation-tolerant.
    """
    matches("Exit.", "shutdown")
    matches("exit.", "shutdown")
    matches("Exit!", "shutdown")
    matches("exit?", "shutdown")
    matches("Shutdown.", "shutdown")
    matches("Quit.", "shutdown")
    matches("Turn off.", "shutdown")
    matches("What time is it?", "get_time")
    matches("Hide.", "hide_avatar")
    matches("Come back!", "show_avatar")
    matches("Screenshot.", "read_screen")
    matches("Read my screen.", "read_screen")


def test_phrase_sets_do_not_match_with_trailing_words():
    """Trailing PUNCTUATION is stripped; trailing WORDS still must not
    capture (otherwise "exit and save" would route to shutdown)."""
    no_match("exit and save my work")
    no_match("quit later")
    no_match("shut down the laptop")  # "the laptop" suffix means it's a computer_task


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
