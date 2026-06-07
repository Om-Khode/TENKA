"""
regex_router.py — Zero-LLM pre-routing for common commands.

Sits between shortcut matching and the intent classifier.
Returns IntentResult on match, None to fall through to LLM.
Covers ~40-50% of daily commands with zero API cost.
"""

import re
from . import config
from .intent import IntentResult

# ─── Phrase Sets ─────────────────────────────────────────────────────────────

_TIME_PHRASES = frozenset({
    "what time is it", "what's the time", "whats the time",
    "what is the time", "tell me the time", "current time",
    "what time", "time please", "time",
})

_SCREEN_PHRASES = frozenset({
    "take a screenshot", "screenshot", "what's on my screen",
    "whats on my screen", "what is on my screen", "whats on screen",
    "read my screen", "look at my screen", "screen capture",
    "capture screen", "what do you see on my screen",
})

_HIDE_PHRASES = frozenset({
    "hide", "hide yourself", "go away", "disappear",
    "hide for now", "minimize yourself", "hide please",
})

_SHOW_PHRASES = frozenset({
    "come back", "show yourself", "reappear", "come here",
    "show up", "unhide", "appear", "show avatar",
})

_SHUTDOWN_PHRASES = frozenset({
    "shut down", "shutdown", "exit", "quit",
    "exit program", "quit program", "close program",
    "stop the assistant", "stop assistant",
    "turn off", "turn yourself off",
    "close the assistant", "close assistant",
})

# Words that block "open/launch/start {X}" → computer_task.
# These indicate file/link/navigation contexts or intent keywords
# that have their own dedicated handlers.
_NON_APP_WORDS = frozenset({
    "file", "document", "folder", "link", "url",
    "page", "website", "site",
    "recording", "record",
})

# ─── Compiled Patterns ───────────────────────────────────────────────────────

# URL: starts with http(s)://, www., or word.tld
_URL_RE = re.compile(
    r"^(?:https?://\S+|www\.\S+|\w[\w\-]*\.(?:com|org|net|io|dev|gov|edu|in|uk)(?:/\S*)?)$",
    re.I,
)

# "go to / visit / navigate to / browse to {url}"
_GOTO_RE = re.compile(
    r"^(?:go\s+to|visit|navigate\s+to|browse\s+to)\s+(.+)$",
    re.I,
)

# Compound action after a navigation target — "and search for X", "then click Y"
# means multi-step task, not a simple URL open.
_COMPOUND_ACTION_RE = re.compile(
    r"\b(?:and|then)\s+(?:search|click|type|fill|find|select|submit|enter"
    r"|download|upload|sign|log|navigate|scroll|read|extract|copy|write)\b",
    re.I,
)

# "open / launch / start {app}"
_OPEN_APP_RE = re.compile(r"^(?:open|launch|start)\s+(.+)$", re.I)

# Music play: "play {X}" or "start playing {X}"
_PLAY_RE = re.compile(r"^(?:play|start\s+playing)\s+(.+)$", re.I)

# Music transport controls (no argument needed)
_MUSIC_CTRL_RE = re.compile(
    r"^(?:pause|resume|unpause"
    r"|next\s*(?:song|track)?"
    r"|previous\s*(?:song|track)?|prev\s*(?:song|track)?"
    r"|skip\s*(?:song|track)?"
    r"|stop\s+(?:the\s+)?(?:music|song|playback)"
    r"|volume\s+(?:up|down)"
    r"|turn\s+(?:it\s+)?(?:up|down))$",
    re.I,
)

# Chrome CDP setup — bidirectional. The verb and the chrome/cdp noun can
# appear in either order, separated by up to ~40 chars of intervening words.
# Critical safety net: this fires BEFORE intent classification, so a goal
# like "set up chrome" can never fall through to computer_task → vision loop.
_CDP_SETUP_VERBS = r"set\s*up|setup|configure|enable|prepare|prep|activate"
_CDP_NOUNS = r"chrome|cdp|remote\s+debug(?:ging)?(?:\s+port)?|debug\s+port"
_CDP_SETUP_RE = re.compile(
    rf"\b(?:{_CDP_SETUP_VERBS})\b[\w\s,'\"-]{{0,40}}\b(?:{_CDP_NOUNS})\b"
    rf"|"
    rf"\b(?:{_CDP_NOUNS})\b[\w\s,'\"-]{{0,40}}\b(?:{_CDP_SETUP_VERBS})\b",
    re.I,
)

# Undo requires an unambiguous CDP-related noun (not bare "chrome" — that
# would catch "remove chrome" meaning uninstall the browser).
_CDP_UNDO_VERBS = r"undo|reverse|unset|revert|restore|deactivate"
_CDP_UNDO_NOUNS = r"chrome\s+(?:setup|cdp|debug(?:ging)?)|cdp|remote\s+debug(?:ging)?(?:\s+port)?|debug\s+port"
_CDP_UNDO_RE = re.compile(
    rf"\b(?:{_CDP_UNDO_VERBS})\b[\w\s,'\"-]{{0,40}}\b(?:{_CDP_UNDO_NOUNS})\b"
    rf"|"
    rf"\b(?:{_CDP_UNDO_NOUNS})\b[\w\s,'\"-]{{0,40}}\b(?:{_CDP_UNDO_VERBS})\b",
    re.I,
)

_CDP_PREVIEW_MARKERS = ("preview", "show me", "what would", "what will", "dry run", "dry-run")

# Recording
_START_REC_RE = re.compile(
    r"^(?:start|begin)\s+(?:a\s+)?recording$", re.I,
)
_STOP_REC_RE = re.compile(
    r"^(?:stop|end|finish)\s+(?:the\s+)?recording$", re.I,
)

# Reminders — knowledge-graph C-Q1: require time anchor OR 'to <verb>' clause to avoid
# false positives on statements like "remind me Aanya works at X" which are
# meant as memory store, not reminders.
_REMINDER_TIME_ANCHOR = (
    r"(?:"
    r"\btoday\b|\btomorrow\b|\btonight\b|"
    r"\bnext\s+(?:week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b|"
    r"\bon\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
        r"january|february|march|april|may|june|july|august|september|october|november|december)\b|"
    r"\bat\s+\d|"
    r"\bin\s+\d+\s+(?:minute|hour|day|week|month)|"
    r"\b\d{1,2}\s*(?:am|pm|a\.m\.|p\.m\.)\b"
    r")"
)
_REMINDER_RE = re.compile(
    r"^remind(?:\s+me)?\s+"
    rf"(?=.*(?:\bto\s+\w+|{_REMINDER_TIME_ANCHOR}))"
    r"(.+)$",
    re.I,
)
_CANCEL_REMINDER_RE = re.compile(
    r"(?:cancel|stop|delete|remove|clear)\s+(?:(?:the|my|all)\s+)?reminder",
    re.I,
)

# Web search
_SEARCH_RE = re.compile(
    r"^(?:search(?:\s+for)?|google|look\s+up|find)\s+(.+)$",
    re.I,
)
_FILE_EXT_RE = re.compile(r"\.\w{1,5}(?:\s|$)")
_FILE_KEYWORDS = frozenset({
    "file", "files", "folder", "folders", "directory", "desktop",
    "documents", "downloads", "drive",
})

# Create note (file)
_NOTE_RE = re.compile(
    r"^(?:note\s+(?:that|down)?|write\s+down(?:\s+that)?|make\s+a\s+note(?:\s+that)?)\s+(.+)$",
    re.I,
)

# Forget / delete a stored fact — must precede _REMEMBER_FACT_RE so
# "forget that cilantro is bad" doesn't match "don't forget that ..."
_FORGET_MEMORY_RE = re.compile(
    r"^(?:forget|delete|remove|erase)\s+(?:that\s+|the\s+(?:fact|memory)\s+(?:that\s+)?|about\s+)?(.+)$",
    re.I,
)

# Store fact to memory — all "remember X" variants.
# Falls through to LLM if content contains "and" — multi-fact sentences
# like "remember X and I hate Y" need the planner to split into steps.
# Single-list "and" ("allergic to peanuts and shellfish") pays one LLM
# call but never misroutes — acceptable tradeoff.
_REMEMBER_FACT_RE = re.compile(
    r"^(?:remember\s+(?:that\s+)?|keep\s+in\s+mind\s+(?:that\s+)?|don'?t\s+forget\s+(?:that\s+)?)\s*(.+)$",
    re.I,
)

# "remember my face" / "remember my voice" — these are camera/mic actions, not facts
_REMEMBER_ACTION_RE = re.compile(
    r"^(?:remember|don'?t\s+forget)\s+(?:my|this|the)\s+(?:face|voice)\b",
    re.I,
)

# Memory query — only unambiguous recall patterns.
# "what's my X" is too broad (matches "what's my IP address?" which needs
# code_executor). The LLM + Guard 5 handle the ambiguous "my X" cases.
_MEMORY_RE = re.compile(
    r"^(?:what\s+do\s+you\s+(?:know|remember)\s+about"
    r"|do\s+you\s+(?:know|remember)"
    r"|recall)\s+(.+)$",
    re.I,
)

# knowledge-graph livetest follow-up: first-person commitment / promise recall.
# Without this, "what did I commit to this week" was getting routed to
# web_search by the LLM intent classifier — the classifier doesn't know
# we have a kg_commitments table, so it treated the question as a
# general-knowledge query. Fast-path here so the recall always lands on
# memory_query, where _is_commitment_query then surfaces OPEN PROMISES.
# Third-party variants ("what does Aanya owe me") are NOT covered — they
# need the LLM classifier to learn about KG entities (v1.1).
_COMMITMENT_RECALL_RE = re.compile(
    r"^what\s+(?:did|do|have|had|am)\s+i\s+"
    r"(?:promised?|commit(?:ted)?(?:\s+to)?|owe[d]?|agreed?(?:\s+to)?|"
    r"pledged?|supposed\s+to(?:\s+\w+)?)\b",
    re.I,
)

# Preference statements — route to small_talk so the LLM doesn't
# interpret "I prefer Chrome" as "set default browser to Chrome".
_PREFERENCE_STMT_RE = re.compile(
    r"^i\s+(?:prefer|like|enjoy|love)\s+(?:using\s+|to\s+use\s+)?\w+",
    re.I,
)


# ─── Procedure Management ────────────────────────────────────────────────────

_PROC_LIST_RE = re.compile(
    r"^(?:(?:list|show)\s+(?:(?:my|all)\s+)*(?:procedures?|routines?)"
    r"|(?:what|which)\s+(?:procedures?|routines?)\s+(?:do\s+I\s+have|are\s+there|exist))",
    re.I,
)

_PROC_DELETE_RE = re.compile(
    r"^(?:delete|remove|forget|forgot|drop)\s+(?:the\s+)?"
    r"(?:(?:procedure|routine)\s+(.+)|(.+?)\s+(?:procedure|routine))\s*$",
    re.I,
)

_PROC_RENAME_RE = re.compile(
    r"^(?:rename|change)\s+(?:the\s+)?(?:procedure|routine)\s+"
    r"(.+)\s+(?:to|into)\s+(.+)$",
    re.I,
)

_PROC_EDIT_RE = re.compile(
    r"^(?:edit|modify|update|re-?teach|redo)\s+(?:the\s+)?"
    r"(?:(?:procedure|routine)\s+(.+)|(.+?)\s+(?:procedure|routine))\s*$",
    re.I,
)

# ─── Event monitors ─────────────────────────────────────────
_MONITOR_LIST_RE = re.compile(
    r"^(?:list|show)\s+(?:(?:my|all)\s+)?(?:event\s+)?monitors?", re.I
)
_MONITOR_CRUD_RE = re.compile(
    r"^(?:pause|disable|resume|enable|unpause|delete|remove|cancel|clear)\s+(?:.*?)monitor",
    re.I,
)

# ─── Scheduled Tasks ────────────────────────────────────────
_SCHEDULE_CREATE_RE = re.compile(
    r"^schedule\s+.+", re.I
)
_SCHEDULE_LIST_RE = re.compile(
    r"^(?:list|show)\s+(?:(?:my|all)\s+)?schedules?", re.I
)
_SCHEDULE_CANCEL_RE = re.compile(
    r"^(?:cancel|delete|remove)\s+(?:.*?)schedule", re.I
)
_SCHEDULE_PAUSE_RE = re.compile(
    r"^(?:pause|disable)\s+(?:.*?)schedule", re.I
)
_SCHEDULE_RESUME_RE = re.compile(
    r"^(?:resume|enable)\s+(?:.*?)schedule", re.I
)


def match_procedure_command(text: str) -> IntentResult | None:
    """
    Detect procedure management commands (list/delete/rename/edit).
    Must run BEFORE match_trigger() to prevent false procedure execution.
    Zero LLM cost.
    """
    t = text.strip()

    if _PROC_LIST_RE.match(t):
        return IntentResult(
            intent="manage_procedure", response=t,
            params={"action": "list"},
        )

    m = _PROC_DELETE_RE.match(t)
    if m:
        name = (m.group(1) or m.group(2)).strip()
        return IntentResult(
            intent="manage_procedure", response=t,
            params={"action": "delete", "name": name},
        )

    m = _PROC_RENAME_RE.match(t)
    if m:
        name = m.group(1).strip()
        new_trigger = m.group(2).strip()
        return IntentResult(
            intent="manage_procedure", response=t,
            params={"action": "rename", "name": name, "new_trigger": new_trigger},
        )

    m = _PROC_EDIT_RE.match(t)
    if m:
        name = (m.group(1) or m.group(2)).strip()
        return IntentResult(
            intent="manage_procedure", response=t,
            params={"action": "edit", "name": name},
        )

    return None


# ─── manifest pre-route ────────────────────────────────────────────────

def _try_manifest_match(phrase: str, response_text: str) -> IntentResult | None:
    """Consult the manifest phrase registry. Returns IntentResult or None.

    Caller must pass the lowercased, stripped utterance — does not normalize.
    `response_text` is the (stripped, original-case) text used for IntentResult.response.

    Returns a match only when (a) a phrase exactly matches AND (b) the
    active window matches the manifest's `match` clause. Otherwise falls
    through to existing regex patterns / LLM classifier.

    Note: callers should batch utterances within ~1s — detect_active_app()
    is uncached and may hit win32 APIs on every call.
    """
    from .automation.manifest_registry import get_singleton
    from .automation.router import detect_active_app

    reg = get_singleton()
    if reg is None:
        return None
    if not reg.all_manifests():  # In-memory cache snapshot; skip per-utterance SQLite hit when none loaded.
        return None

    candidates = reg.lookup_phrase(phrase)
    if not candidates:
        return None

    active = detect_active_app()
    # F12: a process can have multiple manifests (e.g. two manifests both
    # matching notepad.exe with different intents). The previous single-
    # returning get_for_active_app() silently dropped any manifest that
    # wasn't the first match — so a phrase belonging to manifest B was
    # invisible whenever manifest A happened to load first. Walk ALL
    # matching manifests and route to whichever one OWNS the phrase.
    active_manifests = reg.get_all_for_active_app(active)
    if not active_manifests:
        # Phrase matched but no active app — let the classifier handle
        # the ambiguity (spec §4.1)
        return None

    active_app_ids = {am.app_id for am in active_manifests}
    for app_id, intent_id in candidates:
        if app_id in active_app_ids:
            return IntentResult(
                intent="manifest_dispatch",
                response=response_text,
                params={"app_id": app_id, "intent_id": intent_id, "slots": {}},  # TODO(manifest-based Session 4+): wire slot_extraction.extract_slots() here.
            )
    return None


# ─── Router ──────────────────────────────────────────────────────────────────

def pre_route(text: str) -> IntentResult | None:
    """
    Try to match text against known patterns without an LLM call.
    Returns IntentResult on match, None to fall through to classifier.
    """
    t = text.strip()
    tl = t.lower()
    # STT (Whisper) routinely appends ".", "?", "!" to short utterances —
    # "Exit." → would miss _SHUTDOWN_PHRASES{"exit"} and get LLM-classified
    # as computer_task with goal "Exit", triggering a vision-loop runaway.
    # Strip once and reuse for every exact-phrase set check below.
    tl_norm = tl.rstrip(".?!,;: ")

    # ── Chrome CDP setup — checked first so the goal can never fall
    # through to computer_task (which would route to the vision loop and
    # type the literal goal string into search bars in a runaway). Undo
    # is checked before setup because "undo chrome setup" contains both
    # the undo-verb and the setup-verb.
    if _CDP_UNDO_RE.search(tl):
        return IntentResult(intent="browser_cdp_setup", response=t, params={"mode": "undo"})
    if _CDP_SETUP_RE.search(tl):
        mode = "preview" if any(p in tl for p in _CDP_PREVIEW_MARKERS) else "setup"
        return IntentResult(intent="browser_cdp_setup", response=t, params={"mode": mode})

    # ── Exact phrase sets — zero regex overhead ───────────────────────────
    # Use tl_norm so STT-appended terminal punctuation doesn't bypass these.
    if tl_norm in _TIME_PHRASES:
        return IntentResult(intent="get_time", response=t, params={})

    if tl_norm in _SCREEN_PHRASES:
        return IntentResult(intent="read_screen", response=t, params={})

    if tl_norm in _HIDE_PHRASES:
        return IntentResult(intent="hide_avatar", response=t, params={})

    if tl_norm in _SHOW_PHRASES:
        return IntentResult(intent="show_avatar", response=t, params={})

    if tl_norm in _SHUTDOWN_PHRASES:
        return IntentResult(intent="shutdown", response=t, params={})

    # try manifest phrase before URL/play/app routing. Placed AFTER
    # the safety-critical exact-phrase sets above (shutdown/hide/show/etc.)
    # so a manifest can never silently capture system-level commands even
    # if a background process happens to match its `process_names`.
    manifest_match = _try_manifest_match(tl, t)
    if manifest_match is not None:
        return manifest_match

    # ── URL navigation — before "open" app check ─────────────────────────
    # open_browser = webbrowser.open() — just opens the tab.
    # browse_url = Jina scrape + LLM summary — only for explicit "read this page" requests.
    m = _GOTO_RE.match(t)
    if m:
        _goto_target = m.group(1).strip()
        if not _COMPOUND_ACTION_RE.search(_goto_target):
            return IntentResult(intent="open_browser", response=t, params={"url": _goto_target})

    # Bare URL (1-3 tokens that look like a URL)
    if len(t.split()) <= 3 and _URL_RE.match(t):
        return IntentResult(intent="open_browser", response=t, params={"url": t})

    # ── Music playback — before open-app check ("start playing" uses "start") ─
    if _PLAY_RE.match(t):
        return IntentResult(intent="code_executor", response=t, params={"goal": t})

    if _MUSIC_CTRL_RE.match(t):
        return IntentResult(intent="code_executor", response=t, params={"goal": t})

    # ── Recording — before app-launch to prevent "start recording" → app ─
    if _START_REC_RE.match(t):
        return IntentResult(intent="start_recording", response=t, params={})
    if _STOP_REC_RE.match(t):
        return IntentResult(intent="stop_recording", response=t, params={})

    # ── App launching — "open/launch/start {X}" ──────────────────────────
    m = _OPEN_APP_RE.match(t)
    if m:
        target = m.group(1).strip()
        target_lower = target.lower()
        # If the target itself looks like a URL, open it in the browser
        if _URL_RE.match(target):
            return IntentResult(intent="open_browser", response=t, params={"url": target})
        # Block file/link/page words — those belong to other intents
        target_words = set(target_lower.split())
        if target_words.isdisjoint(_NON_APP_WORDS):
            return IntentResult(intent="computer_task", response=t, params={"goal": t})

    # ── Reminders — cancel before set to avoid prefix collision ──────────
    if _CANCEL_REMINDER_RE.search(tl):
        return IntentResult(intent="cancel_reminder", response=t, params={"goal": t})

    m = _REMINDER_RE.match(t)
    if m:
        return IntentResult(intent="set_reminder", response=t, params={"goal": t})

    # ── Web search ────────────────────────────────────────────────────────
    m = _SEARCH_RE.match(t)
    if m:
        query = m.group(1).strip()
        query_lower = query.lower()
        query_words = set(query_lower.split())
        if not (_FILE_EXT_RE.search(query) or
                query_words & _FILE_KEYWORDS or
                query_words & config.BROWSER_NAMES):
            return IntentResult(intent="web_search", response=t, params={"query": query})

    # ── Event monitor CRUD must precede _FORGET_MEMORY_RE ──────────────────
    # "delete all monitors" / "remove the song monitor" would otherwise
    # match the forget-fact pattern's leading verb and route to forget_memory.
    if _MONITOR_LIST_RE.match(t):
        return IntentResult(intent="manage_monitor", response=t, params={"goal": t})
    if _MONITOR_CRUD_RE.match(t):
        return IntentResult(intent="manage_monitor", response=t, params={"goal": t})

    # ── Forget / delete a stored fact ──────────────────────────────────────
    m = _FORGET_MEMORY_RE.match(t)
    if m:
        content = m.group(1).strip()
        if not re.match(r"(?:my|this|the)\s+(?:face|voice)\b", content, re.I):
            return IntentResult(
                intent="forget_memory",
                response=t,
                params={"content": content},
            )

    # ── Store fact to memory ─────────────────────────────────────────────
    # Skip "remember my face/voice" — those are camera/mic enrollment actions
    # Skip sentences with "and" — may be multi-fact, let LLM/planner decide
    if not _REMEMBER_ACTION_RE.match(t):
        m = _REMEMBER_FACT_RE.match(t)
        if m and " and " not in m.group(1).lower():
            return IntentResult(
                intent="store_memory",
                response=t,
                params={"content": m.group(1).strip()},
            )

    # ── Create note ───────────────────────────────────────────────────────
    m = _NOTE_RE.match(t)
    if m:
        return IntentResult(
            intent="create_note",
            response=t,
            params={"content": m.group(1).strip()},
        )

    # ── Memory query ──────────────────────────────────────────────────────
    m = _MEMORY_RE.match(t)
    if m:
        return IntentResult(intent="memory_query", response=t, params={"query": m.group(1).strip()})

    # first-person commitment recall fast-path. Passes the FULL text
    # as the query so memory_search._is_commitment_query can detect the
    # commitment shape (it greps for the verb, not the captured tail).
    if _COMMITMENT_RECALL_RE.match(t):
        return IntentResult(intent="memory_query", response=t, params={"query": t})

    # ── Scheduled tasks ──
    if _SCHEDULE_LIST_RE.match(t):
        return IntentResult(intent="manage_schedule", response=t, params={"action": "list"})
    if _SCHEDULE_CANCEL_RE.match(t):
        return IntentResult(intent="manage_schedule", response=t, params={"goal": t, "action": "cancel"})
    if _SCHEDULE_PAUSE_RE.match(t):
        return IntentResult(intent="manage_schedule", response=t, params={"goal": t, "action": "toggle"})
    if _SCHEDULE_RESUME_RE.match(t):
        return IntentResult(intent="manage_schedule", response=t, params={"goal": t, "action": "toggle"})
    if _SCHEDULE_CREATE_RE.match(t):
        return IntentResult(intent="manage_schedule", response=t, params={"goal": t, "action": "create"})

    # ── Preference statements — not action requests ──────────────────────
    if _PREFERENCE_STMT_RE.match(tl):
        return IntentResult(intent="small_talk", response=t, params={})

    return None
