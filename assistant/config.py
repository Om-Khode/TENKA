"""
config.py — Centralized configuration for the TENKA Voice Assistant.

All settings are defined here so you can tweak them in one place.
Environment variables override defaults where noted.
"""

import os
from pathlib import Path

# ─── Project Paths ────────────────────────────────────────────────────────────

# Root of the TENKA project (parent of this 'assistant' folder)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ─── .env Loading ────────────────────────────────────────────────────────────
# Load .env before any os.getenv() calls so running directly (no bat file) works.
# Existing environment variables are never overridden (shell/system vars take precedence).
_env_path = PROJECT_ROOT / ".env"
if _env_path.exists():
    with open(_env_path, "r", encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                _k, _v = _k.strip(), _v.strip()
                if _k and _k not in os.environ:
                    os.environ[_k] = _v

# Sandbox directory for file operations (notes, read_file, etc.)
# All file tools are restricted to this folder for safety.
SANDBOX_DIR = Path.home() / "TENKA"

# Notes subdirectory inside the sandbox
NOTES_DIR = SANDBOX_DIR / "Notes"
SESSIONS_DIR = SANDBOX_DIR / "Sessions"
# per-app manifest YAMLs (auto-discovered, never shipped)
MANIFESTS_DIR = SANDBOX_DIR / "manifests"


# ─── Runtime Config ──────────────────────────────────────────────────────────
# Setting resolution (DB → env → default) lives in core/runtime_config.py.
# No SQLite is touched at import time — DB reads are lazy (post init_db()).

from .core.runtime_config import setting as _runtime_setting
from .core.runtime_config import REGISTRY as RUNTIME_SETTINGS_REGISTRY


# ─── Debug ───────────────────────────────────────────────────────────────────

DEBUG_LOG: bool = os.getenv("DEBUG_LOG", "true").lower() in ("1", "true", "yes")

# ─── Assistant Identity ──────────────────────────────────────────────────────
# Single source of truth for the assistant's name. Used throughout:
#   - personality system prompts
#   - intent classifier examples
#   - voice shortcut filler words (case-insensitive)
#   - wake word model filename (models/{lowercase}.onnx)
#
# Change via `/set assistant_name Luna` (restart required) or ASSISTANT_NAME=Luna
# in .env. Default "TENKA" matches the project's codename (Transformative Evolving
# Neural Knowledge Agent) — forks can rename to anything else.

ASSISTANT_NAME: str = _runtime_setting(
    "assistant_name", "TENKA", cast=str,
    description="The assistant's display / wake / persona name. Used in prompts, "
                "shortcut matching, and wake word model path. "
                "Note: renaming does NOT wipe conversation memory, so the persona "
                "may take a few turns to fully 'settle in' with the new name.",
    needs_restart=True,
)
ASSISTANT_NAME_LOWER: str = ASSISTANT_NAME.lower()


def _display_name(name: str) -> str:
    """Return a prompt-friendly form of the assistant name.

    If the user typed an all-lowercase name ("luna"), capitalize the first
    letter so the LLM treats it as a proper name ("Luna"). Any user-provided
    casing with at least one uppercase letter is preserved verbatim, so
    "McKay", "DJ", "XAI" etc. round-trip unchanged.
    """
    if not name:
        return name
    if any(c.isupper() for c in name):
        return name
    return name[:1].upper() + name[1:]


# Display form used in user-facing prompts. Keep stored value verbatim for
# filler/reserved matching — only the display form is title-cased.
ASSISTANT_NAME_DISPLAY: str = _display_name(ASSISTANT_NAME)


# ─── TCP Bridge ───────────────────────────────────────────────────────────────

# Port on which Python SENDS commands TO Unity (Python = server, Unity = client)
UNITY_COMMAND_PORT = 7777

# Port on which Python RECEIVES events FROM Unity (Python = server, Unity = client)
UNITY_EVENT_PORT = 7778
MESSAGING_BRIDGE_PORT = 7780

# How often to retry connecting / accepting (seconds)
BRIDGE_RECONNECT_INTERVAL = 2.0

# Master switch for the Unity frontend. False = terminal-only mode: no TCP bridge,
# avatar/expression/animation calls become no-ops, subtitles echo to the console.
UNITY_ENABLED: bool = _runtime_setting(
    "unity_enabled", True, cast=bool,
    description="Enable the Unity avatar frontend and TCP bridge. "
                "Off = terminal-only: TTS/STT/wake word still work, avatar commands are no-ops.",
    needs_restart=True,  # bridge is instantiated once at boot
)


# ─── Speech-to-Text (STT) ────────────────────────────────────────────────────

# Which STT backend to use: "faster_whisper" or "whisper_cpp"
#   - "faster_whisper" : Uses the faster-whisper Python library (CTranslate2).
#                        Downloads its own model on first run (~150 MB for "base.en").
#   - "whisper_cpp"    : Calls your existing whisper.cpp HTTP server.
#                        Reuses the ggml-base.bin model you already have.
STT_BACKEND = os.getenv("STT_BACKEND", "faster_whisper")

# faster-whisper settings (only used if STT_BACKEND == "faster_whisper")
FASTER_WHISPER_MODEL = os.getenv("FASTER_WHISPER_MODEL", "small.en")
FASTER_WHISPER_DEVICE = "cpu"  # "cpu" or "cuda"
FASTER_WHISPER_COMPUTE_TYPE = "int8"  # "int8", "float16", "float32"

# whisper.cpp server settings (only used if STT_BACKEND == "whisper_cpp")
WHISPER_CPP_URL = os.getenv("WHISPER_CPP_URL", "http://127.0.0.1:8080")

# Path to the existing whisper.cpp server executable + model (for auto-start)
WHISPER_CPP_DIR = PROJECT_ROOT / "Assets" / "StreamingAssets" / "VoiceAssistant" / "whisper-cpp"
WHISPER_CPP_EXE = WHISPER_CPP_DIR / "whisper-server.exe"
WHISPER_CPP_MODEL = WHISPER_CPP_DIR / "models" / "ggml-base.bin"
WHISPER_CPP_PORT = 8080

# Recording settings
RECORD_SAMPLE_RATE = 16000   # 16 kHz mono — standard for Whisper
RECORD_CHANNELS = 1
MAX_RECORD_SECONDS = 15      # safety cap


# ─── Text-to-Speech (TTS) ────────────────────────────────────────────────────

# Kokoro voice ID (e.g. "af_bella", "af_sarah", "am_adam")
TTS_VOICE = os.getenv("TTS_VOICE", "af_bella")

# Speech speed multiplier (0.5 – 2.0)
TTS_SPEED = float(os.getenv("TTS_SPEED", "1.0"))

# Sample rate for Kokoro output
TTS_SAMPLE_RATE = 24000


# ─── Vocal Voice Post-Processing ─────────────────────────────────────────────

# Base Kokoro voicepack for vocal voice (used for ALL emotions when enabled)
VOCAL_VOICE_BASE = os.getenv('VOCAL_VOICE_BASE', 'af_heart')

# Master switch — False falls back to old voicepack swapping
VOCAL_VOICE_ENABLED = os.getenv('VOCAL_VOICE_ENABLED', 'true').lower() == 'true'

# Casual/crude language mode — adds mild curse words to the assistant's vocabulary
# Makes her feel more like a real friend who roasts you. Set to 'false' to keep it clean.
VOCAL_CASUAL_LANGUAGE = os.getenv('VOCAL_CASUAL_LANGUAGE', 'false').lower() == 'true'

# Per-emotion audio effect profiles applied ON TOP of the single base voice.
#
# How it works:
#   pitch      = semitones shift via scipy resample (character identity + emotion)
#   speed      = passed to Kokoro's native speed param (natural pacing, no artifacts)
#                NOTE: speed is ALSO used to compensate for the duration change from
#                pitch shifting. Final Kokoro speed = speed / 2^(pitch/12).
#   volume     = gain multiplier (emotion intensity)
#   tremolo_hz / tremolo_depth = voice shaking for vulnerable emotions
#   eq_boost_db = high-frequency brightness boost (2-6kHz, anime quality)
#
EMOTION_VOICE_PROFILES = {
    "neutral":   {"pitch": 2.5,  "speed": 0.95,  "volume": 1.0,  "tremolo_hz": 0,   "tremolo_depth": 0,    "eq_boost_db": 3},
    "happy":     {"pitch": 3.5,  "speed": 1.15, "volume": 1.1,  "tremolo_hz": 0,   "tremolo_depth": 0,    "eq_boost_db": 5},
    "excited":   {"pitch": 4.0,  "speed": 1.25, "volume": 1.2,  "tremolo_hz": 0,   "tremolo_depth": 0,    "eq_boost_db": 5},
    "sad":       {"pitch": 1.5,  "speed": 0.85, "volume": 0.75, "tremolo_hz": 3,   "tremolo_depth": 0.15, "eq_boost_db": 1},
    "angry":     {"pitch": 2.0,  "speed": 0.95, "volume": 1.3,  "tremolo_hz": 0,   "tremolo_depth": 0,    "eq_boost_db": 3},
    "sarcastic": {"pitch": 2.5,  "speed": 0.95, "volume": 1.05, "tremolo_hz": 0,   "tremolo_depth": 0,    "eq_boost_db": 2},
    "worried":   {"pitch": 2.5,  "speed": 0.9,  "volume": 0.85, "tremolo_hz": 4,   "tremolo_depth": 0.12, "eq_boost_db": 3},
    "surprised": {"pitch": 4.0,  "speed": 1.1,  "volume": 1.15, "tremolo_hz": 0,   "tremolo_depth": 0,    "eq_boost_db": 5},
}

VALID_EMOTIONS = frozenset(EMOTION_VOICE_PROFILES.keys())

# Legacy (voice, speed) mapping for non-vocal fallback mode.
LEGACY_EMOTION_VOICE_MAP = {
    "neutral":  ("af_heart", 1.0),
    "excited":  ("af_bella", 1.15),
    "calm":     ("af_aoede",  0.9),
    "sad":      ("af_nicole",  0.85),
}

# Emotion → Unity avatar expression name.
UNITY_EXPRESSION_MAP = {
    "neutral":   "neutral",
    "happy":     "happy",
    "excited":   "happy",
    "sad":       "sad",
    "angry":     "angry",
    "sarcastic": "happy",      # smirk — closest Unity expression
    "worried":   "worried",
    "surprised": "surprised",
}


# ─── Browser Detection ────────────────────────────────────────────────────────

from .core.known_apps import get_apps_by_category as _get_apps_by_cat
BROWSER_NAMES = frozenset(_get_apps_by_cat("browser")) | {"browser"}  # "browser" = generic term


# ─── Memory Governance ────────────────────────────────────────────────────────
MEMORY_EXPIRY_DAYS_FACT = 30
MEMORY_EXPIRY_DAYS_HOW_TO = 14
MEMORY_EXPIRY_DAYS_BLOCKER = 14
VALID_MEMORY_TYPES = frozenset({"preference", "identity", "fact", "how_to", "blocker"})


# ─── LLM ──────────────────────────────────────────────────────────────────────

# Primary: Google Gemini (unified multimodal: text + vision)
# Get your key at: https://aistudio.google.com/apikey
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_MODEL_LITE = os.getenv("GEMINI_MODEL_LITE", "gemini-2.5-flash-lite")

# Groq cloud API (free-tier fallback)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# Cerebras (free-tier fallback — synthesis workhorse when Gemini down)
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "")
CEREBRAS_MODEL = os.getenv("CEREBRAS_MODEL", "gpt-oss-120b")

# Local fallback: Ollama
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

# Tavily Search API keys (rotation supported — add up to 9 keys)
TAVILY_API_KEYS: list[str] = [
    v for i in range(1, 10)
    if (v := os.getenv(f"TAVILY_API_KEY_{i}", ""))
]
# Also support single key for simplicity
_tavily_single = os.getenv("TAVILY_API_KEY", "")
if _tavily_single and _tavily_single not in TAVILY_API_KEYS:
    TAVILY_API_KEYS.append(_tavily_single)

# LLM timeout in seconds
LLM_TIMEOUT = 30


# ─── Prompt Builders (moved to llm/prompts.py) ───────────────────────────────
# Re-exports for callers that still reference config.build_personality_prompt etc.

def build_personality_prompt():
    from .llm.prompts import build_personality_prompt as _bp
    return _bp()

def build_intent_prompt(scope: str | None = None,
                        active_intents: set[str] | None = None):
    from .llm.prompts import build_intent_prompt as _bi
    return _bi(scope=scope, active_intents=active_intents)


def _get_llm_system_prompt():
    from .llm.prompts import get_system_prompt
    return get_system_prompt()

LLM_SYSTEM_PROMPT = _get_llm_system_prompt()

# ─── Intent Detection ────────────────────────────────────────────────────────

# The list of intents the LLM can classify into
INTENTS = [
    "small_talk",
    "unknown",
    "create_note",
    "open_browser",
    "get_time",
    "computer_task",
    "read_screen",
    "find_and_click",
    "code_executor",
    "memory_query",
    "start_recording",
    "stop_recording",
    "get_recording",
    "summarize_recording",
    "web_search",
    "browse_url",
    "file_task",
    "set_reminder",
    "cancel_reminder",
    "hide_avatar",
    "show_avatar",
    "meet_face",
    "recognize_face",
    "forget_face",
    "camera_look",
    "planner",
    "manage_shortcut",
    "manage_procedure",
    "manage_schedule",
    "manage_monitor",
    "enroll_voice",
    "forget_voice",
    "browser_cdp_setup",
    "store_memory",
    "forget_memory",
    "shutdown",
    "manifest_dispatch",  # synthetic intent fired by regex_router
]

# System prompt for the intent-detection LLM call
INTENT_SYSTEM_PROMPT = """\
You are an intent classifier. Output ONE JSON object and nothing else:
{"intent": "<name>", "params": {<key-values>}}
No markdown, no prose, no explanation.

TOP PRIORITY RULES (read first — these settle the common edge cases):
 1. If the task CAN be done via code or an API — music, email, messaging, calendar, weather, math, system info, volume, any third-party service — the intent MUST be code_executor. NOT computer_task. EXCEPTION: tasks that need an interactive web session (booking, reserving, purchasing, ordering, registering, or any goal requiring navigating a website, choosing from options, and filling forms) → planner, because they need browser automation, not API code.
 2. computer_task is for visible-window GUI work: opening desktop apps (Settings, Notepad, Calculator, File Explorer), typing into fields, clicking menus, filling forms, dragging. "Open X" where X is a desktop app → computer_task.
 3. find_and_click is ONLY for clicking UI text already visible on screen ("click Submit", "click Accept"). Opening an app → computer_task. Clicking inside a named app ("click play on the music app") → computer_task.
 4. web_search = current, recent, or time-sensitive info (news, scores, prices, who currently holds a position). small_talk = stable well-known knowledge (math, science concepts, history, how things work, jokes, opinions).
 5. open_browser = navigate/go to a URL (just opens the tab). browse_url = READ or SUMMARIZE a page's content ("summarize X", "what does X say about Y"). If user says "go to", "open", or "visit" a site → open_browser. If user wants to READ the content → browse_url.
 6. planner = task needs MULTIPLE tools OR an interactive web session. EXPLICIT (2+ actions joined by "and"/"then"/commas) OR IMPLICIT (needs camera + code, vision + search, scan + analyze) OR INTERACTIVE (booking, reserving, purchasing, ordering — needs browser navigation + form filling, not a simple API call).
 7. When torn between code_executor and computer_task, pick code_executor — it auto-falls-back to GUI when GUI is actually required.
 8. Any goal / query / text param MUST be the user's exact spoken words. Never copy descriptions or phrasing from this prompt.
 9. For ANY file work (find, read, list, open, write, rename, move, delete) use file_task.
10. unknown is ONLY for truly unintelligible or empty input. Never fall back to unknown for real questions — use small_talk or web_search instead.
11. IMPORTANT: "remember my X is Y" / "remember the X is Y" / "remember that X" → store_memory for a SINGLE fact. If the sentence contains multiple facts joined by "and"/"also"/"plus" (e.g. "remember X and Y", "remember X, also Y"), use planner instead — it will split into separate store_memory steps. NOT create_note, NOT manage_shortcut, NOT meet_face. Only use meet_face when the user is introducing themselves face-to-face ("this is Sarah", "I'm Alex"), not for "remember my name is X".
12. "forget X" / "delete the fact about X" / "remove the memory of X" → forget_memory. User wants to delete a previously stored fact. NOT forget_face/forget_voice (those delete biometric data).
13. "schedule X" / "every N minutes check X" / "daily at X" → manage_schedule (time-based, cron). "remind me X" → set_reminder (one-time alert). "when X happens do Y" / "skip songs that..." / "notify me when..." / "watch for..." → manage_monitor (event-driven, reacts to OS events like media changes or window focus).

Intent catalog (intent | params | when to use):
small_talk          | {}              | casual chat, greetings, jokes, stable knowledge (math, science, history)
create_note         | title, content  | save a note
open_browser        | url             | navigate/go to/visit a URL (just opens tab, doesn't read content)
get_time            | {}              | current time
computer_task       | goal            | GUI: open desktop apps, type/click inside named apps, fill forms, drag
read_screen         | {}              | describe what's currently visible on screen
find_and_click      | text            | click UI text ALREADY visible on screen (not inside a named app)
code_executor       | goal            | API/code-doable: music, email, messaging, weather, math, volume, services
memory_query        | query           | recall stored facts or past conversations ("what did I ask earlier", "do I have food restrictions?", "what's my wifi password?", "what did I tell you about X?")
start_recording     | {}              | begin dictation/recording
stop_recording      | {}              | end recording
get_recording       | session_id?     | retrieve a session ("latest" if user says last/recent)
summarize_recording | session_id?     | summarize a session (defaults to "latest")
web_search          | query           | current/recent/time-sensitive info (news, scores, prices, who-currently-X)
browse_url          | url             | READ/SUMMARIZE a page's content (infer URL if only site named)
file_task           | goal            | ALL file ops: find, read, list, open, write, rename, move, delete
set_reminder        | goal            | one-time reminder — include ALL time words in goal
cancel_reminder     | goal            | cancel one or all reminders
hide_avatar         | {}              | avatar hides/disappears
show_avatar         | {}              | avatar reappears
camera_look         | {}              | look through webcam, describe view or answer visual question
meet_face           | name            | learn/save face during face-to-face intro ("this is Sarah", "I'm Alex")
recognize_face      | {}              | identify who is on camera
forget_face         | name            | delete a saved face
planner             | goal            | multi-step: 2+ tools joined by "and"/"then" OR implicit (camera+code) OR interactive web (booking, purchasing, reserving)
manage_shortcut     | goal            | create/delete/list voice shortcuts ("when I say X, do Y")
manage_procedure    | goal            | list/delete/rename/edit taught procedures
manage_schedule     | goal            | schedule/list/cancel/pause/resume recurring monitors
manage_monitor      | goal            | create/list/pause/resume/delete event monitors ("when X happens, do Y")
enroll_voice        | {}              | register voice for speaker verification
forget_voice        | {}              | delete saved voiceprint
store_memory        | content         | "remember my/the X is Y" — storing a fact (NOT meet_face, NOT reminder)
forget_memory       | content         | "forget about X" / "delete memory of X" — removing a stored fact (NOT forget_face/forget_voice)
browser_cdp_setup   | mode            | configure browser --remote-debugging-port (setup/undo/preview)
unknown             | {}              | truly unintelligible/empty input ONLY

Param rules:
  - goal / query / text → verbatim user speech. Don't paraphrase. Don't copy from this prompt.
  - url → full URL. If user only named a site, infer the URL (wikipedia, bbc, github, etc.).
  - name → extracted person name, or empty string if unclear.
  - session_id → "latest" when user says last/recent/latest; otherwise the specific ID.
  - time expressions on set_reminder must be INCLUDED in goal (e.g. "in 5 minutes", "at 9 PM").

Few-shot (ambiguous cases):
  "what's my battery level"            → {"intent":"code_executor","params":{"goal":"what's my battery level"}}
  "open settings"                      → {"intent":"computer_task","params":{"goal":"open settings"}}
  "open google.com"                    → {"intent":"open_browser","params":{"url":"https://google.com"}}
  "summarize wikipedia black holes"    → {"intent":"browse_url","params":{"url":"https://en.wikipedia.org/wiki/Black_hole"}}
  "click the Submit button"            → {"intent":"find_and_click","params":{"text":"Submit"}}
  "click play on the music app"        → {"intent":"computer_task","params":{"goal":"click play on the music app"}}
  "remember my fav browser is Firefox" → {"intent":"store_memory","params":{"content":"my favorite browser is Firefox"}}
  "forget about cilantro"              → {"intent":"forget_memory","params":{"content":"cilantro"}}
  "remember my name is Alex and my birthday is March 5" → {"intent":"planner","params":{"goal":"remember my name is Alex and my birthday is March 5"}}
  "this is Sarah"                      → {"intent":"meet_face","params":{"name":"Sarah"}}
  "check messages and if Mom messaged reply saying I will be late" → {"intent":"planner","params":{"goal":"check messages and if Mom messaged reply saying I will be late"}}
  "book movie tickets for 2 people"    → {"intent":"planner","params":{"goal":"book movie tickets for 2 people"}}
  "schedule a web search for AI every morning" → {"intent":"manage_schedule","params":{"goal":"schedule a web search for AI every morning"}}
  "skip Japanese songs on Spotify" → {"intent":"manage_monitor","params":{"goal":"skip Japanese songs on Spotify"}}
  "show my monitors" → {"intent":"manage_monitor","params":{"goal":"show my monitors"}}
  "remind me in 5 minutes to drink water" → {"intent":"set_reminder","params":{"goal":"remind me in 5 minutes to drink water"}}

Output the JSON object only. Empty params → {}."""


# ─── Policy Engine ────────────────────────────────────────────────────────────

# Whitelist of allowed intents (anything not here is denied)
ALLOWED_INTENTS = set(INTENTS)

# File write safety mode
# True  (default): write op is restricted to SANDBOX_DIR only
# False           : write op allowed to any user-specified path (still requires confirmation)
FILE_WRITE_SAFE_MODE = os.getenv("FILE_WRITE_SAFE_MODE", "true").lower() == "true"

# Code-executor: inject per-service knowledge entries into LLM code-gen prompts.
# Defaults OFF for v1.0 (CE-DYN post-mortem 2026-05-30 found stale "never"
# entries become permanent dogma and mislead future generations — entries are
# auto-saved on structural failures but have no TTL or confidence scoring).
# WRITES still happen (collection continues for v1.1 analysis), only READS
# are gated. Flip back to "true" once knowledge hygiene lands in v1.1.
CODE_EXECUTOR_INJECT_KNOWLEDGE = os.getenv(
    "CODE_EXECUTOR_INJECT_KNOWLEDGE", "false"
).lower() == "true"

# ─── knowledge-graph kill switches ────────────────────────────────────────────────────
# Set either to "false" / "0" / "no" to disable that side of the KG layer
# without code revert. Both default ON. See spec §12.
# NOTE: knowledge_graph.py also reads these env vars directly so monkeypatching
# env in tests works without re-importing config.
KG_INGEST_ENABLED = os.environ.get("KG_INGEST_ENABLED", "true").lower() not in {"false", "0", "no"}
KG_QUERY_INJECTION_ENABLED = os.environ.get("KG_QUERY_INJECTION_ENABLED", "true").lower() not in {"false", "0", "no"}

# Blacklist of dangerous words/patterns in user input or parameters.
# Patterns are regexes (re.IGNORECASE). Use \b word-boundaries on bare command
# names so they match the command, not random substrings — e.g. "rm " used to
# substring-match inside "form with" / "alarm went off", which broke any goal
# mentioning a form field. Patterns with mandatory punctuation (rm/, exec()
# already self-anchor and don't need \b.
DANGEROUS_PATTERNS = [
    r"\brm\b", r"rm/", r"\brmdir\b",
    r"\bshell\b", r"exec\(", r"eval\(", r"execute\(", r"\bcmd\b", r"\bcommand\b",
    r"\bsudo\b", r"\badmin\b", r"\broot\b",
    r"\bformat\b", r"\bfdisk\b", r"\bmkfs\b",
    r"\bshutdown\b", r"\breboot\b", r"\brestart\b",
    r"\bkill\b", r"\bterminate\b", r"\btaskkill\b",
]


# ─── Wake Word (openWakeWord) ─────────────────────────────────────────────────
# openWakeWord is fully open-source — NO API key needed!
# https://github.com/dscripka/openWakeWord

# Enable/disable wake word detection
WAKE_WORD_ENABLED = os.getenv("WAKE_WORD_ENABLED", "true").lower() == "true"

# Path to a custom trained wake-word model (.onnx file)
# Train via Google Colab: https://colab.research.google.com/drive/1q1oe2zOyZp7UsB3jJiQ1IFn8z5YfjwEb
# Use target phrases that match your assistant name (e.g. ["tenka", "ten-ka", "Tenka"]).
# Download the .onnx file and place it at assistant/models/{assistant_name_lower}.onnx
WAKE_WORD_MODEL_PATH = PROJECT_ROOT / "assistant" / "models" / f"{ASSISTANT_NAME_LOWER}.onnx"

# Built-in wake word fallback (OPT-IN). Empty by default because built-in models
# emit much higher per-frame scores than custom .onnx models — using them with
# the default WAKE_WORD_THRESHOLD (tuned for custom) causes false triggers that
# can loop on TTS output. If you don't have a custom model, set this in .env to
# one of the openWakeWord built-ins: "hey_jarvis_v0.1", "alexa_v0.1",
# "hey_mycroft_v0.1" — and ALSO raise WAKE_WORD_THRESHOLD (try 0.5).
WAKE_WORD_BUILTIN = os.getenv("WAKE_WORD_BUILTIN", "")

# Inference framework: "onnx" (Windows/Linux) or "tflite" (Linux only)
WAKE_WORD_INFERENCE_FRAMEWORK = os.getenv("WAKE_WORD_FRAMEWORK", "onnx")

# Detection threshold (0.0 to 1.0)
# With sliding window accumulation, this is the SUM of scores over ~1.2 seconds.
# A custom-trained openWakeWord model typically produces ~0.02-0.08 per frame spike,
# so a single wake-word utterance accumulates ~0.10-0.15 total.
# Tune up if you get false positives, tune down if it doesn't trigger.
WAKE_WORD_THRESHOLD = float(os.getenv("WAKE_WORD_THRESHOLD", "0.02"))

# Audio chunk size in samples (1280 = 80ms at 16kHz, recommended by openWakeWord)
WAKE_WORD_CHUNK_SIZE = int(os.getenv("WAKE_WORD_CHUNK_SIZE", "1280"))

# Cooldown in seconds after a detection (prevents rapid re-triggers)
WAKE_WORD_COOLDOWN = float(os.getenv("WAKE_WORD_COOLDOWN", "2.0"))

# How many seconds to record after wake word detection (auto-stop timer)
WAKE_WORD_RECORD_SECONDS  = float(os.getenv("WAKE_WORD_RECORD_SECONDS",  "5.0"))
FOLLOW_UP_LISTEN_SECONDS  = float(os.getenv("FOLLOW_UP_LISTEN_SECONDS",  "5.0"))


# ─── Keyboard Trigger ────────────────────────────────────────────────────────

# Key to press to start/stop recording (push-to-talk).
# Single character (v, j, ...) or pynput Key name (home, end, page_up,
# f1..f12, insert, delete, scroll_lock, etc.). Case-insensitive.
# needs_restart=True because the listener captures this once at startup.
PUSH_TO_TALK_KEY = _runtime_setting(
    "push_to_talk_key", "v", str,
    description="Key to start/stop recording (single char or 'home', 'end', 'f1', etc.)",
    needs_restart=True,
)


# ─── Avatar Behavior ─────────────────────────────────────────────────────────

AVATAR_PEEK_DURATION = 3.5        # seconds avatar stays visible during a peek - Default 3.5
AVATAR_PEEK_INTERVAL_MIN = 10    # seconds minimum between auto-peeks - Default 300
AVATAR_PEEK_INTERVAL_MAX = 15    # seconds maximum between auto-peeks - Default 600
AVATAR_HIDDEN_SLIVER = 40         # pixels of avatar visible when hidden - Default 40
AVATAR_LERP_SPEED = 6.0           # window slide speed - Default 6.0


# ─── Camera ──────────────────────────────────────────────────────────────────

CAMERA_ENABLED = os.getenv("CAMERA_ENABLED", "true").lower() == "true"
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
CAMERA_MAX_WIDTH = int(os.getenv("CAMERA_MAX_WIDTH", "1280"))


# ─── Face Recognition ─────────────────────────────────────────────────────────
FACE_RECOGNITION_TOLERANCE = float(os.getenv("FACE_RECOGNITION_TOLERANCE", "0.5"))
# Lower = stricter matching (fewer false positives). Range: 0.4 (strict) to 0.6 (loose).
FACE_MAX_ENCODINGS = int(os.getenv("FACE_MAX_ENCODINGS", "5"))


# ─── Speaker Verification ────────────────────────────────────────────────────

# Master switch
SPEAKER_VERIFY_ENABLED = os.getenv("SPEAKER_VERIFY_ENABLED", "true").lower() == "true"

# Base cosine similarity threshold (0.0 to 1.0)
# Higher = stricter. With VAD trimming, owner scores should be 0.5+.
SPEAKER_VERIFY_THRESHOLD = float(os.getenv("SPEAKER_VERIFY_THRESHOLD", "0.50"))

# Absolute floor — dynamic adjustments never go below this
SPEAKER_VERIFY_THRESHOLD_FLOOR = float(os.getenv("SPEAKER_VERIFY_THRESHOLD_FLOOR", "0.40"))

# Maximum enrollment samples stored (like FACE_MAX_ENCODINGS)
SPEAKER_MAX_ENROLLMENTS = int(os.getenv("SPEAKER_MAX_ENROLLMENTS", "10"))

# Enrollment recording settings
SPEAKER_ENROLL_RECORD_SECONDS = float(os.getenv("SPEAKER_ENROLL_RECORD_SECONDS", "3.0"))
SPEAKER_ENROLL_NUM_SAMPLES = int(os.getenv("SPEAKER_ENROLL_NUM_SAMPLES", "5"))

# Minimum audio duration (seconds) for verification attempt
# Shorter audio → weaker embeddings → fail-open
SPEAKER_MIN_AUDIO_SECONDS = float(os.getenv("SPEAKER_MIN_AUDIO_SECONDS", "1.0"))

# Voiceprint storage
SPEAKER_VOICEPRINT_PATH = SANDBOX_DIR / "memory" / "voiceprint.npz"


# ─── Code Executor ───────────────────────────────────────────────────────────

# True = Tier 3 (unrestricted Python). False = Tier 1 sandbox only.
CODE_EXECUTOR_POWER_MODE = os.getenv("CODE_EXECUTOR_POWER_MODE", "false").lower() == "true"

# Service Knowledge Files
# "immediate" = ask for approval right after successful retry
# "deferred"  = ask on next unrelated interaction (better UX, less intrusive)
KNOWLEDGE_APPROVAL_MODE = os.getenv("KNOWLEDGE_APPROVAL_MODE", "immediate")

# ─── Proactive Nudges ────────────────────────────────────────────────────────

# Master switch
PROACTIVE_ENABLED = os.getenv("PROACTIVE_ENABLED", "true").lower() == "true"
 
# "always"    — nudge fires immediately when ready (fully proactive)
# "idle_only" — nudge fires only when assistant is not processing a request
PROACTIVE_MODE = os.getenv("PROACTIVE_MODE", "always")
 
# How often (in minutes) the background analyzer re-runs
PROACTIVE_INTERVAL_MINUTES = int(os.getenv("PROACTIVE_INTERVAL_MINUTES", "30"))
 
# How many minutes of silence before an idle nudge is sent
PROACTIVE_IDLE_THRESHOLD_MINUTES = int(os.getenv("PROACTIVE_IDLE_THRESHOLD_MINUTES", "10"))

# ─── Messaging Notifications ──────────────────────────────────────────────────

MESSAGING_NOTIFY_DEBOUNCE: float = float(os.getenv("MESSAGING_NOTIFY_DEBOUNCE", "5.0"))
MESSAGING_SUPPRESS_WINDOW: float = float(os.getenv("MESSAGING_SUPPRESS_WINDOW", "300.0"))

# Auto-connect messaging services at startup
# Comma-separated list of services (e.g. "whatsapp,telegram")
# If empty, defaults to all registered services with saved sessions.
MESSAGING_AUTO_CONNECT = os.environ.get("MESSAGING_AUTO_CONNECT", "").strip()

# Read vs summarize threshold for incoming messages
# If <= this many messages, read them out. If more, LLM-summarize.
INCOMING_READ_THRESHOLD: int = 3


# ─── Runtime Config — remaining settings ─────────────────────────────────────
# Each _runtime_setting() call registers metadata for /config and returns the
# current value (DB → env → hardcoded default via core/runtime_config.py).

# ─── Voice I/O ────
LISTEN_TO_EVERYONE: bool = _runtime_setting(
    "listen_to_everyone", False, cast=bool,
    description="If true, speaker verification is disabled — anyone can issue commands. "
                "Also toggled by the 'listen to everyone' voice phrase.",
)
FOLLOWUP_TIMER: float = _runtime_setting(
    "followup_timer", FOLLOW_UP_LISTEN_SECONDS, cast=float,
    description="Seconds the assistant listens for a follow-up utterance after TTS finishes.",
)
FOLLOW_UP_LISTEN_SECONDS = FOLLOWUP_TIMER  # keep legacy name in sync

TTS_SPEED = _runtime_setting(
    "tts_speed", TTS_SPEED, cast=float,
    description="TTS speech rate multiplier (0.5–2.0). Lower = slower & clearer, higher = faster.",
)
VOCAL_VOICE_ENABLED = _runtime_setting(
    "vocal_voice_enabled", VOCAL_VOICE_ENABLED, cast=bool,
    description="Vocal voice post-processing (pitch shift, EQ, tremolo). Off = plain Kokoro voice.",
)
VOCAL_CASUAL_LANGUAGE = _runtime_setting(
    "vocal_casual_language", VOCAL_CASUAL_LANGUAGE, cast=bool,
    description="Let the assistant use mild curses (damn, crap, dumbass...). Persona flavor, not hostility.",
    needs_restart=True,  # tsundere prompt reads env at PersonalityLoader init
)
ACTIVE_PERSONALITY = _runtime_setting(
    "personality", "warm_honest", cast=str,
    description="Active personality base (warm_honest, tsundere, minimal). "
                "Changes take effect immediately — no restart needed.",
)

# ─── Wake Word ────
WAKE_WORD_SENSITIVITY: float = _runtime_setting(
    "wake_word_sensitivity", WAKE_WORD_THRESHOLD, cast=float,
    description="openWakeWord detection threshold (0.0–1.0). Lower = more sensitive.",
)
WAKE_WORD_THRESHOLD = WAKE_WORD_SENSITIVITY  # keep legacy name in sync

WAKE_WORD_COOLDOWN = _runtime_setting(
    "wake_word_cooldown", WAKE_WORD_COOLDOWN, cast=float,
    description="Seconds to ignore the wake word after it fires. Raise if rapid re-triggers happen.",
)
WAKE_WORD_ENABLED = _runtime_setting(
    "wake_word_enabled", WAKE_WORD_ENABLED, cast=bool,
    description="Master switch for wake word detection. Off = push-to-talk only.",
    needs_restart=True,  # gates listener startup
)

# ─── Speaker Verification ────
SPEAKER_VERIFY_ENABLED = _runtime_setting(
    "speaker_verify_enabled", SPEAKER_VERIFY_ENABLED, cast=bool,
    description="Master switch for speaker verification. Off = any speaker is trusted.",
)
SPEAKER_VERIFY_THRESHOLD = _runtime_setting(
    "speaker_verify_threshold", SPEAKER_VERIFY_THRESHOLD, cast=float,
    description="SV cosine similarity threshold (0.0–1.0). Lower if it rejects you often, "
                "higher if impostors slip through.",
)

# ─── Camera / Face ────
CAMERA_ENABLED = _runtime_setting(
    "camera_enabled", CAMERA_ENABLED, cast=bool,
    description="Camera + face recognition. Off saves CPU and improves privacy.",
    needs_restart=True,  # gates camera subsystem init
)
FACE_RECOGNITION_TOLERANCE = _runtime_setting(
    "face_recognition_tolerance", FACE_RECOGNITION_TOLERANCE, cast=float,
    description="Face match strictness (0.4 strict – 0.6 loose). Lower = fewer false positives.",
)

# ─── Proactive Nudges ────
PROACTIVE_ENABLED = _runtime_setting(
    "proactive_enabled", PROACTIVE_ENABLED, cast=bool,
    description="Master switch for unprompted nudges / reflection.",
    needs_restart=True,  # gates analyzer thread start
)
PROACTIVE_MODE = _runtime_setting(
    "proactive_mode", PROACTIVE_MODE, cast=str,
    description="'always' fires immediately when ready; 'idle_only' waits until assistant is idle.",
)
PROACTIVE_INTERVAL_MINUTES = _runtime_setting(
    "proactive_interval_minutes", PROACTIVE_INTERVAL_MINUTES, cast=int,
    description="How often the nudge analyzer re-checks (minutes).",
    needs_restart=True,  # captured in the analyzer thread's sleep loop
)
PROACTIVE_IDLE_THRESHOLD_MINUTES = _runtime_setting(
    "proactive_idle_threshold_minutes", PROACTIVE_IDLE_THRESHOLD_MINUTES, cast=int,
    description="Minutes of silence before an idle nudge fires.",
)

# ─── Verification Layer ────
VERIFY_ENABLED = _runtime_setting(
    "verify_enabled", True, cast=bool,
    description="Master switch for step verification. "
                "Off = skip all pre-checks and post-verifies — fastest, but silent failures possible.",
)
VERIFY_BROWSER_STEPS = _runtime_setting(
    "verify_browser_steps", True, cast=bool,
    description="Verify browser (Playwright) steps. Off only if you're profiling latency.",
)
VERIFY_APP_STEPS = _runtime_setting(
    "verify_app_steps", True, cast=bool,
    description="Verify native app (Terminator) steps. Off only if you're profiling latency.",
)
VERIFY_VISION_FALLBACK = _runtime_setting(
    "verify_vision_fallback", True, cast=bool,
    description="When code-tier verification is ambiguous (e.g. click outcomes), "
                "escalate to a Gemini Flash vision call. Off = treat ambiguous as ok.",
)
VERIFY_STRICT_TEXT_MATCH = _runtime_setting(
    "verify_strict_text_match", False, cast=bool,
    description="True = exact text match on field readback (catches autocomplete drift but causes "
                "false fails on phone/email auto-formatting). False = case-insensitive contains.",
)
VERIFY_MIN_CONFIDENCE = _runtime_setting(
    "verify_min_confidence", 0.5, cast=float,
    description="Vision-tier confidence threshold to count as a real failure. "
                "Lower = more retries, higher = more silent passes.",
)
VERIFY_MAX_RETRIES = _runtime_setting(
    "verify_max_retries", 1, cast=int,
    description="How many self-heal attempts after a verify_failed (per step). "
                "Hard-capped at 1 in policy — endless retry is the antipattern that bricks demos.",
)

# ─── Plan-and-Execute Hardening ────
# Three independent kill-switches for the plan-and-execute hardening work that fixes
# the verifier-/updater-hallucinates-completion class of bugs. Each flag
# isolates one sub-feature so it can be reverted to prior behaviour without
# code removal during a live regression. Flip to False only as a debugging /
# rollback aid; default-on is the supported configuration.
DROPDOWN_COMMIT_GUARD_ENABLED = _runtime_setting(
    "dropdown_commit_guard_enabled", True, cast=bool,
    description="Auto-inject keyboard_press(enter) when an action batch "
                "navigates a dropdown via arrow keys without a commit action. Off = "
                "trust the planner's batch as-is (will regress on Down×N selections).",
)
DETERMINISTIC_MATCHING_ENABLED = _runtime_setting(
    "deterministic_matching_enabled", True, cast=bool,
    description="Action-signature TODO marking with vision-confirm for "
                "select TODOs. Off = revert to text-only LLM updater (will "
                "regress on dropdown completion hallucinations).",
)
DYNAMIC_BUDGET_ENABLED = _runtime_setting(
    "dynamic_budget_enabled", True, cast=bool,
    description="Dynamic loop budget sized from TODO count + dropdowns "
                "(capped at 15) plus stuck-step detector (3 zero-progress batches "
                "→ abort). Off = MAX_LOOPS=8 fixed, no stuck detection.",
)

# ─── Dialog-Engagement Gate ────
# Suppresses the overlay-dismiss action when the agent has recently
# engaged with the modal element (typed into a field, clicked into it, etc.).
# Fixes the 2026-04-26 Truein form bug where the overlay handler misclassified a form-modal
# the agent was filling as an unwanted overlay and clicked its close X.
# Disable to restore pre-2026-04-26 behaviour.
DIALOG_ENGAGEMENT_GATE_ENABLED = _runtime_setting(
    "dialog_engagement_gate_enabled", True, cast=bool,
    description="Dialog-engagement gate: refuse to dismiss overlays when recent agent "
                "actions show successful engagement with the modal surface. "
                "Off = pre-fix behaviour (dismisses regardless of "
                "engagement; can close form-modals the agent is filling).",
)

# ─── Chrome CDP Attach (browser_cdp.py) ────
# Connect to user's already-open Chrome instead of always launching the
# bundled Chromium. Requires Chrome to be launched with --remote-debugging-port.
# When CDP is unreachable, all paths fall back to bundled Chromium silently.
BROWSER_PREFER_CDP = _runtime_setting(
    "browser_prefer_cdp", True, cast=bool,
    description="When True (default), TENKA tries to attach to a running "
                "Chrome with --remote-debugging-port=9222 before launching "
                "its own bundled Chromium. Off = always use bundled.",
)
BROWSER_CDP_PORT = _runtime_setting(
    "browser_cdp_port", 9222, cast=int,
    description="Port to probe for Chrome's CDP endpoint. Default 9222 is "
                "the Chrome convention. Change only if you launch Chrome "
                "with a non-default --remote-debugging-port.",
)
BROWSER_CDP_PROBE_TTL = _runtime_setting(
    "browser_cdp_probe_ttl", 30.0, cast=float,
    description="How long (seconds) the CDP availability probe result is "
                "cached. Lower = more probes (slight latency); higher = "
                "stale state risks (e.g. user closed Chrome mid-session). "
                "30s is the sweet spot.",
)

# ─── DOM accessibility-tree perception (browser_dom.py) ────
BROWSER_DOM_TREE_TOKEN_BUDGET = _runtime_setting(
    "browser_dom_tree_token_budget", 4000, cast=int,
    description="Max tokens the perceived element tree may consume in the "
                "DOM planner prompt. When exceeded, the perceiver drops "
                "bounds → drops placeholders → prunes off-viewport "
                "elements. Budget keeps the planner call cheap and fast.",
)
BROWSER_DOM_CACHE_TTL = _runtime_setting(
    "browser_dom_cache_ttl", 10.0, cast=float,
    description="Tree-cache TTL in seconds. Pages with periodic mutations "
                "(timers, polling) shouldn't rely on a stale tree. 10s "
                "balances reuse against staleness. Manual invalidation "
                "happens on click/press/navigation regardless.",
)

# ─── Routing decision (desktop_automation._choose_browser_mode) ────
BROWSER_DOM_MODE_ENABLED = _runtime_setting(
    "browser_dom_mode_enabled", True, cast=bool,
    description="Master switch for the DOM-aware planner path. When False, "
                "browser-content goals always route to the legacy vision-"
                "loop fallback regardless of CDP availability. Useful "
                "kill-switch if DOM-mode regresses against a specific site.",
)

# ─── Messaging ────
MESSAGING_NOTIFY_DEBOUNCE = _runtime_setting(
    "messaging_notify_debounce", MESSAGING_NOTIFY_DEBOUNCE, cast=float,
    description="Wait window (seconds) before announcing a new message. Use 20–30 in real life.",
)
MESSAGING_SUPPRESS_WINDOW = _runtime_setting(
    "messaging_suppress_window", MESSAGING_SUPPRESS_WINDOW, cast=float,
    description="After reading messages, stay silent for this long (seconds) on new ones from the same chat.",
)
INCOMING_READ_THRESHOLD = _runtime_setting(
    "incoming_read_threshold", INCOMING_READ_THRESHOLD, cast=int,
    description="≤N messages → read verbatim; more → LLM-summarize.",
)

# ─── Event monitor settings ───────────────────────────────────────────
EVENT_MONITOR_ENABLED: bool = True
EVENT_MONITOR_MAX_ACTIVE: int = 20
EVENT_MONITOR_LLM_COOLDOWN: int = 10
EVENT_MONITOR_DEBOUNCE_SECS: float = 1.0

# ─── Telemetry settings ───────────────────────────────────────────────────────
TELEMETRY_RETENTION_DAYS: int = 90


def reload_runtime_settings() -> None:
    """Re-read every registered runtime setting from the DB and refresh module globals.

    Called:
      1. Once at startup after init_db() so persisted values override the
         defaults captured during initial import.
      2. After every /set or /reset command so in-memory callers see the change.
    """
    from .core.runtime_config import reload_all

    new_values = reload_all()

    global ASSISTANT_NAME, ASSISTANT_NAME_LOWER, ASSISTANT_NAME_DISPLAY
    global LISTEN_TO_EVERYONE, FOLLOWUP_TIMER, FOLLOW_UP_LISTEN_SECONDS
    global WAKE_WORD_SENSITIVITY, WAKE_WORD_THRESHOLD, WAKE_WORD_COOLDOWN, WAKE_WORD_ENABLED
    global TTS_SPEED, VOCAL_VOICE_ENABLED, VOCAL_CASUAL_LANGUAGE
    global SPEAKER_VERIFY_ENABLED, SPEAKER_VERIFY_THRESHOLD
    global CAMERA_ENABLED, FACE_RECOGNITION_TOLERANCE
    global PROACTIVE_ENABLED, PROACTIVE_MODE, PROACTIVE_INTERVAL_MINUTES
    global PROACTIVE_IDLE_THRESHOLD_MINUTES
    global MESSAGING_NOTIFY_DEBOUNCE, MESSAGING_SUPPRESS_WINDOW, INCOMING_READ_THRESHOLD
    global UNITY_ENABLED, PUSH_TO_TALK_KEY
    global VERIFY_ENABLED, VERIFY_BROWSER_STEPS, VERIFY_APP_STEPS
    global VERIFY_VISION_FALLBACK, VERIFY_STRICT_TEXT_MATCH
    global VERIFY_MIN_CONFIDENCE, VERIFY_MAX_RETRIES

    ASSISTANT_NAME = new_values.get("assistant_name", ASSISTANT_NAME)
    ASSISTANT_NAME_LOWER = ASSISTANT_NAME.lower()
    ASSISTANT_NAME_DISPLAY = _display_name(ASSISTANT_NAME)
    LISTEN_TO_EVERYONE = new_values.get("listen_to_everyone", LISTEN_TO_EVERYONE)
    FOLLOWUP_TIMER = new_values.get("followup_timer", FOLLOWUP_TIMER)
    FOLLOW_UP_LISTEN_SECONDS = FOLLOWUP_TIMER
    WAKE_WORD_SENSITIVITY = new_values.get("wake_word_sensitivity", WAKE_WORD_SENSITIVITY)
    WAKE_WORD_THRESHOLD = WAKE_WORD_SENSITIVITY
    WAKE_WORD_COOLDOWN = new_values.get("wake_word_cooldown", WAKE_WORD_COOLDOWN)
    WAKE_WORD_ENABLED = new_values.get("wake_word_enabled", WAKE_WORD_ENABLED)
    TTS_SPEED = new_values.get("tts_speed", TTS_SPEED)
    VOCAL_VOICE_ENABLED = new_values.get("vocal_voice_enabled", VOCAL_VOICE_ENABLED)
    VOCAL_CASUAL_LANGUAGE = new_values.get("vocal_casual_language", VOCAL_CASUAL_LANGUAGE)
    SPEAKER_VERIFY_ENABLED = new_values.get("speaker_verify_enabled", SPEAKER_VERIFY_ENABLED)
    SPEAKER_VERIFY_THRESHOLD = new_values.get("speaker_verify_threshold", SPEAKER_VERIFY_THRESHOLD)
    CAMERA_ENABLED = new_values.get("camera_enabled", CAMERA_ENABLED)
    FACE_RECOGNITION_TOLERANCE = new_values.get("face_recognition_tolerance", FACE_RECOGNITION_TOLERANCE)
    PROACTIVE_ENABLED = new_values.get("proactive_enabled", PROACTIVE_ENABLED)
    PROACTIVE_MODE = new_values.get("proactive_mode", PROACTIVE_MODE)
    PROACTIVE_INTERVAL_MINUTES = new_values.get("proactive_interval_minutes", PROACTIVE_INTERVAL_MINUTES)
    PROACTIVE_IDLE_THRESHOLD_MINUTES = new_values.get("proactive_idle_threshold_minutes", PROACTIVE_IDLE_THRESHOLD_MINUTES)
    MESSAGING_NOTIFY_DEBOUNCE = new_values.get("messaging_notify_debounce", MESSAGING_NOTIFY_DEBOUNCE)
    MESSAGING_SUPPRESS_WINDOW = new_values.get("messaging_suppress_window", MESSAGING_SUPPRESS_WINDOW)
    INCOMING_READ_THRESHOLD = new_values.get("incoming_read_threshold", INCOMING_READ_THRESHOLD)
    UNITY_ENABLED = new_values.get("unity_enabled", UNITY_ENABLED)
    PUSH_TO_TALK_KEY = new_values.get("push_to_talk_key", PUSH_TO_TALK_KEY)
    VERIFY_ENABLED = new_values.get("verify_enabled", VERIFY_ENABLED)
    VERIFY_BROWSER_STEPS = new_values.get("verify_browser_steps", VERIFY_BROWSER_STEPS)
    VERIFY_APP_STEPS = new_values.get("verify_app_steps", VERIFY_APP_STEPS)
    VERIFY_VISION_FALLBACK = new_values.get("verify_vision_fallback", VERIFY_VISION_FALLBACK)
    VERIFY_STRICT_TEXT_MATCH = new_values.get("verify_strict_text_match", VERIFY_STRICT_TEXT_MATCH)
    VERIFY_MIN_CONFIDENCE = new_values.get("verify_min_confidence", VERIFY_MIN_CONFIDENCE)
    VERIFY_MAX_RETRIES = new_values.get("verify_max_retries", VERIFY_MAX_RETRIES)