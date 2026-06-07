"""
proactive.py — Proactive Nudge Analyzer for TENKA (Phase 3F)

Runs a background analyzer thread that periodically scans conversation
history and facts to surface timely nudges:
  - deadline   : upcoming or past deadlines/events mentioned in conversation
  - pattern    : recurring behavioral patterns (e.g. "you ask about weather every Monday")
  - preference : repeated tool/app usage preferences learned over time
  - idle       : gentle check-in after a long silence

Architecture:
  - Background thread posts nudge strings to _proactive_queue
  - main.py drains _proactive_queue in its existing 0.1s loop
  - Delivery respects PROACTIVE_MODE: "always" speaks immediately,
    "idle_only" speaks only when _is_processing is False
  - Nudges are deduplicated within a session via _seen_nudge_hashes

Personality reflection engine lives in reflection.py (split from here).

Usage:
    from . import proactive
    proactive.start_analyzer()   # call once after memory.init_memory()
    proactive.stop_analyzer()    # call on shutdown
"""

import hashlib
import json
import logging
import queue
import threading
import time
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger("proactive")

# ─── Public queue — main.py drains this ──────────────────────────────────────

_proactive_queue: queue.Queue = queue.Queue()

# ─── Internal state ───────────────────────────────────────────────────────────

_analyzer_thread: Optional[threading.Thread] = None
_stop_event = threading.Event()
_seen_nudge_hashes: set[str] = set()   # dedup within session
_last_nudge_time: float = 0.0          # throttle back-to-back nudges

# How many recent turns / days of facts to analyze (keeps LLM prompt bounded)
_MAX_TURNS = 50
_MAX_FACTS_DAYS = 30


# ─── Public API ───────────────────────────────────────────────────────────────


def start_analyzer() -> None:
    """
    Start the background proactive analyzer thread.
    Safe to call multiple times — only one thread runs at a time.
    Call once after memory.init_memory().

    Also starts the personality reflection engine (reflection.py),
    which runs independently of PROACTIVE_ENABLED.
    """
    global _analyzer_thread
    from . import config

    # Always start reflection engine, even if proactive nudges are off
    from . import reflection
    reflection.start()

    if not getattr(config, "PROACTIVE_ENABLED", True):
        logger.info("[PROACTIVE] Disabled via config — skipping nudge analyzer")
        return

    if _analyzer_thread and _analyzer_thread.is_alive():
        logger.debug("[PROACTIVE] Analyzer already running")
        return

    _stop_event.clear()
    _analyzer_thread = threading.Thread(
        target=_analyzer_loop,
        name="proactive-analyzer",
        daemon=True,
    )
    _analyzer_thread.start()
    logger.info("[PROACTIVE] Analyzer started")


def stop_analyzer() -> None:
    """Signal the background thread to stop. Call on assistant shutdown."""
    _stop_event.set()
    if _analyzer_thread:
        _analyzer_thread.join(timeout=5)
    logger.info("[PROACTIVE] Analyzer stopped")

    # Stop reflection engine
    from . import reflection
    reflection.stop()


def get_queue() -> queue.Queue:
    """Return the proactive nudge queue (for main.py to drain)."""
    return _proactive_queue


# ─── Background Loop ──────────────────────────────────────────────────────────


def _analyzer_loop() -> None:
    """
    Main loop: runs at startup, then every PROACTIVE_INTERVAL_MINUTES.
    Posts at most one nudge per run to _proactive_queue.
    """
    from . import config

    interval_seconds = getattr(config, "PROACTIVE_INTERVAL_MINUTES", 30) * 60

    # Run immediately at startup (slight delay to let assistant finish init)
    time.sleep(8.0)
    _run_analysis()

    while not _stop_event.wait(timeout=interval_seconds):
        _run_analysis()


def _run_analysis() -> None:
    """
    One analysis pass:
      1. Load recent history from SQLite
      2. Check for idle nudge (no LLM call needed)
      3. Ask LLM to find one meaningful nudge from history
      4. Post to _proactive_queue if something noteworthy found
    """
    global _last_nudge_time
    from . import config, memory

    # Throttle: don't nudge more than once every 5 minutes regardless of interval
    if time.time() - _last_nudge_time < 300:
        logger.debug("[PROACTIVE] Throttled — nudged too recently")
        return

    try:
        turns = _load_recent_turns(memory)
        facts = _load_recent_facts(memory)

        # ── Idle nudge (no LLM needed) ────────────────────────────────────
        idle_threshold = getattr(config, "PROACTIVE_IDLE_THRESHOLD_MINUTES", 10) * 60
        idle_nudge = _check_idle(turns, idle_threshold)
        if idle_nudge:
            _post_nudge(idle_nudge)
            return  # one nudge per run

        # ── Skip LLM analysis if history is too thin ──────────────────────
        if len(turns) < 3:
            logger.debug("[PROACTIVE] Not enough history for pattern analysis yet")
            return

        # ── LLM-based analysis ────────────────────────────────────────────
        nudge = _analyze_with_llm(turns, facts)
        if nudge:
            _post_nudge(nudge)

    except Exception as e:
        logger.warning(f"[PROACTIVE] Analysis run failed: {e}")


# ─── Data Loading ─────────────────────────────────────────────────────────────


def _load_recent_turns(memory_module) -> list[dict]:
    """Load the last _MAX_TURNS conversation turns from SQLite."""
    try:
        from .storage.db import get_db
        db = get_db()
        rows = db.execute(
            "SELECT timestamp, user_input, intent, response "
            "FROM conversations ORDER BY id DESC LIMIT ?",
            (_MAX_TURNS,),
        ).fetchall()
        # Return chronological order (oldest first)
        return [dict(r) for r in reversed(rows)]
    except Exception as e:
        logger.warning(f"[PROACTIVE] Failed to load turns: {e}")
        return []


def _load_recent_facts(memory_module) -> list[dict]:
    """Load facts from the last _MAX_FACTS_DAYS days."""
    try:
        cutoff = (datetime.now() - timedelta(days=_MAX_FACTS_DAYS)).isoformat()
        from .storage.db import get_db
        db = get_db()
        rows = db.execute(
            "SELECT key, value, timestamp FROM facts "
            "WHERE timestamp >= ? ORDER BY id DESC",
            (cutoff,),
        ).fetchall()
        return [dict(r) for r in rows]
    except Exception as e:
        logger.warning(f"[PROACTIVE] Failed to load facts: {e}")
        return []


# ─── Idle Check (no LLM) ──────────────────────────────────────────────────────


def _check_idle(turns: list[dict], threshold_seconds: float) -> Optional[str]:
    """
    Return an idle nudge if the last conversation turn was more than
    threshold_seconds ago. Returns None if not idle or nudge already seen.
    """
    if not turns:
        return None

    last_turn = turns[-1]
    try:
        last_time = datetime.fromisoformat(last_turn["timestamp"])
    except (ValueError, KeyError):
        return None

    elapsed = (datetime.now() - last_time).total_seconds()
    if elapsed < threshold_seconds:
        return None

    minutes = int(elapsed // 60)
    nudge = f"Hey, it's been about {minutes} minutes — just checking in! Is there anything I can help you with?"
    return nudge if not _is_seen(nudge) else None


# ─── LLM Analysis ─────────────────────────────────────────────────────────────


def _analyze_with_llm(turns: list[dict], facts: list[dict]) -> Optional[str]:
    """
    Ask the LLM to find one meaningful proactive nudge from history.
    Uses llama-3.1-8b-instant (cheapest/fastest). Returns spoken nudge or None.
    """
    from .core.asyncio_utils import call_async
    from .llm.contracts import ask_for_intent

    from . import config

    history_text = _format_turns_for_prompt(turns)
    facts_text = _format_facts_for_prompt(facts)
    now = datetime.now()
    day_name = now.strftime("%A")
    time_str = now.strftime("%I:%M %p")
    date_str = now.strftime("%B %d, %Y")
    name = config.ASSISTANT_NAME_DISPLAY

    prompt = f"""You are analyzing a voice assistant's conversation history to find ONE proactive, helpful nudge for the user.

Current date/time: {day_name}, {date_str} at {time_str}

CONVERSATION HISTORY (last {len(turns)} turns):
{history_text}

KNOWN FACTS:
{facts_text if facts_text else "(none yet)"}

Look for exactly ONE of these nudge types (in priority order):
1. DEADLINE — Did the user mention a deadline, meeting, submission, appointment, or time-sensitive task? If it's approaching or past, remind them.
2. PATTERN — Is there a recurring behavior (same request on same day/time)? Only flag if you see it at least twice.
3. PREFERENCE — Has the user repeatedly used a specific app, tool, or workflow? Suggest it proactively if contextually relevant now.
4. IDLE — (skip — handled separately)

Rules:
- If nothing noteworthy is found, return exactly: null
- Only return a nudge if you are genuinely confident it's useful RIGHT NOW
- The nudge must be 1-2 spoken sentences, friendly and natural, as if {name} is saying it
- Do NOT make up deadlines or patterns that aren't clearly in the history
- Do NOT nudge about things from more than 7 days ago unless it's a recurring pattern

Respond with ONLY a JSON object:
{{"type": "deadline"|"pattern"|"preference", "nudge": "spoken text here"}}
Or if nothing useful: null
Do NOT include any other text."""

    try:
        # Run the async LLM call from this sync thread via main event loop
        response = call_async(ask_for_intent(prompt, max_tokens=120))

        if not response or response == "__LLM_UNAVAILABLE__":
            return None

        # Strip markdown fences if present
        cleaned = response.strip().strip("```json").strip("```").strip()

        if cleaned.lower() == "null" or cleaned == "":
            logger.debug("[PROACTIVE] LLM found nothing noteworthy")
            return None

        data = json.loads(cleaned)
        nudge_text = data.get("nudge", "").strip()
        nudge_type = data.get("type", "unknown")

        if not nudge_text:
            return None

        logger.info(f"[PROACTIVE] LLM nudge ({nudge_type}): {nudge_text[:80]}")
        return nudge_text if not _is_seen(nudge_text) else None

    except json.JSONDecodeError:
        logger.debug(f"[PROACTIVE] LLM returned non-JSON: {response[:100]}")
        return None
    except Exception as e:
        logger.warning(f"[PROACTIVE] LLM analysis failed: {e}")
        return None


# ─── Prompt Formatters ────────────────────────────────────────────────────────


def _format_turns_for_prompt(turns: list[dict]) -> str:
    """Format turns into a compact prompt-friendly string."""
    if not turns:
        return "(no history)"
    lines = []
    for t in turns:
        ts = t.get("timestamp", "")[:16]  # trim to minute precision
        intent = t.get("intent", "")
        user = t.get("user_input", "")
        lines.append(f"[{ts}] ({intent}) User: {user}")
    return "\n".join(lines)


def _format_facts_for_prompt(facts: list[dict]) -> str:
    """Format facts into a compact prompt-friendly string."""
    if not facts:
        return ""
    seen = set()
    lines = []
    for f in facts:
        key = f.get("key", "")
        val = f.get("value", "")
        entry = f"{key}: {val}"
        if entry not in seen:
            seen.add(entry)
            lines.append(entry)
    return "\n".join(lines[:20])  # cap at 20 unique facts


# ─── Nudge Dedup & Posting ────────────────────────────────────────────────────


def _is_seen(nudge: str) -> bool:
    """Return True if this nudge was already delivered this session."""
    h = hashlib.md5(nudge.encode()).hexdigest()
    if h in _seen_nudge_hashes:
        logger.debug("[PROACTIVE] Nudge already delivered this session — skipping")
        return True
    _seen_nudge_hashes.add(h)
    return False


def _post_nudge(nudge: str) -> None:
    """Post a nudge to the delivery queue and record the time."""
    global _last_nudge_time
    _proactive_queue.put(nudge)
    _last_nudge_time = time.time()
    logger.info(f"[PROACTIVE] Queued nudge: {nudge[:80]}")
