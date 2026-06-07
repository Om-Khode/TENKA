"""
reminders.py — Timed reminder system for TENKA

Parses natural language reminder requests, stores them in SQLite,
and fires them at the right time via the proactive queue.

Architecture:
  - Reminders stored in SQLite table `reminders` (survives restarts)
  - Background thread polls every 10 seconds — zero LLM, zero I/O cost
  - Fires via proactive._proactive_queue — no new main.py changes needed
  - set_reminder intent parsed by LLM (one-shot, cheap model)
  - cancel_reminder uses LLM synonym expansion for fuzzy matching

Usage:
    from . import reminders
    reminders.start()    # call once after memory.init_memory()
    reminders.stop()     # call on shutdown
"""

import logging
import threading
from datetime import datetime, date, timedelta
from typing import Optional

logger = logging.getLogger("reminders")

_stop_event = threading.Event()
_thread: Optional[threading.Thread] = None


# ─── DB Setup ─────────────────────────────────────────────────────────────────


from .storage.db import get_db as _get_conn


def init_reminders_table() -> None:
    """Create the reminders table if it doesn't exist."""
    conn = _get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS reminders (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at  TEXT NOT NULL,
            remind_at   TEXT NOT NULL,
            message     TEXT NOT NULL,
            fired       INTEGER NOT NULL DEFAULT 0
        )
    """)
    conn.commit()
    logger.info("[REMINDERS] Table ready")


# ─── Public API ───────────────────────────────────────────────────────────────


def add_reminder(remind_at: datetime, message: str) -> int:
    """
    Save a reminder to SQLite.
    Returns the new reminder's row id.
    """
    conn = _get_conn()
    conn.execute(
        "INSERT INTO reminders (created_at, remind_at, message, fired) VALUES (?, ?, ?, 0)",
        (datetime.now().isoformat(), remind_at.isoformat(), message),
    )
    row_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.commit()
    logger.info(f"[REMINDERS] Saved: '{message}' at {remind_at.strftime('%I:%M %p')}")
    return row_id


def list_pending() -> list[dict]:
    """Return all unfired reminders, ordered by soonest first."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM reminders WHERE fired = 0 ORDER BY remind_at ASC"
    ).fetchall()
    return [dict(r) for r in rows]


def list_pending_summary() -> str:
    """Return a short human-readable summary of pending reminders for TTS."""
    pending = list_pending()
    if not pending:
        return "You have no pending reminders."

    if len(pending) == 1:
        r = pending[0]
        t = datetime.fromisoformat(r["remind_at"]).strftime("%I:%M %p").lstrip("0")
        return f"You have 1 reminder: {r['message']} at {t}."

    # Group by message for recurring reminders
    from collections import defaultdict
    groups: dict = defaultdict(list)
    for r in pending:
        groups[r["message"]].append(datetime.fromisoformat(r["remind_at"]))

    lines = [f"You have {len(pending)} pending reminders:"]
    for msg, times in groups.items():
        times.sort()
        if len(times) == 1:
            t = times[0].strftime("%I:%M %p").lstrip("0")
            lines.append(f"  {msg} at {t}")
        else:
            first = times[0].strftime("%I:%M %p").lstrip("0")
            last = times[-1].strftime("%I:%M %p").lstrip("0")
            lines.append(f"  {msg} — {len(times)} times from {first} to {last}")

    return "\n".join(lines)


def _mark_fired(reminder_id: int) -> None:
    conn = _get_conn()
    conn.execute("UPDATE reminders SET fired = 1 WHERE id = ?", (reminder_id,))
    conn.commit()


def cancel_all_pending() -> int:
    """Cancel all unfired reminders. Returns count cancelled."""
    conn = _get_conn()
    count = conn.execute(
        "SELECT COUNT(*) FROM reminders WHERE fired = 0"
    ).fetchone()[0]
    conn.execute("UPDATE reminders SET fired = 1 WHERE fired = 0")
    conn.commit()
    logger.info(f"[REMINDERS] Cancelled all {count} pending reminders")
    return count


def cancel_by_message(keyword: str) -> int:
    """
    Cancel unfired reminders whose message contains keyword.
    Returns count cancelled.
    """
    conn = _get_conn()
    pattern = f"%{keyword}%"
    count = conn.execute(
        "SELECT COUNT(*) FROM reminders WHERE fired = 0 AND message LIKE ?",
        (pattern,)
    ).fetchone()[0]
    conn.execute(
        "UPDATE reminders SET fired = 1 WHERE fired = 0 AND message LIKE ?",
        (pattern,)
    )
    conn.commit()
    logger.info(f"[REMINDERS] Cancelled {count} reminders matching '{keyword}'")
    return count


def cancel_by_time(time_24: str) -> int:
    """
    Cancel unfired reminders scheduled at a specific hour:minute.
    Matches any reminder on any date with that HH:MM.
    Returns count cancelled.
    """
    try:
        hour, minute = map(int, time_24.split(":"))
    except ValueError:
        logger.warning(f"[REMINDERS] Invalid time_24: {time_24}")
        return 0

    conn = _get_conn()
    pending = conn.execute(
        "SELECT id, remind_at FROM reminders WHERE fired = 0"
    ).fetchall()

    cancelled = 0
    for row in pending:
        try:
            remind_at = datetime.fromisoformat(row["remind_at"])
            if remind_at.hour == hour and remind_at.minute == minute:
                conn.execute(
                    "UPDATE reminders SET fired = 1 WHERE id = ?", (row["id"],)
                )
                cancelled += 1
        except ValueError:
            continue

    conn.commit()
    logger.info(f"[REMINDERS] Cancelled {cancelled} reminders at {time_24}")
    return cancelled


def cancel_by_id(reminder_id: int) -> bool:
    """Cancel a specific reminder by ID. Returns True if found and cancelled."""
    conn = _get_conn()
    result = conn.execute(
        "UPDATE reminders SET fired = 1 WHERE id = ? AND fired = 0",
        (reminder_id,)
    )
    conn.commit()
    return result.rowcount > 0


# ─── Background Poller ────────────────────────────────────────────────────────


def start() -> None:
    """Start the reminder polling thread. Call once after memory.init_memory()."""
    global _thread
    init_reminders_table()

    if _thread and _thread.is_alive():
        return

    _stop_event.clear()
    _thread = threading.Thread(target=_poll_loop, name="reminder-poller", daemon=True)
    _thread.start()
    logger.info("[REMINDERS] Poller started (checking every 10s)")


def stop() -> None:
    """Stop the reminder polling thread."""
    _stop_event.set()
    if _thread:
        _thread.join(timeout=5)
    logger.info("[REMINDERS] Poller stopped")


def _poll_loop() -> None:
    """Check for due reminders every 10 seconds. Zero LLM, zero I/O cost."""
    from . import proactive

    while not _stop_event.wait(timeout=10):
        try:
            now = datetime.now()
            pending = list_pending()

            for reminder in pending:
                remind_at = datetime.fromisoformat(reminder["remind_at"])
                if now >= remind_at:
                    msg = f"⏰ Reminder: {reminder['message']}"
                    proactive._proactive_queue.put(msg)
                    _mark_fired(reminder["id"])
                    logger.info(f"[REMINDERS] Fired: '{reminder['message']}'")

        except Exception as e:
            logger.warning(f"[REMINDERS] Poll error: {e}")


# ─── Recurring Pattern Extractor (Python-side, no LLM) ───────────────────────


def _extract_recurring_params(text: str) -> Optional[tuple[int, int]]:
    """
    Try to extract (interval_minutes, total_minutes) from text using regex.
    Returns None if pattern not found — falls back to LLM.

    Handles: mins, minutes, hours, hrs, seconds, secs
    Examples:
      'every 2 mins for 10 mins'    → (2, 10)
      'every 1 hour for 3 hours'    → (60, 180)
      'every 30 seconds for 5 mins' → (1, 5)  [seconds rounded up to 1 min]
    """
    import re

    unit_to_minutes = {
        "sec": 1/60, "secs": 1/60, "second": 1/60, "seconds": 1/60,
        "min": 1, "mins": 1, "minute": 1, "minutes": 1,
        "hr": 60, "hrs": 60, "hour": 60, "hours": 60,
    }

    units_pattern = "|".join(unit_to_minutes.keys())
    pattern = re.compile(
        rf'every\s+(\d+)\s*({units_pattern})'
        rf'.{{0,40}}?for\s+(\d+)\s*({units_pattern})',
        re.IGNORECASE
    )

    m = pattern.search(text)
    if not m:
        return None

    interval_val = int(m.group(1))
    interval_unit = m.group(2).lower()
    total_val = int(m.group(3))
    total_unit = m.group(4).lower()

    interval_mins = max(1, round(interval_val * unit_to_minutes[interval_unit]))
    total_mins = max(1, round(total_val * unit_to_minutes[total_unit]))

    return interval_mins, total_mins


# ─── Natural Language Parser ──────────────────────────────────────────────────


async def parse_and_save(user_input: str) -> str:
    """
    Parse a natural language reminder request, save it, and return
    a spoken confirmation string.

    Strategy:
      1. Try regex for recurring pattern — if matched, only use LLM for message
      2. Otherwise use LLM with 3-field structured JSON (type/value/message)
      3. All datetime arithmetic always done in Python via timedelta
    """
    from .llm.contracts import ask_for_intent
    import json

    now = datetime.now()
    today_str = now.strftime("%Y-%m-%d")
    time_24 = now.strftime("%H:%M")
    day_str = now.strftime("%A")

    # ── Fast path: recurring pattern detected by regex ────────────────────
    recurring_params = _extract_recurring_params(user_input)
    if recurring_params:
        interval_mins, total_mins = recurring_params

        # LLM only extracts the short message — no time math involved
        msg_prompt = (
            f"The user said: \"{user_input}\"\n"
            f"Extract ONLY the short action they want to be reminded about "
            f"(e.g. 'water plants', 'stir soup', 'take medicine'). "
            f"Return ONLY the short phrase. No JSON, no punctuation, no extra text."
        )
        message = await ask_for_intent(
            msg_prompt, max_tokens=20,
            system_prompt="You are a text extraction utility. Return only the requested text, nothing else.",
        )
        message = (message or "reminder").strip().strip('"').strip("'")

        saved_times = []
        t = now + timedelta(minutes=interval_mins)
        end = now + timedelta(minutes=total_mins)

        while t <= end:
            add_reminder(t, message)
            saved_times.append(t.strftime("%I:%M %p").lstrip("0"))
            t += timedelta(minutes=interval_mins)

        if not saved_times:
            return (
                f"That interval ({interval_mins} min) is longer than "
                f"the total duration ({total_mins} min) — nothing to remind."
            )

        # Keep TTS short — summarize instead of listing all times
        if len(saved_times) <= 2:
            times_summary = f"at {' and '.join(saved_times)}"
        else:
            times_summary = f"starting at {saved_times[0]}, last at {saved_times[-1]}"

        return (
            f"Got it! I'll remind you to {message} every "
            f"{interval_mins} minute{'s' if interval_mins != 1 else ''} "
            f"for the next {total_mins} minutes — "
            f"{len(saved_times)} reminder{'s' if len(saved_times) != 1 else ''} "
            f"{times_summary}."
        )

    # ── LLM path: single reminder (offset or absolute) ────────────────────
    prompt = (
        f"Current time: {time_24} (24-hour) on {day_str} {today_str}\n"
        f"User said: \"{user_input}\"\n\n"
        f"Return ONLY this JSON with exactly 3 fields:\n"
        f"{{\n"
        f"  \"message\": \"short action phrase\",\n"
        f"  \"type\": \"offset\" | \"absolute\",\n"
        f"  \"value\": <see rules below>\n"
        f"}}\n\n"
        f"Rules for 'value' based on type:\n"
        f"- type='offset'   → value = number of minutes from now (integer)\n"
        f"                    'in 10 minutes' → 10\n"
        f"                    'after 2 mins' → 2\n"
        f"                    'in an hour' → 60\n"
        f"                    'half an hour' → 30\n"
        f"                    'a quarter hour' → 15\n"
        f"                    'a couple minutes' → 2\n"
        f"                    'a few minutes' → 3\n"
        f"                    'in 90 minutes' → 90\n"
        f"                    'after 2 hours' → 120\n"
        f"                    ANY 'in X' or 'after X' phrasing → always offset\n"
        f"- type='absolute' → value = 'HH:MM' in 24-hour format\n"
        f"                    '11 PM' → '23:00'\n"
        f"                    '9 AM' → '09:00'\n"
        f"                    'midnight' → '00:00'\n"
        f"                    '9:30 PM' → '21:30'\n"
        f"                    'noon' → '12:00'\n\n"
        f"Return ONLY the JSON. No explanation, no markdown, no extra text."
    )

    response = await ask_for_intent(
        prompt, max_tokens=60,
        system_prompt="You are a JSON extraction utility. Return only the requested JSON, nothing else.",
    )

    if not response or response == "__LLM_UNAVAILABLE__":
        return "Sorry, I couldn't parse that reminder. Try: 'remind me at 9 PM to call mum'."

    try:
        cleaned = response.strip().strip("```json").strip("```").strip()
        data = json.loads(cleaned)

        message = (data.get("message") or "reminder").strip().strip('"').strip("'")
        reminder_type = data.get("type", "")
        value = data.get("value")

        if not reminder_type or value is None:
            logger.warning(f"[REMINDERS] LLM returned no usable fields — raw: {response[:100]}")
            return "I couldn't figure out when to remind you. Try: 'remind me in 10 minutes to check the oven'."

        # ── Offset: relative time ─────────────────────────────────────────
        if reminder_type == "offset":
            try:
                offset_mins = int(value)
            except (TypeError, ValueError):
                return "I couldn't understand the time offset. Try: 'remind me in 10 minutes'."
            if offset_mins <= 0:
                return "The offset needs to be greater than zero minutes."
            remind_at = now + timedelta(minutes=offset_mins)

        # ── Absolute: clock time ──────────────────────────────────────────
        elif reminder_type == "absolute":
            try:
                hour, minute = map(int, str(value).split(":"))
                remind_at = datetime(now.year, now.month, now.day, hour, minute)
                # If time already passed today, roll to tomorrow
                if remind_at <= now:
                    remind_at += timedelta(days=1)
            except (ValueError, AttributeError):
                return "I couldn't parse that time. Try: 'remind me at 9 PM to call mum'."

        else:
            logger.warning(f"[REMINDERS] Unknown type '{reminder_type}' — raw: {response[:100]}")
            return "I couldn't figure out when to remind you. Try: 'remind me in 10 minutes to check the oven'."

        # ── Sanity check: reject times more than 2 min in the past ───────
        if (now - remind_at).total_seconds() > 120:
            return (
                f"That time ({remind_at.strftime('%I:%M %p')}) has already passed. "
                f"Could you give me a future time?"
            )

        add_reminder(remind_at, message)

        time_label = remind_at.strftime("%I:%M %p").lstrip("0")
        when_str = (
            f"today at {time_label}"
            if remind_at.date() == date.today()
            else remind_at.strftime(f"%A at {time_label}")
        )
        return f"Got it! I'll remind you to {message} {when_str}."

    except (json.JSONDecodeError, ValueError, KeyError) as e:
        logger.warning(f"[REMINDERS] Parse failed: {e} | response: {response[:100]}")
        return "I had trouble understanding that reminder. Try: 'remind me at 9 PM to turn off the oven'."


# ─── Cancel Handler ───────────────────────────────────────────────────────────


async def cancel_reminders(goal: str) -> str:
    """
    Cancel reminders by scope (all), keyword (specific), or time.
    Uses LLM for scope classification + synonym expansion for fuzzy matching.
    Called from actions.handle_cancel_reminder.
    """
    from .llm.contracts import ask_for_intent
    import json

    # ── Step 1: Classify scope ────────────────────────────────────────────
    classify_prompt = (
        f"The user said: \"{goal}\"\n\n"
        f"Return ONLY this JSON:\n"
        f"{{\"scope\": \"all\" | \"specific\" | \"time\", "
        f"\"keyword\": \"keyword or null\", "
        f"\"time_24\": \"HH:MM or null\"}}\n\n"
        f"Rules:\n"
        f"- scope='all'      → user says 'all reminders', 'everything', no specific task\n"
        f"- scope='specific' → user names a specific task (e.g. 'pills', 'water plants')\n"
        f"- scope='time'     → user refers to a reminder by its time, not its task\n"
        f"                     e.g. 'cancel my 11 PM reminder', 'remove the midnight one'\n"
        f"- When in doubt between specific and time, check if a clock time is mentioned\n"
        f"- keyword: core task noun if scope=specific, else null\n"
        f"- time_24: 24-hour HH:MM if scope=time, else null\n"
        f"           '11 PM' → '23:00', '9 AM' → '09:00', 'midnight' → '00:00'\n\n"
        f"Examples:\n"
        f"  'cancel all reminders'             → {{\"scope\": \"all\", \"keyword\": null, \"time_24\": null}}\n"
        f"  'stop all my reminders'            → {{\"scope\": \"all\", \"keyword\": null, \"time_24\": null}}\n"
        f"  'cancel the pills reminder'        → {{\"scope\": \"specific\", \"keyword\": \"pills\", \"time_24\": null}}\n"
        f"  'cancel my water plants reminders' → {{\"scope\": \"specific\", \"keyword\": \"water plants\", \"time_24\": null}}\n"
        f"  'remove the medicine reminder'     → {{\"scope\": \"specific\", \"keyword\": \"medicine\", \"time_24\": null}}\n"
        f"  'cancel my 11 PM reminder'         → {{\"scope\": \"time\", \"keyword\": null, \"time_24\": \"23:00\"}}\n"
        f"  'remove the midnight reminder'     → {{\"scope\": \"time\", \"keyword\": null, \"time_24\": \"00:00\"}}\n"
        f"  'cancel the 9 AM one'              → {{\"scope\": \"time\", \"keyword\": null, \"time_24\": \"09:00\"}}\n"
        f"Return ONLY the JSON."
    )

    try:
        classify_response = await ask_for_intent(
            classify_prompt, max_tokens=60
        )
        classify_data = json.loads(
            classify_response.strip().strip("```json").strip("```").strip()
        )
        scope = classify_data.get("scope", "all")
        keyword = classify_data.get("keyword")
        time_24 = classify_data.get("time_24")
    except Exception as e:
        logger.warning(f"[REMINDERS] Classify failed: {e} — defaulting to cancel all")
        scope = "all"
        keyword = None
        time_24 = None

    # ── Step 2: Cancel all ────────────────────────────────────────────────
    if scope == "all" or (not keyword and not time_24):
        count = cancel_all_pending()
        if count == 0:
            return "You don't have any pending reminders to cancel."
        return f"Cancelled all {count} pending reminder{'s' if count != 1 else ''}."

    # ── Step 3: Cancel by time ────────────────────────────────────────────
    if scope == "time" and time_24:
        count = cancel_by_time(time_24)
        if count == 0:
            pending = list_pending()
            if not pending:
                return "You don't have any pending reminders."
            names = list(dict.fromkeys(r["message"] for r in pending))[:3]
            names_str = ", ".join(names)
            more = f" and {len(pending) - 3} more" if len(pending) > 3 else ""
            return (
                f"I couldn't find any reminder at that time. "
                f"Your pending reminders are: {names_str}{more}."
            )
        # Convert back to 12h for spoken response
        h, m = map(int, time_24.split(":"))
        spoken_time = datetime(2000, 1, 1, h, m).strftime("%I:%M %p").lstrip("0")
        return f"Cancelled {count} reminder{'s' if count != 1 else ''} at {spoken_time}."

    # ── Step 4: Cancel by keyword with synonym expansion ──────────────────
    if scope == "specific" and keyword:
        expand_prompt = (
            f"The user wants to cancel a reminder related to: \"{keyword}\"\n"
            f"List up to 4 short synonym or alias words the reminder message might use.\n"
            f"Return ONLY a JSON array of strings.\n"
            f"Examples:\n"
            f"  'medicine' → [\"medicine\", \"pills\", \"medication\", \"tablet\"]\n"
            f"  'water plants' → [\"water plants\", \"water\", \"plants\", \"watering\"]\n"
            f"  'sleep' → [\"sleep\", \"bed\", \"bedtime\", \"rest\"]\n"
            f"No explanation, no markdown."
        )

        try:
            expand_response = await ask_for_intent(
                expand_prompt, max_tokens=50
            )
            synonyms = json.loads(
                expand_response.strip().strip("```json").strip("```").strip()
            )
            if not isinstance(synonyms, list):
                synonyms = [keyword]
            # Always include original keyword
            if keyword not in synonyms:
                synonyms.insert(0, keyword)
        except Exception as e:
            logger.warning(f"[REMINDERS] Synonym expansion failed: {e}")
            synonyms = [keyword]

        logger.info(f"[REMINDERS] Cancelling by synonyms: {synonyms}")

        # Try each synonym until something matches
        total_cancelled = 0
        matched_keyword = None
        for syn in synonyms:
            n = cancel_by_message(syn)
            if n > 0:
                total_cancelled += n
                matched_keyword = matched_keyword or syn

        if total_cancelled == 0:
            pending = list_pending()
            if not pending:
                return (
                    f"I don't have any reminder matching '{keyword}', "
                    f"and there are no other pending reminders."
                )
            names = list(dict.fromkeys(r["message"] for r in pending))[:3]
            names_str = ", ".join(names)
            more = f" and {len(pending) - 3} more" if len(pending) > 3 else ""
            return (
                f"I couldn't find any reminders matching '{keyword}'. "
                f"Your pending reminders are: {names_str}{more}."
            )

        return (
            f"Cancelled {total_cancelled} "
            f"reminder{'s' if total_cancelled != 1 else ''} "
            f"for '{matched_keyword}'."
        )

    # ── Fallback ──────────────────────────────────────────────────────────
    return "I wasn't sure which reminders to cancel. Try: 'cancel all reminders' or 'cancel my medicine reminder'."