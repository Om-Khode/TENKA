"""
reflection.py — Personality Reflection Engine

Runs on its own background thread, independent of PROACTIVE_ENABLED.
Checks periodically whether a personality reflection cycle is due, based on
conversation count or elapsed time since last reflection.

When triggered, it:
  - Gathers recent interaction summaries from memory
  - Sends them to the LLM with current trait state + known preferences
  - Parses trait deltas and discovered preferences from the response
  - Applies trait changes via personality.update_traits()
  - Processes discovered preferences via preferences
  - Runs preference decay if overdue

Usage (called from proactive.py):
    from . import reflection
    reflection.start()    # call once after memory.init_memory()
    reflection.stop()     # call on shutdown
"""

import json
import logging
import threading
import time
from datetime import datetime
from typing import Optional

# Keep logger name for log continuity
logger = logging.getLogger("proactive")

# ─── Reflection constants ────────────────────────────────────────────────────

# Reflection cycle triggers — whichever comes first
REFLECTION_INTERVAL_CONVERSATIONS = 20   # every N conversations
REFLECTION_INTERVAL_HOURS = 24           # OR every N hours
_REFLECTION_CHECK_SECONDS = 30 * 60      # how often to CHECK if reflection is due (30 min)
_REFLECTION_MEMORY_LIMIT = 20            # how many recent turns to pull for context
_PREFERENCE_DECAY_INTERVAL_DAYS = 30     # how often to run preference decay

# ─── Thread state ─────────────────────────────────────────────────────────────

_reflection_thread: Optional[threading.Thread] = None
_reflection_stop_event = threading.Event()


# ─── Public API ──────────────────────────────────────────────────────────────


def start() -> None:
    """Start the personality reflection background thread."""
    global _reflection_thread

    if _reflection_thread and _reflection_thread.is_alive():
        logger.debug("[REFLECTION] Already running")
        return

    _reflection_stop_event.clear()
    _reflection_thread = threading.Thread(
        target=_reflection_loop,
        name="personality-reflection",
        daemon=True,
    )
    _reflection_thread.start()
    logger.info("[REFLECTION] Personality reflection engine started")


def stop() -> None:
    """Stop the personality reflection background thread."""
    _reflection_stop_event.set()
    if _reflection_thread:
        _reflection_thread.join(timeout=5)
    logger.info("[REFLECTION] Personality reflection engine stopped")


# ─── Background Loop ────────────────────────────────────────────────────────


def _reflection_loop() -> None:
    """
    Background loop: checks every _REFLECTION_CHECK_SECONDS whether a
    personality reflection cycle should run.
    """
    # Delay startup to let everything initialize
    time.sleep(15.0)

    while not _reflection_stop_event.is_set():
        try:
            _maybe_run_reflection()
        except Exception as e:
            logger.warning(f"[REFLECTION] Error in reflection check: {e}")

        # Sleep in small increments so we can respond to stop_event quickly
        for _ in range(_REFLECTION_CHECK_SECONDS):
            if _reflection_stop_event.is_set():
                return
            time.sleep(1)


def _maybe_run_reflection() -> None:
    """
    Check if it's time for a personality reflection cycle.
    Triggers when EITHER condition is met:
      - conversation_count >= REFLECTION_INTERVAL_CONVERSATIONS
      - hours since last reflection >= REFLECTION_INTERVAL_HOURS
    """
    from . import personality

    # Check conversation count trigger
    conv_count = personality.get_conversation_count()
    count_ready = conv_count >= REFLECTION_INTERVAL_CONVERSATIONS

    # Check time trigger
    last_reflection = personality.get_metadata("last_reflection_at")
    time_ready = False
    if last_reflection:
        try:
            last_dt = datetime.fromisoformat(last_reflection)
            hours_elapsed = (datetime.now() - last_dt).total_seconds() / 3600
            time_ready = hours_elapsed >= REFLECTION_INTERVAL_HOURS
        except (ValueError, TypeError):
            time_ready = True  # corrupted timestamp → trigger reflection
    else:
        time_ready = True  # no record → first run

    if not count_ready and not time_ready:
        logger.debug(
            f"[REFLECTION] Not yet — {conv_count}/{REFLECTION_INTERVAL_CONVERSATIONS} convos, "
            f"time_ready={time_ready}"
        )
        return

    # Check for absence before reflection
    try:
        from . import personality
        personality.check_absence()
    except Exception as e:
        logger.debug(f"[REFLECTION] Absence check failed: {e}")

    logger.info(
        f"[REFLECTION] Triggering reflection cycle "
        f"(convos={conv_count}, time_ready={time_ready})"
    )

    _run_reflection_cycle()


def _run_reflection_cycle() -> None:
    """
    Execute one personality reflection cycle:
      1. Pull recent conversation turns
      2. Load current trait state + current preferences
      3. Send to 70b with reflection prompt
      4. Parse trait deltas + discovered preferences
      5. Apply trait changes via personality.update_traits()
      6. Process discovered preferences via preferences
      7. Run preference decay if due
      8. Reset conversation counter and update last_reflection_at
    """
    from . import memory, personality
    from .core.asyncio_utils import call_async
    from .llm.contracts import ask_for_personality_reflection

    # Import preference store
    try:
        from . import preferences
        has_pref_store = True
    except Exception:
        has_pref_store = False

    # 1. Gather recent interaction summaries
    memory_summaries = _gather_reflection_context(memory)
    if not memory_summaries:
        logger.info("[REFLECTION] No recent memories to reflect on — skipping")
        # Still reset counters so we don't keep trying with no data
        personality.reset_conversation_count()
        personality.set_metadata("last_reflection_at", datetime.now().isoformat())
        return

    # 2. Load current traits
    traits = personality.get_current_traits()
    if not traits:
        logger.warning("[REFLECTION] No traits found — skipping")
        return

    traits_json = json.dumps(traits, indent=2)

    # Load current preferences for context
    preferences_json = "[]"
    if has_pref_store:
        try:
            all_prefs = preferences.get_all_preferences()
            if all_prefs:
                # Compact format: just key, value, confidence
                pref_summary = [
                    {"key": p["key"], "value": p["value"], "confidence": p["confidence"]}
                    for p in all_prefs
                ]
                preferences_json = json.dumps(pref_summary, indent=2)
        except Exception as e:
            logger.debug(f"[REFLECTION] Failed to load preferences for context: {e}")

    # 3. Build reflection prompt (now includes preferences)
    prompt = _build_reflection_prompt(traits_json, memory_summaries, preferences_json)

    # 4. Call LLM via main event loop
    try:
        response = call_async(
            ask_for_personality_reflection(
                prompt,
                system_prompt=(
                    "You are a personality analysis system. "
                    "Respond ONLY with valid JSON. No markdown, no explanation."
                ),
                max_tokens=500,
                temperature=0.3,
            )
        )
    except Exception as e:
        logger.warning(f"[REFLECTION] LLM call failed: {e}")
        return

    if not response or response == "__LLM_UNAVAILABLE__":
        logger.warning("[REFLECTION] LLM unavailable for reflection")
        return

    # 5. Parse response (now returns preferences too)
    deltas, reasoning, discovered_preferences = _parse_reflection_response(response)

    # 6. Apply trait deltas
    if deltas:
        changed = personality.update_traits(
            deltas, reasoning, trigger="reflection_cycle"
        )
        if changed:
            logger.info(f"[REFLECTION] Applied trait changes: {changed}")
        else:
            logger.info("[REFLECTION] LLM suggested no meaningful changes")
    else:
        logger.info("[REFLECTION] No valid deltas parsed from reflection")

    # 7. Process discovered preferences
    if has_pref_store and discovered_preferences:
        _process_discovered_preferences(discovered_preferences)

    # 8. Run preference decay if due
    if has_pref_store:
        _maybe_run_preference_decay()

    # 9. Reset counters regardless of whether traits changed
    personality.reset_conversation_count()
    personality.set_metadata("last_reflection_at", datetime.now().isoformat())
    logger.info("[REFLECTION] Cycle complete — counters reset")


# ─── Preference Processing ──────────────────────────────────────────────────


def _process_discovered_preferences(discovered_prefs: list[dict]) -> None:
    """
    Process preferences discovered by the reflection cycle.

    For each discovered preference:
      - If new: create with the discovered confidence
      - If existing with same value: bump confidence by +0.15
      - If existing with different value: overwrite (evidence contradicts)
    """
    from . import preferences

    for pref in discovered_prefs:
        key = pref["key"]
        value = pref["value"]
        category = pref["category"]
        confidence = pref["confidence"]
        evidence = pref.get("evidence", "Discovered by reflection")

        try:
            existing = preferences.get_preference(key)

            if existing:
                if existing["value"] == value:
                    # Same value observed again — bump confidence
                    new_conf = preferences.bump_confidence(key, delta=0.15)
                    logger.info(
                        f"[REFLECTION] Preference '{key}' reconfirmed "
                        f"(confidence → {new_conf})"
                    )
                else:
                    # Contradicting existing preference — update with new evidence
                    preferences.set_preference(
                        key=key,
                        value=value,
                        category=category,
                        confidence=confidence,
                        source="reflection",
                        reason=evidence,
                    )
                    logger.info(
                        f"[REFLECTION] Preference '{key}' changed: "
                        f"'{existing['value']}' → '{value}'"
                    )
            else:
                # New preference discovered
                preferences.set_preference(
                    key=key,
                    value=value,
                    category=category,
                    confidence=confidence,
                    source="reflection",
                    reason=evidence,
                )
                logger.info(
                    f"[REFLECTION] New preference discovered: "
                    f"{key}={value} (confidence={confidence})"
                )

        except Exception as e:
            logger.warning(f"[REFLECTION] Failed to process preference '{key}': {e}")


def _maybe_run_preference_decay() -> None:
    """
    Run preference decay if it hasn't been run in _PREFERENCE_DECAY_INTERVAL_DAYS.
    Uses the personality metadata table to track last decay run.
    """
    from . import personality, preferences

    try:
        last_decay = personality.get_metadata("last_preference_decay_at")

        if last_decay:
            try:
                last_dt = datetime.fromisoformat(last_decay)
                days_elapsed = (datetime.now() - last_dt).days
                if days_elapsed < _PREFERENCE_DECAY_INTERVAL_DAYS:
                    return  # not due yet
            except (ValueError, TypeError):
                pass  # corrupted timestamp → run decay

        # Run decay
        decayed = preferences.decay_unused_preferences()
        personality.set_metadata(
            "last_preference_decay_at", datetime.now().isoformat()
        )

        if decayed:
            logger.info(f"[REFLECTION] Preference decay complete: {decayed} prefs decayed")

    except Exception as e:
        logger.debug(f"[REFLECTION] Preference decay check failed: {e}")


# ─── Context Gathering ──────────────────────────────────────────────────────


def _gather_reflection_context(memory_module) -> str:
    """
    Gather recent interaction summaries for the reflection prompt.
    Only includes turns from the current personality session to prevent
    cross-personality contamination.
    """
    try:
        recent = memory_module.get_recent(_REFLECTION_MEMORY_LIMIT)
        if not recent:
            return ""

        from assistant.storage.db import get_db
        from assistant import config as _cfg
        _assistant_name = _cfg.ASSISTANT_NAME_DISPLAY

        since = None
        db = get_db()
        if db is not None:
            row = db.fetchone(
                "SELECT updated_at FROM metadata WHERE key = 'active_personality'"
            )
            if row:
                since = row["updated_at"]

        lines = []
        for turn in recent:
            ts = turn.get("timestamp", "")
            if since and ts < since:
                continue
            intent = turn.get("intent", "")
            user = turn.get("user_input", "")
            response = turn.get("response", "")[:100]
            lines.append(f"[{ts[:16]}] ({intent}) User: {user} → {_assistant_name}: {response}")

        if not lines:
            return ""

        return "\n".join(lines)

    except Exception as e:
        logger.warning(f"[REFLECTION] Failed to gather context: {e}")
        return ""


# ─── Prompt Building ────────────────────────────────────────────────────────


def _build_reflection_prompt(traits_json: str, memory_summaries: str,
                              preferences_json: str = "[]") -> str:
    """Build the reflection prompt using active personality's reflection hints."""
    from assistant.personalities import get_active_loader
    from assistant import config

    loader = get_active_loader()
    hints = loader.get_reflection_hints()
    identity = hints.get("identity", "A desktop assistant.")
    drift_check = hints.get("drift_check", "")
    anchor = hints.get("character_anchor", "")

    return f"""You are analyzing interaction patterns for a personality evolution system AND a user preference learning system.
{identity}
Based on the recent interactions, suggest how personality traits should shift AND identify any user preferences.

CHARACTER ANCHOR: {anchor}
DRIFT CHECK: {drift_check}

CURRENT TRAIT STATE (each 0.0 to 1.0):
{traits_json}

TRAIT MEANINGS:
- trust: How much the assistant trusts the user. Low=guarded. High=open and sharing.
- warmth: Affection level. Low=reserved. High=caring and attentive.
- sass: Wit/sarcasm intensity. Low=straightforward. High=playful pushback.
- openness: Emotional vulnerability. Low=factual. High=shares perspective freely.
- patience: Tolerance for mistakes. Low=curt. High=gently redirects.
- playfulness: Humor frequency. Low=serious helper. High=witty and fun.

CURRENT KNOWN PREFERENCES:
{preferences_json}

RECENT INTERACTIONS (last {_REFLECTION_MEMORY_LIMIT} turns):
{memory_summaries}

Output ONLY a JSON object with this exact format:
{{
  "deltas": {{
    "trust": 0.00,
    "warmth": 0.00,
    "sass": 0.00,
    "openness": 0.00,
    "patience": 0.00,
    "playfulness": 0.00
  }},
  "reasoning": "Brief explanation of why these changes make sense.",
  "preferences": [
    {{
      "key": "example_key",
      "value": "example_value",
      "category": "app_routing",
      "confidence": 0.4,
      "evidence": "Why this preference was detected."
    }}
  ]
}}

Trait rules:
1. Each delta must be between -0.05 and 0.05
2. Most deltas should be 0.00 — only change what the evidence supports
3. Changes should be TINY — personality evolves over weeks, not minutes
4. Trait relationships matter — if trust rises, openness can naturally follow
5. Never suggest changes that contradict the interaction evidence
6. If interactions were routine/neutral with no emotional signals, ALL deltas should be 0.00
7. Look for: personal sharing (trust+), banter (sass+, playfulness+), frustration (patience+), greetings/check-ins (warmth+), emotional questions (openness+), long absence (warmth-)

Preference discovery rules:
1. Only suggest preferences with CLEAR evidence from interactions
2. If a pattern appeared only once, do NOT create a preference
3. Minimum 3 occurrences of a pattern to suggest with confidence 0.4
4. If the user explicitly stated a preference ("always use X"), confidence = 0.8
5. Categories: app_routing, contact_routing, response_style, task_defaults, schedule, environment
6. Do NOT re-suggest preferences that already exist with confidence >= 0.7
7. If evidence contradicts an existing preference, suggest it with the NEW value
8. "preferences" array should be empty ([]) if no new patterns are found
9. Examples of discoverable preferences:
   - User always uses the same music app → key=music_app, value=<app_name>, category=app_routing
   - User keeps responses short → key=verbosity, value=brief, category=response_style"""


# ─── Response Parsing ────────────────────────────────────────────────────────


def _parse_reflection_response(response: str) -> tuple[dict[str, float], str, list[dict]]:
    """
    Parse the 70b reflection response into (deltas_dict, reasoning_string, preferences_list).
    Returns ({}, "", []) on any parse failure.

    Now also parses the "preferences" array from the response.
    """
    try:
        # Strip markdown fences if present
        cleaned = response.strip().strip("```json").strip("```").strip()
        data = json.loads(cleaned)

        deltas = data.get("deltas", {})
        reasoning = data.get("reasoning", "Reflection cycle")
        preferences = data.get("preferences", [])

        if not isinstance(deltas, dict):
            logger.warning(f"[REFLECTION] 'deltas' is not a dict: {type(deltas)}")
            return {}, "", []

        # Validate and clean deltas
        valid_traits = {"trust", "warmth", "sass", "openness", "patience", "playfulness"}
        cleaned_deltas = {}

        for trait, delta in deltas.items():
            if trait not in valid_traits:
                continue
            try:
                delta_f = float(delta)
                # Skip zero deltas
                if delta_f == 0.0:
                    continue
                # Clamp to ±0.05 (belt-and-suspenders — update_traits also clamps)
                delta_f = max(-0.05, min(0.05, delta_f))
                cleaned_deltas[trait] = delta_f
            except (ValueError, TypeError):
                continue

        # Validate preferences
        valid_categories = {
            "app_routing", "contact_routing", "response_style",
            "task_defaults", "schedule", "environment"
        }
        cleaned_preferences = []

        if isinstance(preferences, list):
            for pref in preferences:
                if not isinstance(pref, dict):
                    continue
                key = pref.get("key", "").strip()
                value = pref.get("value", "").strip()
                category = pref.get("category", "").strip()
                evidence = pref.get("evidence", "").strip()

                if not key or not value or not category:
                    continue
                if category not in valid_categories:
                    logger.debug(f"[REFLECTION] Ignoring preference with invalid category: {category}")
                    continue

                confidence = pref.get("confidence", 0.4)
                try:
                    confidence = max(0.0, min(1.0, float(confidence)))
                except (ValueError, TypeError):
                    confidence = 0.4

                cleaned_preferences.append({
                    "key": key,
                    "value": value,
                    "category": category,
                    "confidence": confidence,
                    "evidence": evidence[:200],
                })

        if cleaned_preferences:
            logger.info(
                f"[REFLECTION] Discovered {len(cleaned_preferences)} preference(s): "
                f"{[p['key'] for p in cleaned_preferences]}"
            )

        return cleaned_deltas, str(reasoning)[:200], cleaned_preferences

    except json.JSONDecodeError:
        logger.warning(f"[REFLECTION] Failed to parse JSON: {response[:150]}")
        return {}, "", []
    except Exception as e:
        logger.warning(f"[REFLECTION] Parse error: {e}")
        return {}, "", []
