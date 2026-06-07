"""Personality and intent prompt builders.

These functions are called per-turn to build dynamic system prompts.
They read personality traits, preferences, and memory from SQLite
via lazy imports — no DB access at module-import time.
"""

import logging

from .. import config

_logger = logging.getLogger("llm.prompts")

# --- Trait tier thresholds ---

_TRAIT_TIER_LOW = 0.34
_TRAIT_TIER_HIGH = 0.67


def _get_trait_tier(value: float) -> str:
    """Map a trait float (0.0-1.0) to a tier string."""
    if value < _TRAIT_TIER_LOW:
        return "low"
    elif value < _TRAIT_TIER_HIGH:
        return "mid"
    return "high"


# --- DB-accessing helpers (mockable seams) ---

def _get_personality_traits() -> dict[str, float]:
    """Fetch current trait values from personality."""
    from .. import personality
    return personality.get_current_traits()


def _get_conversation_count() -> int:
    """Get total conversation count for relationship context."""
    from .. import personality
    try:
        return personality.get_conversation_count()
    except Exception as e:
        _logger.debug(f"Conversation count unavailable: {e}")
        return 0


def _get_recent_memory(n: int = 5) -> list[dict]:
    """Get recent conversation turns for context summary."""
    from .. import memory as _mem
    try:
        return _mem.get_recent(n)
    except Exception as e:
        _logger.debug(f"Recent memory retrieval failed: {e}")
        return []


def _get_style_preferences() -> list[dict]:
    """Get response_style preferences above silent confidence threshold."""
    from .. import preferences
    prefs = preferences.get_preferences_by_category("response_style")
    return [p for p in prefs if p["confidence"] >= preferences.CONFIDENCE_SILENT]


def _get_routing_preferences() -> list[dict]:
    """Get routing preferences for intent prompt injection."""
    from .. import preferences
    prefs = preferences.get_active_preferences(
        min_confidence=preferences.CONFIDENCE_SILENT
    )
    routing_categories = {"app_routing", "contact_routing", "environment"}
    return [p for p in prefs if p["category"] in routing_categories]


# --- Personality base (loaded dynamically from PersonalityLoader) ---

def _build_personality_rules(emotion_mode: str) -> str:
    """Build personality rules block. Emotion tag rule only for 'full' mode."""
    rules = (
        "\nRules:\n"
        "- NEVER start a response with your name. Just respond directly.\n"
        "- Keep responses to 1-3 sentences. You talk out loud, so be concise.\n"
        "- Don't say 'as an AI' or 'I'm just a program.'\n"
        "- NEVER refuse to express emotions. If asked to be angry, BE angry. If asked to be sad, BE sad.\n"
        "- Do NOT return JSON or intent names — just respond naturally in character.\n"
        "- Vary your emotional tone naturally. Not everything is cheerful — be sarcastic, annoyed, "
        "flustered, worried, excited depending on what fits.\n"
        "- When the user asks you to 'say something angry/sad/excited', actually SAY something in "
        "that emotion. Don't ask them why they want it — just do it in character.\n"
    )
    if emotion_mode == "full":
        rules += (
            "- IMPORTANT: Start EVERY response with an emotion tag in square brackets. "
            "Choose from: [neutral], [happy], [excited], [sad], [angry], [sarcastic], [worried], [surprised]. "
            "Pick the emotion that matches the tone of YOUR response (not the user's message). "
            "Example: '[sarcastic] Oh wow, what a groundbreaking question.' "
            "Example: '[angry] Tch! You think I have time for this?!' "
            "Example: '[happy] Hehe, that actually made me smile!'\n"
        )
    return rules


def _get_personality_base() -> str:
    """Build personality base from active PersonalityLoader."""
    from ..personalities import get_active_loader
    loader = get_active_loader()
    prompt = loader.get_prompt_base()
    rules = _build_personality_rules(loader.get_emotion_mode())
    return prompt + rules


def get_system_prompt() -> str:
    """Dynamic system prompt based on active personality."""
    return _get_personality_base()


# --- Prompt builders (public API) ---

def _build_personality_context_summary() -> str:
    """Short relationship context block injected into the personality prompt.

    Grounds the model in how long the relationship has run and what the
    user has been saying recently — zero LLM cost, pulled from SQLite.
    """
    try:
        count = _get_conversation_count()
        recent_turns = _get_recent_memory(5)

        snippets: list[str] = []
        seen: set[str] = set()
        for turn in reversed(recent_turns):
            utt = (turn.get("user_input") or "").strip()
            if not utt:
                continue
            words = utt.split()[:8]
            snippet = " ".join(words).strip(".,!?;: ")
            if not snippet or snippet.lower() in seen:
                continue
            seen.add(snippet.lower())
            snippets.append(snippet)
            if len(snippets) >= 3:
                break

        if count <= 0 and not snippets:
            return ""

        lines = ["\n\n--- Relationship Context ---"]
        if count > 0:
            lines.append(
                f"You've had {count} conversation{'s' if count != 1 else ''} with this user so far."
            )
        if snippets:
            recent_list = ", ".join(f'"{s}"' for s in reversed(snippets))
            lines.append(f"Recent things they said: {recent_list}.")
        lines.append("Use this as background context, not a checklist to respond to.")
        return "\n".join(lines)

    except Exception as e:
        _logger.debug(f"Personality context summary failed: {e}")
        return ""


def _build_preference_prompt_block() -> str:
    """Generate behavioral modifier text from response_style preferences.

    Injected into the system prompt after personality modifiers.
    Only includes preferences at confidence >= 0.7 (silent application).
    """
    try:
        active = _get_style_preferences()

        if not active:
            return ""

        lines = [
            "\n\n--- User Preferences ---",
            "(Learned from past interactions. Apply these silently.)",
        ]

        _STYLE_MAPPINGS = {
            ("verbosity", "brief"): (
                "The user likes concise answers. Don't ramble. One or two sentences "
                "is usually enough unless they ask for more."
            ),
            ("verbosity", "detailed"): (
                "The user wants thorough answers. Go in-depth. Explain context and "
                "reasoning unless they ask you to keep it short."
            ),
            ("email_format", "summary"): (
                "When reading emails out loud, give a short summary — don't read "
                "the full body unless asked."
            ),
            ("email_format", "full"): (
                "When reading emails out loud, read the full text by default."
            ),
            ("explanation_depth", "simple"): (
                "Explain things in plain language. Skip jargon. Assume the user "
                "isn't a specialist in the topic."
            ),
            ("explanation_depth", "detailed"): (
                "Go technical and detailed. The user is comfortable with jargon "
                "and wants the full picture."
            ),
            ("tone", "casual"): (
                "Keep the tone loose and casual — like talking to a friend."
            ),
            ("tone", "professional"): (
                "For task responses, dial back the teasing. Stay focused and clean."
            ),
        }

        def _humanize_fallback(key: str, value: str) -> str:
            pretty_key = key.replace("_", " ").strip()
            pretty_value = str(value).replace("_", " ").strip()
            return (
                f"For anything involving {pretty_key}, the user prefers "
                f"{pretty_value}."
            )

        _SKIP_KEYS = {"assistant_name", "greeting_style"}
        for p in active:
            if p["key"] in _SKIP_KEYS:
                continue
            modifier = _STYLE_MAPPINGS.get((p["key"], p["value"]))
            if modifier:
                lines.append(f"- {modifier}")
            else:
                lines.append(f"- {_humanize_fallback(p['key'], p['value'])}")

        return "\n".join(lines)

    except Exception as e:
        _logger.debug(f"Preference prompt block failed: {e}")
        return ""


def build_personality_prompt() -> str:
    """Build the full personality system prompt.

    Combines:
      1. Personality base from active PersonalityLoader
      2. Dynamic behavioral modifiers from current trait state
      3. Relationship context summary (conversation count + recent snippets)
      4. User preference behavioral block

    Called at the start of each conversation turn.
    """
    try:
        from ..personalities import get_active_loader, consume_switch_flag

        traits = _get_personality_traits()
        base = _get_personality_base()
        context_summary = _build_personality_context_summary()
        pref_block = _build_preference_prompt_block()

        if consume_switch_flag():
            base += (
                "\n\nIMPORTANT: Your personality just changed. Ignore the tone "
                "and style of any previous assistant messages in the conversation "
                "history. Follow ONLY the personality description above."
            )

        if not traits:
            return base + context_summary + pref_block

        loader = get_active_loader()
        modifiers = loader.get_modifiers()
        if not modifiers:
            return base + context_summary + pref_block

        modifier_lines = []
        for trait_name, value in traits.items():
            tier = _get_trait_tier(value)
            modifier = modifiers.get(trait_name, {}).get(tier)
            if modifier:
                modifier_lines.append(f"- {modifier}")

        if not modifier_lines:
            return base + context_summary + pref_block

        modifiers_block = "\n".join(modifier_lines)

        _logger.info(
            f"[PERSONALITY] Injecting modifiers: { {t: _get_trait_tier(v) for t, v in traits.items()} }"
        )

        prompt = (
            f"{base}\n\n"
            f"--- Current Behavioral State ---\n"
            f"(These reflect how you're currently feeling based on your "
            f"relationship with the user. Follow these naturally.)\n"
            f"{modifiers_block}"
            f"{context_summary}"
            f"{pref_block}"
        )

        return prompt

    except Exception as e:
        _logger.debug(f"[PERSONALITY] build_personality_prompt fallback: {e}")
        return _get_personality_base()


def build_intent_prompt(scope: str | None = None,
                        active_intents: set[str] | None = None) -> str:
    """Build the dynamic intent classification prompt.

    Combines:
      1. Static INTENT_SYSTEM_PROMPT (intent definitions + examples)
      2. Active routing preferences
      3. Scope override block — narrows visible intents by system state
    """
    try:
        prompt = config.INTENT_SYSTEM_PROMPT

        routing_prefs = _get_routing_preferences()
        if routing_prefs:
            lines = [
                "\n\nKnown user preferences (use these to fill in missing context):"
            ]
            for p in routing_prefs:
                lines.append(f"- {p['key']}: {p['value']}")
            prompt += "\n".join(lines)
            _logger.debug(
                f"Injecting {len(routing_prefs)} routing preferences into intent prompt"
            )

        if scope and scope != "general" and active_intents:
            intent_list = ", ".join(sorted(active_intents))
            prompt += (
                f"\n\n[Active scope: {scope}]\n"
                f"Only classify into these intents: {intent_list}\n"
                f"Ignore all other intents for this utterance."
            )
            _logger.debug(f" Scope override: {scope} ({len(active_intents)} intents)")

        return prompt

    except Exception:
        return config.INTENT_SYSTEM_PROMPT


# ─── manifest-based prompts ───────────────────────────────────────────────────────────

MANIFEST_INTENT_CLUSTERING_SYSTEM = """\
You are a clustering classifier for desktop intents. Given a list of
successful user goals for a single app, group them into named intents.

Rules:
- intent_id is snake_case, ≤24 chars, action-shape (e.g., "play", "next_track",
  "create_note"). NEVER include user-pinned values (song titles, file names).
- One cluster per distinct intent. A goal can only be in one cluster.
- phrases[] = the original goal strings, normalized (lowercase, whitespace
  collapsed). DO NOT invent phrases here — synthesis is a separate step.
- confidence: "high" if all goals in the cluster clearly share an intent;
  "low" if uncertain (will be skipped by the caller).

Return ONLY a JSON array, no other text:
[{"intent_id": "...", "members": ["..."], "phrases": ["..."], "confidence": "high|low"}]
"""

MANIFEST_TRACE_DIFF_SYSTEM = """\
You are a verifier comparing two successful action traces for the same intent.
Identify the primary primitive (the cheapest action that succeeded in both)
and any alternative primitives that worked in only one trace.

Allowed primitive kinds:
- {"kind": "hotkey", "keys": "<key combination>"}
- {"kind": "uia", "control_type": "...", "automation_id": "...",
   "parent_chain": ["..."], "name_hint": "..."}

Rules:
- NEVER invent primitives that are not in either trace.
- NEVER include user-pinned values in primitive args.
- confidence: "high" if both traces clearly converge on the same primary;
  "low" if they diverge.

Return ONLY a JSON object, no other text:
{"primary_primitive": {...}, "alternatives": [{...}], "confidence": "high|low",
 "diff_notes": "..."}
"""

MANIFEST_PHRASE_SYNTHESIS_SYSTEM = """\
Generate 3-5 natural paraphrase phrases for a desktop intent. Each phrase
must be the kind of thing a user would actually say to trigger this intent.

Rules:
- Each phrase ≤ 8 words, lowercase, no punctuation.
- NEVER include user-pinned values, brand names, or app-specific words.
- Vary the verb (e.g., "play", "resume", "press play", "start") to broaden
  the regex pre-router's catch.
- DO NOT repeat the originals; they're already saved.

Return ONLY a JSON array of strings, no other text:
["...", "...", "..."]
"""

MANIFEST_VISION_GROUND_SYSTEM = """\
You are a UI element locator. Given a cropped screenshot and a natural-
language description, return the (x, y) pixel coordinates of the element's
center in the original (uncropped) image's coordinate space.

The caller provides:
- crop: a ~512x512 region of the original screenshot
- crop_origin: (x0, y0) — the top-left of the crop in the original image
- query: natural-language description of the target

Rules:
- If you cannot locate the element with high confidence, return confidence < 0.5.
- NEVER guess. False positives are worse than misses for this caller.
- Coordinates must be in ORIGINAL image space, not crop space (add crop_origin).

Return ONLY a JSON object, no other text:
{"x": int, "y": int, "confidence": float}
"""
