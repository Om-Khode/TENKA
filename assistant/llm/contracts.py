"""Task-shaped LLM wrappers.

Each function maps 1:1 to a task_type in TASK_MODEL_MAP. Action handlers
call these instead of ``get_llm_response`` directly, so the task_type is
always correct and per-task defaults live in one place.

Vision calls (``get_vision_response``) are NOT covered here — future pass.
"""

import json
import logging
import re

from .router import get_llm_response, get_vision_response

logger = logging.getLogger(__name__)


async def ask_for_intent(
    prompt: str, *,
    system_prompt: str | None = None,
    max_tokens: int = 256,
    temperature: float | None = None,
    json_mode: bool = False,
) -> str:
    """Intent classification, JSON extraction, structured parsing."""
    result = await get_llm_response(
        prompt,
        system_prompt=system_prompt,
        task_type="intent",
        max_tokens=max_tokens,
        temperature=temperature,
        json_mode=json_mode,
    )
    return result.text


async def ask_for_synthesis(
    prompt: str, *,
    system_prompt: str | None = None,
    max_tokens: int = 256,
    temperature: float | None = None,
    json_mode: bool = False,
) -> str:
    """Personality-bearing content summarization."""
    result = await get_llm_response(
        prompt,
        system_prompt=system_prompt,
        task_type="synthesis",
        max_tokens=max_tokens,
        temperature=temperature,
        json_mode=json_mode,
    )
    return result.text


async def ask_for_plan(
    prompt: str, *,
    system_prompt: str | None = None,
    max_tokens: int = 256,
    temperature: float | None = None,
    json_mode: bool = False,
) -> str:
    """Structured multi-step planning (agent, DOM, vision)."""
    result = await get_llm_response(
        prompt,
        system_prompt=system_prompt,
        task_type="agent_plan",
        max_tokens=max_tokens,
        temperature=temperature,
        json_mode=json_mode,
    )
    return result.text


async def ask_for_code_gen(
    prompt: str, *,
    system_prompt: str | None = None,
    max_tokens: int = 256,
    temperature: float | None = None,
    json_mode: bool = False,
) -> str:
    """Code generation (strongest reasoning model)."""
    result = await get_llm_response(
        prompt,
        system_prompt=system_prompt,
        task_type="code_gen",
        max_tokens=max_tokens,
        temperature=temperature,
        json_mode=json_mode,
    )
    return result.text


async def ask_for_small_talk(
    prompt: str, *,
    system_prompt: str | None = None,
    max_tokens: int = 256,
    temperature: float | None = None,
    json_mode: bool = False,
) -> str:
    """Personality-bearing casual conversation."""
    result = await get_llm_response(
        prompt,
        system_prompt=system_prompt,
        task_type="small_talk",
        max_tokens=max_tokens,
        temperature=temperature,
        json_mode=json_mode,
    )
    return result.text


async def ask_for_personality_reflection(
    prompt: str, *,
    system_prompt: str | None = None,
    max_tokens: int = 256,
    temperature: float | None = None,
    json_mode: bool = False,
) -> str:
    """Nightly personality trait analysis."""
    result = await get_llm_response(
        prompt,
        system_prompt=system_prompt,
        task_type="personality_reflection",
        max_tokens=max_tokens,
        temperature=temperature,
        json_mode=json_mode,
    )
    return result.text


async def ask_for_agent_verify(
    prompt: str, *,
    system_prompt: str | None = None,
    max_tokens: int = 256,
    temperature: float | None = None,
    json_mode: bool = False,
) -> str:
    """Goal achievement verification (text fallback path)."""
    result = await get_llm_response(
        prompt,
        system_prompt=system_prompt,
        task_type="agent_verify",
        max_tokens=max_tokens,
        temperature=temperature,
        json_mode=json_mode,
    )
    return result.text


async def ask_for_default(
    prompt: str, *,
    system_prompt: str | None = None,
    max_tokens: int = 256,
    temperature: float | None = None,
    json_mode: bool = False,
) -> str:
    """Short utility calls (default model chain)."""
    result = await get_llm_response(
        prompt,
        system_prompt=system_prompt,
        task_type="default",
        max_tokens=max_tokens,
        temperature=temperature,
        json_mode=json_mode,
    )
    return result.text


# ─── Streaming Contracts ────────────────────────────────────────────────────

from .router import get_llm_response_stream


async def stream_for_synthesis(
    prompt: str, *,
    system_prompt: str | None = None,
    max_tokens: int = 256,
    temperature: float | None = None,
):
    """Streaming personality-bearing content summarization."""
    stream = await get_llm_response_stream(
        prompt, system_prompt=system_prompt, task_type="synthesis",
        max_tokens=max_tokens, temperature=temperature,
    )
    async for chunk in stream:
        yield chunk


async def stream_for_small_talk(
    prompt: str, *,
    system_prompt: str | None = None,
    max_tokens: int = 256,
    temperature: float | None = None,
    messages: list[dict] | None = None,
):
    """Streaming personality-bearing casual conversation."""
    stream = await get_llm_response_stream(
        prompt, system_prompt=system_prompt, task_type="small_talk",
        max_tokens=max_tokens, temperature=temperature,
        messages=messages,
    )
    async for chunk in stream:
        yield chunk


# ─── Memory Classification ──────────────────────────────────────────────────

_VALID_MEMORY_TYPES = frozenset({"preference", "identity", "fact", "how_to", "blocker"})

_MEMORY_TYPE_PROMPT = """\
Classify this memory fact into exactly one type.

Types:
- preference: user preferences and choices (no expiry)
- identity: facts about who the user is (no expiry)
- fact: general situational knowledge (expires in 30 days)
- how_to: informal procedural knowledge (expires in 14 days)
- blocker: known issues or limitations (expires in 14 days)

Key: {key}
Value: {value}

Respond with ONLY the type name, nothing else."""


async def ask_for_memory_type(key: str, value: str) -> str:
    """Classify a fact into a memory type. Flash-Lite call, single token."""
    try:
        raw_result = await get_llm_response(
            _MEMORY_TYPE_PROMPT.format(key=key, value=value),
            system_prompt="You are a classification utility. Return only the type name.",
            task_type="synthesis",
            max_tokens=10,
            temperature=0.0,
        )
        raw = raw_result.text
        result = raw.strip().lower()
        if result in _VALID_MEMORY_TYPES:
            return result
    except Exception:
        pass
    return "fact"


# ─── Session Summary ────

_SESSION_SUMMARY_PROMPT = """Summarize this conversation in JSON.
- task_summary: one sentence, what the user was doing (max 100 chars)
- blocker: one sentence if something was left unfinished or failed, else null

Respond ONLY with valid JSON. No markdown, no explanation.

Conversation:
{turns_text}"""


async def ask_for_session_summary(turns: list[dict]) -> dict:
    """Summarize a session's turns into task_summary + blocker. Flash-Lite."""
    import json

    turns_text = "\n".join(
        f"User: {t['user_input']}\nAssistant: {t['response']}"
        for t in turns
    )
    try:
        raw_result = await get_llm_response(
            _SESSION_SUMMARY_PROMPT.format(turns_text=turns_text),
            system_prompt="You are a summarization utility. Return only JSON.",
            task_type="synthesis",
            max_tokens=100,
            temperature=0.0,
        )
        raw = raw_result.text
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        result = json.loads(cleaned)
        if "task_summary" in result:
            return {
                "task_summary": str(result["task_summary"])[:100],
                "blocker": result.get("blocker"),
            }
    except (json.JSONDecodeError, KeyError, TypeError, Exception):
        pass
    return {"task_summary": "General conversation", "blocker": None}


# ─── Context Compression ────────────────────────────────────────────────────

_CONTEXT_COMPRESSION_PROMPT = """\
Compress this conversation into a concise summary (3-5 sentences).
Preserve: key facts, user names, decisions, unresolved questions, user preferences.
Drop: greetings, filler, repeated information, pleasantries, the assistant's name.
Refer to the assistant as "the assistant", never by name.

Conversation:
{turns_text}"""


async def ask_for_context_compression(turns: list[dict]) -> str:
    """Compress conversation turns into a brief summary. Flash-Lite."""
    turns_text = "\n".join(
        f"User: {t['user_input']}\nAssistant: {t['response']}"
        for t in turns
    )
    result = await get_llm_response(
        _CONTEXT_COMPRESSION_PROMPT.format(turns_text=turns_text),
        system_prompt="You are a conversation summarizer. Be concise.",
        task_type="synthesis",
        max_tokens=150,
        temperature=0.0,
    )
    text = result.text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    return text


# ─── Schedule Parsing  ──────────────────────────────────────

_SCHEDULE_PARSE_PROMPT = """\
Extract a scheduled task from this request.
Current time: {now}

Request: "{goal}"

Respond with JSON:
{{
  "name": "short label for this task (3-5 words)",
  "cron_expr": "minute hour day month weekday",
  "task_type": "web_search" or "http_check" or "procedure",
  "goal": "search query, URL to check, or procedure name",
  "notify_mode": "always" or "on_match_only" or "on_change",
  "condition_text": "condition for notification (null if always/on_change)"
}}

Rules:
- "every morning" = "0 9 * * *"
- "every evening" = "0 18 * * *"
- "every day at Xam/pm" = "0 X * * *"
- "every Monday" = "0 9 * * 1"
- "every hour" = "0 * * * *"
- Default notify_mode is "on_match_only" unless user says "always" or "if anything changes"
- If user says "only tell me if..." or "only when...", extract that as condition_text
- task_type is "http_check" if goal is a URL (http:// or https://)
- task_type is "procedure" only if user names a taught procedure
- Otherwise task_type is "web_search"
"""


async def ask_for_schedule_parse(goal: str) -> dict | None:
    """Extract structured schedule from natural language. Returns None on failure."""
    import json
    from datetime import datetime

    try:
        raw_result = await get_llm_response(
            _SCHEDULE_PARSE_PROMPT.format(goal=goal, now=datetime.now().isoformat()),
            system_prompt="You are a schedule extraction utility. Return only JSON.",
            task_type="synthesis",
            max_tokens=256,
            temperature=0.0,
        )
        raw = raw_result.text
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        result = json.loads(cleaned)
        required = ("name", "cron_expr", "task_type", "goal", "notify_mode")
        if all(k in result for k in required):
            return result
    except (json.JSONDecodeError, KeyError, TypeError, Exception):
        pass
    return None


_CONDITION_CHECK_PROMPT = """\
A scheduled monitor just ran and got this result:

{result}

The user wants to be notified only if: {condition_text}

Should the user be notified?
Respond with JSON: {{"notify": true/false, "summary": "one short sentence (under 100 chars)"}}
"""


async def ask_for_condition_check(result: str, condition_text: str) -> dict:
    """Evaluate whether a monitor result meets the notification condition."""
    import json

    try:
        raw_result = await get_llm_response(
            _CONDITION_CHECK_PROMPT.format(
                result=result[:2000], condition_text=condition_text
            ),
            system_prompt="You are a condition evaluator. Return only JSON.",
            task_type="synthesis",
            max_tokens=80,
            temperature=0.0,
        )
        raw = raw_result.text
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        parsed = json.loads(cleaned)
        if "notify" in parsed and "summary" in parsed:
            parsed["summary"] = str(parsed["summary"])[:100]
            return parsed
    except (json.JSONDecodeError, KeyError, TypeError, Exception):
        pass
    return {"notify": False, "summary": "Could not evaluate condition"}


# ─── Event Monitor Definition ──────────────────────────────────────────

_MONITOR_PARSE_PROMPT = """\
You are defining an event monitor. The user wants: "{goal}"

Available event types:
- media_changed: fires when the current song/media track changes.
  Event fields: title, artist, album, source_app, playback_status
- window_focus: fires when the active window changes.
  Event fields: source_app, window_title, prev_app, prev_title
- window_title: fires when the foreground window's title text changes.
  Event fields: source_app, window_title

Return ONE JSON object:
{{
  "name": "short human-readable name for this monitor",
  "event_type": "one of: media_changed, window_focus, window_title",
  "source_filter": "app name to scope to, or null if any app",
  "condition_code": "Python expression using event field names as variables, or null",
  "condition_prompt": "natural language condition for LLM evaluation, or null if code handles it",
  "action_type": "code_executor or tts_notify",
  "action_payload": "goal string or TTS message with {{field}} placeholders",
  "cooldown_secs": 5
}}

Rules:
- source_filter matches the PROCESS name (e.g. "spotify", "brave", "chrome", "firefox", "discord"), NOT a website or service name. For website-specific matching, use condition_code on window_title instead. For websites, set source_filter to null (websites can be opened in any browser).
- source_filter does case-insensitive substring matching at runtime. Do NOT put app-matching logic in condition_code — source_filter handles it.
- If there is no content-level condition beyond matching an app, set condition_code to null. Null means "fire on every matching event".
- condition_code is a Python EXPRESSION (not statement). Available builtins: any, all, len, ord, chr, min, max, int, float, str, bool, isinstance, range, sorted.
- For substring matching in condition_code, use `in` (e.g. `'youtube' in window_title.lower()`), not `startswith`. window_title is the page/app TITLE text (e.g. "Video Name - YouTube - Brave"), NOT the URL. Use display names like "YouTube", "GitHub", not domains like "youtube.com".
- Prefer condition_code over condition_prompt when possible.
- action_payload for code_executor: a natural language goal string.
- action_payload for tts_notify: a message template with {{field}} placeholders.
- If you cannot map the request to any event_type, set event_type to null.

Return ONLY the JSON object. No markdown, no explanation."""


async def ask_for_monitor_definition(goal: str) -> dict | None:
    """Decompose a natural language goal into a monitor definition. Returns None on failure."""
    import json

    try:
        raw_result = await get_llm_response(
            _MONITOR_PARSE_PROMPT.format(goal=goal),
            system_prompt="You are a monitor definition utility. Return only JSON.",
            task_type="synthesis",
            max_tokens=512,
            temperature=0.0,
        )
        raw = raw_result.text
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        result = json.loads(cleaned)
        required = ("name", "event_type", "action_type", "action_payload")
        if all(k in result for k in required):
            return result
    except (json.JSONDecodeError, KeyError, TypeError, Exception):
        pass
    return None


# ─── manifest-based contracts ─────────────────────────────────────────────────────────


async def ask_for_intent_clustering(*, app: str, goals: list[str]) -> list[dict]:
    """Cluster goal slugs for an app into named intents.

    Returns ``[{intent_id, members, phrases, confidence}, ...]`` on success.
    Returns ``[]`` on any failure (parse error, empty input, LLM unavailable)
    so the promoter cycle keeps going for other apps.
    """
    from .prompts import MANIFEST_INTENT_CLUSTERING_SYSTEM
    from ..core.json_utils import extract_json_array

    if not goals:
        return []

    user_msg = (
        f"Cluster these goals for app '{app}' into intents:\n"
        + "\n".join(f"- {g}" for g in goals)
    )
    try:
        result = await get_llm_response(
            user_msg,
            system_prompt=MANIFEST_INTENT_CLUSTERING_SYSTEM,
            task_type="intent",
            max_tokens=512,
            temperature=0.0,
        )
        return extract_json_array(result.text, sanitize=True)
    except Exception:
        return []


async def ask_for_trace_diff_verification(*, traces: list[list[dict]]) -> dict:
    """Verify that 2+ action traces converge on a canonical primitive.

    Returns ``{primary_primitive, alternatives, confidence, diff_notes}``.
    On parse failure / LLM unavailability returns a low-confidence sentinel
    so the promoter cycle keeps going.
    """
    import json
    from .prompts import MANIFEST_TRACE_DIFF_SYSTEM
    from ..core.json_utils import extract_json_object

    fallback = {
        "primary_primitive": None,
        "alternatives": [],
        "confidence": "low",
        "diff_notes": "parse failure",
    }

    if not traces:
        return fallback

    try:
        trace_blocks = []
        for i, trace in enumerate(traces):
            trace_blocks.append(
                f"Trace {i + 1}:\n" + json.dumps(trace, ensure_ascii=False)
            )
        user_msg = (
            "Compare these action traces and report whether they converge:\n\n"
            + "\n\n".join(trace_blocks)
        )
        result = await get_llm_response(
            user_msg,
            system_prompt=MANIFEST_TRACE_DIFF_SYSTEM,
            task_type="agent_verify",
            max_tokens=512,
            temperature=0.0,
        )
        raw_obj = extract_json_object(result.text, sanitize=True, repair=True)
        if not raw_obj:
            return fallback
        parsed = json.loads(raw_obj)
        if not isinstance(parsed, dict):
            return fallback
        return {
            "primary_primitive": parsed.get("primary_primitive"),
            "alternatives": parsed.get("alternatives") or [],
            "confidence": parsed.get("confidence") or "low",
            "diff_notes": parsed.get("diff_notes") or "",
        }
    except Exception:
        return fallback


async def ask_for_phrase_synthesis(
    *, intent_id: str, originals: list[str]
) -> list[str]:
    """Generate 3-5 paraphrase phrases for an intent.

    Returns ``[]`` on any failure so the promoter cycle keeps going.
    Filters out empty strings and original phrases from the result.
    """
    from .prompts import MANIFEST_PHRASE_SYNTHESIS_SYSTEM
    from ..core.json_utils import extract_json_array

    if not intent_id:
        return []

    originals_block = "\n".join(f"- {p}" for p in originals) if originals else "(none)"
    user_msg = (
        f"intent_id: {intent_id}\n"
        f"original phrases:\n{originals_block}\n\n"
        f"Generate 3-5 fresh paraphrases."
    )
    try:
        result = await get_llm_response(
            user_msg,
            system_prompt=MANIFEST_PHRASE_SYNTHESIS_SYSTEM,
            task_type="default",
            max_tokens=256,
            temperature=0.7,
        )
        parsed = extract_json_array(result.text, sanitize=True)
        if not isinstance(parsed, list):
            return []
        seen_originals = {p.strip().lower() for p in originals}
        cleaned: list[str] = []
        for item in parsed:
            if not isinstance(item, str):
                continue
            phrase = item.strip()
            if not phrase or phrase.lower() in seen_originals:
                continue
            cleaned.append(phrase)
        return cleaned
    except Exception:
        return []


async def ask_for_vision_ground_coords(
    *, crop_bytes: bytes, query: str, crop_origin: tuple[int, int],
) -> dict:
    """Ground a natural-language UI query to (x, y) via Gemini vision.

    Tier-2 healer path. Returns ``{"x": int, "y": int, "confidence": float}``.
    On any failure (LLM unavailable, malformed JSON, missing keys) returns
    ``{"confidence": 0.0}`` so the healer's threshold check (< 0.7) naturally
    rejects bad calls.

    Args:
        crop_bytes: raw PNG/JPEG bytes of a ~512x512 crop around the target.
        query: natural-language description (e.g. "play button").
        crop_origin: (x0, y0) — top-left of the crop in original-image space.
    """
    import base64
    import json
    from .prompts import MANIFEST_VISION_GROUND_SYSTEM
    from ..core.json_utils import extract_json_object

    fallback = {"confidence": 0.0}
    try:
        image_b64 = base64.b64encode(crop_bytes).decode("ascii")
        user_msg = (
            f"crop_origin: {crop_origin}\nquery: {query}\n"
            "Locate the element and return JSON only."
        )
        result = await get_vision_response(
            image_base64=image_b64,
            prompt=user_msg,
            system_prompt=MANIFEST_VISION_GROUND_SYSTEM,
            json_mode=True,
            max_tokens=128,
        )
        raw_obj = extract_json_object(result.text, sanitize=True, repair=True)
        if not raw_obj:
            return fallback
        parsed = json.loads(raw_obj)
        if not (isinstance(parsed, dict) and "x" in parsed and "y" in parsed
                and "confidence" in parsed):
            return fallback
        try:
            return {
                "x": int(parsed["x"]),
                "y": int(parsed["y"]),
                "confidence": float(parsed["confidence"]),
            }
        except (ValueError, TypeError):
            return fallback
    except Exception:
        pass
    return fallback


# ─── Entity / Fact / Relationship Extraction ──────────────────────────


def _build_kg_extraction_prompt(
    source: str, context_hint: str | None = None,
) -> str:
    """Pure prompt-builder for entity/fact/relationship extraction.

    Extracted from ask_for_entity_extraction so tests can assert prompt
    contents directly without async/LLM round-tripping.

    Source biases first-person ("user_msg") vs third-person ("tenka_resp")
    interpretation but does not change the JSON schema. context_hint, when
    non-empty, is injected as a CONVERSATION CONTEXT section so the LLM can
    resolve referents ("she", "he", "it") to the active topic — fixes the
    knowledge-graph Session 1 livetest gap where pronoun-led turns produced no
    subject-facts on the referenced person.
    """
    context_section = ""
    if context_hint and context_hint.strip():
        context_section = (
            f"\nCONVERSATION CONTEXT (use to resolve pronouns like she/he/it "
            f"in the TEXT below to the referenced entity before extraction):\n"
            f"{context_hint.strip()}\n"
        )
    return (
        f"{context_section}"
        "Extract a knowledge graph from the text. Return ONLY this JSON:\n"
        '{"entities":[{"type":"person|project|tool|place|concept|event","name":"...","confidence":0.0-1.0}],'
        '"facts":[{"subject":"entity name","predicate":"snake_case_verb","object":"value","confidence":0.0-1.0,'
        '"event_at":"YYYY-MM-DD or YYYY-MM or YYYY or omit"}],'
        '"relationships":[{"from":"entity name","to":"entity name","type":"manages|uses|part_of|related_to|parent_of|knows","confidence":0.0-1.0}],'
        '"commitments":[{"owner":"entity name (default user)","promise":"what was promised, free text","when_due":"YYYY-MM-DD or omit"}]}\n'
        "Use type values from the closed lists above. Confidence: 1.0 for "
        "explicitly stated; <0.5 for inferred. If nothing extractable, return "
        "empty arrays.\n"
        "Do NOT invent entities from templated key:value confirmation phrases "
        "(e.g. 'Got it, I'll remember that. X: Y' — X is a storage label, not "
        "an entity). Skip storage-key-style phrases formatted as Title Case "
        "with underscores or 'Foo Bar:' prefixes. Only extract names that "
        "refer to real-world people, places, projects, tools, concepts, or "
        "events the speaker actually mentions.\n"
        "Do NOT extract pronouns or generic referents (she, he, they, them, "
        "their, you, your, it, this, that, we, us, our) as entities or as "
        "endpoints of relationships — they are not first-class entities. "
        "If a sentence uses a pronoun for a referent, resolve it from earlier "
        "context if obvious; otherwise skip that entity/relationship.\n"
        "\n"
        "SUBJECT-FACT LIFTING: when a person is described with a job, role, "
        "or family relation, emit the fact ON the person (subject = the "
        "person). Do NOT bury personal attributes in concept-to-concept "
        "relationships.\n"
        "  Person + employer + role: 'Priya works at Acme as a finance "
        "analyst' → facts: (Priya, works_at, Acme), (Priya, has_role, "
        "finance analyst). NOT relationships between 'finance analyst' and "
        "'Acme'.\n"
        "  Family relations (brother, sister, mother, father, parent, son, "
        "daughter, uncle, aunt, cousin): 'my uncle Damien lives in Madrid' "
        "→ facts: (Damien, is_a, uncle), (user, has_uncle, Damien), "
        "(Damien, lives_in, Madrid).\n"
        "\n"
        "TEMPORAL GROUNDING (event_at): when the text states or implies "
        "WHEN a fact happened with an absolute or recoverable calendar "
        "marker (date, month, year, e.g. 'last March' or 'in 2024'), set "
        "event_at on that fact to the most specific ISO form you can derive "
        "(YYYY-MM-DD > YYYY-MM > YYYY). Do NOT resolve relative phrases like "
        "'yesterday' / 'last week' / 'recently' to absolute dates — omit "
        "event_at unless the calendar anchor is recoverable from the text "
        "itself. Omit event_at when no temporal info is present.\n"
        "\n"
        "COMMITMENTS: a commitment is a PROMISE someone made — either the "
        "user committing to do something ('I'll send the report by Friday', "
        "'I promised to call mom') or someone committing to the user "
        "('Priya said she'll review it tomorrow'). Set owner to the person "
        "who made the promise (default 'user' when the speaker is the user "
        "and no other person is named). promise is free text describing the "
        "thing promised. when_due follows the same date rules as event_at "
        "(omit relative phrases like 'soon' / 'later'). "
        "Do NOT emit a commitment for: requests addressed to the assistant "
        "(remind me to X / set a reminder for Y), past tense without a "
        "promise verb, hypotheticals ('I should probably...', 'maybe I'll'), "
        "or already-fulfilled actions. Empty array when nothing qualifies.\n"
        "\n"
        "Return ONLY the JSON, no prose."
    )


async def ask_for_entity_extraction(
    text: str, source: str, context_hint: str | None = None,
) -> dict:
    """Extract entities + facts + relationships from a turn.

    Returns:
        {
          "entities":      [{"type", "name", "confidence"}],
          "facts":         [{"subject", "predicate", "object", "confidence", "event_at"}],
          "relationships": [{"from", "to", "type", "confidence"}],
        }

    `source` is "user_msg" or "tenka_resp"; biases the prompt toward
    first-person vs third-person interpretation.

    Never raises. Malformed payload → empty arrays.
    """
    EMPTY = {"entities": [], "facts": [], "relationships": [], "commitments": []}
    if not text or not text.strip():
        return EMPTY

    source_hint = (
        "The text is the USER'S message — extract first-person facts about the user "
        "and people/places/things they mention."
        if source == "user_msg" else
        "The text is the ASSISTANT'S response — extract facts and entities the "
        "assistant asserts as true."
    )

    system_prompt = _build_kg_extraction_prompt(source, context_hint=context_hint)
    user_message = f"IMPORTANT: {source_hint}\n\nTEXT:\n{text}"

    try:
        result = await get_llm_response(
            user_message,
            system_prompt=system_prompt,
            task_type="kg_extraction",
            max_tokens=400,
            json_mode=False,
        )
    except Exception as e:
        logger.debug(f"[KG] ask_for_entity_extraction failed (non-critical): {e}")
        return EMPTY

    raw = (result.text or "").strip()
    if not raw or raw == "{}":
        return EMPTY

    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:].lstrip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return EMPTY
        try:
            parsed = json.loads(m.group(0))
        except json.JSONDecodeError:
            return EMPTY

    if not isinstance(parsed, dict):
        return EMPTY
    return {
        "entities":      list(parsed.get("entities", []))      if isinstance(parsed.get("entities"), list)      else [],
        "facts":         list(parsed.get("facts", []))         if isinstance(parsed.get("facts"), list)         else [],
        "relationships": list(parsed.get("relationships", [])) if isinstance(parsed.get("relationships"), list) else [],
        "commitments":   list(parsed.get("commitments", []))   if isinstance(parsed.get("commitments"), list)   else [],
    }


# ─── knowledge-graph D — Multi-hop COT validator ──────────────────────────────────────


_KG_FOLLOWUP_SYSTEM_PROMPT = (
    "You are a knowledge-graph traversal helper. The user asked a question "
    "and the assistant has gathered some KG context blocks already. Decide "
    "whether the current context is SUFFICIENT to answer the question, OR "
    "whether one more entity should be looked up.\n"
    "\n"
    "Return ONLY a single JSON object — no prose, no markdown — with this "
    "exact shape:\n"
    '  {"sufficient": true,  "follow_up": null}\n'
    'OR\n'
    '  {"sufficient": false, "follow_up": "<entity name to look up next>"}\n'
    "\n"
    "Pick a follow_up only when:\n"
    "- the current blocks mention an entity by name but do not say enough "
    "about it to answer the question, AND\n"
    "- that named entity is the most likely next hop.\n"
    "If the question is already answerable, set sufficient=true and "
    "follow_up=null. If nothing useful can be looked up, also set "
    "sufficient=true (do NOT invent names that are not in the blocks)."
)


async def ask_for_kg_followup(question: str, current_context: str) -> dict:
    """D — Cognee multi-hop validator.

    Inputs:
        question: the user's original question
        current_context: the joined KG context blocks gathered so far
                         (the same string format produced by build_kg_context)

    Returns:
        {"sufficient": bool, "follow_up": str | None}

    Never raises. Malformed payload → {"sufficient": True, "follow_up": None}
    (graceful stop). Empty inputs → same graceful stop.
    """
    STOP = {"sufficient": True, "follow_up": None}
    if not question or not question.strip():
        return STOP
    if not current_context or not current_context.strip():
        return STOP

    user_message = (
        f"QUESTION:\n{question}\n\n"
        f"CURRENT KG CONTEXT BLOCKS:\n{current_context}"
    )

    try:
        result = await get_llm_response(
            user_message,
            system_prompt=_KG_FOLLOWUP_SYSTEM_PROMPT,
            task_type="kg_followup",
            max_tokens=80,
            json_mode=False,
        )
    except Exception as e:
        logger.debug(f"[KG] ask_for_kg_followup failed (non-critical): {e}")
        return STOP

    raw = (result.text or "").strip()
    if not raw:
        return STOP
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:].lstrip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return STOP
        try:
            parsed = json.loads(m.group(0))
        except json.JSONDecodeError:
            return STOP

    if not isinstance(parsed, dict):
        return STOP

    sufficient = bool(parsed.get("sufficient", True))
    follow_up_raw = parsed.get("follow_up")
    follow_up: str | None = None
    if isinstance(follow_up_raw, str):
        s = follow_up_raw.strip()
        if s and s.lower() not in {"null", "none", ""}:
            follow_up = s
    # Coherence: if sufficient=False but no follow_up name, treat as stop.
    if not sufficient and follow_up is None:
        sufficient = True
    return {"sufficient": sufficient, "follow_up": follow_up}
