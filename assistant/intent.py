"""
intent.py — Intent detection using the LLM.

Sends the user's transcribed speech to the LLM with a structured JSON prompt.
The LLM classifies the input into one of the defined intents and extracts
any parameters (e.g., note title, URL, filename, goal, text, query).

Intents:
  - small_talk      : casual conversation
  - unknown         : unrecognized input
  - create_note     : save a note (params: title, content)
  - open_browser    : open a URL (params: url)
  - get_time        : get current time
  - get_weather     : get weather info
  - computer_task   : agent controls the computer (params: goal)
  - read_screen     : OCR + summarize what's on screen
  - find_and_click  : find text on screen and click it (params: text)
  - code_executor   : system info / computations via code (params: goal)
  - memory_query    : recall past conversations (params: query)
"""

import json
import logging
import re
from dataclasses import dataclass, field

from . import config
from . import llm

logger = logging.getLogger("intent")


@dataclass
class IntentResult:
    """
    Structured result from intent detection.
    Mirrors the C# IntentResult class.
    """
    intent: str = "unknown"
    response: str = ""
    params: dict = field(default_factory=dict)

    def get_param(self, key: str, default: str = "") -> str:
        """Get a parameter value by key, or default if not found."""
        return self.params.get(key, default)


async def detect_intent(
    transcribed_text: str,
    scope: str | None = None,
    active_intents: set[str] | None = None,
    topic_hint: str | None = None,
) -> IntentResult:
    """
    Classify the transcribed text into an intent using the LLM.

    Args:
        transcribed_text: The user's speech as text (may be topic-resolved).
        scope: Active intent scope name from intent-scoping (e.g. "browser_mode").
        active_intents: Set of intents available in the current scope.
        topic_hint: Active topic string from topic-tracking (e.g. "Active topic: WW2").

    Returns:
        An IntentResult with the detected intent and extracted parameters.
    """
    if not transcribed_text or not transcribed_text.strip():
        return IntentResult(intent="unknown", response="I didn't catch that.")

    logger.info(f'Classifying: "{transcribed_text}"')

    try:
        user_prompt = f"User said: {transcribed_text.strip()}"
        if topic_hint:
            user_prompt += f"\n[{topic_hint}]"

        system_prompt = llm.build_intent_prompt(
            scope=scope, active_intents=active_intents,
        )

        raw_response = await llm.ask_for_intent(
            user_prompt, system_prompt=system_prompt,
        )

        logger.debug(f"LLM raw intent response: {raw_response}")

        result = _parse_intent_response(raw_response, transcribed_text)

        result = _post_correct_intent(result, transcribed_text)

        logger.info(f"Detected intent: {result.intent} | params: {result.params}")
        return result

    except Exception as e:
        logger.error(f"Intent detection error: {e}")
        return IntentResult(
            intent="unknown",
            response="Sorry, an error occurred during intent detection.",
        )


def _parse_intent_response(raw: str, original_text: str) -> IntentResult:
    """
    Parse the LLM's JSON response into an IntentResult.

    The LLM should respond with something like:
      {"intent": "create_note", "params": {"title": "Shopping", "content": "milk"}}

    If parsing fails, falls back to "small_talk" with the original text.
    """
    # Try to extract JSON from the response (LLM might add extra text around it)
    from .core.json_utils import extract_json_object
    json_str = extract_json_object(raw)

    if not json_str:
        logger.warning(f"No JSON found in LLM response: {raw}")
        return IntentResult(intent="small_talk", response=original_text)

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.warning(f"Invalid JSON from LLM: {e} | raw: {json_str}")
        return IntentResult(intent="small_talk", response=original_text)

    intent = data.get("intent", "unknown").strip().lower()
    params = data.get("params", {})

    # Validate the intent is one we know about
    if intent not in config.INTENTS:
        logger.warning(f"LLM returned unknown intent '{intent}', treating as 'unknown'")
        intent = "unknown"

    # Make sure params is a dict
    if not isinstance(params, dict):
        params = {}

    return IntentResult(
        intent=intent,
        response=original_text,  # Keep the original text as context
        params=params,
    )




# ─── Post-Classification Corrections ────────────────────────────────────────

# Pattern: "verb ... on/in [app_name]" — should never be find_and_click
from .core.known_apps import KNOWN_APPS as _KNOWN_APPS_INTENT, get_apps_by_category as _get_apps_by_cat

_app_names_for_re = sorted(
    (name for entry in _KNOWN_APPS_INTENT.items()
     for name in [entry[0]] + entry[1].aliases),
    key=len, reverse=True,
)
_APP_TARGETED_RE = re.compile(
    r'\b(?:on|in|from|with|using)\s+'
    r'(' + '|'.join(re.escape(n) for n in _app_names_for_re)
    + r'|calculator|settings|file\s*explorer|word|excel|powerpoint)\b',
    re.IGNORECASE,
)

# "click/tap/press X" = explicit GUI action → computer_task, never code_executor
_GUI_VERB_RE = re.compile(r'\b(click|tap|press|drag|scroll|type|right.?click|double.?click)\b', re.IGNORECASE)

# Guard 4: system query keywords (Bug 6)
_SYS_KEYWORDS = re.compile(
    r'\b(wifis?|bluetooth|battery|networks?|brightness|volume|display|screen'
    r'|disk|cpu|ram|memory usage|process(?:es)?)\b', re.IGNORECASE,
)
_SYS_VERBS = re.compile(
    r'\b(list|show|get|check|scan|turn\s+on|turn\s+off|enable|disable|toggle)\b',
    re.IGNORECASE,
)

# Guard 5: personal recall patterns (Bug 9)
_PERSONAL_RE = re.compile(
    r'\b(my|i have|do i|did i|what\'?s my|what did i|i told you|you know about'
    r'|i mentioned|i said|remember.{0,15}(my|i|about))\b',
    re.IGNORECASE,
)
_RECALL_RE = re.compile(
    r'\b(restrictions?|allerg|password|preference|diet|favorite'
    r'|tell you|told you|know about|remember)\b',
    re.IGNORECASE,
)
_TIME_SENSITIVE_RE = re.compile(
    r'\b(today|right now|current|latest|live|trending|recent|this week'
    r'|this month|score|price|news|weather|ip address)\b',
    re.IGNORECASE,
)


def _post_correct_intent(result: IntentResult, text: str) -> IntentResult:
    """Regex-based post-correction for 8b classifier misroutes."""
    original = result.intent
    text_lower = text.lower().strip()

    # Guard 1: find_and_click with an app name → always computer_task
    # "click X on spotify" is a GUI action targeting a specific app, not OCR-click
    if result.intent == "find_and_click":
        app_match = _APP_TARGETED_RE.search(text)
        if app_match:
            result.intent = "computer_task"
            result.params = {"goal": text_lower}
            logger.info(f"[INTENT] Post-corrected {original} → computer_task (app target: {app_match.group(1)})")

    # Guard 2: code_executor with explicit GUI verb ("click", "tap", etc.) + app → computer_task
    # "click play on spotify" is GUI, not API. The word "click" means Terminator, not spotipy.
    if result.intent == "code_executor" and _GUI_VERB_RE.search(text):
        app_match = _APP_TARGETED_RE.search(text)
        if app_match:
            result.intent = "computer_task"
            result.params = {"goal": text_lower}
            if original != result.intent:
                logger.info(f"[INTENT] Post-corrected {original} → computer_task (GUI verb + app target)")

    # Guard 3: explicit browser mention → computer_task (routes to browser backend)
    # "search weather on chrome", "check scores in firefox", etc.
    if result.intent in ("code_executor", "find_and_click", "browse_url"):
        _browser_alt = '|'.join(re.escape(b) for b in sorted(
            _get_apps_by_cat("browser"), key=len, reverse=True
        ))
        browser_match = re.search(
            rf'\b(?:on|in|using)\s+(?:the\s+)?({_browser_alt}|browser|the\s+web|google|internet)\b',
            text, re.IGNORECASE,
        )
        if browser_match:
            result.intent = "computer_task"
            result.params = {"goal": text_lower}
            if original != result.intent:
                logger.info(f"[INTENT] Post-corrected {original} → computer_task (explicit browser: {browser_match.group(1)})")

    # Guard 4: system query keywords → code_executor (Bug 6)
    if result.intent in ("file_task", "computer_task", "unknown"):
        if _SYS_KEYWORDS.search(text) and _SYS_VERBS.search(text):
            if not re.search(r'\bfile\s+\w+\.\w+', text, re.IGNORECASE):
                result.intent = "code_executor"
                result.params = {"goal": text_lower}
                if original != result.intent:
                    logger.info(f"[INTENT] Post-corrected {original} → code_executor (system query)")

    # Guard 5: personal recall → memory_query (Bug 9)
    if result.intent in ("web_search", "code_executor", "small_talk"):
        if _PERSONAL_RE.search(text) and _RECALL_RE.search(text):
            if not _TIME_SENSITIVE_RE.search(text):
                result.intent = "memory_query"
                result.params = {"query": text_lower}
                if original != result.intent:
                    logger.info(f"[INTENT] Post-corrected {original} → memory_query (personal recall)")

    return result
