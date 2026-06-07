"""
planner.py — Planner Agent for TENKA.

Orchestrates multi-step goals by decomposing them into sequential
tool invocations, passing context between steps, and handling failures.

Sits ABOVE code_executor, computer_agent, and all other tools.
Does not replace any tool — composes them.

Architecture:
  1. needs_planning()  — regex gate: does this goal need multi-step planning?
  2. _generate_plan()  — LLM decomposes goal into ordered PlanSteps
  3. executor.execute_step() — dispatches each step to the existing tool handler
  4. _step_failed()    — deterministic check: did the step actually succeed?
  5. _synthesize_result() — combines all step outputs into a final spoken response

Key design principles:
  - ZERO tool-specific code. Adding a new tool = one dict entry.
  - Step dispatch goes through actions.execute() — gets all existing
    sentinel handling (OAuth, device auth, GUI handoff) for free.
  - Context passing via $step_N string references in goal text.
  - Cascading failure: if step N fails, all steps that depend on N are skipped.
  - "synthesize" pseudo-tool: planner-internal LLM call for mid-plan analysis.
  - Interactive tool awareness: tools that need user confirmation are flagged.
"""

import logging
import time
import json
import re
from dataclasses import dataclass, field

from ...core.known_apps import KNOWN_APPS

logger = logging.getLogger("planner")


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PlanStep:
    """A single step in a plan."""
    step_id: int                    # 1-based index
    tool: str                       # tool name from TOOL_MANIFEST or "synthesize"
    goal: str                       # natural-language goal for this step
    depends_on: list[int] = field(default_factory=list)  # step_ids this depends on
    condition: str | None = None    # optional: "if $step_1 contains 'Mom'"
    status: str = "pending"         # pending | running | success | failed | skipped
    output: str = ""                # result from tool execution
    error: str = ""                 # error message if failed


@dataclass
class Plan:
    """A complete execution plan."""
    original_goal: str
    steps: list[PlanStep]
    status: str = "pending"         # pending | executing | completed | failed
    created_at: float = field(default_factory=time.time)
    context: dict = field(default_factory=dict)  # accumulated outputs keyed by step_N


# ═══════════════════════════════════════════════════════════════════════════════
#  PLAN SUSPENSION — pause mid-plan when a step needs user input
#
#  Generic design: instead of checking individual pending state variables,
#  we snapshot ALL pending states BEFORE a step runs and compare AFTER.
#  If any NEW pending state appeared, the step triggered an interactive flow.
#
#  Future-proof: adding a new pending state in actions/__init__.py automatically
#  works — no planner changes needed. Just register it via pending_registry.
# ═══════════════════════════════════════════════════════════════════════════════

_suspended_plan: Plan | None = None
_suspended_step_index: int = 0
_suspended_llm_func = None
_suspended_tts_func = None
_suspended_bridge = None


def has_suspended_plan() -> bool:
    """Check if there's a plan waiting to resume after user interaction."""
    return _suspended_plan is not None


def clear_suspended_plan() -> None:
    """Clear any suspended plan (e.g., if user changes topic)."""
    global _suspended_plan, _suspended_step_index
    global _suspended_llm_func, _suspended_tts_func, _suspended_bridge
    if _suspended_plan:
        logger.info("[PLANNER] Clearing suspended plan")
    _suspended_plan = None
    _suspended_step_index = 0
    _suspended_llm_func = None
    _suspended_tts_func = None
    _suspended_bridge = None


def _suspend_plan(plan, resume_from_index, llm_func, tts_func, bridge):
    """Save plan state for later resumption."""
    global _suspended_plan, _suspended_step_index
    global _suspended_llm_func, _suspended_tts_func, _suspended_bridge
    _suspended_plan = plan
    _suspended_step_index = resume_from_index
    _suspended_llm_func = llm_func
    _suspended_tts_func = tts_func
    _suspended_bridge = bridge
    logger.info(
        f"[PLANNER] Plan SUSPENDED at step {resume_from_index + 1}/"
        f"{len(plan.steps)} — waiting for user interaction"
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  TOOL MANIFEST — what the planner knows about available tools
#  Adding a new tool = adding one entry. No logic changes.
# ═══════════════════════════════════════════════════════════════════════════════

TOOL_MANIFEST = {
    "code_executor": {
        "description": "Run Python code to interact with APIs and services. "
                       "Handles: weather, music, email, messaging, system info, "
                       "math, data processing, volume control, any task solvable "
                       "with a single API call. NOT for tasks that need a browser "
                       "(booking, purchasing, reserving, filling web forms).",
        "param_key": "goal",
        "interactive": False,
    },
    "computer_task": {
        "description": "Control the computer via GUI — click buttons, type in fields, "
                       "navigate menus, interact with visible application windows.",
        "param_key": "goal",
        "interactive": False,
    },
    "browser_action": {
        "description": "Automate browser tasks — navigate websites, fill forms, "
                       "extract page content, click web elements, book tickets, "
                       "make reservations, purchase items, or do anything that "
                       "requires interacting with a website. Faster and more "
                       "reliable than computer_task for any web task. Opens its "
                       "own browser — does not interfere with the user's browser.",
        "param_key": "goal",
        "interactive": False,
    },
    "app_action": {
        "description": "Automate native Windows desktop applications — click buttons, "
                       "type text, read UI elements in any running app (Calculator, "
                       "Notepad, Settings, File Explorer, etc). Uses accessibility "
                       "selectors — faster and more reliable than computer_task "
                       "for tasks targeting specific app UI elements.",
        "param_key": "goal",
        "interactive": False,
    },
    "web_search": {
        "description": "Search the web for current events, news, facts, prices, scores.",
        "param_key": "query",
        "interactive": False,
    },
    "browse_url": {
        "description": "Fetch and summarize a specific webpage URL.",
        "param_key": "url",
        "interactive": False,
    },
    "file_task": {
        "description": "File operations — find, read, list, open files. "
                       "NOTE: write/rename/move/delete require user confirmation "
                       "and cannot be auto-confirmed in a plan.",
        "param_key": "goal",
        "interactive": True,  # destructive ops need confirmation
    },
    "camera_look": {
        "description": "Capture an image from the webcam and describe what is seen.",
        "param_key": "goal",
        "interactive": False,
    },
    "read_screen": {
        "description": "OCR the current screen and describe what is displayed.",
        "param_key": "goal",
        "interactive": False,
    },
    "memory_query": {
        "description": "Search past conversations and stored facts.",
        "param_key": "query",
        "interactive": False,
    },
    "create_note": {
        "description": "Save a text note to disk. Needs title and content.",
        "param_key": "goal",
        "interactive": False,
    },
    "open_browser": {
        "description": "Open a URL in the default browser.",
        "param_key": "url",
        "interactive": False,
    },
    "set_reminder": {
        "description": "Set a timed reminder.",
        "param_key": "goal",
        "interactive": False,
    },
    "recognize_face": {
        "description": "Look at the webcam and identify who is visible.",
        "param_key": "goal",
        "interactive": False,
    },
    "synthesize": {
        "description": "Analyze, summarize, extract information from, or transform "
                       "the output of previous steps using the LLM. Use when you need "
                       "to think about or process earlier results before the next step. "
                       "Example: 'extract urgent emails from $step_1' or "
                       "'summarize the key points from $step_2'.",
        "param_key": "goal",
        "interactive": False,
    },
    "vision_analyze": {
        "description": "Capture a camera image and send it to the vision LLM with a "
                       "structured prompt. Returns the LLM's interpretation. "
                       "Use for SEMANTIC understanding: identifying objects, describing "
                       "scenes, reading emotions, answering 'what is this?'. "
                       "Do NOT use for precise data extraction (exact colors, pixel "
                       "measurements, counting grid cells, reading barcodes) — use "
                       "camera_preview + code_executor for those tasks instead.",
        "param_key": "goal",
        "interactive": False,
    },
    "camera_preview": {
        "description": "Open a live camera preview window with optional overlay guides. "
                       "User aligns the target object visually, presses SPACE to capture. "
                       "Returns the file path of the captured frame. "
                       "Use with code_executor for PRECISE visual analysis: color "
                       "detection, pixel sampling, contour analysis, HSV classification, "
                       "text extraction via OCR, barcode/QR reading, measurements. "
                       "This is the deterministic vision tier — zero API calls, no "
                       "hallucination, works offline. "
                       "Overlay options: 'grid_3x3', 'grid_4x4', 'crosshair', 'rectangle'. "
                       "Mention the overlay type in the goal text. "
                       "Example: 'Open camera with 3x3 grid overlay for cube face alignment' "
                       "or 'Show camera with crosshair overlay for barcode scanning'.",
        "param_key": "goal",
        "interactive": False,
    },
    "prompt_user": {
        "description": "Speak a message to the user via TTS and pause for a few seconds "
                       "to let them perform a physical action (rotate an object, hold "
                       "something up to the camera, flip a page, etc.). The plan resumes "
                       "automatically after the pause. Use between camera_preview or "
                       "vision_analyze steps when you need the user to reposition "
                       "something. "
                       "Example: 'Now rotate the cube to show the right face' or "
                       "'Hold the next page up to the camera'.",
        "param_key": "goal",
        "interactive": False,
    },
    "store_memory": {
        "description": "Store a fact, preference, or piece of information the user wants "
                       "remembered. Use for 'remember X', 'my X is Y', 'keep in mind that'. "
                       "NOT for notes (use create_note for titled documents).",
        "param_key": "content",
        "interactive": False,
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
#  COMPLEXITY DETECTION — regex gate, zero API cost
#
#  Core idea: split on conjunctions ("and", "then") and check if BOTH
#  sides contain action verbs or action nouns. If yes → multi-step.
#  No brittle override lists that break on new phrasings.
# ═══════════════════════════════════════════════════════════════════════════════

_ACTION_VERBS = frozenset({
    "check", "read", "send", "reply", "forward", "draft", "email",
    "find", "search", "open", "download", "play", "pause", "stop",
    "take", "capture", "look", "save", "create", "write", "delete",
    "move", "rename", "set", "cancel", "summarize", "tell", "identify",
    "list", "get", "show", "describe", "analyze", "browse",
    "recognize", "remember", "forget",
    "type", "click", "press", "close", "launch", "start", "run",
    "switch", "navigate", "enable", "disable", "mute", "unmute",
    "calculate", "compute", "install", "update", "restart", "copy",
})

_STATIC_ACTION_NOUNS = frozenset({
    "weather", "forecast", "temperature",
    "email", "emails", "inbox",
    "music", "song", "songs", "playlist",
    "message", "messages",
    "photo", "camera", "picture", "webcam",
    "reminder", "alarm", "timer",
    "note", "notes",
    "file", "files", "folder", "document",
    "screen", "screenshot",
    "battery", "cpu", "ram", "disk", "volume",
})

_APP_WORDS = frozenset(
    word for name in KNOWN_APPS for word in name.split()
)
_ACTION_NOUNS = _STATIC_ACTION_NOUNS | _APP_WORDS


def needs_planning(goal: str) -> bool:
    """
    Fast check: does this goal need multi-step planning?

    Strategy:
      1. Split on conjunctions ("and", "then", "also", etc.)
      2. Check if BOTH sides contain action verbs or action nouns
      3. If yes → multi-step. If no → single-step.

    Returns False for single-tool goals (vast majority of requests).
    False positives are cheap — planner generates a 1-step plan,
    returns None, and caller falls back to normal routing.
    """
    goal_stripped = goal.strip()
    words = goal_stripped.lower().split()

    if len(words) < 5:
        return False

    goal_lower = goal_stripped.lower()

    # ── Strong multi-step signals (explicit sequencing) ────────────
    if re.search(r'\b(and then|after that|afterwards|once done|then also)\b', goal_lower):
        return True

    if re.search(r'\bif\s+.{3,60}\b(then|,)\s*\w.{3,}', goal_lower):
        return True

    # ── Primary check: conjunction splits two action clauses ───────
    for conj in ("and", "then", "also", "plus"):
        pattern = rf'\b{conj}\b'
        parts = re.split(pattern, goal_lower, maxsplit=1)
        if len(parts) == 2:
            left_words = set(parts[0].split())
            right_words = set(parts[1].split())

            # "and" appears in song/app names ("Beauty and a Beat") —
            # require verbs on BOTH sides, not just nouns.
            if conj == "and":
                left_verbs = left_words & _ACTION_VERBS
                right_verbs = right_words & _ACTION_VERBS
                if left_verbs and right_verbs:
                    logger.info(
                        f"[PLANNER] Multi-step: '{conj}' connects "
                        f"{left_verbs} ↔ {right_verbs}"
                    )
                    return True
            else:
                left_actions = (left_words & _ACTION_VERBS) | (left_words & _ACTION_NOUNS)
                right_actions = (right_words & _ACTION_VERBS) | (right_words & _ACTION_NOUNS)
                if left_actions and right_actions:
                    logger.info(
                        f"[PLANNER] Multi-step: '{conj}' connects "
                        f"{left_actions} ↔ {right_actions}"
                    )
                    return True

    # ── Secondary: comma-separated clauses with different actions ──
    clauses = [c.strip() for c in goal_stripped.split(",") if c.strip()]
    if len(clauses) >= 2:
        clause_actions = []
        for clause in clauses:
            cw = set(clause.lower().split())
            actions = (cw & _ACTION_VERBS) | (cw & _ACTION_NOUNS)
            if actions:
                clause_actions.append(actions)
        if len(clause_actions) >= 2:
            all_actions = set()
            for a in clause_actions:
                all_actions.update(a)
            if len(all_actions) >= 2:
                return True

    # ── Tertiary: 3+ distinct action verbs anywhere ────────────────
    all_verbs = set(words) & _ACTION_VERBS
    if len(all_verbs) >= 3:
        return True

    return False


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP OUTPUT VERIFICATION — deterministic, zero API cost
# ═══════════════════════════════════════════════════════════════════════════════

_FAILURE_PHRASES = [
    "couldn't access", "couldn't find", "couldn't open", "couldn't connect",
    "couldn't read", "couldn't complete", "couldn't send",
    "not available", "not connected", "not installed", "not found",
    "not configured", "not enabled", "not supported",
    "failed to", "unable to",
    "camera is currently disabled",
    "no recording", "no face", "no file",
    "llm unavailable", "__llm_unavailable__",
    "package problem:", "all retries exhausted",
    "i don't have any saved faces",
    "no transcription received",
    "sorry, i couldn't", "sorry, an error",
    "sorry, that command didn't work",
    "cancelled", "message cancelled", "aborted",
    "make up your mind", "could decide before",
    "skip the", "skipping",
    "i won't", "i'll skip",
    "no contact found", "try using a phone number",
    "contact not found", "no match found",
    "multiple contacts match",
    "no contacts found", "no matching contact",
    "didn't work", "did not work", "doesn't work",
    "no results", "no data",
    "an unexpected error occurred",
    "timed out", "connection timed out", "read timed out",
    "connection timeout", "connect timeout",
    "err_name_not_resolved", "err_connection_refused",
    "err_internet_disconnected", "err_connection_reset",
    "err_connection_closed", "err_ssl_protocol_error",
    "net::err_", "page.goto:", "locator.click:",
    "locator.fill:", "locator.select_option:",
    "error running steps", "error extracting text",
    "timeout 10000ms exceeded", "timeout 30000ms exceeded",
    "subtree intercepts pointer events",
    "404: page not found", "page not found",
]

_FAILURE_PREFIXES = (
    "ERROR:", "BLOCKED:", "TIMEOUT", "Error:", "Traceback",
    "__NEEDS_OAUTH__", "__NEEDS_DEVICE_AUTH__",
    "__CONFIRM_SEND__", "__SEND_ERROR__",
    "VERIFY_FAILED|", "APP_NOT_READY|",
)


def _step_failed(output: str) -> bool:
    """
    Determine if a step's output indicates failure.
    Uses deterministic regex/string matching — zero API cost.
    """
    if not output or output.strip() == "(no output)":
        return True

    if any(output.startswith(p) for p in _FAILURE_PREFIXES):
        return True

    output_lower = output.lower()
    for phrase in _FAILURE_PHRASES:
        if phrase in output_lower:
            return True

    return False


# ═══════════════════════════════════════════════════════════════════════════════
#  PLAN GENERATION PROMPT
# ═══════════════════════════════════════════════════════════════════════════════

_PLAN_SYSTEM_PROMPT = """\
You are a task planner for a desktop AI assistant. Given a complex goal, \
break it into sequential steps using the available tools.

AVAILABLE TOOLS:
{tool_descriptions}

RULES:
1. MINIMUM steps needed. If a goal needs only ONE tool, return a single step. \
Never over-decompose. "Read my messages" is ONE step, not "connect" + "read".
2. Each step has: tool name, goal description, and optional dependencies.
3. Use "depends_on" when a step needs output from a specific earlier step.
4. Reference previous outputs with $step_N in the goal text. \
Example: "search the web for: $step_1" uses step 1's output as the search query.
5. Use "condition" ONLY when a step should be skipped based on a previous \
step's result. Format: "if $step_N contains 'keyword'" or \
"if $step_N does not contain 'keyword'".
6. For code_executor goals: copy the user's EXACT words for that sub-task, \
with $step_N references where context from an earlier step is needed.
7. NEVER split a single API operation into multiple steps.
8. Use "synthesize" tool when you need to analyze, extract, or transform \
earlier step outputs before proceeding. It calls the LLM to think about results.
9. file_task with write/rename/move/delete will ask the user for confirmation. \
If a plan includes destructive file ops, put them as the LAST step.
10. NEVER merge separate user goals into one step. If the user says \
"send a message AND check the weather", those are TWO independent steps. \
Do NOT put weather info inside the message. Do NOT use $step_N to inject \
one task's output into an unrelated task's goal.
11. Preserve the user's intended order. If they say "do X and Y", execute X \
first, then Y. Do NOT reorder unless there is a clear dependency.
12. When passing vision_analyze or camera_look output to code_executor, ALWAYS \
add a "synthesize" step in between to normalize the data into a clean, \
parseable format. Vision output is unpredictable prose — code needs structured \
data. The synthesize step should strip all commentary and output ONLY the \
extracted data in a consistent format.
13. Use "prompt_user" when you need the user to physically do something between \
steps (rotate an object, hold up a document, move to a position). It speaks \
the message via TTS and pauses automatically. Do NOT use it for questions — \
only for physical actions.
14. PREFER camera_preview + code_executor over vision_analyze for tasks that \
need PRECISE data extraction (exact colors, pixel measurements, counting \
cells in a grid, reading barcodes/QR codes, OCR). camera_preview lets the \
user align the target with a visual overlay, captures a clean frame to disk, \
and code_executor processes it with deterministic OpenCV — zero hallucination. \
Reserve vision_analyze for SEMANTIC tasks only ("what is this?", "describe \
the scene"). When using camera_preview, the code_executor step receives the \
file path via $step_N and loads the image with cv2.imread(path). No synthesize \
step is needed between camera_preview and code_executor — the file path is \
already structured data.
15. NEVER use synthesize to reformat structured data that will be passed to \
code_executor. Code can parse raw data itself — pass $step_N references \
directly in the code_executor goal text. Use synthesize ONLY for final \
user-facing summaries. If code_executor needs results from multiple earlier \
steps, list them all in the goal: "Using data: LABEL_A=$step_2 LABEL_B=$step_5".
16. For tasks that require MULTIPLE camera captures (scanning multiple sides \
of an object, multiple pages), use the pattern: \
camera_preview → code_executor → prompt_user → camera_preview → code_executor \
→ ... → code_executor (final processing with all $step_N refs). \
Each capture+process pair handles one view. prompt_user tells the user to \
reposition between captures.
17. Maximum 3 steps per plan. If the task genuinely requires more than 3 steps, \
pack as much work as possible into steps 1-2, then end with a "synthesize" \
step whose goal is: "Summarize what was accomplished and list what still needs \
to be done to fully complete the original task." The system will automatically \
re-plan the remaining work using your summary as context.
18. When the goal contains "remember" or "keep in mind" clauses, use the \
"store_memory" tool for each fact. Do NOT use "create_note" for storing facts. \
"remember X and remember Y" = two store_memory steps. create_note is for \
titled documents, not fact storage.

{region_hint}

Respond ONLY with a JSON array of step objects. No explanation. No markdown.

Step format:
{{"step_id": 1, "tool": "tool_name", "goal": "description with $step_N refs", "depends_on": [], "condition": null}}

EXAMPLES:

Goal: "Check my messages and if Mom messaged, reply saying I'll be home by 7"
[
  {{"step_id": 1, "tool": "code_executor", "goal": "read my messages", "depends_on": [], "condition": null}},
  {{"step_id": 2, "tool": "code_executor", "goal": "send a message to Mom: I'll be home by 7", "depends_on": [1], "condition": "if $step_1 contains 'Mom'"}}
]

Goal: "What's the weather and play some music"
[
  {{"step_id": 1, "tool": "code_executor", "goal": "what is the weather", "depends_on": [], "condition": null}},
  {{"step_id": 2, "tool": "code_executor", "goal": "play some music", "depends_on": [], "condition": null}}
]

Goal: "Take a photo and search the web for what it is"
[
  {{"step_id": 1, "tool": "camera_look", "goal": "take a photo and describe what you see", "depends_on": [], "condition": null}},
  {{"step_id": 2, "tool": "web_search", "goal": "$step_1", "depends_on": [1], "condition": null}}
]

Goal: "Read my emails and tell me which ones are urgent"
[
  {{"step_id": 1, "tool": "code_executor", "goal": "read my unread emails", "depends_on": [], "condition": null}},
  {{"step_id": 2, "tool": "synthesize", "goal": "From these emails: $step_1 — which ones are urgent or need immediate attention?", "depends_on": [1], "condition": null}}
]

Goal: "Scan this document and extract the text"
[
  {{"step_id": 1, "tool": "camera_preview", "goal": "Open camera with rectangle overlay for document alignment", "depends_on": [], "condition": null}},
  {{"step_id": 2, "tool": "code_executor", "goal": "Read the image at $step_1. Use OpenCV to detect document edges, apply perspective transform, then extract text with pytesseract. Print the extracted text.", "depends_on": [1], "condition": null}}
]

Goal: "Book movie tickets for 2 people for tonight"
[
  {{"step_id": 1, "tool": "browser_action", "goal": "book movie tickets for 2 people for tonight", "depends_on": [], "condition": null}}
]
"""


# ═══════════════════════════════════════════════════════════════════════════════
#  SELF-DIAGNOSIS & RECOVERY
#
#  When a step fails, the planner asks the LLM to diagnose the failure and
#  suggest recovery steps. Only ONE re-plan attempt per failed step.
#  Uses the same tool manifest — no magic, just a different approach.
#  Cost: 1 extra 70b call per failed step (only on failure).
# ═══════════════════════════════════════════════════════════════════════════════

_REPLAN_SYSTEM_PROMPT = """\
You are a task recovery planner. A step in a multi-step plan FAILED. \
Analyze the failure and suggest recovery steps using the available tools.

AVAILABLE TOOLS:
{tool_descriptions}

FAILED STEP:
  Tool: {failed_tool}
  Goal: {failed_goal}
  Error: {failed_error}

ORIGINAL USER GOAL: {original_goal}
{region_hint}
COMPLETED STEPS SO FAR:
{completed_context}

RULES:
1. Suggest 1-3 recovery steps that could fix the problem and achieve \
the failed step's goal. Return an empty array [] if the failure is \
unrecoverable (e.g. service not set up, hardware unavailable).
2. Common recovery patterns:
   - "No active device" → add a step to open/launch the app first, then retry
   - "No contact found" → try using code_executor to search contacts with a \
broader query, then retry send with the found name/number
   - "File not found" → search with different name/location
   - "Permission denied" → try a different approach
   - "API error 4xx" → the tool already retried internally, return []
3. Do NOT retry the exact same step with the exact same goal — that already \
failed. Change the approach: add a preparatory step, use a different tool, \
modify the goal.
4. If the error is a fundamental capability issue (no API key, hardware \
disabled, service not configured), return [] — these need user action.
5. Use the same JSON step format. step_id should continue from {next_step_id}.

Respond ONLY with a JSON array of recovery steps, or [] if unrecoverable. \
No explanation. No markdown.
"""

_UNRECOVERABLE_PATTERNS = [
    "camera is currently disabled",
    "not installed",
    "llm unavailable", "__llm_unavailable__",
    "__needs_oauth__",
    "__needs_device_auth__",
    "package problem:",
    "no saved faces",
    "blocked:",
    "all retries exhausted",
]


async def _attempt_recovery(
    failed_step: PlanStep,
    plan: Plan,
    llm_func,
) -> list[PlanStep]:
    """
    Attempt to recover from a failed step.

    Asks the LLM to suggest alternative approaches. Returns a list of
    recovery PlanSteps (empty if unrecoverable).

    Only called ONCE per failed step — no recursive recovery.
    """
    error_lower = failed_step.error.lower()
    for pattern in _UNRECOVERABLE_PATTERNS:
        if pattern in error_lower:
            logger.info(
                f"[PLANNER] Skipping recovery — unrecoverable: {pattern}"
            )
            return []

    tool_desc_parts = []
    for name, info in TOOL_MANIFEST.items():
        tool_desc_parts.append(f"  - {name}: {info['description']}")
    tool_descriptions = "\n".join(tool_desc_parts)

    completed_parts = []
    for s in plan.steps:
        if s.status == "success":
            output = re.sub(
                r'^\[(?:neutral|happy|excited|sad|angry|sarcastic|worried|surprised)\]\s*',
                '', s.output
            )
            completed_parts.append(
                f"  Step {s.step_id} [{s.tool}]: {s.goal} → {output[:200]}"
            )
    completed_context = "\n".join(completed_parts) if completed_parts else "  (none)"

    max_step_id = max(s.step_id for s in plan.steps)

    from ...core.geolocation import get_cached_region, format_region_hint
    _region_hint = format_region_hint(get_cached_region())

    prompt = _REPLAN_SYSTEM_PROMPT.format(
        tool_descriptions=tool_descriptions,
        failed_tool=failed_step.tool,
        failed_goal=failed_step.goal,
        failed_error=failed_step.error[:300],
        original_goal=plan.original_goal,
        region_hint=_region_hint,
        completed_context=completed_context,
        next_step_id=max_step_id + 1,
    )

    raw = await llm_func(
        "The step failed. Suggest recovery.",
        system_prompt=prompt,
        task_type="agent_plan",
        max_tokens=400,
        temperature=0,
    )

    if raw == "__LLM_UNAVAILABLE__":
        return []

    try:
        steps_data = _extract_json_array_parsed(raw, sanitize=True)

        if not steps_data:
            logger.info("[PLANNER] No recovery steps suggested")
            return []

        recovery_steps = []
        for sd in steps_data:
            tool = sd.get("tool", "")
            if tool not in TOOL_MANIFEST:
                continue
            max_step_id += 1
            recovery_steps.append(PlanStep(
                step_id=max_step_id,
                tool=tool,
                goal=sd.get("goal", ""),
                depends_on=sd.get("depends_on") or [],
                condition=sd.get("condition"),
            ))

        if recovery_steps:
            logger.info(
                f"[PLANNER] Recovery plan — {len(recovery_steps)} steps:"
            )
            for rs in recovery_steps:
                logger.info(
                    f"  Recovery step {rs.step_id}: [{rs.tool}] {rs.goal[:80]}"
                )

        return recovery_steps

    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning(f"[PLANNER] Recovery parse error: {e}")
        return []


# ═══════════════════════════════════════════════════════════════════════════════
#  CONTEXT RESOLUTION
# ═══════════════════════════════════════════════════════════════════════════════

_STEP_REF_RE = re.compile(r'\$step_(\d+)')


def _resolve_references(text: str, plan: Plan) -> str:
    """Replace $step_N references with actual outputs from completed steps."""
    def _replace(match):
        step_id = int(match.group(1))
        for step in plan.steps:
            if step.step_id == step_id and step.status == "success":
                output = step.output
                output = re.sub(r'^\[(?:neutral|happy|excited|sad|angry|sarcastic|worried|surprised)\]\s*', '', output)
                if len(output) > 1500:
                    output = output[:1500] + "\n... (truncated)"
                return output
        return match.group(0)
    return _STEP_REF_RE.sub(_replace, text)


def _evaluate_condition(condition: str, plan: Plan) -> bool:
    """
    Evaluate a step condition like "if $step_1 contains 'Mom'".
    Returns True if the step should EXECUTE, False if it should be SKIPPED.
    """
    if not condition:
        return True

    resolved = _resolve_references(condition, plan)

    m = re.match(
        r"if\s+(.+?)\s+contains\s+['\"](.+?)['\"]",
        resolved, re.IGNORECASE
    )
    if m:
        haystack = m.group(1).lower()
        needle = m.group(2).lower()
        return needle in haystack

    m = re.match(
        r"if\s+(.+?)\s+does\s+not\s+contain\s+['\"](.+?)['\"]",
        resolved, re.IGNORECASE
    )
    if m:
        haystack = m.group(1).lower()
        needle = m.group(2).lower()
        return needle not in haystack

    logger.warning(f"[PLANNER] Unknown condition format: {condition}")
    return True


# ═══════════════════════════════════════════════════════════════════════════════
#  UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_note_params(goal: str) -> dict:
    """Extract title and content from a natural-language note goal."""
    m = re.match(r"title:\s*(.+?),\s*content:\s*(.+)", goal, re.IGNORECASE | re.DOTALL)
    if m:
        return {"title": m.group(1).strip(), "content": m.group(2).strip()}

    m = re.search(
        r"title\s+['\"](.+?)['\"](?:\s+and)?\s+content\s+['\"]?(.+)",
        goal, re.IGNORECASE | re.DOTALL
    )
    if m:
        return {"title": m.group(1).strip(), "content": m.group(2).strip().rstrip("'\"").strip()}

    m = re.search(
        r"(?:titled?|called?|named?)\s+['\"](.+?)['\"][\s,]+(?:with\s+)?(?:content\s+)?(.+)",
        goal, re.IGNORECASE | re.DOTALL
    )
    if m:
        return {"title": m.group(1).strip(), "content": m.group(2).strip()}

    words = goal.split()
    if len(words) > 5:
        return {"title": " ".join(words[:4]), "content": goal}
    return {"title": "Plan Note", "content": goal}


def _brief(text: str, max_words: int = 8) -> str:
    """Shorten a goal string for TTS announcement."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "..."


# ═══════════════════════════════════════════════════════════════════════════════
#  PLAN GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

async def _generate_plan(goal: str, llm_func) -> Plan | None:
    """Generate a plan from a natural-language goal using the LLM."""
    tool_desc_parts = []
    for name, info in TOOL_MANIFEST.items():
        tool_desc_parts.append(f"  - {name}: {info['description']}")
    tool_descriptions = "\n".join(tool_desc_parts)

    from ...core.geolocation import get_cached_region, format_region_hint
    _region_hint = format_region_hint(get_cached_region())
    prompt = _PLAN_SYSTEM_PROMPT.format(tool_descriptions=tool_descriptions, region_hint=_region_hint)

    conv_context = ""
    try:
        from ... import memory
        from ...session import get_current_session_id
        conv_context = memory.build_recent_context(
            limit=8,
            header="RECENT CONVERSATION (for reference resolution only — do NOT replay these tasks):",
            session_id=get_current_session_id(),
        )
    except Exception as e:
        logger.debug(f"[PLANNER] conversation context unavailable: {e}")

    from ...core.datetime_utils import date_context_line
    date_ctx = date_context_line()
    user_message = f"{conv_context}\n\n{date_ctx}\nGoal: {goal}" if conv_context else f"{date_ctx}\nGoal: {goal}"

    raw = await llm_func(
        user_message,
        system_prompt=prompt,
        task_type="agent_plan",
        max_tokens=2000,
        temperature=0,
    )

    if raw == "__LLM_UNAVAILABLE__":
        logger.warning("[PLANNER] LLM unavailable for plan generation")
        return None

    try:
        steps_data = _extract_json_array_parsed(raw, sanitize=True)

        if not steps_data:
            logger.warning(f"[PLANNER] Invalid plan format: {raw[:200]}")
            return None

        steps = []
        for sd in steps_data:
            tool = sd.get("tool", "")
            if tool not in TOOL_MANIFEST:
                logger.warning(f"[PLANNER] Unknown tool '{tool}' in plan — skipping")
                continue
            steps.append(PlanStep(
                step_id=sd.get("step_id", len(steps) + 1),
                tool=tool,
                goal=sd.get("goal", ""),
                depends_on=sd.get("depends_on") or [],
                condition=sd.get("condition"),
            ))

        if not steps:
            return None

        plan = Plan(original_goal=goal, steps=steps)

        interactive_steps = [
            s for s in steps
            if TOOL_MANIFEST.get(s.tool, {}).get("interactive", False)
        ]
        if interactive_steps:
            tools = ", ".join(s.tool for s in interactive_steps)
            logger.info(
                f"[PLANNER] Plan includes interactive tools: {tools} "
                f"— may require user confirmation mid-plan"
            )

        return plan

    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning(f"[PLANNER] Plan parse error: {e} | raw: {raw[:200]}")
        return None


from ...core.json_utils import extract_json_array as _extract_json_array_parsed


# ═══════════════════════════════════════════════════════════════════════════════
#  RESULT SYNTHESIS
# ═══════════════════════════════════════════════════════════════════════════════

async def _synthesize_result(plan: Plan, llm_func) -> str:
    """
    Synthesize a final spoken response from all step outputs.
    Uses Cerebras (synthesis task type) — cheap.
    """
    parts = []
    for step in plan.steps:
        if step.status == "success" and step.output:
            parts.append(f"[{step.tool}] {step.output}")
        elif step.status == "failed":
            parts.append(f"[{step.tool}] FAILED: {step.error[:150]}")
        elif step.status == "skipped":
            reason = step.error or "condition not met"
            parts.append(f"[{step.tool}] Skipped: {reason}")

    if not parts:
        return "I tried to work on that but couldn't complete any of the steps."

    results_text = "\n".join(parts)

    all_failed = all(
        s.status in ("failed", "skipped") for s in plan.steps
    )

    # Surface planner failures to telemetry so action_outcome reflects reality.
    # We mark failure when ANY step failed (not just all_failed) — partial
    # failures still indicate the plan didn't fully succeed.
    failed_steps = [s for s in plan.steps if s.status == "failed"]
    if failed_steps:
        try:
            from ... import telemetry as _telemetry
            first = failed_steps[0]
            reason = (
                f"{len(failed_steps)} step(s) failed; "
                f"first: step {first.step_id} [{first.tool}] "
                f"{(first.error or '')[:120]}"
            )
            _telemetry.mark_action_failure(
                "PlannerStepFailed" if not all_failed else "PlannerAllStepsFailed",
                reason,
            )
        except Exception:
            pass

    synth_prompt = (
        f'The user asked: "{plan.original_goal}"\n\n'
        f'Results:\n{results_text}\n\n'
        f'Give a concise natural spoken summary (2-4 sentences). '
        f'Focus on what was accomplished or what went wrong. '
        f'Do NOT list step numbers or tool names — speak naturally. '
        f'If steps failed, explain briefly what happened.'
    )

    result = await llm_func(
        synth_prompt,
        task_type="synthesis",
        max_tokens=300,
    )

    if result == "__LLM_UNAVAILABLE__":
        for step in reversed(plan.steps):
            if step.status == "success":
                return step.output
        return "Sorry, I couldn't complete that task."

    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

async def execute_plan(
    goal: str,
    llm_func,
    tts_func=None,
    bridge=None,
    _depth: int = 0,
) -> str | None:
    """
    Main entry point for the planner.

    1. Generate a plan from the goal
    2. Execute each step sequentially
    3. Verify each step's output
    4. Synthesize a final response from all step outputs

    Returns:
        Final synthesized response string, or None if the planner decides
        this is a single-step task (caller should fall back to normal routing).
    """
    from .executor import execute_step

    try:
        logger.info(f'[PLANNER] Goal: "{goal}"')

        # ── Step 1: Generate plan ──────────────────────────────────────
        plan = await _generate_plan(goal, llm_func)

        if not plan or not plan.steps:
            logger.warning("[PLANNER] Plan generation failed — falling back")
            return None

        if len(plan.steps) == 1:
            tool = plan.steps[0].tool
            step_goal = plan.steps[0].goal
            logger.info(
                f"[PLANNER] Single-step plan → bypassing planner, "
                f"direct to {tool}"
            )
            return {"bypass": tool, "goal": step_goal}

        logger.info(f"[PLANNER] Plan: {len(plan.steps)} steps")
        for s in plan.steps:
            cond = f" [if: {s.condition}]" if s.condition else ""
            deps = f" [needs: step {s.depends_on}]" if s.depends_on else ""
            logger.info(
                f"  Step {s.step_id}: [{s.tool}] {s.goal[:80]}{deps}{cond}"
            )

        # ── Step 2: Execute steps ──────────────────────────────────────
        plan.status = "executing"
        _step_idx = 0
        while _step_idx < len(plan.steps):
            from ...core.abort import abort, UserAborted
            if abort.is_aborted():
                raise UserAborted(abort.reason)
            step = plan.steps[_step_idx]
            from ...io.status_broadcaster import status, StatusPhase
            _total = len(plan.steps)
            # Detail uses the step intent (e.g. "browser_action") replacing
            # underscores with spaces. Empty if missing — step chip carries N/M.
            _intent = getattr(step, "intent", "") or ""
            _detail = str(_intent).replace("_", " ")[:32]
            status.set(StatusPhase.PLANNING,
                       detail=_detail,
                       step=(_step_idx + 1, _total))
            if step.depends_on:
                skip = False
                for dep_id in step.depends_on:
                    dep_step = next(
                        (s for s in plan.steps if s.step_id == dep_id), None
                    )
                    if dep_step and dep_step.status in ("failed", "skipped"):
                        step.status = "skipped"
                        step.error = (
                            f"dependency step {dep_id} "
                            f"{dep_step.status}: {dep_step.error[:80]}"
                        )
                        logger.info(
                            f"[PLANNER] Step {step.step_id} SKIPPED: "
                            f"{step.error}"
                        )
                        skip = True
                        break
                if skip:
                    _step_idx += 1
                    continue

            await execute_step(
                step, plan,
                llm_func=llm_func,
                bridge=bridge,
                tts_func=tts_func,
            )

            # ── Check if step is waiting for user interaction ─────────
            if step.status == "waiting":
                current_index = plan.steps.index(step)
                resume_index = current_index + 1

                if resume_index < len(plan.steps):
                    _suspend_plan(plan, resume_index, llm_func, tts_func, bridge)
                    return step.output
                else:
                    step.status = "success"
                    logger.info(
                        f"[PLANNER] Step {step.step_id} was last step, "
                        f"no suspension needed"
                    )

            # ── Mark origin as recovered if all recovery steps done ──
            if (step.status == "success"
                    and hasattr(plan, '_recovery_step_ids')
                    and step.step_id == plan._recovery_step_ids[-1]):
                origin = next(
                    (s for s in plan.steps if s.step_id == plan._recovery_origin),
                    None,
                )
                if origin and origin.status == "failed":
                    origin.status = "recovered"
                    logger.info(
                        f"[PLANNER] Step {plan._recovery_origin} marked "
                        f"'recovered' — all recovery steps succeeded"
                    )

            # ── Attempt recovery on failure ──────────────────────────
            if step.status == "failed":
                if not hasattr(plan, '_recovery_attempted'):
                    plan._recovery_attempted = False

                if not plan._recovery_attempted:
                    plan._recovery_attempted = True
                    logger.info(
                        f"[PLANNER] Attempting recovery for step "
                        f"{step.step_id}"
                    )

                    if tts_func:
                        from ...automation import verification as _ver
                        parsed = _ver.parse_verify_failed(step.error or "")
                        if parsed:
                            await tts_func(
                                f"{_ver.format_failure_for_user(parsed)} Trying again."
                            )
                        else:
                            await tts_func("Hmm, that didn't work. Let me try a different approach.")

                    recovery_steps = await _attempt_recovery(
                        step, plan, llm_func
                    )

                    if recovery_steps:
                        for i, rs in enumerate(recovery_steps):
                            plan.steps.insert(_step_idx + 1 + i, rs)
                        plan._recovery_origin = step.step_id
                        plan._recovery_step_ids = [rs.step_id for rs in recovery_steps]
                        logger.info(
                            f"[PLANNER] Inserted {len(recovery_steps)} "
                            f"recovery steps after step {step.step_id}"
                        )
                        _step_idx += 1
                        continue
                else:
                    logger.info(
                        f"[PLANNER] Recovery already attempted this plan "
                        f"— skipping for step {step.step_id}"
                    )

                for later in plan.steps:
                    if (later.status == "pending"
                            and step.step_id in later.depends_on):
                        later.status = "skipped"
                        later.error = (
                            f"dependency step {step.step_id} failed: "
                            f"{step.error[:80]}"
                        )
                        logger.info(
                            f"[PLANNER] Step {later.step_id} SKIPPED: "
                            f"dependency failed"
                        )

            _step_idx += 1

        # ── Step 3: 3D re-plan if step limit was hit ──────────────────
        plan.status = "completed"
        last_step = plan.steps[-1] if plan.steps else None
        if (
            _depth == 0
            and last_step is not None
            and last_step.tool == "synthesize"
            and last_step.status == "success"
            and len(plan.steps) >= 3
        ):
            continuation_goal = (
                f"{goal}\n\n"
                f"Progress so far: {last_step.output}\n"
                f"Continue completing the remaining work."
            )
            logger.info("[PLANNER] 3D: Step limit hit — re-planning continuation")
            continuation_result = await execute_plan(
                continuation_goal, llm_func, tts_func, bridge, _depth=1
            )
            if continuation_result:
                return continuation_result

        # ── Step 4: Synthesize final response ──────────────────────────
        return await _synthesize_result(plan, llm_func)
    finally:
        pass


async def resume_plan(interaction_result: str = "") -> str | None:
    """
    Resume a suspended plan after user interaction completes.

    Called from main.py after a pending handler resolves. Continues
    executing remaining steps from where the plan was suspended.
    """
    from .executor import execute_step

    global _suspended_plan

    if _suspended_plan is None:
        return None

    plan = _suspended_plan
    resume_from = _suspended_step_index
    llm_func = _suspended_llm_func
    tts_func = _suspended_tts_func
    bridge = _suspended_bridge

    clear_suspended_plan()

    if resume_from > 0:
        waiting_step = plan.steps[resume_from - 1]
        if waiting_step.status == "waiting":
            if interaction_result and not _step_failed(interaction_result):
                waiting_step.status = "success"
                waiting_step.output = interaction_result
                plan.context[f"step_{waiting_step.step_id}"] = interaction_result
                logger.info(
                    f"[PLANNER] Suspended step {waiting_step.step_id} "
                    f"resolved: SUCCESS"
                )
            else:
                waiting_step.status = "failed"
                waiting_step.error = interaction_result[:300] if interaction_result else "cancelled"
                waiting_step.output = interaction_result or ""
                logger.info(
                    f"[PLANNER] Suspended step {waiting_step.step_id} "
                    f"resolved: FAILED"
                )

    logger.info(
        f"[PLANNER] Resuming plan from step {resume_from + 1}/"
        f"{len(plan.steps)}"
    )

    if tts_func:
        remaining = len(plan.steps) - resume_from
        await tts_func(
            f"Alright, continuing. "
            f"{remaining} step{'s' if remaining != 1 else ''} left."
        )

    # ── Continue executing remaining steps ─────────────────────────
    try:
        for i in range(resume_from, len(plan.steps)):
            step = plan.steps[i]

            if step.depends_on:
                skip = False
                for dep_id in step.depends_on:
                    dep_step = next(
                        (s for s in plan.steps if s.step_id == dep_id), None
                    )
                    if dep_step and dep_step.status in ("failed", "skipped"):
                        step.status = "skipped"
                        step.error = (
                            f"dependency step {dep_id} "
                            f"{dep_step.status}: {dep_step.error[:80]}"
                        )
                        skip = True
                        break
                if skip:
                    continue

            await execute_step(
                step, plan,
                llm_func=llm_func,
                bridge=bridge,
                tts_func=tts_func,
            )

            if step.status == "waiting":
                current_index = plan.steps.index(step)
                next_index = current_index + 1
                if next_index < len(plan.steps):
                    _suspend_plan(plan, next_index, llm_func, tts_func, bridge)
                    return step.output
                else:
                    step.status = "success"

            # Attempt recovery on failure in resumed plan too
            if step.status == "failed":
                if not hasattr(plan, '_recovery_attempted'):
                    plan._recovery_attempted = False

                if not plan._recovery_attempted:
                    plan._recovery_attempted = True
                    logger.info(
                        f"[PLANNER] Attempting recovery for step "
                        f"{step.step_id} (in resumed plan)"
                    )
                    recovery_steps = await _attempt_recovery(
                        step, plan, llm_func
                    )
                    if recovery_steps:
                        for rs in recovery_steps:
                            await execute_step(
                                rs, plan,
                                llm_func=llm_func,
                                bridge=bridge,
                                tts_func=tts_func,
                            )
                            if rs.status == "waiting":
                                next_rs_idx = plan.steps.index(rs) + 1
                                if next_rs_idx < len(plan.steps):
                                    _suspend_plan(plan, next_rs_idx, llm_func, tts_func, bridge)
                                    return rs.output
                            if rs.status == "failed":
                                break
                        continue

                for later in plan.steps:
                    if (later.status == "pending"
                            and step.step_id in later.depends_on):
                        later.status = "skipped"
                        later.error = (
                            f"dependency step {step.step_id} failed: "
                            f"{step.error[:80]}"
                        )

        plan.status = "completed"
        return await _synthesize_result(plan, llm_func)
    finally:
        pass
