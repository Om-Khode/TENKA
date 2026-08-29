---
paths:
  - "assistant/llm/**"
  - "assistant/config.py"
  - "assistant/intent.py"
  - "assistant/regex_router.py"
  - "assistant/actions/**"
  - "assistant/personality.py"
  - "assistant/reflection.py"
  - "assistant/personalities/**"
---

# LLM dispatch, intents, and personality

## Dispatch

All LLM calls go through `llm/contracts.py` task-shaped wrappers (`ask_for_intent`,
`ask_for_plan`, `ask_for_synthesis`, …). **Never call `llm.router.get_llm_response`
directly from a handler.** Providers live in `llm/providers/` and self-register via
`provider_registry`.

Full task-to-model table: `ARCHITECTURE.md` §3; authoritative source is
`llm/router.py:TASK_MODEL_MAP`. Operationally:

- **Gemini Flash** (thinking disabled): `code_gen`, `agent_plan`, `personality_reflection`,
  `small_talk`, `agent_verify`.
- **Gemini Flash-Lite**: `intent`, `synthesis`, `kg_extraction`, `kg_followup`, `default`.
- **Vision**: dedicated `get_vision_response()`. **Not** in `TASK_MODEL_MAP` — there is no
  `"vision"` key.
- **Groq + Cerebras**: defensive fallbacks so public forks work without a Gemini key. Groq
  vision is `llama-4-scout` only.
- **Save API calls.** Free-tier quotas are small. Cache, dedupe, and skip the LLM whenever a
  deterministic path exists.

Deterministic paths that already exist and must not be replaced by an LLM call:
`regex_router.pre_route()` (~40–50% of daily commands, zero LLM) and
`planner.needs_planning()` (conjunction split + action-verb sets, zero LLM).

## Intents

`config.INTENTS` is the single source of truth (40 entries); `ALLOWED_INTENTS = set(INTENTS)`
is the policy whitelist.

```
small_talk, unknown, create_note, open_browser, get_time, computer_task, read_screen,
find_and_click, code_executor, memory_query, start_recording, stop_recording, get_recording,
summarize_recording, web_search, browse_url, file_task, set_reminder, cancel_reminder,
hide_avatar, show_avatar, meet_face, recognize_face, forget_face, camera_look, planner,
manage_shortcut, manage_procedure, manage_schedule, manage_monitor, manage_backup,
enroll_voice, forget_voice, browser_extension_setup, browser_tabs, store_memory,
forget_memory, shutdown, manifest_dispatch, self_knowledge
```

`browser_action` and `app_action` are registered handlers but **internal routing targets** —
correctly absent from `INTENTS`, reached only from an intent that already passed the gate.

**Adding an intent touches five places in sync:** `config.INTENTS`, the intent system
prompt's catalogue (also `config.py`), `TENKA_Capabilities.md`,
`core/intent_capabilities.py` — **twice**: `REQUIRED_CAPABILITY`, and exactly one of
`PERSISTS_AUTHORITY` / `TRANSIENT_AUTHORITY` — and `core/intent_scopes.py` when the intent
only makes sense in one scope. The capability entries are required by the fail-closed
default: an unclassified intent requires `EXECUTE` and is refused over every transport, and
the authority pair is exhaustive with no default in either direction.

**A new row competes with the ones already there.** `browser_tabs` first claimed an `open`
action; `open_browser` already opened a URL in the user's default browser, so two rows
described one request and the classifier picked neither — sending "open a tab for X" to a
GUI vision loop. Before adding a verb, check nothing else already answers it, and say in
the row what the intent is *not* for.

Easy to confuse: `manage_schedule` = time-based/cron. `manage_monitor` = event-driven.
`set_reminder` = one-time alert. `manage_backup` = TENKA's own data durability, not a user
automation.

## Handler conventions

`handle_<intent_name>`, self-registered via `@tool_registry.decorator("intent")`. Dispatch
in `actions/__init__.py` uses `tool_registry.get(intent)`. Pending-state handlers:
`handle_pending_<state>`. No `_tool_*` / `_handle_*` anywhere. The `_TOOLS` dict no longer
exists — `tool_registry` replaced it in the RG-1 refactor.

`from ..io.audio import tts` goes **inside** handler functions, not at module top. The
original import cycle is gone, but the deferred-import convention is what the codebase does
everywhere.

## Prompt hygiene

- **The LLM never computes datetimes.** All date math in Python
  (`core/datetime_utils.py`); pass literal strings into prompts.
- **Groq's 8b intent router ignores system prompts.** Only bites on the Groq fallback now.
  Critical hints go in the user message with an `IMPORTANT:` prefix.
- **TTS messages: <120 chars, no paths, no error codes.** Long error strings become
  50-second audio nightmares.

## Personality

Personality changes **how** she communicates, never **what is true**:

```
system truth → personality → expression        never        personality → truth
```

She never claims a capability she lacks, never claims success without verification, and
stays in character while refusing, failing, or expressing uncertainty. Avoid unnecessary
"as an AI" disclaimers on ordinary character questions; answer honestly if asked directly.

Personality evolution must be bounded, evidence-based, reversible, inspectable. Do not
expand nightly self-modification. Note that `reflection.py` writes **preferences**, which
`automation/router.py` reads as routing priority 1 gated only on confidence — the `source`
field is never consulted (KI-31).
