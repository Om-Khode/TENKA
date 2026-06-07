# TENKA — Architecture

The public architecture overview. Companion to the [README](./README.md) (what TENKA is), [CONFIGURATION.md](./CONFIGURATION.md) (runtime knobs), [SETUP.md](./SETUP.md) (how to install), and [CONTRIBUTING.md](./CONTRIBUTING.md) (how to change code without breaking things). This document explains *how* TENKA is built and *why* it's built that way.

If you're forking or contributing, read this end-to-end at least once. If you're just curious, the [first three sections](#1-overview) are enough.

---

## 1. Overview

**TENKA** — **T**ransformative **E**volving **N**eural **K**inetic **A**gent — is a standalone Python voice agent for Windows. The name is the thesis: **she evolves without dependency**. Adding support for a new app, service, or domain must not require code changes. TENKA discovers, learns, and adapts to it at runtime.

TENKA exposes a TCP wire protocol that any frontend can implement. The reference frontend is the upstream **Mate Engine** Unity avatar project, shipped and licensed separately. TENKA itself contains no avatar, no rendering, no Unity code — only the brain.

### Tech stack

| Layer | Technology |
| --- | --- |
| Language | Python 3.11.9 (asyncio-heavy, Windows-only) |
| LLM providers | Gemini (primary), Groq + Cerebras (cloud fallback), Ollama (local) |
| STT | faster-whisper or whisper.cpp |
| TTS | Kokoro + optional vocal voice post-processing (pitch shift, EQ, tremolo) |
| Wake word | openWakeWord (ONNX/TFLite) |
| Speaker verification | ECAPA-TDNN (SpeechBrain) |
| Storage | SQLite (single DB) + FAISS (semantic memory) |
| Browser automation | Playwright (bundled Chromium) + CDP |
| Native automation | Terminator (Windows UI Automation tree) |
| Vision automation | pyautogui + LLM vision (fallback only) |
| Frontend | Any TCP client. Reference impl: Mate Engine Unity. |

### Wire protocol

| Port | Direction | Format | Purpose |
| --- | --- | --- | --- |
| 7777 | TENKA → frontend | length-prefixed JSON | expressions, animations, subtitles |
| 7778 | frontend → TENKA | length-prefixed JSON | listening triggers, chat input, clicks |
| 7780 | external → TENKA | HTTP | messaging-bridge adapter API |

A frontend doesn't need to implement all three. Run with `UNITY_ENABLED=false` in `.env` to use TENKA in terminal-only mode (no TCP, subtitles echo to console).

---

## 2. Core design principles

These principles are load-bearing. Every architectural choice in TENKA derives from them.

### 2.1 Generic over specific (THE-rule)

**No hardcoded app-specific rules or functions. All solutions must be generic and future-proof.**

If a user switches from Spotify to YouTube Music to a local player, TENKA must adapt with zero code changes.

- ❌ `if app == "spotify": handle_spotify_play(...)` — never.
- ❌ A module named `spotify_handler.py`.
- ❌ A regex that mentions a brand name.
- ✅ A generic `play_media(query, service=None)` that dispatches by user preference or runtime discovery.
- ✅ A row in `core/known_apps.py`'s `_KNOWN_APPS` data table.
- ✅ A user-taught procedure that captures "this is how Spotify works for me" without TENKA's source code knowing the word "Spotify".

**Behavior is data; code is mechanism.** When a feature seems to need an app-specific branch, the right answer is almost always to lift it into a data row or fall through to a generic mechanism (browser DOM, native AT tree, vision agent).

### 2.2 Regex-first before LLM

Any pattern that can be parsed structurally is parsed structurally. LLMs are for ambiguity, not for parsing `(\d{1,2}):(\d{2})\s*(am|pm)`. Regex is free, deterministic, and never rate-limited. `assistant/regex_router.py` is where most "TENKA understood me instantly" magic happens.

### 2.3 Code-level fixes over prompt-level

When a behavior is wrong, fix the code that constructs the prompt — not the prompt itself. Prompts drift; code is testable. Reach for prompt edits only when the model fundamentally misunderstands the task and no amount of input shaping helps.

### 2.4 The LLM never computes datetimes

All date math happens in Python. Pass literal strings into prompts. LLMs hallucinate dates with high confidence; Python's `datetime` does not.

### 2.5 Save API calls

Free-tier quotas are real and small. Cache, dedupe, and skip the LLM whenever a deterministic path exists. The provider chain is shaped to use the cheapest capable model for each task.

### 2.6 Diagnostic over speculation

When a partial fix doesn't resolve an issue, don't iterate on guesses. Add INFO logs, re-run, read the log line by line, then fix from evidence.

---

## 3. LLM provider strategy

TENKA uses **Gemini as primary** with **Groq + Cerebras free tiers as defensive fallbacks**, and Ollama for offline survival. The chain ensures TENKA stays functional even without a paid LLM key.

### Provider selection rationale

| Provider | Role | Why |
| --- | --- | --- |
| **Gemini** | Primary (text + vision) | Unified API, generous free tier. Flash-Lite is the cheapest capable model. |
| **Groq** | Defensive fallback | Free tier is real and fast. Forks work without a paid Gemini key. Vision via llama-4-scout. |
| **Cerebras** | Defensive fallback | High RPM ceiling on `gpt-oss-120b`. Synthesis + KG extraction. |
| **Ollama** | Local last resort | Offline survival. Quality drops but TENKA stays alive. |

### Task-to-model mapping

| Task | Primary | Fallback chain |
| --- | --- | --- |
| `code_gen` | gemini-2.5-flash | groq kimi-k2 → llama-3.3-70b → qwen3-32b |
| `intent` | gemini-2.5-flash-lite | groq llama-3.1-8b-instant |
| `agent_plan` | gemini-2.5-flash | groq llama-3.3-70b-versatile |
| `agent_verify` | gemini-2.5-flash | groq llama-4-scout-17b |
| `small_talk` | gemini-2.5-flash | groq llama-3.3-70b-versatile |
| `synthesis` | gemini-2.5-flash-lite | cerebras gpt-oss-120b |
| `kg_extraction` | gemini-2.5-flash-lite | cerebras gpt-oss-120b → groq 8b-instant |
| `default` | gemini-2.5-flash-lite | cerebras gpt-oss-120b |
| `vision` | gemini-2.5-flash (vision) | groq llama-4-scout-17b |

**Flash vs Flash-Lite:** Flash handles tasks that benefit from reasoning capacity (code generation, planning, reflection, small talk, verification). Flash-Lite is the cheaper, faster route for intent / synthesis / default — sufficient for those task shapes.

**Vision dispatch:** the `vision` task bypasses `TASK_MODEL_MAP` and uses a dedicated `get_vision_response()` in `llm/router.py` with its own provider chain.

**All LLM calls go through `llm/contracts.py` task-shaped wrappers** (`ask_for_intent`, `ask_for_plan`, `ask_for_synthesis`, …). Handlers never call `llm/router.py` directly — that's an internal boundary.

---

## 4. Layered architecture

TENKA is a strictly layered Python package with one orchestrating pipeline. Each layer depends only on the layers below it.

```
┌──────────────────────────────────────────────────────────────┐
│  main.py — pipeline orchestrator (the only place that wires) │
└──────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────┐   ┌──────────────┐   ┌────────────┐
│  actions/   │   │ automation/  │   │    io/     │
│ (handlers)  │   │ (3 stacks)   │   │ (audio,TCP)│
└─────────────┘   └──────────────┘   └────────────┘
        │                 │                 │
        ▼                 ▼                 │
┌──────────────────────────────────┐        │
│ domain modules (top-level .py)   │        │
│ intent, policy, personality,     │        │
│ preferences, memory              │        │
└──────────────────────────────────┘        │
        │                                    │
        ▼                                    │
┌──────────┐   ┌──────────┐                  │
│   llm/   │   │ storage/ │                  │
└──────────┘   └──────────┘                  │
        │           │                        │
        ▼           ▼                        ▼
┌─────────────────────────────────────────────┐
│        core/ + config.py (no deps)          │
└─────────────────────────────────────────────┘
```

### Pipeline (`main.py`)

```
audio/text input
  → shortcut check (zero-API trigger match)
  → speaker verification (if enrolled)
  → intent detection
  → pending-state handlers (multi-turn dialogs)
  → policy (which action to dispatch)
  → action handler (actions/<intent>.py)
      → optional: automation/* (browser / native / vision)
      → optional: llm/contracts (ask_for_*)
  → memory write + knowledge-graph extraction
  → personality event bumps
  → preference correction detection
  → TTS
```

---

## 5. Folder structure

```
assistant/
  main.py                  # The only place that wires io/ to handlers.
  config.py                # Static settings, intents, emotion profiles.

  core/                    # Cross-cutting utilities. Imports nothing else.
    registry.py            # RegistryBase[T] — generic thread-safe registry
    known_apps.py          # _KNOWN_APPS data table — THE-rule's extension point
    runtime_config.py      # Three-tier setting resolution: DB → env → default
    asyncio_utils.py       # call_maybe_async() and friends
    datetime_utils.py      # Date parsing and math
    abort.py               # Universal abort controller (ESC → propagates)

  llm/                     # LLM dispatch.
    router.py              # get_llm_response + TASK_MODEL_MAP + fallback chains
    contracts.py           # Task-shaped wrappers: ask_for_intent, ask_for_plan, …
    prompts.py             # System prompt templates
    providers/             # Provider registry. Each self-registers on import.

  storage/                 # Persistence. Single SQLite + per-domain repos.
    db.py                  # Singleton DB, WAL mode, versioned schema migrations
    repos/                 # personality, preference, procedure, settings,
                           # shortcut, memory, knowledge_graph, schedule, monitor,
                           # automation_cache, …

  io/                      # External boundaries. NEVER imports actions/ or main.py.
    unity_bridge.py        # Length-prefixed JSON over TCP (7777/7778)
    messaging_bridge.py    # HTTP server (7780)
    screen.py              # Screen capture (mss) + OCR
    audio/                 # stt, tts, wake_word, speaker_verify, streaming
    overlay/               # Cursor visibility overlay (status pill)
    esc_monitor.py         # Universal abort key listener

  automation/              # Three "do a step on screen" stacks + shared layer.
    router.py              # Preference → URL regex → process → keyword routing
    manifest_dispatcher.py # Learned per-app UI manifests (the "evolving" core)
    manifest_runtime.py    # Singleton holder + Terminator adapter
    manifest_store.py      # YAML serde for manifests under ~/TENKA/manifests/
    manifest_primitives.py # Send-key / find-element / click primitives
    promoter.py            # Cold-path: cluster successful caches → manifests
    healer.py              # Selector self-healing (UI fingerprint + vision)
    recovery.py            # Shared recovery policy across tiers
    verification.py        # Shared verifier interface
    step_cache.py          # Automation step caching (zero-cost replay)
    vision_cap.py          # Daily vision-call cap counter
    native.py              # Terminator wrapper (Windows UI Automation tree)
    event_bus.py           # Event-driven monitor pump
    event_sources/         # Media (SMTC), Window (WinEventHook), …
    browser/               # Playwright + CDP
    vision/                # Vision-loop fallback (computer_task last resort)

  actions/                 # Intent handlers. Self-register via @tool_registry.decorator().
    registry.py            # tool_registry = RegistryBase[Handler]("tool")
    simple.py              # small_talk, get_time, create_note, reminders, …
    da_handlers.py         # computer_task, planner, browser/app action routing
    web.py                 # web_search, browse_url
    file_ops.py            # file_task (sandbox-safe)
    camera.py              # camera_look, meet/recognize/forget_face
    voice.py               # enroll_voice, forget_voice
    memory_search.py       # memory_query, store_memory, forget_memory
    recording.py           # start/stop/get/summarize_recording
    schedule.py            # manage_schedule (cron-style)
    monitors.py            # manage_monitor (event subscriptions)
    procedures.py          # manage_procedure (taught recipes)
    shortcuts.py           # manage_shortcut
    teaching.py            # Interactive teach-me-X sessions
    manifest_dispatch.py   # Synthetic handler for manifest-routed phrases
    planner/               # Multi-step goal orchestration
    pending_handlers.py    # Multi-turn confirmations
    browser_cdp_setup.py   # CDP attach helper

  code_executor/           # Sandboxed Python execution pipeline
  personalities/           # Swappable personality bases
  models/                  # ML model files (ONNX/TFLite). Read-only at runtime.

  # Top-level domain modules — small, cohesive, no folder needed
  intent.py policy.py personality.py preferences.py memory.py
  knowledge.py knowledge_graph.py procedures.py reminders.py
  recording.py settings.py shortcuts.py pending.py proactive.py
  reflection.py regex_router.py slash_commands.py credentials.py
  topic_tracker.py event_monitoring.py telemetry.py scheduler.py
  overlay_manager.py faces.py

tests/                     # All tests live here. None at repo root, none in assistant/.
```

---

## 6. Three-tier desktop automation

The most distinctive piece of TENKA. When she needs to act on the screen, three tiers escalate in order of cost.

| Tier | Mechanism | Cost | When |
| --- | --- | --- | --- |
| **1. `browser_action`** | Playwright + CDP, DOM-level | Zero vision calls | Websites — booking, search, form fill |
| **2. `app_action`** | Terminator (Windows UI Automation tree) | Zero vision calls | Native apps with good accessibility (Settings, Notepad, File Explorer, Calculator) |
| **3. `computer_task`** | pyautogui + LLM vision loop | 3–10 vision calls per task | Last resort — anything the cheaper tiers can't handle |

### Routing is zero-LLM-cost

`assistant/automation/router.py` picks the tier without an LLM call: user preferences → URL regex → running process → launch keyword → fallback. No regex anywhere mentions an app name (the dispatch is data-driven).

### Manifest learning (the "evolves" part)

On top of the three tiers, TENKA learns **per-app manifests** from observation. When tier 2 successfully drives an app twice (configurable promotion gate), the patterns are clustered into a YAML manifest under `~/TENKA/manifests/` and the manifest dispatcher becomes the new fast path. Next time, that app's selector chain runs at tier-1 cost — zero LLM calls, zero vision calls.

Selectors self-heal via `automation/healer.py`:
- **Tier-1 heal:** UI-tree fingerprint similarity (`at_fingerprint.py`).
- **Tier-2 heal:** if tier-1 misses, fall back to an LLM-vision re-ground, then re-resolve a stable selector.

When manifests don't cover a goal or the dispatcher escalates, the handler falls through to `computer_task` — the user still gets their result.

---

## 7. Registry system

All pluggable components self-register via `RegistryBase[T]` from `core/registry.py`. Adding a new component = one file + one `register("key", obj)` call. No central file edits.

| Registry | Location | Type | Used for |
| --- | --- | --- | --- |
| `tool_registry` | `actions/registry.py` | Handler | Intent handlers |
| `provider_registry` | `llm/providers/__init__.py` | Provider | Gemini, Groq, Cerebras, Ollama |
| `channel_registry` | `io/channels/__init__.py` | Channel | WhatsApp + future adapters |
| `source_registry` | `automation/event_sources/__init__.py` | EventSource | SMTC media, Window events |

### Example: adding a Telegram adapter

```python
# io/adapters/telegram.py
from ..channels import channel_registry

class TelegramAdapter:
    name = "telegram"
    async def send(self, message, recipient=None): ...
    async def start(self): ...
    async def stop(self): ...
    def execute(self, action, params): ...

channel_registry.register("telegram", TelegramAdapter())
# Done. No other files need editing.
```

---

## 8. Layering and import boundaries

```
core/  →  config  →  storage/, llm/  →  domain  →  automation/  →  actions/  →  main.py
```

`io/` is parallel to domain and may import `core/`, `config` only. **`io/` may NEVER import `actions/` or `main.py`.** `config.py` may NEVER read SQLite directly.

These boundaries are enforced by **`import-linter`** via `pyproject.toml` and run on every commit through a pre-commit hook. Violations are CI-blocking. Run `lint-imports` locally before pushing.

---

## 9. Knowledge graph

TENKA's memory is more than vector-similarity search. The knowledge graph (under `assistant/knowledge_graph.py` + `storage/repos/knowledge_graph.py`) holds:

- **Entities** — people, places, things mentioned in conversation
- **Facts** — typed assertions about entities ("X is my brother", "Y costs $50")
- **Relationships** — typed edges between entities
- **Commitments** — promises with optional due times ("I'll call her tomorrow")
- **Temporal grounding** — when facts were asserted, and their validity windows

Retrieval is hybrid (FAISS semantic + SQLite FTS5 with RRF fusion) and multi-hop (chain-of-thought style expansion). Extraction runs after every meaningful turn and is gated to avoid wasting API calls on trivial inputs. Contradiction handling is structural: new facts that conflict with old ones invalidate the old, with provenance preserved for audit.

This subsystem is its own world — see `assistant/knowledge_graph.py` and `assistant/storage/repos/knowledge_graph.py` for the canonical shape.

---

## 10. Personalities

Personality is data, not code. Three bases ship:

- `warm_honest` (default) — direct, empathetic, no filler
- `tsundere` — prickly, secretly fond
- `minimal` — terse, low-resource

Each personality is a folder under `assistant/personalities/<name>/` with a `traits.json`, a system prompt fragment, and an optional opener pool. They're swappable at runtime via `/set active_personality <name>` and a restart.

Per-axis modifiers (openness, patience, playfulness, sass, trust, warmth) layer on top of the base. **Nightly reflection** (`assistant/reflection.py`) reads the day's conversations and updates the axis values to drift TENKA toward how you actually interact with her — within bounds the base allows.

The personality system never touches business logic. It only shapes the system-prompt prefix and TTS phrasing.

---

## 11. Where data lives

| Path | Purpose | Lifetime |
| --- | --- | --- |
| Repo root | Code, config templates, `.gitmessage`, `LICENSE`, docs | Tracked in git |
| `.env` (repo root) | Your local secrets — keys, region, model overrides | gitignored, per-machine |
| `.tenka_setup.json` (repo root) | Setup wizard's progress marker (schema-versioned) | gitignored, per-machine |
| `~/TENKA/memory/tenka.db` | Single SQLite — facts, settings, schedules, monitors, manifests metadata, automation cache | Persistent runtime data |
| `~/TENKA/memory/*.faiss` + `*.idmap.json` | FAISS indexes for memory, recordings, facts | Persistent runtime data |
| `~/TENKA/memory/voiceprint.npz` | Speaker enrollment | Persistent runtime data |
| `~/TENKA/manifests/` | Per-app UI manifests (YAML, learned) | Persistent runtime data |
| `~/TENKA/Notes/` | `create_note` output (Markdown) | Persistent runtime data |
| `~/TENKA/Sessions/` | Session transcripts (one per conversation) | Persistent runtime data |
| `~/TENKA/browser-cache/` | Playwright Chromium | Persistent runtime data |

Everything under `~/TENKA/` is safe to delete to start over. `.env` and `.tenka_setup.json` live at the repo root, not under `~/TENKA/`.

---

## 12. Adding a feature — the right path

When you want TENKA to do something new, ask in this order:

1. **Can it be a preference?** Try `/set <key> <value>` or add a row to a preference repo.
2. **Can it be a taught procedure?** Open a teaching session and walk TENKA through it once. The recipe lives as data afterward.
3. **Can it be a `_KNOWN_APPS` row?** If you're teaching the dispatcher to recognize a new browser/app, one row in `core/known_apps.py` covers it.
4. **Can it be a YAML manifest?** Write it by hand at `~/TENKA/manifests/<app>.yaml` (the schema is in `assistant/automation/manifest_schema.py`) or let TENKA learn it from N successful runs of the cheap-tier automation.
5. **Can it be a new event source / channel / provider?** Drop a file in `event_sources/`, `io/adapters/`, or `llm/providers/`. Self-register. No central edits.
6. **New intent.** Only if none of the above fit. Add the handler under `actions/`, register it, add a regex fast-path in `regex_router.py` if applicable, list it in `config.py`'s allowed-intent set, and add it to `TENKA_Capabilities.md`. Write a test.

The bias is strong toward 1–5. The architecture is shaped to make 6 rarely necessary.

---

## 13. Anti-patterns

These are the things to *not* do:

- **THE-rule violations.** Hardcoded app names, brand-specific regexes, app-specific modules. See section 2.1.
- **Direct LLM calls from handlers.** Always go through `llm/contracts.py`.
- **Direct SQLite from `config.py`.** Settings resolution happens in `core/runtime_config.py` lazily, not at import.
- **Cross-layer imports.** Don't import from a higher layer into a lower one. Run `lint-imports` if unsure.
- **New top-level folders under `assistant/`.** The nine subpackages are the contract.
- **Tests outside `tests/`.** Never at repo root, never inside `assistant/`.
- **Prompt-level fixes for code-level problems.** If the model gets the input wrong, fix the input builder.
- **Skipping the regex check** when the pattern is structural. LLMs are not a parser of last resort.

---

## 14. Glossary

- **Manifest** — a YAML file under `~/TENKA/manifests/` describing how to drive a specific app via UI Automation. Learned, not coded.
- **Intent** — a high-level user goal mapped to a handler (e.g. `web_search`, `planner`). Listed exhaustively in [TENKA_Capabilities.md](./TENKA_Capabilities.md).
- **THE-rule** — the non-negotiable principle that TENKA contains no app-specific code. See 2.1.
- **Three-tier automation** — the escalation pattern `browser_action → app_action → computer_task`. See section 6.
- **Pending state** — a multi-turn dialog flow (OAuth, confirmation, disambiguation) tracked by `pending.py`.
- **Healer** — the layer that recovers from broken selectors via UI fingerprint similarity and vision re-grounding.
- **Promoter** — the cold-path job that clusters cached automation steps into reusable manifests.
- **Reflection** — nightly LLM pass that updates personality trait values from the day's conversations.

---

## 15. References inside the repo

- [README.md](./README.md) — what TENKA is, quickstart, headline capabilities.
- [SETUP.md](./SETUP.md) — install walkthrough + troubleshooting.
- [CONFIGURATION.md](./CONFIGURATION.md) — every runtime knob, explained.
- [TENKA_Capabilities.md](./TENKA_Capabilities.md) — the full intent catalogue.
- [TENKA_Known_Issues.md](./TENKA_Known_Issues.md) — current punch list.
- [CONTRIBUTING.md](./CONTRIBUTING.md) — how to send a PR that lands cleanly.
- [LICENSE](./LICENSE) — Apache 2.0.
- [NOTICE](./NOTICE) — attribution + bundled third-party licenses.
