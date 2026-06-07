# TENKA Capabilities

The full list of intents TENKA can dispatch on, grouped by what they're for. Each row is one entry in `assistant/regex_router.py` + a handler under `assistant/actions/`. Most intents have a regex fast-path (zero LLM cost) and fall back to the LLM classifier when the wording is ambiguous.

If you find yourself wanting to add a new intent, read [Adding capabilities](#adding-capabilities) at the bottom first. Most "new intent" ideas are better expressed as preferences, taught procedures, or app manifests — TENKA is designed to grow as data, not as code.

---

## Voice & interaction

| Intent | What it does |
| --- | --- |
| `small_talk` | Free-form conversation with personality. The default fallback when nothing else matches. |
| `get_time` | "What time is it?" — answered in Python, never by the LLM. |
| `hide_avatar` | Hide the Unity avatar (if a frontend is connected). |
| `show_avatar` | Show the Unity avatar again. |
| `shutdown` | "Shut down" / "go to sleep" — graceful exit, snapshots the session first. |

---

## Memory & knowledge

| Intent | What it does |
| --- | --- |
| `memory_query` | "What did I tell you about X?" — hybrid retrieval over the knowledge graph + FAISS semantic memory. |
| `store_memory` | "Remember that I prefer dark mode" — explicit fact insertion. |
| `forget_memory` | "Forget what I said about X" — soft-deletes the matching facts. |
| `create_note` | "Take a note: …" — writes a Markdown file to `~/TENKA/Notes/`. |

---

## Web & search

| Intent | What it does |
| --- | --- |
| `web_search` | Tavily-backed search. Returns a short synthesized answer with citations. |
| `browse_url` | Fetch and summarise a specific URL. |
| `open_browser` | Just open a URL in your real browser (no automation). |

---

## Productivity

| Intent | What it does |
| --- | --- |
| `set_reminder` | "Remind me to call mom at 6 pm" — stored in SQLite, polled every 10 s. |
| `cancel_reminder` | "Cancel my 6 pm reminder." |
| `manage_shortcut` | Add, list, or delete voice shortcuts ("when I say X, do Y"). |
| `manage_procedure` | Add, list, edit, or delete taught procedures. Multi-step recipes TENKA learns from you. |
| `manage_schedule` | Cron-style recurring tasks with optional conditions — "every weekday at 8 am, if I have a meeting today, remind me." Create, list, cancel, or toggle. |
| `manage_monitor` | Subscribe to real-time desktop events — "tell me when my media player changes track" or "ping me when this window appears." Backed by event sources (SMTC media, WinEventHook, etc.). |

---

## Recording & summarisation

| Intent | What it does |
| --- | --- |
| `start_recording` | Begin a long-form recording session (meeting, lecture, conversation). |
| `stop_recording` | End the current session. |
| `get_recording` | List or read back a past session. |
| `summarize_recording` | LLM-summarise a session into key points, decisions, and action items. |

---

## Desktop & automation

These are the three-tier automation system plus the supporting tools. Tier escalation is automatic — TENKA picks the cheapest route that can do the job.

| Intent | What it does |
| --- | --- |
| `planner` | Multi-step goals — "book movie tickets for Saturday" — decomposed into a step graph and executed. Composes the other tools below. |
| `code_executor` | Sandboxed Python execution for anything solvable via an API: music control, weather, system info, math, messaging, calendar. |
| `browser_action` (via planner) | Playwright-driven website automation — fill forms, click buttons, scrape pages. Used internally by the planner. |
| `app_action` (via planner) | Terminator-driven native Windows app automation — click buttons, type text, read UI elements via Accessibility Tree. Faster than `computer_task` when an app exposes good UIA. |
| `computer_task` | Vision-loop fallback for desktop UI when no cheaper tier applies. A few LLM-vision calls per task. |
| `read_screen` | "What's on my screen?" — captures the screen and describes it. |
| `find_and_click` | "Click the submit button" — finds and clicks a named UI element. |
| `file_task` | Find, read, list, or open files. Write/move/delete require confirmation. |

---

## Identity & recognition

| Intent | What it does |
| --- | --- |
| `enroll_voice` | Train TENKA on your voice for speaker verification. |
| `forget_voice` | Delete the stored voiceprint. |
| `meet_face` | "This is Aarav, my brother" — adds a face to the recognition store. |
| `recognize_face` | "Who is this?" — identifies the person in front of the camera. |
| `forget_face` | Remove a face from the recognition store. |
| `camera_look` | Look through the webcam and describe what's seen. |

---

## System

| Intent | What it does |
| --- | --- |
| `browser_cdp_setup` | One-time setup helper for connecting TENKA to your real Chrome over the DevTools Protocol (optional advanced feature). |
| `unknown` | Sentinel — the classifier explicitly said "no idea what this is." Triggers a polite "I didn't catch that, can you rephrase?". |

---

## Routing model

Intents are detected in three layers, in order of cost:

1. **Regex fast-path** (`assistant/regex_router.py`) — zero LLM cost. Catches `~70%` of utterances by pattern alone.
2. **Manifest pre-route** — if the user's phrase matches a learned per-app manifest, dispatch goes straight to `manifest_dispatch` (still zero LLM cost).
3. **LLM intent classifier** — Gemini Flash-Lite (fallback: Groq llama-3.1-8b-instant). Returns a JSON `{intent, params}` object scoped to currently-relevant intents only (I6 scope detection).

After classification, the matched intent's handler runs. Some handlers can suspend mid-flow and resume after user input — credential prompts, confirmations, multi-turn clarifications — without losing context.

---

## Adding capabilities

The right way to extend TENKA almost never involves adding a new intent. In order of preference:

1. **A new preference row** — for one-off behavior knobs ("default music app is X").
2. **A new taught procedure (TP-1)** — for multi-step recipes specific to your workflow. Tell TENKA: "here's how I check the weather."
3. **A new app manifest** — for UI automation on a specific app. Auto-learned from your behavior after N=2 successful runs, or you can write a YAML manifest by hand at `~/TENKA/manifests/`.
4. **A new known-app row** — for the dispatch router to recognise a new browser/app.
5. **A new intent** — only when none of the above fit and the capability is genuinely orthogonal to everything that exists.

See [CONFIGURATION.md](./CONFIGURATION.md) for the runtime knobs around each of these.
