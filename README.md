# TENKA

**Transformative Evolving Neural Kinetic Agent** — a local-first Python voice agent for Windows. She listens, thinks, remembers, and acts on the desktop.

The name is the thesis: **TENKA evolves without dependency**. Adding support for a new app, service, or domain must not require code changes. Apps are discovered at runtime, learned from your behavior, and stored as data.

Created by **Om Khode**. Licensed under [Apache 2.0](./LICENSE).

> Status: v1.0.0 released 2026-06-07. Active development continues — expect rough edges. Bug reports and PRs welcome.

---

## What TENKA does

- **Talks back to you.** Wake-word ("TENKA") or push-to-talk, then full streaming voice replies via Kokoro TTS. An optional anime-style vocal voice mode (pitch shift, EQ, tremolo) is included.
- **Acts on your desktop.** Books tickets, fills forms, plays music, controls apps — through three escalating tiers (browser → native UI → vision fallback). No app-specific code.
- **Remembers conversations.** A real knowledge graph (entities, facts, relationships, commitments) backed by SQLite + FAISS. Survives restarts.
- **Has a personality.** Three swappable bases (warm-honest, tsundere, …) with per-axis modifiers. Nightly self-reflection updates traits from how you actually interact.
- **Runs on free tiers.** Gemini primary, Groq + Cerebras as fallback. Falls through to local Ollama if everything else fails. You don't need a paid LLM key.

### Things you can ask her

```
"Open YouTube and search for piano tutorials"     → multi-step planner
"Play some lo-fi"                                 → code_executor (Spotify API)
"Search the web for the latest Python release"    → web_search
"Take a note: groceries — eggs, milk, bread"      → create_note
"Remind me to call mom at 6 pm"                   → set_reminder
"Record this meeting"                             → start_recording
"Summarise what we recorded"                      → summarize_recording
"What did I tell you about my brother?"           → memory_query (KG recall)
"This is Aarav, my brother"                       → meet_face (enrol identity)
"Who is this?" (looking at the camera)            → recognize_face
```

The full list of ~38 intents — grouped by category with descriptions — lives in **[TENKA_Capabilities.md](./TENKA_Capabilities.md)**.

---

## Requirements

| Requirement | Notes |
| --- | --- |
| Windows 10/11 | Linux/macOS unsupported. Win32 APIs and Terminator are load-bearing. |
| Python 3.11.9 | Other versions untested. `pyenv-win` recommended. |
| ~4 GB RAM | At rest. The Whisper model and embeddings are the heavy hitters. |
| GPU optional | NVIDIA helps Kokoro TTS. A GTX 1650 is enough. CPU works for everything else. |
| Microphone | Required for voice. Terminal-only mode also works without one. |

A Gemini API key is recommended but not required — TENKA will degrade gracefully through Groq → Cerebras → Ollama.

---

## Quick start

```powershell
# 1. Clone
git clone https://github.com/<you>/tenka.git
cd tenka

# 2. Set up Python 3.11.9 (pyenv-win example)
pyenv install 3.11.9
pyenv local 3.11.9

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install Playwright's bundled Chromium into TENKA's isolated cache
$env:PLAYWRIGHT_BROWSERS_PATH = "$HOME\TENKA\browser-cache"
python -m playwright install chromium

# 5. Drop API keys into .env (see .env.example for the full list)
copy .env.example .env
notepad .env

# 6. Run
.\start_assistant.bat
```

The first launch creates `~/TENKA/` and walks you through a brief setup wizard (region, timezone, wake-word model). All persistent data — DB, FAISS indexes, manifests, recordings, notes — lives under that single directory.

Press **V** to push-to-talk, or say the wake word once you've enrolled one. **ESC** aborts any in-flight automation cleanly.

For the full walkthrough — wizard explanation, manual setup, optional wake-word and CDP config, and a troubleshooting catalogue — see **[SETUP.md](./SETUP.md)**.

---

## Configuration

Every runtime knob is documented in **[CONFIGURATION.md](./CONFIGURATION.md)** — env vars, runtime `/set` commands, defaults, and where each setting reads from. Start there for tuning STT/TTS, wake-word thresholds, LLM routing, file safety, and feature flags.

Common edits live in `.env` (loaded on startup) — `.env.example` is the canonical list.

---

## Architecture in 60 seconds

TENKA is a single Python process with a few long-lived domains:

```
┌────────────────────────────────────────────────────────────────┐
│  io/  (audio in/out, overlay, screen, status broadcaster)      │
├────────────────────────────────────────────────────────────────┤
│  llm/        — provider chain + task-shaped contracts          │
│  storage/    — single SQLite DB + FAISS indexes                │
│  automation/ — three-tier desktop control + manifest learning  │
│  actions/    — intent handlers (small_talk, planner, …)        │
│  core/       — config resolution, abort, scopes                │
└────────────────────────────────────────────────────────────────┘
```

The three automation tiers, in order of cost:

1. **`browser_action`** — Playwright + CDP. Websites. Zero vision calls.
2. **`app_action`** — Terminator (Windows Accessibility Tree). Native apps. Zero vision calls.
3. **`computer_task`** — pyautogui + LLM vision. Last resort. A few vision calls per task.

Routing is zero-LLM-cost: user preferences → URL regex → running process → launch keyword → fallback. New apps are never coded in — they're added as data rows, learned from observation, or taught directly via voice.

For the deeper picture — layering rules, registry system, knowledge-graph design, manifest learning, the full folder map, and the "right path" for adding features — see **[TENKA_Architecture.md](./TENKA_Architecture.md)**.

---

## LLM provider strategy

TENKA dispatches different tasks to different models. The full routing table lives in `assistant/llm/router.py`. In short:

| Task | Primary | Fallback |
| --- | --- | --- |
| Intent classification | Gemini Flash-Lite | Groq llama-3.1-8b-instant |
| Small talk | Gemini Flash | Groq llama-3.3-70b |
| Code generation | Gemini Flash | Groq Kimi-K2 → llama-3.3-70b |
| Synthesis / default | Gemini Flash-Lite | Cerebras gpt-oss-120b |
| Vision verification | Gemini Flash (vision) | Groq llama-4-scout |
| KG extraction | Gemini Flash-Lite | Cerebras → Groq |
| Last resort | — | Ollama (local) |

Every chain is defensive: any provider can be down or missing a key and TENKA keeps working. Cerebras retired `llama3.1-8b` mid-2026 — the current default is `gpt-oss-120b` on their free tier.

---

## Frontends

TENKA is just the brain. It exposes a TCP wire protocol (JSON over ports 7777/7778) and runs fine in terminal-only mode out of the box — subtitles echo to the console, voice still works.

Any frontend can connect. The reference implementation is the **Mate Engine** Unity avatar project (VRM model + animations + subtitles), shipped and licensed separately. To run terminal-only, set `UNITY_ENABLED=false` in `.env`.

A messaging bridge listens on port 7780 (HTTP) for adapter integrations (e.g. WhatsApp).

---

## Project layout

```
assistant/                 # the brain — all TENKA code lives here
  core/                    # config resolution, abort, scopes
  llm/                     # provider chain + task contracts
  storage/                 # SQLite + FAISS repos
  io/                      # audio, overlay, screen, status
  automation/              # three-tier desktop control + manifests
  actions/                 # intent handlers
  code_executor/           # sandboxed Python execution
  models/                  # wake-word ONNX, etc.
  personalities/           # persona definitions
CONFIGURATION.md           # every setting, explained
.env.example               # canonical env var list
start_assistant.bat        # launcher
```

Persistent runtime data (DB, manifests, recordings) lives under `~/TENKA/`, not in the repo.

---

## Known issues

See **[TENKA_Known_Issues.md](./TENKA_Known_Issues.md)** for the current punch list of minor bugs and rough edges. The pre-release period catches new ones constantly; that file is the canonical record.

---

## License

TENKA is released under the **Apache License, Version 2.0**. See [LICENSE](./LICENSE) for the full text and [NOTICE](./NOTICE) for attribution requirements.

In short: you can use, modify, fork, and redistribute TENKA — commercially or not. You must keep the LICENSE and NOTICE files in any redistribution, and credit the project in source-form documentation. Contributing back is welcome but optional.

---

## Acknowledgements

TENKA stands on a stack of generous open-source and free-tier work:

- **Speech & audio** — [faster-whisper](https://github.com/SYSTRAN/faster-whisper), [Kokoro TTS](https://github.com/hexgrad/kokoro), [openWakeWord](https://github.com/dscripka/openWakeWord), ECAPA-TDNN via [SpeechBrain](https://github.com/speechbrain/speechbrain).
- **Automation** — [Playwright](https://github.com/microsoft/playwright-python), [Terminator](https://github.com/mediar-ai/terminator).
- **Memory** — [FAISS](https://github.com/facebookresearch/faiss), [sentence-transformers](https://github.com/UKPLab/sentence-transformers).
- **LLM providers** — [Google Gemini](https://ai.google.dev/) (primary), [Groq](https://groq.com/) and [Cerebras](https://www.cerebras.ai/) (defensive fallback), [Ollama](https://ollama.com/) (local last-resort). TENKA's free-tier-first routing is a deliberate design choice; without their generous free quotas a hobbyist-built personal agent like this wouldn't be possible.
