# TENKA — Setup Guide

The README's [Quick start](./README.md#quick-start) is the 5-step happy path. This document is the deeper walkthrough: every step explained, every common pitfall flagged, and a manual path for people who don't want a wizard.

Two ways through:

1. **[The wizard](#the-wizard-recommended)** — interactive, idempotent, handles 80% of edge cases. **Recommended for first install.**
2. **[Manual setup](#manual-setup)** — every step done by hand. Pick this if the wizard's prompts don't suit you, or if you're scripting deployment.

After install, jump to **[First launch](#first-launch)** and the **[Troubleshooting](#troubleshooting)** section.

---

## Requirements (recap)

| Requirement | Detail |
| --- | --- |
| **OS** | Windows 10 (build 1903+) or Windows 11. Linux/macOS unsupported. |
| **Python** | **Exactly 3.11.x** (tested on 3.11.9). Other 3.x versions are untested and likely to break. |
| **Disk** | ~3 GB free (Whisper model + Playwright Chromium + embedding model + Kokoro voices). |
| **RAM** | 4 GB at rest, ~6 GB peak during STT inference. |
| **GPU** | Optional. NVIDIA accelerates Kokoro TTS but isn't required. |
| **Microphone** | Required for voice. Terminal-only mode works without one. |
| **Internet** | Required at install (model downloads + LLM API keys) and at runtime (cloud LLM providers). |

---

## The wizard (recommended)

```powershell
cd <repo>
python scripts/setup.py
```

The wizard runs 7 idempotent steps. You can re-run it any time — it tracks progress in `.tenka_setup.json` and skips already-completed steps. Pass `--force` to redo everything from scratch, or `--no-launch` to suppress the "launch now?" prompt at the end.

### What the wizard does

1. **Python version check** — confirms you're on 3.11.x. Refuses to continue on 3.10 or 3.12+ (won't try to "make it work" with the wrong version).
2. **`pip install -r requirements.txt`** — pulls every runtime dependency. ~250 MB download. Slow first time, fast on re-runs (pip cache).
3. **Playwright Chromium** — installs the bundled browser into `~/TENKA/browser-cache/`. Isolated from your system Chrome so TENKA's automation doesn't fight with your day-to-day browsing.
4. **API keys** — interactively asks for Gemini, Groq, Cerebras, Tavily, Jina, Hugging Face. All optional except *something* (TENKA falls through Gemini → Groq → Cerebras → Ollama until one works). Press Enter to skip any you don't have.
5. **Region + timezone** — auto-detected from `kernel32!GetUserDefaultGeoName` and `tzlocal`. Wizard asks you to confirm or override.
6. **Write `.env`** — merges your answers into the existing `.env` (preserving anything you already set) using atomic write. Backed up if there's a write race.
7. **Done** — offers to launch TENKA immediately. Subsequent runs are `start_assistant.bat` or `python -m assistant.main`.

If the wizard fails partway through, fix the underlying problem (network, key typo, etc.) and re-run — it'll pick up where it left off.

---

## Manual setup

Skip this section if you used the wizard.

### 1. Install Python 3.11.9

Recommended via `pyenv-win`:

```powershell
# Install pyenv-win itself (one-time)
Invoke-WebRequest -UseBasicParsing -Uri "https://raw.githubusercontent.com/pyenv-win/pyenv-win/master/pyenv-win/install-pyenv-win.ps1" -OutFile install-pyenv-win.ps1
.\install-pyenv-win.ps1
# Restart the shell so PATH picks up

# Then for TENKA
pyenv install 3.11.9
cd <repo>
pyenv local 3.11.9
python --version   # should print: Python 3.11.9
```

Alternative: install Python 3.11.9 from [python.org](https://www.python.org/downloads/release/python-3119/) directly.

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

If pip complains about a wheel that needs a C compiler, install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) (C++ workload) and retry.

### 3. Install Playwright's bundled Chromium

```powershell
$env:PLAYWRIGHT_BROWSERS_PATH = "$HOME\TENKA\browser-cache"
python -m playwright install chromium
```

The `$env:PLAYWRIGHT_BROWSERS_PATH` line is critical — it isolates Playwright's cache from the system default at `%LOCALAPPDATA%\ms-playwright\`. Without it, TENKA fights other Playwright instances (IDE-side MCP servers, other projects) over the shared lock and may hang on launch.

> **PowerShell vs cmd.exe:** in PowerShell, plain `set NAME=value` only sets a shell variable, not an environment variable — child processes won't inherit it. Use `$env:NAME = "value"` instead.

### 4. Drop API keys into `.env`

```powershell
copy .env.example .env
notepad .env
```

Recommended order of how-to-get for each provider:

| Provider | Get a key | Free tier? | Used for |
| --- | --- | --- | --- |
| **Gemini** | https://aistudio.google.com/apikey | Yes, generous | Primary route for everything (intent, planning, vision, synthesis). |
| **Groq** | https://console.groq.com/ | Yes, ~1K req/day on 70b | First fallback. Llama 3.3 70b for small talk + planning. |
| **Cerebras** | https://cloud.cerebras.ai/ | Yes, 5 req/min on `gpt-oss-120b` | Second fallback. Synthesis + KG extraction. |
| **Tavily** | https://app.tavily.com/ | Yes, 1K req/month | `web_search` intent. |
| **Jina** | https://jina.ai/reranker/ | Yes | Optional reranker for memory retrieval. |
| **Hugging Face** | https://huggingface.co/settings/tokens | N/A | Only needed for gated-model downloads. Default builds don't require one. |

You don't need all of them. The chain degrades gracefully — TENKA will keep working as long as *one* cloud provider responds, and falls back to a local Ollama if everything is down (provided Ollama is running with `llama3.1:8b` pulled).

### 5. Region and timezone

If you skip the wizard, set these in `.env`:

```
USER_REGION=IN
USER_TIMEZONE=Asia/Kolkata
```

`USER_REGION` is a 2-letter ISO 3166-1 alpha-2 country code. Services like Spotify return `item:null` without it. `USER_TIMEZONE` is an IANA name — TENKA does all date math in Python, never in the LLM.

---

## First launch

```powershell
.\start_assistant.bat
```

What to expect on the first run:

1. **Console banner.** TENKA's startup banner + config summary (STT backend, TTS voice, wake-word state, configured LLM providers, sandbox dir).
2. **One-time model downloads.**
   - `faster-whisper` small.en — ~150 MB, downloaded into your HF cache.
   - Kokoro TTS — ~300 MB, downloaded into the Python package install.
   - ECAPA-TDNN (speaker verification) — ~80 MB, into HF cache.
   - openWakeWord base models — ~10 MB, into the package.
   - sentence-transformers all-MiniLM-L6-v2 (embeddings) — ~80 MB.
   These all happen lazily — the first launch is slow (~30–60 s extra); subsequent launches are fast.
3. **Persistent data location.** Everything writes under `~/TENKA/`. Subdirs created automatically: `memory/`, `manifests/`, `Notes/`, `Sessions/`, `browser-cache/`.
4. **The ready prompt.** Once you see `Ready! Waiting for Unity to connect, say the wake word, or press V to talk...`, TENKA is live.

### Trying it out

- **Push-to-talk** — press **V** to start recording, press V again to stop. Or, type a message in the console and press Enter.
- **Wake word** — by default, TENKA looks for a custom model at `assistant/models/tenka.onnx`. If you renamed the assistant via `ASSISTANT_NAME`, the filename must match too — see [Option B — custom model](#option-b--custom-model) below. If the file is absent, push-to-talk still works. See [Wake-word setup](#optional-wake-word-setup) below to enable a built-in or train your own.
- **Abort** — hold **ESC** to abort any in-flight automation, browser action, or TTS playback. Works mid-sentence.

---

## Optional: wake-word setup

TENKA ships with `openWakeWord` as the inference framework but no model. You have three options:

### Option A — built-in model (fastest)

```
WAKE_WORD_BUILTIN=hey_jarvis_v0.1
WAKE_WORD_THRESHOLD=0.5
```

Available built-ins: `hey_jarvis_v0.1`, `alexa_v0.1`, `hey_mycroft_v0.1`. The threshold has to be *higher* for built-ins than for custom models — built-ins emit much higher per-frame scores.

### Option B — custom model

Train one at [openWakeWord-cli](https://github.com/dscripka/openWakeWord) or use the [Hugging Face training space](https://huggingface.co/spaces/davidscripka/openWakeWord). Save the resulting `.onnx` to `assistant/models/<your_word>.onnx`, then set `ASSISTANT_NAME=<your_word>` in `.env`. TENKA looks for `assistant/models/{lowercase_name}.onnx` on startup.

Tune `WAKE_WORD_THRESHOLD` empirically — start at `0.02` and adjust by `0.01` until false positives match your tolerance.

### Option C — push-to-talk only

Set `WAKE_WORD_ENABLED=false`. Use **V** to talk.

---

## Optional: persistent Chrome (CDP mode)

TENKA's planner can attach to your *real* Chrome via the DevTools Protocol instead of launching its own Playwright Chromium. This is useful if you want TENKA to see your logged-in sessions, bookmarks, and tabs.

```powershell
python -m assistant.main
# Then in the chat: "set up Chrome CDP"
```

This runs the `browser_cdp_setup` intent, which edits your Chrome shortcuts to add `--remote-debugging-port=9222`. Reversible via "undo Chrome setup". See `assistant/automation/browser/setup.py` for the gory details.

---

## Troubleshooting

### "Wake word disabled — expected custom model at ... but file is missing"

Push-to-talk still works. Add a built-in or custom model — see [Wake-word setup](#optional-wake-word-setup).

### Playwright hangs on `async_playwright().start()`

You forgot to set `PLAYWRIGHT_BROWSERS_PATH` before running `playwright install`, so Playwright is fighting with another instance over the shared cache directory. Fix:

```powershell
$env:PLAYWRIGHT_BROWSERS_PATH = "$HOME\TENKA\browser-cache"
python -m playwright install chromium
```

And re-launch TENKA.

### Gemini 429 / "prepayment credits depleted"

Either get a fresh Gemini key from [AI Studio](https://aistudio.google.com/apikey) (the free tier resets) or rely on the Groq + Cerebras fallback chain — TENKA will fall through automatically and keep working.

### Cerebras 404 — "Model llama3.1-8b does not exist"

Cerebras retired `llama3.1-8b` mid-2026. The default is now `gpt-oss-120b`. If you have a manual override in `.env` referencing the old model, update it:

```
CEREBRAS_MODEL=gpt-oss-120b
```

### SpeechBrain crash on Windows ("symlink failed")

Already handled by a monkey-patch in `assistant/io/audio/speaker_verify.py` (replaces `os.symlink` with `shutil.copy2` during model loading). If you see this anyway, your SpeechBrain version may be ahead of what we patch. Check the speaker_verify module for the patch site.

### `faster-whisper` is slow on CPU

Drop to a smaller model in `.env`:

```
FASTER_WHISPER_MODEL=base.en
```

`small.en` is the default and is reasonably accurate; `base.en` is roughly twice as fast at the cost of more transcription errors. `tiny.en` is for desperate situations.

### Voiceprint enrollment refuses to recognize me

Voiceprints accumulate up to `SPEAKER_MAX_ENROLLMENTS` (default 10) per speaker. If the first few enrollments were noisy and now your reliable samples are getting averaged with them, run:

```
forget my voice
```

…and re-enroll in a quiet environment.

### Kokoro TTS sounds robotic / cuts off mid-sentence

Speed too high. Try `TTS_SPEED=1.0` or lower. Also check `VOCAL_VOICE_ENABLED=false` if you want the plain Kokoro voice instead of the anime-style post-processing.

### "Wake word enabled" but it never triggers

Three things to check, in order:
1. `WAKE_WORD_THRESHOLD` — too high for a custom model. Default `0.02` works for most custom models; built-ins need `~0.5`.
2. Your `tenka.onnx` (or whichever model) is in `assistant/models/`, not somewhere else.
3. Your microphone is being picked up — see "Microphone troubleshooting" below.

### Microphone not detected / silence threshold issue

TENKA auto-calibrates the noise floor at startup (the `[STT] Noise floor: ...` log line). If that number is suspiciously high (>0.05), your mic is either picking up too much ambient noise or you've selected the wrong input device. Fix in Windows Sound Settings; TENKA uses the default input device.

### "Database is locked"

Something else has `~/TENKA/memory/tenka.db` open — another TENKA instance, a SQLite browser, etc. Close it. TENKA uses a singleton SQLite connection on a single thread; concurrent processes will collide.

### Need a clean slate

Delete `~/TENKA/` and re-run `start_assistant.bat`. The setup wizard's marker (`.tenka_setup.json` at repo root) is separate from runtime data — touch that too if you want to re-do the install steps.

---

## Where things live

```
<repo>/                     # the code
  .env                      # your local secrets (gitignored)
  .tenka_setup.json         # setup wizard's progress marker (gitignored)
  start_assistant.bat       # launcher

~/TENKA/                    # all runtime data (per user)
  memory/
    tenka.db                # SQLite — facts, settings, schedules, monitors, etc.
    *.faiss                 # FAISS indexes for memory, recordings, facts
    *.idmap.json            # FAISS id → row mappings
    voiceprint.npz          # speaker enrollment (if enrolled)
  manifests/                # learned per-app UI manifests (YAML, one per app)
  Notes/                    # create_note output (Markdown files)
  Sessions/                 # session transcripts (one per conversation)
  browser-cache/            # Playwright's bundled Chromium (~280 MB)
```

Everything in `~/TENKA/` is safe to delete to start over — TENKA will rebuild on next launch. The wizard marker and your `.env` live at the repo root, not under `~/TENKA/`.

---

## Next steps

Once setup is done, browse:

- **[CONFIGURATION.md](./CONFIGURATION.md)** — every runtime knob, what it does, when you'd touch it.
- **[TENKA_Capabilities.md](./TENKA_Capabilities.md)** — the 38 intents grouped by category. Useful for "what can I ask?".
- **[TENKA_Known_Issues.md](./TENKA_Known_Issues.md)** — current punch list.
