# Configuration

Every knob TENKA exposes, what it does, and when you'd touch it.

## How settings layer

TENKA reads settings from three places, in order of precedence:

1. **Runtime DB** (highest) — set with `/set <name> <value>` while TENKA is running. Persists in `~/TENKA/memory/tenka.db`. Survives restarts.
2. **Environment variable** — set in `.env` at the repo root, or in your shell. Overrides the hardcoded default on startup. Loaded by `assistant/config.py:17-28`.
3. **Hardcoded default** (lowest) — defined in `assistant/config.py`.

Some settings are env-only (you have to edit `.env` and restart). Some are runtime-only (no env var, only `/set`). Most are both — the column **Layer** below tells you which.

| Layer | What it means |
|---|---|
| `env` | Set in `.env`, read once on startup |
| `runtime` | Set with `/set`, applied immediately (or after restart if marked) |
| `env + runtime` | Either works; runtime value wins if both are present |

To list every runtime-settable name from inside TENKA: `/list`. To reset one back to the default: `/reset <name>`.

---

## Quickstart — keys most users care about

If you only touch six things, touch these. Everything else has a sensible default.

| Var | Layer | Default | Purpose |
|---|---|---|---|
| `GEMINI_API_KEY` | env | _(empty)_ | Primary LLM. Without this, TENKA falls back to Groq/Cerebras free tiers (rate-limited). Get a key at https://aistudio.google.com/apikey |
| `GROQ_API_KEY` | env | _(empty)_ | Free-tier fallback (~1K req/day on llama-3.3-70b). https://console.groq.com/ |
| `CEREBRAS_API_KEY` | env | _(empty)_ | Free-tier fallback for synthesis. Default model now `gpt-oss-120b` (Cerebras retired `llama3.1-8b` mid-2026). https://cloud.cerebras.ai/ |
| `TAVILY_API_KEY` | env | _(empty)_ | Web search backend. Needed for the `web_search` intent. https://app.tavily.com/ |
| `USER_REGION` | env | auto-detected | 2-letter ISO country code (`US`, `IN`, `GB`, ...). Used by services that need a market hint (Spotify, YouTube). |
| `USER_TIMEZONE` | env | auto-detected | IANA timezone name (`America/New_York`, `Asia/Kolkata`, ...). Used for date arithmetic in prompts. |

`USER_REGION` and `USER_TIMEZONE` are filled in by the setup wizard. Override manually only if the wizard guessed wrong.

---

## LLM providers

TENKA dispatches different tasks to different models — see `assistant/llm/router.py` `TASK_MODEL_MAP` for the routing table.

| Var | Layer | Default | Purpose |
|---|---|---|---|
| `GEMINI_API_KEY` | env | _(empty)_ | Primary key. Required for the default cost path. |
| `GEMINI_MODEL` | env | `gemini-2.5-flash` | Model id for the standard Gemini route (code_gen, agent_plan, vision). |
| `GEMINI_MODEL_LITE` | env | `gemini-2.5-flash-lite` | Model id for the lite route (intent, synthesis, default — cheaper, faster). |
| `GROQ_API_KEY` | env | _(empty)_ | Free-tier fallback. Also accepts numbered rotation form (see [Multi-key rotation](#multi-key-rotation)). |
| `GROQ_MODEL` | env | `llama-3.3-70b-versatile` | Model id used on Groq. |
| `CEREBRAS_API_KEY` | env | _(empty)_ | Free-tier fallback. |
| `CEREBRAS_MODEL` | env | `gpt-oss-120b` | Model id used on Cerebras. (`llama3.1-8b` was removed mid-2026; `gpt-oss-120b` is the current free-tier production model. `zai-glm-4.7` is also available as a Preview.) |
| `OLLAMA_URL` | env | `http://127.0.0.1:11434` | Local Ollama daemon URL. Used only when cloud providers are unreachable. |
| `OLLAMA_MODEL` | env | `llama3` | Local model name as `ollama` knows it. Pull beforehand: `ollama pull llama3`. |
| `TAVILY_API_KEY` | env | _(empty)_ | Web search. Also accepts numbered rotation form. |
| `JINA_API_KEY` | env | _(empty)_ | Optional. Reranker for memory retrieval (`storage/repos/memory.py:85`). Without it, recall falls back to local-only rerank. |
| `HF_TOKEN` | env (library) | _(empty)_ | Optional. Read by `huggingface_hub` directly for gated-model downloads. Most TENKA defaults are open models, so leave blank unless you've gated something yourself. |

### Multi-key rotation

Groq and Tavily support up to 9 numbered keys for round-robin rotation, useful for dodging per-key rate limits.

```
GROQ_API_KEY_1=gsk_...
GROQ_API_KEY_2=gsk_...
GROQ_API_KEY_3=gsk_...

TAVILY_API_KEY_1=tvly_...
TAVILY_API_KEY_2=tvly_...
```

Implementation: `assistant/llm/providers/groq.py:24-33`, `assistant/config.py:248-256`. The bare `GROQ_API_KEY` / `TAVILY_API_KEY` form is also accepted and is appended to the rotation list as a single-entry fallback.

---

## Region and timezone

| Var | Layer | Default | Purpose |
|---|---|---|---|
| `USER_REGION` | env | auto-detected | 2-letter ISO 3166-1 alpha-2 country code. Passed as a `market` / `region` hint to services that return null without it (e.g. Spotify track search). |
| `USER_TIMEZONE` | env | auto-detected | IANA name. The LLM never computes datetimes — `assistant/core/datetime_utils.py` does it in Python and passes literal strings into prompts. This var feeds that. |

Auto-detection happens in `scripts/setup.py`. On Windows it uses `kernel32!GetUserDefaultGeoName` for region and `tzlocal` for TZ (with a legacy-IANA-alias normalisation table — see `_TZ_ALIAS_TO_CANONICAL` in the wizard). If detection fails, the wizard prompts you.

---

## Logging

| Var | Layer | Default | Purpose |
|---|---|---|---|
| `DEBUG_LOG` | env | `true` | Master switch for verbose debug logging to `assistant/debug.log`. Set to `false` to suppress. |

Set to `false` only if you're profiling — the log is overwritten per run, doesn't grow unboundedly.

---

## Speech-to-Text (STT)

| Var | Layer | Default | Purpose |
|---|---|---|---|
| `STT_BACKEND` | env | `faster_whisper` | Which engine to use. Options: `faster_whisper` (CTranslate2, in-process, downloads model on first run) or `whisper_cpp` (HTTP call to a separate whisper.cpp server). |
| `FASTER_WHISPER_MODEL` | env | `small.en` | Model size for the `faster_whisper` backend. Larger = slower but more accurate. Try `base.en` if `small.en` is too slow on CPU, `medium.en` if you want more accuracy. |
| `WHISPER_CPP_URL` | env | `http://127.0.0.1:8080` | HTTP endpoint for the `whisper_cpp` backend, if you're using it. |

---

## Text-to-Speech (TTS)

| Var | Layer | Default | Purpose |
|---|---|---|---|
| `TTS_VOICE` | env + runtime | `af_bella` | Kokoro voice id. Other built-ins include `af_heart`, `af_sarah`, `am_adam`. |
| `TTS_SPEED` | env + runtime | `1.0` | Speech rate multiplier (0.5–2.0). Lower = slower & clearer. |

Runtime: `/set tts_speed 1.15`.

---

## Wake word (openWakeWord)

| Var | Layer | Default | Purpose |
|---|---|---|---|
| `WAKE_WORD_ENABLED` | env + runtime | `true` | Master switch. Off = push-to-talk only. **Restart required** when flipped via `/set`. |
| `WAKE_WORD_BUILTIN` | env | _(empty)_ | Opt-in built-in model name (`hey_jarvis_v0.1`, `alexa_v0.1`, `hey_mycroft_v0.1`). If set, also raise `WAKE_WORD_THRESHOLD` to ~0.5 — built-ins emit much higher per-frame scores than custom `.onnx` models. |
| `WAKE_WORD_FRAMEWORK` | env | `onnx` | Inference backend. `onnx` (Windows/Linux) or `tflite` (Linux only). |
| `WAKE_WORD_THRESHOLD` | env + runtime | `0.02` | Detection threshold. With sliding-window accumulation, this is the *sum* of frame scores over ~1.2 s. A custom-trained openWakeWord model typically produces 0.02–0.08 per frame, so a single utterance accumulates ~0.10–0.15. Tune up for false positives, down for missed triggers. |
| `WAKE_WORD_CHUNK_SIZE` | env | `1280` | Audio chunk in samples. 1280 = 80 ms at 16 kHz (openWakeWord recommendation). |
| `WAKE_WORD_COOLDOWN` | env + runtime | `2.0` | Seconds to ignore the wake word after it fires. Raise if you get rapid re-triggers. |
| `WAKE_WORD_RECORD_SECONDS` | env | `5.0` | How long to record after wake-word detection before auto-stopping. |
| `FOLLOW_UP_LISTEN_SECONDS` | env + runtime | `5.0` | Seconds the assistant listens for a follow-up utterance after TTS finishes. |

The wake-word model file is `assistant/models/<assistant_name>.onnx` — defaults to `tenka.onnx`. Rename or replace to swap the trained wake phrase.

---

## Camera

| Var | Layer | Default | Purpose |
|---|---|---|---|
| `CAMERA_ENABLED` | env + runtime | `true` | Master switch for camera + face recognition. Off saves CPU and improves privacy. **Restart required**. |
| `CAMERA_INDEX` | env | `0` | Which camera to use if you have multiple. |
| `CAMERA_MAX_WIDTH` | env | `1280` | Frame width cap. Lower = less CPU. |
| `FACE_RECOGNITION_TOLERANCE` | env + runtime | `0.5` | Match strictness (0.4 strict — 0.6 loose). Lower = fewer false positives but more rejections of you on bad lighting. |
| `FACE_MAX_ENCODINGS` | env | `5` | Max encodings stored per known face. |

---

## Speaker verification

| Var | Layer | Default | Purpose |
|---|---|---|---|
| `SPEAKER_VERIFY_ENABLED` | env + runtime | `true` | Master switch. Off = anyone speaking near the mic can issue commands. |
| `SPEAKER_VERIFY_THRESHOLD` | env + runtime | `0.50` | Base cosine similarity threshold. Lower if you're being rejected often, raise if impostors slip through. |
| `SPEAKER_VERIFY_THRESHOLD_FLOOR` | env | `0.40` | Absolute floor for dynamic threshold adjustments. They never go below this. |
| `SPEAKER_MAX_ENROLLMENTS` | env | `10` | Max voiceprint samples kept per speaker. |
| `SPEAKER_ENROLL_RECORD_SECONDS` | env | `3.0` | How long each enrollment sample records. |
| `SPEAKER_ENROLL_NUM_SAMPLES` | env | `5` | How many samples are captured at enrollment time. |
| `SPEAKER_MIN_AUDIO_SECONDS` | env | `1.0` | Minimum audio duration for a verification attempt. Shorter clips give weak embeddings and fail-open. |

Voiceprint storage: `~/TENKA/memory/voiceprint.npz`.

---

## Knowledge graph

| Var | Layer | Default | Purpose |
|---|---|---|---|
| `KG_INGEST_ENABLED` | env | `true` | Master switch for extracting entities + facts from conversation. Off = no new facts learned. |
| `KG_QUERY_INJECTION_ENABLED` | env | `true` | Master switch for injecting KG context into LLM prompts. Off = TENKA still ingests but doesn't recall. |
| `KNOWLEDGE_APPROVAL_MODE` | env | `immediate` | Code-executor service-knowledge approval flow. `immediate` asks right after the successful retry; `deferred` asks on the next unrelated interaction (less intrusive). |

Both kill-switches accept `false`, `0`, or `no` to disable. See `assistant/knowledge_graph.py` and knowledge-graph spec for details.

---

## Proactive nudges

Unprompted reflection / suggestion. Off by default for new users; turn on once you trust TENKA's timing.

| Var | Layer | Default | Purpose |
|---|---|---|---|
| `PROACTIVE_ENABLED` | env + runtime | `true` | Master switch. **Restart required**. |
| `PROACTIVE_MODE` | env + runtime | `always` | `always` fires immediately when ready; `idle_only` waits until TENKA is not processing a request. |
| `PROACTIVE_INTERVAL_MINUTES` | env + runtime | `30` | How often the background analyzer re-runs. **Restart required**. |
| `PROACTIVE_IDLE_THRESHOLD_MINUTES` | env + runtime | `10` | Silence required before an idle nudge fires (when `PROACTIVE_MODE=idle_only`). |

---

## Messaging

| Var | Layer | Default | Purpose |
|---|---|---|---|
| `MESSAGING_AUTO_CONNECT` | env | _(empty)_ | Comma-separated services to auto-connect at startup (`whatsapp`, `telegram`, ...). Empty = connect every registered service with a saved session. Set to `none` to disable. |
| `MESSAGING_NOTIFY_DEBOUNCE` | env + runtime | `5.0` | Wait window (seconds) before announcing a new message. Use 20–30 in real life — the 5 s default is for testing. |
| `MESSAGING_SUPPRESS_WINDOW` | env + runtime | `300.0` | After reading messages from a chat, stay silent for this many seconds on further messages from the same chat. |
| `MESSAGING_BRIDGE_PORT` | constant | `7780` | HTTP port the messaging bridge listens on. Not env-overridable currently. |

---

## File operations

| Var | Layer | Default | Purpose |
|---|---|---|---|
| `FILE_WRITE_SAFE_MODE` | env | `true` | `true` restricts writes to `~/TENKA/`. `false` allows any path you specify (still requires confirmation). |

---

## Code executor

| Var | Layer | Default | Purpose |
|---|---|---|---|
| `CODE_EXECUTOR_POWER_MODE` | env | `false` | `true` = Tier 3 unrestricted Python; `false` = Tier 1 sandbox only. Power mode bypasses the import / subprocess guards — use only on your own machine. |
| `CODE_EXECUTOR_INJECT_KNOWLEDGE` | env | `false` | Inject per-service knowledge entries into LLM code-gen prompts. Defaults off in v1.0; will flip back to `true` once knowledge hygiene lands in v1.1. See `assistant/config.py:421-429`. |

---

## Runtime-only settings (no env var)

These can only be changed with `/set <name> <value>` from inside TENKA. They live in the runtime registry (`assistant/core/runtime_config.py`).

| Name | Default | Purpose |
|---|---|---|
| `assistant_name` | `TENKA` | Display / wake / persona name. **Restart required** — wake-word model path is derived from this. |
| `unity_enabled` | `true` | Unity avatar frontend. Off = terminal-only mode: STT/TTS/wake word work, avatar commands no-op. **Restart required**. |
| `personality` | `warm_honest` | Active personality base. Options: `warm_honest`, `tsundere`, `minimal`. Applied immediately, no restart. |
| `listen_to_everyone` | `false` | Disables speaker verification. Same as the spoken phrase "listen to everyone". |
| `push_to_talk_key` | `v` | Recording trigger. Single char or pynput key name (`home`, `f1`, etc.). **Restart required**. |
| `verify_enabled` | `true` | Master switch for step verification (pre-checks + post-verifies). Off = fastest, but silent failures possible. |
| `verify_browser_steps` | `true` | Verify Playwright steps. |
| `verify_app_steps` | `true` | Verify Terminator steps. |
| `verify_vision_fallback` | `true` | When code-tier verification is ambiguous, escalate to a Gemini vision call. |
| `verify_strict_text_match` | `false` | `true` catches autocomplete drift but false-fails on phone/email auto-formatting. `false` uses case-insensitive contains. |
| `verify_min_confidence` | `0.5` | Vision-tier confidence threshold to count as a real failure. |
| `verify_max_retries` | `1` | Self-heal attempts per step. Hard-capped at 1 — endless retry bricks demos. |
| `browser_prefer_cdp` | `true` | Try to attach to a running Chrome with `--remote-debugging-port=9222` before launching bundled Chromium. |
| `browser_cdp_port` | `9222` | Port to probe for Chrome's CDP endpoint. |
| `browser_cdp_probe_ttl` | `30.0` | How long the CDP availability probe is cached (seconds). |
| `browser_dom_mode_enabled` | `true` | Master switch for the DOM-aware browser planner. Off = always use the vision-loop fallback. |
| `browser_dom_tree_token_budget` | `4000` | Max tokens the perceived element tree may consume in the DOM-planner prompt. |
| `browser_dom_cache_ttl` | `10.0` | DOM tree cache TTL (seconds). |
| `incoming_read_threshold` | `3` | ≤N incoming messages → read verbatim; more → LLM-summarize. |
| `dropdown_commit_guard_enabled` | `true` | Auto-inject `keyboard_press(enter)` when a batch navigates a dropdown via arrow keys without a commit. |
| `deterministic_matching_enabled` | `true` | Action-signature TODO marking with vision-confirm for `select` TODOs. |
| `dynamic_budget_enabled` | `true` | Dynamic loop budget sized from TODO count; stuck-step detector. |
| `dialog_engagement_gate_enabled` | `true` | Refuse to dismiss overlays when recent actions show engagement with the modal surface. |

Use `/list` from inside TENKA to see the live values + descriptions, and `/reset <name>` to restore defaults.

---

## Personality-specific vars

Personalities (`warm_honest`, `tsundere`, `minimal`) may introduce their own env vars for tuning their voice. These only matter when that personality is active.

### Tsundere (vocal voice)

| Var | Layer | Default | Purpose |
|---|---|---|---|
| `VOCAL_VOICE_BASE` | env | `af_heart` | Kokoro base voicepack used for all emotions in vocal voice mode. |
| `VOCAL_VOICE_ENABLED` | env + runtime | `true` | Enable per-emotion audio effects (pitch shift, EQ, tremolo) — produces an anime-style vocal character. Off = plain Kokoro voice. |
| `VOCAL_CASUAL_LANGUAGE` | env + runtime | `false` | Let the tsundere personality use mild curses (`damn`, `crap`, `dumbass`). Persona flavor, not hostility. **Restart required** when toggled. |

Other personalities may add their own vars when they ship — check the personality's docstring in `assistant/personalities/`.

---

## Other library / OS vars TENKA respects

These aren't TENKA-owned but TENKA's behavior changes if they're set in your environment.

| Var | Source | Effect |
|---|---|---|
| `NO_COLOR` | de facto standard | If set to any non-empty value, the setup wizard skips ANSI colors. |
| `HF_TOKEN` | `huggingface_hub` library | Used for gated HuggingFace model downloads. Most TENKA models are open. |
| `TOKENIZERS_PARALLELISM` | `transformers` library | Set to `false` in some code paths to silence the fork-after-init warning. |
| `PLAYWRIGHT_BROWSERS_PATH` | Playwright | Override where `playwright install chromium` puts the browser. |
| `LANG` / `LC_ALL` / `LC_CTYPE` | POSIX | Read by the setup wizard as a region-detection fallback. |

---

## Where the runtime DB lives

```
~/TENKA/memory/tenka.db    # five repos share this:
                           # personality, preference,
                           # procedure, settings, shortcut
                           # plus FAISS index + ID-map for memory repo (same folder)
~/TENKA/manifests/         # per-app YAML manifests
~/TENKA/Sessions/          # session transcripts
~/TENKA/Notes/             # create_note output
```

To start over: stop TENKA, delete `~/TENKA/`, restart. The setup wizard's `.tenka_setup.json` marker is separate and lives at the repo root.
