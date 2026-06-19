# Known Issues

Minor issues discovered during testing. Not blocking — features work, but suboptimal. Batch-fix when current work is done.

---

## KI-1: ~~get_text timeout on app automation search tasks~~ FIXED

**Priority:** Low
**Effort:** Medium (deterministic step-plan fix in router.py + multi-scenario retest)
**Fixed:** 2026-06-19 — added `"search"` to `_TYPE_WORDS` in `router.py` Fix A, so search goals (no result-reading words) strip hallucinated `get_text` steps just like type/write goals do. Test: `tests/test_known_issues_fixes.py::TestKI1SearchGetTextStrip`.
**Discovered:** 2026-05-07, during D1+D9 live-test

**Symptom:** When computer_task native automation runs a search task (e.g. "search weather in Berlin on Chrome"), the LLM planner adds a `get_text` step with a hallucinated selector (`name:Weather in Berlin` on window `Berlin Weather - Google Chrome`). The selector doesn't resolve, causing a 15s timeout. The task still succeeds — Chrome opens and searches — but wastes 15 seconds.

**Root cause:** LLM plans an optimistic read-back step assuming it can locate the result element by name. The selector is fabricated (the window title hasn't changed yet, the element name is guessed).

**Existing precedent:** `router.py:995-1004` already strips `get_text` steps from pure type/write tasks. A similar heuristic could strip `get_text` from search-oriented goals (goal contains "search" + no result-reading words like "read", "get", "check", "what").

**Test case:**
```
Input: "Search weather in Berlin on Chrome"
Expected: Chrome opens, types query, presses Enter — no get_text step
Actual: All of the above works, but adds a 15s timeout on hallucinated get_text
```

**Log excerpt:**
```
[DA] LLM planned 4 steps: [..., {'action': 'get_text', 'params': {'selector': 'name:Weather in Berlin', 'window': 'Berlin Weather - Google Chrome'}}]
[APP] get_text timed out after 15s for: name:Weather in Berlin
```

---

## KI-2: ~~Double shutdown log on exit~~ FIXED

**Priority:** Low
**Effort:** Low (find duplicate signal/keyboard handler registration)
**Fixed:** verified resolved 2026-06-19 — `main.py:2086` `signal_handler` now guards on `_shutdown_event.is_set()` (a second Ctrl+C logs `Force shutdown`, not a duplicate), the graceful path logs `Shutting down gracefully...` exactly once, and the `KeyboardInterrupt` catch logs the distinct `Goodbye!`. No code change needed.
**Discovered:** 2026-05-10, during S11 live-test

**Symptom:** On exit, `Shutting down...` is logged twice:
```
12:00:06 [main] INFO: Shutting down...
12:00:07 [main] INFO: Shutting down...
```

**Root cause (suspected):** The shutdown handler is registered on both a signal handler (e.g. SIGINT/SIGTERM) and a keyboard interrupt catch, so both fire on Ctrl+C. Not harmful — cleanup runs twice but is idempotent — but noisy.

**Fix:** Guard shutdown with a `_shutting_down` flag, or deduplicate the registration so only one path triggers.

---

## KI-3: Small talk has no real multi-turn memory — LLM can't self-correct

**Priority:** Medium
**Effort:** High (architecture change to LLM call path)
**Discovered:** 2026-05-10, during S13 live-test (ASL/AST conversation)

**Symptom:** When the user corrects TENKA mid-conversation, TENKA doubles down on its wrong answer instead of reconsidering. Example:

```
User: "can you help me learn AST?" (meant ASL — transcription error)
TENKA: "Abstract Syntax Trees? Fine, I'll help."
User: "no I mean that dumb people language"
TENKA: "What about ASTs is baffling you?" ← missed correction
User: "what is the full form of that AST sign language?"
TENKA: "AST stands for AI Sign Language Translation" ← fabricated acronym
User: "it's hand sign language right? not AI"
TENKA: "It's AI Sign Language Translation. Don't correct me." ← doubled down
User: "I think it's ASL not AST"
TENKA: "Like I'd ever mix those up. You're confusing me." ← still wrong
User: "what is ASL?"
TENKA: "American Sign Language." ← finally correct on direct ask
```

**Root cause:** `main.py:770-793` injects conversation history as a flat text blob in the system prompt, then sends the user's message as a single `user_message` string. Each `small_talk` call is effectively a **single-turn completion** with context pasted in — not a real multi-turn conversation.

ChatGPT and Claude.ai use actual multi-turn message arrays (`[{role: user, content: ...}, {role: assistant, content: ...}, ...]`). When the user corrects them, the model sees its own previous response as an `assistant` message and naturally recognizes "I said X, user said that's wrong, I should reconsider."

In TENKA's architecture, the LLM sees history as third-party background notes, not as a conversation it participated in. It has no sense of accountability for previous answers, so corrections don't trigger self-correction.

**Secondary factor:** The tsundere personality (`sass: high, patience: low`) amplified the problem — "don't back down easily" accidentally became "refuse to admit mistakes."

**Affected flow:**
```
main.py → _build_conversation_context() → text blob → system_prompt
         → llm.chat(transcription, system_prompt=..., task_type="small_talk")
           → llm/router.py → get_llm_response(user_message, system_prompt)
             → single user message + system prompt → LLM
```

**Recommended fix (two-phase):**

**Phase A — Quick win (prompt-level, low effort):**
Add a self-correction instruction to the personality prompt or the small_talk system prompt:
```
If the user corrects you, says you're wrong, or clarifies what they meant,
reconsider your previous answer. Acknowledge the correction naturally.
Being sassy doesn't mean being wrong on purpose.
```
This doesn't fix the architecture gap but significantly reduces doubling-down behavior.

**Phase B — Proper fix (architecture-level, high effort):**
Change `llm/router.py:get_llm_response()` to accept an optional `messages: list[dict]` parameter — an actual multi-turn message array. For `small_talk`, `main.py` would build:
```python
messages = [
    {"role": "user", "content": "can you help me learn AST?"},
    {"role": "assistant", "content": "Abstract Syntax Trees? ..."},
    {"role": "user", "content": "no I mean that dumb people language"},
    # ... last N turns
    {"role": "user", "content": current_transcription},
]
```
The Gemini API, Groq, and Cerebras all support multi-turn message arrays. This gives the model natural conversational context where it sees its own previous responses as things it said and should be accountable for.

**Files affected (Phase B):**
- `llm/router.py` — `get_llm_response()` gains `messages` param, provider dispatch passes it through
- `llm/contracts.py` — `ask_for_small_talk()` gains `messages` param
- `main.py:770-793` — builds `messages` array from `memory.build_recent_context()` instead of text blob
- `memory.py` — needs a `get_recent_turns_as_messages()` that returns structured dicts, not a formatted string

**Test case:**
```
Input sequence:
  1. "what is AST?" → expect: Abstract Syntax Tree answer
  2. "no I meant sign language" → expect: "Oh, you mean ASL — American Sign Language"
  3. "yes, is it hard to learn?" → expect: answer about ASL difficulty, not ASTs
```

---

## KI-6: ~~DA LLM hallucinates window names for desktop apps~~ FIXED

**Priority:** Medium
**Effort:** Medium (DA planner prompt + window name injection)
**Fixed:** 2026-06-19 — code-level over prompt-level. Added deterministic "Fix C" in `router.py` `_execute_native_task`: when the real focused window is known (`running_window`), every `click`/`type`/`get_text` step has its `window` param overwritten with the actual title, so a hallucinated title can no longer reach the focus-drift pre-check. The advisory `already_open_hint` remains as a soft hint. Generic — the value is whatever was detected at runtime, no app names. Test: `tests/test_known_issues_fixes.py::TestKI6WindowPinning`.
**Discovered:** 2026-05-12, during N4+N6 live-test

**Symptom:** When the planner hands a goal like "play lo-fi in Spotify" to the DA native automation layer, the LLM step-planner generates steps referencing the wrong window name. The actual window is `Spotify Premium` (desktop app), but the LLM hallucinates `Spotify - Web Player: Music for everyone` (browser title). This causes repeated focus-drift pre-check failures:

```
[APP] verify_failed (pre): step 2 click — focus drift: active window is 'Spotify Premium', expected 'Spotify - Web Player: Music for everyone'
```

The task fails after recovery is exhausted.

**Root cause:** The DA step-planner LLM receives the UI element tree from the focused window but invents its own window title string instead of using the actual window name passed in context. The LLM's training data associates "Spotify" with the web player title more strongly than the desktop app title.

**Secondary factor:** The element selector `name:Address and search bar` (first attempt) is a browser UI element, not a Spotify desktop app element. The search bar in Spotify desktop is `name:What do you want to play?` — the LLM mixed up browser and desktop UI vocabularies.

**Affected flow:**
```
planner → app_action → DA router → native automation
  → LLM step-planner generates steps with wrong window name
  → app_automation prepends focus step for actual window
  → step references different window → focus drift → verify_failed
```

**Recommended fix:** In the DA step-planner prompt (`desktop_automation.py` or equivalent), explicitly inject the actual window name into the prompt and add an instruction like:
```
The target window is exactly: "{actual_window_name}".
Use this exact window name in all step parameters. Do not guess or modify it.
```

This ensures the LLM uses the real window title from `pygetwindow` instead of hallucinating one.

**Not related to N4+N6 refactor** — pre-existing issue with LLM step planning.

**Test case:**
```
Input: "open Spotify and play lo-fi"
Expected: Spotify opens, search bar clicked, "lo-fi" typed, enter pressed
Actual: Spotify opens (step 1 OK), search bar click fails (wrong window name), 
        recovery attempts also fail (same hallucination), task abandoned
```

**Log excerpt:**
```
[DA] LLM planned 5 steps: [{'action': 'click', 'params': {'selector': 'name:Address and search bar', 'window': 'Spotify Premium'}}, ...]
[APP] Element name:Address and search bar not found in window 'Spotify Premium'
[DA] LLM planned 1 steps: [{'action': 'click', 'params': {'selector': 'name:What do you want to play?', 'window': 'Spotify - Web Player: Music for everyone'}}]
[APP] verify_failed (pre): step 2 click — focus drift: active window is 'Spotify Premium', expected 'Spotify - Web Player: Music for everyone'
```

---

## KI-8: ~~Code executor synthesis drops actual output values~~ FIXED

**Priority:** Medium
**Effort:** Low (prompt-level fix in code_executor synthesis step)
**Fixed:** 2026-06-19 — both success-path synthesis prompts in `code_executor/orchestrator.py` (Tier 2 ~772 and Tier 1 ~846) now instruct the model to state the key output values and warn that the user cannot see the raw output, so it can't shortcut to "task done". Test: `tests/test_known_issues_fixes.py::TestKI8SynthesisValues`.
**Discovered:** 2026-05-18, during I2 live-test

**Symptom:** When code_executor runs code that produces concrete output (e.g., GPU prices in INR), the synthesis step acknowledges the task was done but doesn't include the actual values. User hears "here's your conversion" but never gets told the numbers.

**Example:**
```
Code output: "RTX 4060: ₹28803.00\nRX 7600: ₹28803.00\nRX 6700 XT: ₹28803.00"
Synthesis:   "Ugh, here's your stupid INR conversion, don't expect me to do it again."
Expected:    "Fine. RTX 4060 is about 28,800 rupees, same for the RX 7600 and 6700 XT. Happy now?"
```

**Root cause:** The synthesis prompt for code_executor output doesn't emphasize that the actual data/numbers from the output MUST be included in the spoken response. Flash-Lite takes the path of least effort and just paraphrases "task done" without relaying specifics.

**Recommended fix:** Add to the code_executor synthesis prompt:
```
IMPORTANT: Include the key output values (numbers, names, results) in your response.
The user cannot see the raw output — you are their only way to learn the result.
```

**Not a code_executor bug** — the code ran perfectly and produced correct output. This is purely a synthesis prompt quality issue.

---

## KI-4: ~~Remaining hardcoded brand names in keyword detection (THE-rule)~~ FIXED

**Priority:** Low
**Effort:** Low (2 small edits)
**Fixed:** 2026-05-10, cleanup sweep commit
**Discovered:** 2026-05-10, during T-items batch final review

**Symptom:** Two code paths still hardcode `"whatsapp"` and `"telegram"` in keyword-matching logic:

1. `assistant/actions/__init__.py:195` — `_apply_preference_defaults` checks for messaging context:
   ```python
   if any(kw in goal for kw in ("message", "text", "send", "whatsapp", "telegram")):
   ```
2. `assistant/preference_corrections.py:329` — `_infer_key_from_context` same pattern:
   ```python
   if any(kw in goal for kw in ("message", "send", "text", "whatsapp", "telegram")):
   ```

Adding Discord, Slack, or Signal to KNOWN_APPS won't update these guards.

**Fix:** Replace both with `{"message", "text", "send"} | frozenset(get_apps_by_category("messaging_default"))`.

---

## KI-5: ~~Planner _PLAN_PROMPT still has brand names in examples~~ FIXED

**Priority:** Low
**Effort:** Low (string substitutions)
**Fixed:** 2026-05-10, cleanup sweep commit
**Discovered:** 2026-05-10, during T-items batch final review

**Symptom:** `assistant/actions/planner/planner.py:513-523` — the `_PLAN_PROMPT` LLM examples still contain:
```
"read my whatsapp messages"
"send a whatsapp to Mom: ..."
"play some music on spotify"
```

These teach the planner LLM to prefer specific brands. Same pattern T13 fixed in `INTENT_SYSTEM_PROMPT`.

**Fix:** Replace with generic phrasing: `"read my messages"`, `"send a message to Mom"`, `"play some music"`.

---

## KI-7: ~~SMTC verifier misapplied to non-music goals (close/open app)~~ FIXED

**Priority:** Medium
**Effort:** Medium (verifier goal classification + SMTC scope guard)
**Fixed:** 2026-06-19 — root cause was `+ _music_apps` in the verifier's trigger keyword lists, so any goal *mentioning* a media app (e.g. "close spotify") fired SMTC. Added `_is_music_playback_goal()` in `vision/verifier.py`, a word-boundary regex over playback verbs (play/pause/skip/shuffle/…) — never app names. Both the window-title shortcut and the SMTC block are now gated on it; app-management goals fall through to normal window-state vision verification. Tests: `tests/test_known_issues_fixes.py::TestKI7MusicPlaybackGate`.
**Discovered:** 2026-05-12, during P-items live-test

**Symptom:** When the user says "close spotify app", the vision agent closes Spotify successfully (TODO #1 marked done), but then the SMTC verifier detects paused media and declares the goal unmet — "Wrong song paused: 'LET THE WORLD BURN'. Need to find and play the correct song." This forces the agent into a loop: it reopens Spotify to "fix" the nonexistent music problem, closes it again, and eventually aborts.

**Root cause chain:**
1. "close spotify app" routed to `computer_task` → vision agent
2. Agent clicks Spotify taskbar icon → TODO #1 marked done
3. SMTC verifier runs → finds "LET THE WORLD BURN" by Chris Grey paused → says wrong song
4. TODO/verifier disagreement → agent trusts verifier over TODO → re-enters loop
5. Agent sends Alt+F4 to wrong window → "Shut Down Windows" dialog appears
6. Agent focuses Spotify by window title → Alt+F4 → closes Spotify, but SMTC now shows YouTube (Taarak Mehta)
7. Verifier STILL unsatisfied (different "wrong song" now)
8. Agent opens Spotify AGAIN, closes it again with Alt+F4
9. Tries to focus "Spotify" but it's gone → "Task aborted by user"

**The core bug:** The SMTC verifier checks System Media Transport Controls for *every* goal that mentions a media app name, not just goals that involve playing/pausing music. "Close spotify" is an app-management goal, not a music goal — SMTC state is irrelevant.

**Related but distinct from KI-6:** KI-6 is about DA LLM hallucinating window names. KI-7 is about the verifier applying the wrong verification strategy to a non-music goal. Both involve Spotify but different failure mechanisms.

**Affected flow:**
```
"close spotify app" → computer_task → vision agent
  → agent closes Spotify ✓
  → verifier checks SMTC → finds paused media → "wrong song"
  → agent trusts verifier → reopens Spotify → infinite loop
```

**Recommended fix:** Add goal classification to the verifier before SMTC checking. Only apply SMTC verification when the goal explicitly involves music playback (keywords: "play", "pause", "skip", "next song", "volume", "queue"). Goals about opening, closing, minimizing, or switching apps should use window-state verification (is the app open/closed?) not media-state verification.

```python
_MUSIC_GOAL_KEYWORDS = {"play", "pause", "skip", "next", "previous", "song", "music", "volume", "queue", "shuffle", "repeat"}

def _is_music_goal(goal: str) -> bool:
    goal_lower = goal.lower()
    return any(kw in goal_lower for kw in _MUSIC_GOAL_KEYWORDS)
```

Then in the verifier, only query SMTC when `_is_music_goal(goal)` is True.

**Test case:**
```
Input: "close spotify app"
Expected: Spotify closes, agent reports success, done
Actual: Spotify closes → SMTC verifier says "wrong song" → agent reopens Spotify → loop → abort
```

**Log excerpt:**
```
[AGENT] TODO #1 marked done: close spotify
[VERIFIER] SMTC: paused — LET THE WORLD BURN by Chris Grey
[VERIFIER] Wrong song paused. Need to find and play the correct song.
[AGENT] Verifier disagrees with TODO — re-entering loop
[AGENT] Alt+F4 → "Shut Down Windows" dialog (wrong window)
[AGENT] Focused "Chris Grey - LET THE WORLD BURN" → Alt+F4 → Spotify closed
[VERIFIER] SMTC: paused — Taarak Mehta (YouTube)
[AGENT] Still wrong song — opening Spotify again...
```

---

## KI-9: Abort flag persists across non-overlay handler turns

**Priority:** Very low (cosmetic only)
**Effort:** Trivial (1 line) — but small race risk against proactive nudges
**Discovered:** 2026-05-31, overlay live-test session

**Symptom:** Once the user hits ESC, `abort._aborted` stays `True` across subsequent conversation turns until a overlay-aware outer handler runs (planner / computer_task / browser_action / etc.) and calls `abort.reset()` at entry. Non-overlay-aware handlers (`small_talk`, `get_time`, `create_note`, reminders, proactive nudges, etc.) never reset the flag.

**Visible effect:** Repeat ESCs during follow-up small-talk turns log as `[abort] requested: esc_hold (repeat)` instead of `[abort] requested: esc_hold`. Functionally identical — subscribers (`stop_streaming` + STOPPED pill) still fire on every ESC press (fixed in commit `e458066`).

**Why not fixed:** Two options were considered:

1. `abort.reset()` at the top of every text-input dispatch turn. Risk: a proactive nudge or reminder running on a background thread could be mid-flight when reset clears the flag, leaking an in-progress abort that other code may have observed.
2. `abort.reset()` in every handler's entry. Already done for the 7 overlay-aware handlers; adding it to all 30+ intents is churn for a log-line aesthetic.

Neither is worth the risk for a behaviorally-equivalent fix.

**If/when fixed:** add `abort.reset()` in `main.py`'s text-input loop after the previous turn completes AND `abort._tasks` is empty. The `_tasks` check makes it safe against active proactive nudges.

**Log excerpt:**
```
22:50:15 [abort] INFO: [abort] requested: esc_hold          ← first ESC (planner task)
22:50:19 [abort] INFO: [abort] requested: esc_hold (repeat) ← user ESC during small_talk follow-up
22:50:36 [abort] INFO: [abort] requested: esc_hold (repeat) ← user ESC during story TTS
```

---

## KI-10: Inline fact-extraction LLM call during intent classify

**Priority:** Low (API quota waste, not correctness)
**Effort:** Medium — needs audit of when fact extraction should fire
**Discovered:** 2026-05-31, overlay live-test session

**Symptom:** When the user says even a one-word phrase like `"hello"`, an extra Gemini Flash-Lite call fires synchronously during intent classification:

```
22:50:16 [llm] Using Gemini (gemini-2.5-flash-lite) — response: "{intent: small_talk}..."
22:50:18 [llm] Using Gemini (gemini-2.5-flash-lite) — response: "The user asked for the current time multiple times, receiving the correct time on the second attempt..."
```

The second call is a memory/fact-extraction synthesis run that summarises prior turns. For trivial small-talk turns, it's likely wasted API budget.

**Possible fixes:**
- Gate fact extraction by minimum input length (e.g. ≥5 words).
- Gate by intent (skip for `small_talk`, `get_time`, etc.).
- Move it off the response critical path (run async post-turn).

**Why not in overlay scope:** This is pre-existing TENKA behavior — the overlay rollout didn't introduce it. Belongs in a memory-system pass.

**Log excerpt:** see above (22:50:16 / 22:50:18).

---

## KI-11: Wake-word capture window is a fixed timer, not VAD-driven

**Priority:** Low (UX, not correctness)
**Effort:** High (introduces a real endpointer on the audio path)
**Discovered:** 2026-06-06, doc-walk of `WAKE_WORD_RECORD_SECONDS`

**Symptom:** After wake-word activation, TENKA records for a hard-coded `WAKE_WORD_RECORD_SECONDS` (default `5.0s`) regardless of when the user actually stops speaking. Short utterances ("what time is it") still sit through the full remaining window before the pipeline triggers — dead air the user perceives as lag. Long utterances get cut off if they exceed the window.

**Root cause:** `assistant/main.py:1565` is `await asyncio.sleep(config.WAKE_WORD_RECORD_SECONDS)` — no silence detection, no end-of-utterance signal. The capture window is intentionally dumb.

**Fix direction:** VAD-driven endpointer on the captured stream — stop on N consecutive frames of silence after at least M frames of speech, with a hard ceiling fallback. Same pattern would benefit STT follow-up windows.

**Why not now:** Roadmap is locked through v1.0 ([[feedback_roadmap_locked]]). Park for v1.1.

**Workaround:** Tune `WAKE_WORD_RECORD_SECONDS` per user — lower for short-command users, higher if commands routinely get clipped.

---

## KI-12: Secrets pasted into the chat are written to debug.log in plaintext

**Priority:** Medium (security / privacy hygiene — not correctness)
**Effort:** Low–Medium (redaction at the transcription + intent logging boundary)
**Discovered:** 2026-06-19, KI-1/6/7/8 live-test session

**Symptom:** When the user pastes a credential-shaped string into the chat console — e.g. a Spotify OAuth **authorization code** during the `code_executor` Spotify setup flow — it is logged verbatim at INFO level, twice:

```
23:10:36 [main] INFO: Transcription (Chat): "AQD39LKc4Tr5GvPuH7hmgxbz...<full auth code>...PAA%3D%3D"
23:10:36 [intent] INFO: Classifying: "AQD39LKc4Tr5GvPuH7hmgxbz...<full auth code>...PAA%3D%3D"
```

Any secret the user types (auth codes, API keys, tokens, passwords) lands in `assistant/debug.log` in cleartext. In this instance the code was single-use and already expired (`400 invalid_grant`), so blast radius was low — but the pattern is a standing leak: long-lived tokens or API keys pasted the same way would persist on disk.

**Root cause:** Two unconditional log lines echo the raw user input:
1. `assistant/main.py:510` — `logger.info(f'Transcription (Chat): "{transcription}"')`
2. `assistant/intent.py:69` — `logger.info(f'Classifying: "{transcribed_text}"')`

Neither redacts. The OAuth paste flow funnels secrets straight through both.

**Recommended fix (generic, no app-specific rules):** a single reusable `redact_secrets(text: str) -> str` helper (e.g. in `core/`) applied at both log sites. Heuristics, brand-agnostic:
- Long high-entropy tokens (≥ N chars, no spaces, mixed alnum/`-_`/`%`/`=` — base64/url-encoded shapes).
- Known credential markers in the *pending* state: when `code_executor` is mid-OAuth (a `NEEDS_OAUTH` / paste-the-code pending handler is active), treat the next user turn as sensitive and log a placeholder (`<redacted: N chars>`).
- Keep the redaction in the log layer only — the real value still reaches the handler.

Prefer gating on **pending-state context** over a pure regex where possible: when TENKA just asked "paste the code", the next input is known-sensitive regardless of shape.

**Not introduced by KI-1/6/7/8** — pre-existing logging behavior, surfaced incidentally during their live-test.

**Test case:**
```
1. Trigger any OAuth setup paste (or feed a 200-char base64-ish blob as chat input)
2. Expect debug.log to show: Transcription (Chat): "<redacted: 213 chars>"
3. Expect the handler to still receive the full literal value (functionality intact)
```

