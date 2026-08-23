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

## KI-12: ~~Secrets pasted into the chat are written to debug.log in plaintext~~ FIXED

**Priority:** Medium (security / privacy hygiene — not correctness)
**Effort:** Low–Medium (redaction at the transcription + intent logging boundary)
**Discovered:** 2026-06-19, KI-1/6/7/8 live-test session
**Fixed:** 2026-08-16, milestone 6a.5 — in two parts, because the ticket only described half the problem.

*Redaction* was the part this ticket asked for, and `redact_secrets` was already wired into both named sites. 6a.5 found it leaked on nine realistic shapes an adversarial review proved: pretty-printed JSON with a strong label, lowercase snake_case compounds (`db_pass`, `client_secret`), camelCase, PEM key bodies, values split by a hard line-wrap, `user:pass@host` URLs, all-digit tokens, docker-compose `- VAR=` leads, and `:=`. All nine closed. Over-redaction was fixed in the same pass — a naive fix had blanked 90 working source lines, and an unterminated `BEGIN PRIVATE KEY` marker used to blank every line after it.

*Framing* was the part nobody had noticed. The log lines interpolated without `!r`, so `redact_secrets` — which is about secrets, not newlines — passed `\n` straight through, and 8,000 characters of chat text could write as many fabricated log lines as it liked into the file an operator greps after an incident. All four sites (`main.py` ×2, `intent.py`, `io/audio/tts.py`) now use `{…!r}`, matching what `routes/pairing.py` already did.

Tests: `tests/test_security_pass_6a5_redaction.py`, `tests/test_6a5_api_fixes.py`, `tests/test_redact.py`.

**Was still open, and mis-pointed:** this note said "tracked separately as KI-15", but KI-15 is the unfenced-facts issue -- the storage-write gap had no ticket. Filed and fixed as [KI-29](#ki-29) on 2026-08-22, after it turned out to be wider than this note claimed (every turn, not just extracted facts) and to have already reached cloud backup.

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


---

## KI-13: ~~Pending state has no owner — cross-device confused deputy~~ FIXED

**Priority:** Medium (security)
**Discovered:** 2026-08-16, milestone 6a.5 adversarial review
**Fixed:** 2026-08-19, milestone 6b (`fix/6b-pending-state-owner`, squashed to main `942a411`) — `tests/test_6a5_predispatch_gate.py`'s strict `xfail` removed and now passing.

**What shipped:** `core/principal.py`'s `current_principal` contextvar, set alongside `current_grants` at every place a turn's identity is known (voice/console set `LOCAL_PRINCIPAL`; an authenticated device turn sets `f"device:{device_id}"`, a namespace no device can spoof into since the prefix is never built from caller input). `PendingState` records the principal at arm time; both answer sites in `pending_handlers.py` compare it at answer time before honouring a "yes".

**Two ways the shipped fix differs from the original recommendation, learned by hitting the failure modes:**

1. **A mismatch skips the row rather than refusing the whole turn.** The first cut refused the turn outright, which meant a phone arming a pending and then *the operator at their own keyboard* saying anything got refused by it — for up to 300 seconds. Skipping means a non-owner's message falls through as an ordinary capability-gated turn instead of eating the confirmation slot; the confused deputy is still closed, because the skip happens whether or not the non-owner holds the capability the row is guarded by.
2. **The operator-facing notice lands on the owner's next answer, not on the intruder.** The original "refuse loudly" wording aimed the loudness at the wrong party — there is no useful message to show a device that was silently skipped. `PendingState.note_foreign_attempt()` / `take_foreign_attempts()` record the attempt and surface a short notice the next time the actual owner answers, so the operator learns something else tried, without the row itself ever being disturbed.

One arming site needed moving rather than gating: a code-executor knowledge-approval handler that armed *itself* inside its own confirmation check bypassed the ownership comparison entirely (it never reached the loop-level guard); the fix relocated its arming to proposal-creation time, carrying the creating turn's principal forward, rather than special-casing the loop for one row.

Tests: `tests/test_6a5_predispatch_gate.py` (`xfail` removed), `tests/test_6b_principal.py` (the new suite, including an AST walk over every arming site in `main.py` so a future site cannot silently forget to record a principal).

**Correction, added after the 6b live test (2026-08-20): this closed one of three doors.** KI-13 as shipped covered who may *answer* a pending state. Nothing mirrored it on who may *arm* one, or who may *clear* one, and the live test hit both:

- **Arming.** A device correctly skipped as an answer fell through to ordinary intent classification, reached a handler, and armed the *same* state via the ambient-principal default — taking ownership of it. The operator's own next answer was then refused as foreign against their own request: a denial of service on their own confirmations, observed live.
- **Clearing.** `camera.py::handle_camera_look` discarded an active pending state unconditionally, with nothing upstream confirming the caller owned it.

Both are now fixed. `pending.try_arm` (commit `3865e04`) and `pending.try_clear` (commit `d2c23de`) apply the same ownership condition the answer side already used — across 18 arm sites in nine modules (three of them `teaching_session` sites a grep missed because the state isn't `pending_*`-named) and the one real clear-side gap a 44-site sweep found. Each guard is backed by a tree-wide AST sweep that is itself a test (`test_every_pending_arm_outside_the_two_reasoned_exceptions_uses_try_arm`, `test_every_pending_clear_outside_the_reasoned_exceptions_is_gated` in `tests/test_6b_principal.py`), so a future site added without the guard breaks the sweep rather than shipping silently.

A refused arm deliberately returns the identical text an ordinary first arm would — non-disclosure means refusing alone did not bound how often a remote caller could try, so it could make the machine speak a 13–18 second confirmation challenge on every message. See [KI-23](#ki-23) for how that was bounded, and [KI-27](#ki-27) for the sweep's one known blind spot — it is name-based, and a clear through a loop-local variable is invisible to it.

Tests: `tests/test_6b_principal.py`.

---

## KI-14: A URL delegated to a file is still chosen by the file

**Priority:** Medium (security)
**Effort:** Medium (a pending state plus a resolver, following the existing pending-handler pattern)
**Discovered:** 2026-08-16, milestone 6a.5 adversarial review
**Status:** open. Explicitly out of scope for milestone 6b (spec §1) — not touched, not made worse.

**Symptom:** "Read notes.txt and open the site named in it" is a legitimate request, and 6a.5 keeps it working. But the file picks the destination, and a query string on that URL carries data outward. Reduction (one public http(s) URL, no userinfo, no private hosts) and authorisation (the user's own words must carry a navigation verb) narrow it; neither closes it.

**Recommended fix:** a consent gate — show the resolved URL and require confirmation before the first navigation to a host the user did not name themselves. Deliberately out of 6a.5's scope: it needs a new pending state, and the milestone was already three times its planned size.

---

## KI-15: `KNOWN FACTS ABOUT THE USER` is replayed into the system prompt unfenced

**Priority:** Medium (security)
**Effort:** Low (fence the block, as `render_untrusted_block` already does elsewhere)
**Discovered:** 2026-08-16, milestone 6a.5 adversarial review
**Status:** open. Explicitly out of scope for milestone 6b (spec §1) — not touched, not made worse.

**Symptom:** `_build_facts_context` concatenates stored facts into `build_personality_prompt()` with no delimiter and no untrusted label — the most trusted position in the tree — on every subsequent turn.

**What 6a.5 did:** cut the supply. `store_memory` left the payload class, so planted step output can no longer become a stored fact. The read side is untouched, so anything already stored, or stored by another route, still replays unfenced.

---

## KI-16: Topic resolution rewrites pronouns with the previous turn's trailing noun

**Priority:** Medium (correctness / usability, not security)
**Effort:** Unknown — needs diagnosis before a fix is proposed
**Discovered:** 2026-08-16, during 6a.5 live testing
**Status:** open. Explicitly out of scope for milestone 6b (spec §1) — not touched, not made worse.

**Symptom:** "it" is replaced with a noun phrase lifted from the previous turn, frequently the wrong one. Observed in one session:

| Typed | Classified as |
| --- | --- |
| `...the site named in it` | `...the site named in a public` |
| `...the site named in it` | `...the site named in the site` |
| `...the site named in it` | `...the site named in the shell command` |

The third injected the word "shell", which the then-live `DANGEROUS_PATTERNS` deny-list refused — so the turn failed for a reason unrelated to anything the user typed. It broke three live tests before being recognised as a feature misfiring rather than a test error.

**Related:** `main` @ `228602a` is "fix: stop topic resolution rewriting pasted code" — the same mechanism, a different input class. Worth asking whether the feature earns its keep before repairing it again.

---

## KI-17: ~~A 6b tunnel pointed at the existing port inherits the `local` policy~~ FIXED

**Priority:** was **High — a landmine for the next milestone, not a live defect**
**Discovered:** 2026-08-16, milestone 6a.5 adversarial review
**Fixed:** 2026-08-19, milestone 6b — three independent layers, the third load-bearing on its own.

**What shipped:**

- **L1 — TENKA builds every tunnel's argv.** The target port always comes from the fixed port registry (`io/api/listeners.py`'s `port_for`), never from anything an operator types. A spawn-time assertion checks the real argument vector against the registry: the port is registered, maps to this adapter's own policy name, and is not the port `local` holds.
- **L2 — preflight reconciliation, on every start.** Before binding, the Tailscale adapters read `tailscale serve status --json` and refuse a pre-existing mapping that conflicts, naming the offending entry rather than silently repairing it. The predicate is two checks: unconditionally refuse any entry proxying into `local`'s port (KI-17 proper, from either adapter), and — only within this adapter's own public-port entry — refuse a stale target pointing somewhere other than this transport's local port. A malformed entry degrades that one entry, never the whole sweep.
- **L3 — per-listener `Host` scoping. The load-bearing layer.** The `local` listener's `HostGate` accepts **loopback names only** (`127.0.0.1`, `localhost`, `[::1]`). Every tunnel forwards the public authority in `Host` (`*.ts.net`, `*.trycloudflare.com`), so a tunnelled request arriving on the local port is refused **421 before authentication, before policy lookup, before any route runs** — and this holds even against a tunnel TENKA never launched and knows nothing about, which is more than a kernel-assigned port could ever offer.

**Stated gap, recorded rather than claimed away:** `cloudflared --http-host-header 127.0.0.1:8787` rewrites `Host` to a loopback name and defeats L3. That requires an attacker already executing processes on this machine, at which point the local listener is not the weakest thing available to them. L2 catches the honest-mistake version (a stale hand-set `tailscale serve` mapping); L3 catches everything short of a local shell.

**Two gaps recorded for the adversarial round rather than closed here** (both contained by L3): the preflight sweep is `Web`-only, so a `--tls-terminated-tcp` mapping and Tailscale's `Foreground` map are both invisible to it (see [KI-19](#ki-19)); and a leftover `AllowFunnel` flag can still escape the check under an unrecognised document shape (see [KI-20](#ki-20)).

Tests: `tests/test_6b_ki17.py`, `tests/test_api_hardening.py`'s per-listener host-scoping suite, plus `tests/test_6b_transport_adapters.py` and `tests/test_6b_transport_cloudflare.py` for each adapter's own preflight coverage.

---

## KI-18: ~~The admin gate is not isolated by any test~~ FIXED

**Priority:** was Medium (security — a config-time footgun, not a live defect)
**Discovered:** 2026-08-19, milestone 6b (Task 13's own vacuity audit, generalised by Task 18's review)
**Fixed:** 2026-08-19, milestone 6b (`0e60df2`) — pinned with a new test rather than a code change, because the gate itself (`admin_capability_satisfied`, checked first on capability and second on `policy.admin`) was already correct; only its isolation from the capability check was untested.

**What shipped:** `tests/test_6b_raise_routes.py::test_ki18_the_admin_gate_holds_even_when_the_ceiling_carries_system_control` builds the one shape that isolates the flag — a constructed `ListenerPolicy` (`admin=False`, `ceiling` holding `SYSTEM_CONTROL`) mounted at a synthetic port so no shipped policy is touched. `GET /v1/devices` refuses 403 against it; the test then flips only `admin` to `True` on the identical policy and the same request succeeds, proving the 403 really was `require_admin`'s own gate and not an unrelated refusal. `test_a_raise_never_changes_admin_or_bearer_or_secure_cookie` pins the companion invariant — a live raise on `tailnet` never flips `policy.admin`, `allow_bearer`, or `secure_cookie` — checked both off the module data and over the wire (`GET /v1/devices` still 403, a bearer header still 401 there).

Two independent mutations confirm both halves, re-run 2026-08-20: dropping `policy.admin` from `admin_capability_satisfied`'s predicate (`security.py`) reds the first test; setting `admin=True` on the `tailnet` policy (`policy.py`) reds the second.

The entry's own "What to check first" asked for exactly what now exists — a constructed `ListenerPolicy` whose `ceiling` holds `SYSTEM_CONTROL` while `admin` is `False`, rather than one of the shipped policies. Worth saying so, because it makes the entry's own advice the thing that resolved it. Correction to the entry's earlier count: there are **three** shipped policies now (`local`, `tailnet`, `funnel`), not four — `quick` was removed during the 6b live test. Its ceiling was unreachable by construction (pairing over `quick` is refused outright, `quick` accepts no bearer credential, and cookies are host-scoped so no cookie issued anywhere else is ever sent to it), so no device could ever authenticate over it.

Tests: `tests/test_6b_raise_routes.py`.

---

## KI-19: The `tailscale serve` preflight sweep only sees `Web` mappings

**Priority:** Low (contained by KI-17's layer 3)
**Effort:** Medium (two more parsers: a bare `host:port` string for TCP forwards, and a recursive descent into `Foreground`)
**Discovered:** 2026-08-19, milestone 6b (Task 7's fix rounds)

**Symptom:** KI-17's layer 2 preflight (`transports/tailscale.py`) walks `tailscale serve status --json`'s `Web` map to catch a pre-existing mapping pointed at the local Studio port. Two shapes never reach that walk: a `tailscale serve --tls-terminated-tcp 443 tcp://127.0.0.1:8787` mapping, whose `TCPForward` value is a bare `host:port` string rather than a URL and needs different parsing entirely; and a **foreground** (non-`--bg`) serve, which Tailscale nests under its own `Foreground` map with its own `Web`/`AllowFunnel` pair that neither the sweep nor the `AllowFunnel` check descends into.

**How you would know:** hand-set either shape (`tailscale serve --tls-terminated-tcp 443 tcp://127.0.0.1:8787`, or a foreground `tailscale serve https / http://127.0.0.1:8787` without `--bg`) and start a TENKA transport — preflight will not refuse it, even though the mapping conflicts.

**What to check first:** layer 3 (`HostGate`'s loopback-only local listener) still refuses the resulting tunnelled request with 421 regardless — this is a hole in the honest-mistake catch, not in the load-bearing defence. Confirm layer 3 is still in place before treating this as urgent.

---

## KI-20: A leftover `AllowFunnel` flag can still escape the preflight check

**Priority:** Low (contained by KI-17's layer 3; the one check that exists is correct for its own scope)
**Effort:** Low (the existing top-level `AllowFunnel` read just needs to run before any shape guard, and to descend into `Foreground`)
**Discovered:** 2026-08-19, milestone 6b (Task 7 fix rounds 3–4)

**Symptom:** A leftover `AllowFunnel` entry on tailnet's public port 8443 would publish TENKA's only raisable listener to the open internet while she believes it is tailnet-only. The shipped check reads `AllowFunnel` at the top level, independently of the `Web` shape guard, specifically for `tailnet`'s own public port — closing the case the fold-in was written for. It does not sweep `AllowFunnel` in general: a flag sitting inside a non-dict `Web` entry, or inside the `Foreground` map (see [KI-19](#ki-19)), is still invisible to it.

**How you would know:** a leftover `AllowFunnel: true` under `tailnet`'s public port with a malformed or foreground `Web` shape around it passes preflight silently.

**What to check first:** whether the check has been generalised to hoist above every shape guard and to walk `Foreground` the same way `Web` is walked, rather than being re-scoped narrowly again to `tailnet`'s own document shape.

---

## KI-21: A cancellation helper can leak a non-cancellation exception through its own discard contract

**Priority:** Low (dormant — no reachable path today)
**Effort:** Low (one direct unit test, before the helper gets a second caller)
**Discovered:** 2026-08-19, milestone 6b (Task 9, self-reported, deferred by design)

**Symptom:** `transports/manager.py`'s `_settle_uncancellably(fut)` re-shields `fut` on every loop-cycle cancellation but only catches `asyncio.CancelledError` from the shield; if `fut` itself completes with a **non-cancellation** exception while a caller is mid-await, that exception propagates straight out of `_settle_uncancellably` rather than being handed to the caller to discard. `_wait_uncancellably`, the one caller whose whole contract is "discard `fut`'s outcome, cancellation or not," would then fail to discard it — contradicting its own docstring.

**Why it is dormant, not live:** `_wait_uncancellably`'s sole caller today only ever waits on a task that `_reraise_if_still_cancelling()` guarantees ends `CANCELLED` — never on a task that can raise something else. The `if not fut.cancelled(): fut.exception()` line in `_wait_uncancellably` is consequently dead code with no test exercising the branch it exists for.

**What to check first, before the helper gets a second caller:** a direct unit test on `_wait_uncancellably` against a bare `asyncio.Future` resolved with a non-cancellation exception (e.g. `fut.set_exception(RuntimeError(...))`, no cancellation involved) — confirm the call returns cleanly rather than propagating. If that test does not exist yet, `_wait_uncancellably` is not safe to reuse for a future call site without it.

---

## KI-22: The raise audit trail records reach, not consumption

**Priority:** Medium (a known, deliberate trade — not a defect, but a real limit on incident forensics)
**Effort:** N/A — accepted trade-off, not a bug to fix without a broader design change
**Discovered:** 2026-08-19, milestone 6b (Task 10 review; spec §3.6 amended in place)

**Symptom:** Spec §3.6 asks for "an audit event on every request the raise puts capabilities in reach of." What shipped audits **reach**: `applied = grants - effective(issued, policy)` is a function of the raise and the policy alone, computed once per authenticated request on a raised listener, and never consults what the request actually went on to do. A Studio status poll made while a raise is live gets exactly the same audit line as a request that used the raised `EXECUTE` capability to run code.

**Why it shipped this way:** recording true *consumption* means recording inside `require()` **and** every site that checks a grant without going through it — and enumerating those sites is precisely the class of failure this project has already lost two Criticals to (see `boundaries-need-their-bypasses-enumerated`). Recording once, at the single choke point every raised request passes through regardless, is the defensible trade, and over-recording is the fail-safe direction.

**The cost, stated plainly:** an operator reading the audit trail after a raise's window (up to 7 days) cannot tell "a raised capability was walked through" from "the door was merely open." The method and path in each entry narrow it down after the fact, but do not close the gap.

**What to check first** if this needs to become load-bearing later: whether the sites that check a grant without calling `require()` have since been enumerated and closed — if so, moving the audit point (or adding a second one) inside `require()` becomes a much cheaper change than it was at 6b.

---

## KI-23: ~~A refused arm could still make the machine speak on every remote message~~ FIXED

**Priority:** was Medium (security/availability — a denial-of-service shape, not a disclosure one)
**Discovered:** 2026-08-20, milestone 6b live test (Part 8)
**Fixed:** 2026-08-20, milestone 6b (`d2c23de`).

**What shipped:** a refused arm deliberately returns the same text an ordinary first arm would — that non-disclosure is the point, so a remote caller can't tell "you were skipped" from "you just armed a real confirmation" — but it meant refusing alone did not bound how often a remote device could try. A remote device could make the machine speak a 13–18 second confirmation challenge on every message it sent, with no cap anywhere on the path. Now bounded at the one choke point every remote turn crosses before it can reach dispatch or TTS: `POST /v1/chat` spends the existing per-device rate limiter (`throttle(Capability.CHAT_SEND, "chat_send", ...)` in `routes/chat.py`), already bounded against unbounded growth. A throttled caller gets an explicit 429, never a silent drop and never audio. The keyboard is unthrottled structurally — local sources never enter that route at all.

Tests: `tests/test_api_chat.py::test_repeated_chat_sends_from_one_device_are_throttled`, `::test_a_different_device_is_not_throttled_by_anothers_budget`.

---

## KI-24: ~~A tier-2 re-arm on a fresh thread recorded no owner at all~~ FIXED

**Priority:** was Medium (security — pre-existing, unrelated to 6b, surfaced by the new arm guard)
**Discovered:** 2026-08-20, milestone 6b (found while auditing arm sites for KI-13's third-door fix)
**Fixed:** 2026-08-20, milestone 6b (`d2c23de`).

**What shipped:** `pending_file_search`'s tier-2 re-arm ran on a fresh thread, and a thread inherits no `contextvars` — so it armed with **no principal at all**, unownable by construction. Pre-existing and unrelated to 6b's own work; it was found only because the new arm guard forced someone to ask who owned each arming site. The principal is now captured on the loop and passed explicitly into the thread rather than relying on ambient context to cross a thread boundary it never crosses.

Tests: `tests/test_6b_principal.py::test_the_tier2_rearm_carries_the_searchers_principal`.

---

## KI-25: `package_studio_ui.py` can vendor a bundle from a failed build and stamp it with the current contract

**Priority:** Low (dev tooling, not a runtime security issue)
**Effort:** Low–Medium (a success marker the packager requires, or a refusal when source is newer than `out/`)
**Discovered:** 2026-08-20, caught by accident during the 6b live test — a build failed on a missing dependency and the packager ran anyway.

**Symptom:** `tools/package_studio_ui.py` reopens and checks the finished archive for the marker and the contract hash, but neither check is derived from the bundle's own freshness. `ui.contract_hash()` fingerprints the **daemon's** OpenAPI schema and compares it to the marker written at packaging time — it catches "bundle built against an older API." It cannot catch "bundle content is stale but was packaged just now," because the marker is not derived from the bundle's content at all. So the one freshness guard that exists passes stale content clean.

**How you would know:** let `npm run build:bundled` fail (e.g. a missing dependency after a merge, with `node_modules` not reinstalled) and run `package_studio_ui.py` anyway — it packages whatever is sitting in `out/` from the previous successful build, stamps it with the current daemon's contract hash, and reports success.

**What to check first:** whether either candidate fix has landed — a success marker that `build:bundled` writes into `out/` and the packager requires, or a refusal when anything in the Studio source tree is newer than `out/`. Neither exists today.

---

## KI-26: `PairDeviceDialog.carryState()` special-cases `local` by name

**Priority:** Low (a client-side assumption, not a daemon-side defect)
**Effort:** Low on the daemon side (publish `local` in the transports listing)
**Discovered:** 2026-08-20, milestone 6b live test (Ruling 82)

**Symptom:** `carryState()` (`PairDeviceDialog.tsx:138`, Studio repo) opens with `if (transportName === LOCAL_TRANSPORT) return "ceiling"` — a hardcoded name comparison, the exact pattern removed from the daemon side the same day (see `quick`'s removal under [KI-18](#ki-18)). Not laziness: `GET /v1/transports` lists *transports*, and `local` is a listener with no row, so there is nothing for the dialog to read a ceiling from. The shortcut encodes "local carries everything" as an assumption rather than reading it, and would silently lie if `local`'s ceiling ever narrowed.

**How you would know:** narrow `local`'s ceiling in `io/api/policy.py` and watch the pair dialog keep offering every capability against it anyway — nothing in Studio would notice.

**What to check first:** whether the daemon-side fix has landed — listing `local` in the transports payload (or exposing its policy somewhere the dialog can read) so the client reads a ceiling instead of assuming one.

---

## KI-27: `planner/executor.py`'s auth-failure clear is invisible to the arm/clear AST sweeps

**Priority:** Low (reasoned safe today; a promise gap, not a live defect)
**Effort:** Low once someone decides how to widen the sweep past name-matching
**Discovered:** 2026-08-20, milestone 6b (Ruling 84, live-test wrap-up)

**Symptom:** `planner/executor.py:182` clears a pending state through a **loop-local variable** (`state = pending_registry.get(name); ...; state.clear()`), so it is invisible to both AST sweeps that cover every other arm and clear site in the tree ([KI-13](#ki-13)'s `test_every_pending_arm_outside_the_two_reasoned_exceptions_uses_try_arm` and `test_every_pending_clear_outside_the_reasoned_exceptions_is_gated`). Reasoned safe today — it is same-request teardown of a state that the same call armed — but the sweeps' whole promise is that a new unguarded site cannot ship silently, and this is the one shape where that promise does not hold: the sweeps are name-based, and a variable named `state` rather than `pending_*` sits outside what they can see.

**How you would know:** a refactor that lets this clear reach a state armed by a *different* principal than the one clearing it would not turn any sweep red — nothing today produces that shape, which is exactly why it is unguarded rather than closed.

**What to check first:** whether the sweep has grown a way to follow a variable back to its `pending_registry.get(...)` origin rather than matching on the receiver's name, or whether this one site has been moved onto `try_clear` directly despite the loop-local binding.

---

## KI-28: ~~A turn skipped by a security control could still claim the thing it refused~~ FIXED

**Priority:** was Medium (security/correctness — a false claim about machine state, persisted into memory)
**Discovered:** 2026-08-20, milestone 6b live test (Part 8) — the phone said "cancel" on a foreign confirmation, the control correctly skipped it, and TENKA replied "I've cancelled that deletion" (twice) while the pending delete was still armed.
**Fixed:** 2026-08-20, milestone 6b (`04bfa79`).

**What shipped:** a turn skipped by a security control used to fall through to a fallback that built its reply from conversation history alone, with no knowledge that a refusal had just happened — so it asserted the exact state change the control had refused to make. The snapshot then persisted that sentence verbatim into durable session memory, replayed at the start of the next session. Not two-principal-specific: a narrow device talking to itself hit the same blind fallback through the capability-skip path, which has the same shape and is covered by its own test with one principal on both sides.

Both skip shapes (foreign-principal skip, capability skip) now mark the turn, and a marked turn answers from a fixed string **without calling the model at all** — the false claim is unreachable rather than discouraged, deliberately: a synthesis path that can assert a state change it did not make is a code problem, not a wording problem, and no prompt text was added anywhere. Schema v20 (`storage/db.py`) records which turns were skipped; the session summariser drops them before it reads, by exclusion rather than annotation, so no model is ever asked to interpret a flag correctly.

Tests: `tests/test_reply_cannot_contradict_the_machine.py`.

---

## KI-29: ~~Secrets pasted into the chat were stored in the database in plaintext~~ FIXED

**Priority:** High (a live credential left the machine)
**Effort:** Low — the redactor already existed; only the wiring was missing
**Discovered:** 2026-08-22, while extracting a routing corpus from real history
**Fixed:** 2026-08-22, same day

**Symptom:** `core/redact.py` was wired into **eight** log and preview sites and **zero**
write sites. So a credential pasted into the chat was scrubbed on its way to `debug.log`
and written verbatim to SQLite in the same turn. The file an operator greps after an
incident was the one clean copy of it.

Found the hard way. Building a corpus of real utterances for a routing test pulled three
live Google OAuth values straight out of the database — a client id, a `GOCSPX-` client
secret, and a `4/0…` authorization code, all pasted months earlier during
`manage_backup`'s OAuth setup. They were one `git add` away from a public repository.

**Why storage was worse than the log**, and the reason this is High rather than the
hygiene-grade Medium [KI-12](#ki-12) carried:

- `conversations` is replayed into prompts, so a stored secret is re-sent to a cloud model
  on later turns, indefinitely;
- `io/backup/orchestrator.py:93` snapshots the whole database, so a stored secret **leaves
  the machine** on the next backup. By the time this was found, it already had.

**This was half-known and mis-filed.** [KI-12](#ki-12)'s closing note said:

> Still open, tracked separately as KI-15: redaction runs at log and preview sites, never
> at storage-write sites.

[KI-15](#ki-15) is the unfenced-facts-in-the-system-prompt issue — a different problem. So
the storage-write gap had no ticket at all, only a dangling cross-reference. The note also
understated the scope twice: it named `save_typed_fact` and the knowledge graph, when the
columns actually holding the credentials were `conversations.user_input` and
`interaction_events.transcript` — **every turn**, not just extracted facts.

**What shipped.** `redact_secrets` at five write sites in `storage/repos/`, chosen over
`redact_secrets_strict` on measurement rather than instinct: against the four shapes
actually found in this database the lenient tier catches all of them, while strict's blunt
assignment-shaped-line rule risks eating ordinary conversation. Over-redaction here is
unrecoverable — this is her memory, and a false positive silently deletes something the
user said.

| Site | Column |
| --- | --- |
| `repos/memory.py` `save_turn` | `conversations.user_input`, `.response` |
| `repos/memory.py` `save_typed_fact` | `facts.value` |
| `repos/memory.py` `save_chunk` | `recording_sessions.transcript` |
| `repos/telemetry.py` `create` | `interaction_events.transcript` |
| `repos/knowledge_graph.py` `add_fact` | `kg_facts.object` |

Redaction is at the **repo**, not the facade, so a caller that bypasses `memory.py` is
still covered. `conversations_fts` is filled by an INSERT trigger on `conversations`, so
redacting the base column keeps the search index clean with no second mechanism.

`tools/scrub_stored_secrets.py` cleans rows already stored — dry-run by default, backs the
database up before writing, and rebuilds both FTS indexes afterwards because an UPDATE does
not fire the INSERT triggers that populate them. It reuses the same `redact_secrets`, so
there is exactly one definition of "looks like a secret" rather than a second one that
drifts. Nine rows were scrubbed on 2026-08-22 across `conversations.user_input`,
`conversations.response` and `interaction_events.transcript`.

**Rotation is not optional.** Redacting the copy in this database does not un-leak a secret
that already reached a cloud backup. The client secret found here was rotated.

**Tests:** `tests/test_storage_write_redaction.py` — 35 cases. Both halves are pinned and
both mutations were run: removing the redaction from `save_turn` reds 8 tests (including
the FTS-index copy), and making `redact_secrets` return `[REDACTED]` unconditionally reds
13 — because a redactor that blanks every turn would otherwise pass every secret assertion
while quietly deleting her memory. Live-test the answer, not the refusal.

**Still open:** redaction protects what TENKA *stores*. It does not protect what a user
pastes from being sent to a cloud model in the turn it was pasted — the intent classifier
sees the raw utterance. That is a separate boundary (the Context Builder's egress
filtering) and is not closed by this fix.

---

## KI-30: ~~A live raise could be converted into permanent local `EXECUTE`~~ FIXED

**Priority:** High (a time-bounded control could be made unbounded)
**Effort:** Low–Medium — a second predicate beside the existing one, plus data
**Discovered:** 2026-08-22, auditing the authority model to write `TENKA-v2.md`
**Fixed:** 2026-08-23

**Symptom:** A capability raise is deliberately time-bounded — minted at the keyboard,
`require_admin(SYSTEM_CONTROL)`, scoped to one device and one transport, expiring. But
`manage_monitor`, `manage_schedule`, `manage_procedure`, `manage_shortcut` and
`manage_backup` all **install something that runs later**, and `automation/event_bus.py`
and `scheduler.py` run it with `LOCAL_GRANTS`.

Spend a thirty-minute raise on installing a monitor and the expiry stops mattering: the row
fires on a cadence forever, attributed in every log to `local`. The bound the raise exists
to provide was defeated by the artefact it was spent on.

The chain, every link verified in the tree:

1. a device pairs over `tailnet` with `EXECUTE` ticked. 6b's issue-time fix
   (`routes/pairing.py`) deliberately *stores* it in the vault rather than stripping it, so
   a later raise can reach it — that fix is correct and is not the defect;
2. on ordinary requests `effective(issued, policy, raised=∅)` narrows it away. Refused;
3. the operator mints a raise at the keyboard;
4. during the window the device reaches `manage_monitor`, gated on `EXECUTE`, which now
   passes;
5. `handle_manage_monitor` (`actions/monitors.py:41`) has no other guard — it goes straight
   to `event_monitoring.create_monitor`;
6. the raise expires;
7. the row still fires, with `LOCAL_GRANTS` and `LOCAL_PRINCIPAL`.

**Why it survived four review rounds and a live test.** The installer *was* gated, and
correctly. The gate asks "does this caller hold `EXECUTE` **now**". The fire path asks "did
whoever installed this hold `EXECUTE`" and answers from the first question's result — which
was true, for thirty minutes. `scheduler.py:135` states the assumption in its own comment:

> scheduling one requires EXECUTE (`manage_schedule` in `core/intent_capabilities.py`), so
> whoever installed this task already held it.

Sound when written. The raise mechanism, shipped in a later milestone, is what made it
false. This is the project's recurring shape for the third time — a correct boundary with an
unenumerated path beside it — and the first instance produced by the *interaction* of two
milestones rather than by either alone. Neither milestone's review could have caught it:
6a.5 had no raise, and 6b had no reason to re-read the scheduler.

**What shipped.** A property, not a special case for five intents:

> A capability held only by virtue of a live raise may not be spent on an action whose
> effect outlives the raise.

`RaiseContext` gains `ceiling`, so a turn can be asked what it holds with **no raise in
force** — `issued & ceiling`, which is exactly `effective(issued, policy, raised=∅)`.
`current_grants` cannot answer that: the raise is already folded in and the narrowing that
produced it is gone. `durable_capability_refusal()` reads it, beside
`capability_refusal()` and at the same choke point in `actions/__init__.py`, immediately
after the existing gate.

`ceiling` is a **required** field, not defaulted — the same discipline as the policy
literals. A default would let a call site that forgot it report "holds nothing durably" and
refuse the operator's own keyboard.

**The classification is exhaustive, with no default in either direction.**
`PERSISTS_AUTHORITY` and `TRANSIENT_AUTHORITY` in `core/intent_capabilities.py` partition
all 38 entries of `config.INTENTS`. This is the one place a strong default does not work,
and both obvious choices are wrong: defaulting to "persists" would refuse `code_executor`
to a raised device and destroy the entire purpose of a raise, while defaulting to
"transient" fails **open** for a future intent that installs something. So there is none,
and a test enumerates `config.INTENTS` and fails on any intent in neither set or in both.
Adding an intent is now a five-place change, not four.

`code_executor` is classified transient **deliberately**, and a test says so, because it
looks like an omission. Running code can do anything a shell can inside the window and no
in-process check changes that — which is what granting `EXECUTE` means and why a raise is a
deliberate act. What this gate stops is TENKA's *own* machinery being used to make the
window permanent.

**Audit half.** Schema **v21** adds `installed_by TEXT NOT NULL DEFAULT 'local'` to
`event_monitors`, `schedules`, `user_procedures` and `user_shortcuts`, written from
`core.principal.installer_label()` at the repo write.

*Corrected the next day, and worth recording as its own lesson.* The first commit added the
column and **never wired the write**, while this entry and the commit message both stated
it was populated. Every row would have read `'local'` from the migration default whoever
installed it -- worse than an absent column, because it is confidently wrong. Same shape as
6b's `quick`: individually correct decisions producing unreachable configuration, and the
same "verified the wrong artifact and reported fine" failure this project has already
recorded once. Caught only because the operator asked whether a live test was needed and
the answer required checking what had actually shipped.

The write reads `current_principal` at the repo rather than taking a parameter, for the
reason `redact_secrets` does: a parameter is something a future caller forgets, and a
forgotten one here writes a lie. `None` records `"unknown"`, never `"local"` -- the
migration default is honest for rows predating the column, since a remote device could not
reach these intents before the raise existed, but it is not honest for a new row whose
principal nobody set. An upsert on a shortcut reassigns `installed_by`, because
re-installing is installing. It does **not** gate: a fire-time check
would need a live policy for a device that may not be connected. The default backfills
honestly — before this, a remote device could not reach these intents at all without a
raise, and the raise mechanism is newer than every existing row.

**Tests:** `tests/test_raise_cannot_outlive_itself.py` — 28 cases. Six mutations run, and
the sixth is the one worth recording: the first five all reded, while **deleting the gate's
wiring in `execute()` left all seventeen unit tests green**. The predicate was perfect and
nothing called it. Three dispatch-level tests were added, which red on both removing the
hook and moving it after handler resolution. A perfect predicate nobody calls refuses
nothing.

**The gate covers the whole intent, not just the create — decided, not overlooked.**
Confirmed by the operator on 2026-08-23 after seeing it in a live run.

`manage_monitor` covers create, list, pause, resume and delete; `manage_backup` covers both
"back up now" and "enable scheduled backups". Gating the intent gates all of them, so a
raised device cannot list or delete its own monitors either — the live log shows
`Delete firefox monitor` refused with the durability sentence.

That looks wrong at first glance, because deleting *reduces* authority. It is kept anyway,
on three grounds:

- **The precise version costs the property.** Only the handler knows whether a given call
  creates something (`monitors.py:_detect_action` parses the goal). Moving the check there
  makes every handler responsible for remembering it, which is exactly the shape that left
  five doors unguarded in 6a.5. Duplicating the parse at the choke point would be a second
  source of truth about what "create" means.
- **Managing a durable trigger is a keyboard activity.** A raise is for doing a thing on a
  vetted machine, not for administering what runs on this one afterwards.
- **It fails in the safe direction.** The cost is a raised phone cannot tidy up after
  itself; the alternative risks the gate being narrowed into uselessness.

Pinned by `test_the_gate_covers_management_not_only_creation`, so the next person who reads
the refusal as a bug and narrows the gate gets a red test and this paragraph. Revisit only
by splitting the intents (`create_monitor` vs `manage_monitor`), which is a five-place
intent change per intent and an API contract change — not by moving the check into a
handler.

**Not closed by this:** a raise spent directly on `code_executor` can install an OS-level
scheduled task outside TENKA. No in-process check can prevent that. The raise's value is
that it is deliberate and narrow, not that it is containment.
