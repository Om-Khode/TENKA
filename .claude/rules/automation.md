---
paths:
  - "assistant/automation/**"
  - "assistant/actions/da_handlers.py"
  - "assistant/actions/manifest_dispatch.py"
  - "assistant/actions/planner/**"
  - "assistant/procedure_executor.py"
  - "assistant/code_executor/**"
---

# Automation

Architecture: `ARCHITECTURE.md` §6. THE rule applies here harder than anywhere else — this
is the layer most tempted to special-case an app.

## The tiers, cheapest first

| Tier | Mechanism | Vision cost |
| --- | --- | --- |
| 0. manifest | learned YAML selector chain (`manifest_dispatcher`) | 0 |
| 1. `browser_action` | Playwright + CDP/DOM → websites | 0 |
| 2. `app_action` | Terminator → native Windows apps | 0 |
| 3. `computer_task` | vision loop → **fallback only** | 3–10 calls |

Use the most reliable mechanism available. Arbitrary software cannot be assumed to be
perfectly automatable — `unsupported` and `uncertain` are valid outcomes and must stay
distinguishable from `failed`.

## Routing

Zero-LLM-cost, in `automation/router.py:detect_backend`, in this order:

```
preference → URL regex → running process → launch keyword → app context → fallback
```

**No app-specific routing logic.** No brand name in a regex — that is THE rule, not a
preference. Known violations to fix rather than extend: `_CANVAS_APP_RE` (figma, miro,
excalidraw, tldraw, google slides/docs) and `_BROWSER_PLAN_PROMPT`'s `bbc.com` /
`example.com` worked examples.

Manifests are **learned**: `promoter.py` clusters successful cached steps into YAML under
`~/TENKA/manifests/`; `healer.py` repairs broken selectors (fingerprint first, vision
re-ground second).

## Verification

`automation/verification.py` — three tiers, escalating only when the cheaper one is
inconclusive: tier 0 pre-check (code, ~30ms), tier 1 post-verify (code, ~30ms), tier 2
vision (~600ms, only on ambiguity).

**`VerifyResult.ambiguous()` and `.skip()` both return `ok=True`.** A caller that reads
`.ok` without checking `tier == "ambiguous"` treats "code cannot decide" as success.
`native.py`, `browser/automation.py` and `router.py` escalate correctly;
`recovery.py:~471` does not. `vision_verify` fails open by returning the ambiguous result
unchanged, so a missing screenshot, an LLM error or an exhausted free tier reads as
success. Open as KI-30 — do not add a new `.ok` reader without handling ambiguity.

Never equate *no exception* with *success*.

## Gotchas

- **Terminator SDK (PyO3) is sync.** Wrap in `asyncio.to_thread()`. `Locator.get_text()`
  works only on resolved element chains, not intermediate `Locator` objects.
- **Playwright is not thread-safe.** All calls on the same event loop.
- **Step-planning prompts must NOT include examples that match test cases.** Examples teach
  patterns; matching examples make tests prove nothing. Same shape, different content.
- **User-pinned values are HARD constraints.** "mobile as 99999" is never silently
  substituted. If the form rejects it, bail with a summary.
- **Never swallow `UserAborted` into a string error.** Re-raise so the planner stops cleanly
  (`core/abort.py`).
- **Schema-version every on-disk marker.** Bump on contract changes; reject older markers.
- Six independent execute/retry/replan loops exist (`planner.py`, `vision/agent.py`,
  `browser/dom_orchestrator.py`, `procedure_executor.py`, `code_executor/orchestrator.py`,
  `automation/recovery.py`). Do not add a seventh.

## Live automation tests

`pytest tests/` excludes them by default via `addopts = "-m 'not live_automation'"`. They
drive the real desktop. On 2026-08-08 a bare `pytest tests/` typed into someone's
foreground window. Any new desktop-driving test **must** carry the marker.
