# Contributing to TENKA

Thanks for the interest. TENKA is currently a solo pre-release project — I'm the only person who's shipped code here so far — but the architecture was built with public contributions in mind from day one. This document tells you how to make a change that lands cleanly.

The short version: **read the README first, follow the THE-rule, write a test, follow the commit format, and open a PR.**

---

## Before you start

1. **Read [README.md](./README.md)** — the project's thesis and surface.
2. **Read [SETUP.md](./SETUP.md)** — get TENKA running locally.
3. **Read [TENKA_Architecture.md](./TENKA_Architecture.md)** — layering rules, registry system, three-tier automation, manifest learning. End-to-end, at least once.
4. **Skim [TENKA_Capabilities.md](./TENKA_Capabilities.md)** — know the intents and tiers before adding new ones.
5. **Glance at [TENKA_Known_Issues.md](./TENKA_Known_Issues.md)** — your bug might already be tracked, or your PR might close a known issue (good!).
6. **Get TENKA running and reproduce whatever you're about to change.** Don't write theoretical PRs.

---

## The non-negotiable rule (THE-rule)

> **No hardcoded app-specific rules or functions. All solutions must be generic and future-proof.**

TENKA must adapt to a new app, service, or domain **without code changes**. New apps are discovered at runtime, learned from observation, or taught by the user — they are never hardcoded.

- ❌ `if app == "spotify":` — never.
- ❌ A module or file named after a specific app.
- ❌ A regex that mentions a brand name.
- ✅ A preference row (`/set music_app spotify`).
- ✅ A `_KNOWN_APPS` entry that anyone can extend with one row.
- ✅ A user-taught procedure or a learned per-app manifest under `~/TENKA/manifests/`.

If your feature *seems* to need an app-specific branch, stop and propose first in an issue. The right answer is almost always to lift the specific behavior into data, or to fall through to a generic mechanism (browser DOM, native accessibility tree, vision agent).

---

## Architecture conventions

### Layering

```
core/  →  config  →  storage/, llm/  →  domain  →  automation/  →  actions/  →  main.py
```

`io/` is parallel to domain code and may import `core/` and `config` only. **`io/` may NEVER import `actions/` or `main.py`.** `config` may NEVER read SQLite directly.

These boundaries are enforced by `import-linter` and run on every commit via the pre-commit hook. Run `lint-imports` before pushing to catch violations early.

### Folder structure

The nine approved packages under `assistant/` are:

```
core/  llm/  storage/  io/  automation/  actions/  code_executor/  models/  personalities/
```

**Never create a new top-level folder under `assistant/`** without raising an issue first.

### Reuse before creating

Before writing a JSON parser, path resolver, async wrapper, or pending-state pattern, **search `core/` and `actions/pending_handlers.py`**. Duplicates are how this codebase grew complicated in the first place — please don't add new ones.

### LLM dispatch

All LLM calls go through `llm/contracts.py` task-shaped wrappers (`ask_for_intent`, `ask_for_plan`, `ask_for_synthesis`, …). **Never call `llm.router.get_llm_response` directly from a handler.** Provider chains live in `llm/router.py`'s `TASK_MODEL_MAP`.

### Testing

- Every change needs a corresponding test in `tests/test_<feature>.py`.
- Tests live in `tests/` only — never at repo root, never inside `assistant/`.
- Don't `pytest` tests that hit external services (browsers, search APIs) blindly — they open real windows. Mock the external boundary.
- For UI / live behavior changes, also **live-test against a running TENKA**. Type checks and unit tests verify code correctness; they don't verify feature correctness.

---

## Commit style

TENKA uses **conventional commits with a tsundere TENKA trailer**. Every commit ends with a one-line voice-of-TENKA reaction. The `.gitmessage` template at the repo root explains the format.

```
<type>: <imperative summary, ≤ 72 chars>

<optional body — what & why, wrap at 72>

TENKA ~ "<tsundere one-liner reacting to THIS specific change>"
```

Types: `feat | fix | refactor | chore | docs | test | perf`

**The TENKA line is mandatory on every commit.** It must be:
- In character (tsundere — reluctant, prickly, secretly pleased)
- Reactive to *this specific change*, not a generic line
- A single line in double quotes after `TENKA ~ `

Example:

```
fix: stop browser planner from reusing stale page handle

TENKA ~ "Hmph, fine, I'll remember which tab I was on. Don't get used to me being this thoughtful."
```

**Never use `Co-Authored-By: Claude` or any AI-attribution trailer** — TENKA speaks in her own voice in the commit log.

Set up the template once:

```powershell
git config commit.template .gitmessage
```

---

## Branching

- **`main`** is the public, clean history. Only squash-merges land here.
- **`development`** is the integration branch for general WIP.
- **Feature work** happens on `feat/<short-name>`, branched from `development`.
- **Bug fixes** happen on `fix/<short-name>`, branched from `development`.
- Messy intermediate commits on feature branches are fine — they collapse to one polished commit when squash-merged.

**Never push directly to `main`**. Always go through a PR.

---

## Pull request process

1. **One PR, one purpose.** A bug fix and a feature in the same PR will be sent back to split.
2. **Description should include:**
   - What changed (one-paragraph summary)
   - Why (link an issue if there is one)
   - How tested (unit tests + live behavior, where applicable)
   - Any new env vars / settings / migrations
3. **CI must be green.** `lint-imports` + the test suite are the floor.
4. **Self-review your diff before requesting review.** Read it like you don't know what you wrote.
5. **PR title follows commit convention** — `feat: add X`, `fix: resolve Y` — because that's the final squash-commit subject.

---

## Reporting bugs

Open an issue with:

- **What you ran** — exact command or voice input
- **What happened** — including the relevant lines from `assistant/debug.log`
- **What you expected**
- **Your environment** — Windows version, Python version, GPU yes/no
- **Whether the wizard's `.tenka_setup.json` says setup completed**

For automation bugs specifically, also include:
- Active app / browser URL at the time
- The intent the LLM classifier picked (visible in the log line `[intent] Detected intent: ...`)

---

## Requesting features

Open an issue with:

- **The problem** — what real situation you're hitting, not the proposed solution
- **What you've tried** — preferences, taught procedures, shortcuts
- **A specific scenario** — concrete voice command + expected behavior

Most "I want TENKA to do X" requests turn out to be solvable via **data** (a preference, a taught procedure, a known-app entry, a per-app manifest) without any code change. That's by design — TENKA evolves without dependency. Please check that path before asking for new intents or handlers.

---

## License

By contributing to TENKA, you agree that your contributions will be licensed under the same [Apache License 2.0](./LICENSE) as the project itself.

No CLA. No additional paperwork. The contribution itself is the agreement, per Apache 2.0 §5.

---

## A note on the current pace

This is solo work right now and review may be slow — sometimes a few days, occasionally a couple of weeks during heavier release pushes. If a PR sits without response, ping the issue or open a draft PR to start the conversation. I'd rather merge five small, focused PRs than one large one that's hard to review.

Thanks again for caring enough to consider contributing.
