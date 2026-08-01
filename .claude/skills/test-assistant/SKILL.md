---
name: test-assistant
description: Run a fast smoke test of the TENKA assistant — Python environment, config integrity, module imports, .env keys, bridge ports, and the test suite — then report a pass/fail summary. Use before a live run or after dependency changes.
disable-model-invocation: true
---

# Smoke-test TENKA

A pre-flight check. It answers one question: **would TENKA actually boot right now?**
Run it before a live session, after `pip install`, after a machine rebuild, or when
something fails and you don't yet know which layer broke.

Work through the steps in order and **stop at the first hard failure** — a broken
Python environment makes every later result meaningless. Report findings using the
format at the bottom.

Invoke Python as `py -3.11` (Windows, no venv). Never assume bare `python` works —
on this machine it resolves to a Microsoft Store stub.

---

## 1. Python environment

```bash
py -3.11 --version
```

Must be **3.11.9** (see `.python-version`). Other versions are untested.

Then confirm the packages that gate startup are importable:

```bash
py -3.11 -c "import numpy, requests, pydantic, yaml, sounddevice, psutil; print('core deps OK')"
```

If any are missing: `py -3.11 -m pip install -r requirements.txt`.

Heavy optional deps (`torch`, `speechbrain`, `faiss`, `easyocr`, `terminator`,
`dlib`) are guarded by lazy imports — note them as warnings, not failures.

---

## 2. Config integrity

```bash
py -3.11 -c "from assistant import config; print(len(config.INTENTS), 'intents'); print('sandbox', config.SANDBOX_DIR)"
```

Checks that `config.py` imports without touching SQLite (a hard architectural rule)
and that the intent list loaded.

Then verify intents and handlers agree — the single most common drift:

```bash
py -3.11 -m pytest tests/test_doc_reconciliation.py -q
```

---

## 3. Module imports

Import the pipeline's top layer. This transitively pulls `llm/`, `storage/`,
`automation/`, and `actions/`, so it catches most breakage in one shot:

```bash
py -3.11 -c "import assistant.main; print('pipeline imports OK')"
```

Then confirm handler self-registration actually ran:

```bash
py -3.11 -c "import assistant.actions; from assistant.actions.registry import tool_registry; print(len(tool_registry.keys()), 'handlers registered')"
```

Handler count should be close to the intent count (`shutdown` is pipeline-intercepted
and has no handler; `browser_action` / `app_action` are internal routing targets).

---

## 4. `.env` and API keys

Check the file exists and which providers are configured:

```bash
py -3.11 -c "import os; from pathlib import Path; p=Path('.env'); print('.env present' if p.exists() else '.env MISSING — copy .env.example'); ks=[k for k in ('GEMINI_API_KEY','GROQ_API_KEY','CEREBRAS_API_KEY') if k in p.read_text()] if p.exists() else []; print('keys found:', ks or 'none')"
```

**Never print key values, and never read `.env` into the transcript** — a PreToolUse
hook blocks writes to it, but reads are on you. Report only which names are present.

No Gemini key is a warning, not a failure: Groq/Cerebras/Ollama are defensive
fallbacks so forks stay usable.

---

## 5. Bridge ports

TENKA's external contract. Confirm nothing else already holds them:

| Port | Direction | Purpose |
| --- | --- | --- |
| 7777 | TENKA → frontend | expressions, animations, subtitles |
| 7778 | frontend → TENKA | listening triggers, chat input |
| 7780 | external → TENKA | messaging-bridge HTTP API |

```bash
py -3.11 -c "import socket
for port in (7777, 7778, 7780):
    s=socket.socket(); s.settimeout(0.3)
    busy = s.connect_ex(('127.0.0.1', port))==0; s.close()
    print(f'  {port}: {\"IN USE\" if busy else \"free\"}')"
```

A port in use means either TENKA is already running or something else took it.
Terminal-only mode (`UNITY_ENABLED=false`) needs none of them.

---

## 6. Import boundaries and tests

```bash
lint-imports
```

If `lint-imports` isn't on PATH (its console script lives in Python's `Scripts/`,
which often isn't), call it through Python instead:

```bash
py -3.11 -c "from importlinter.cli import lint_imports; lint_imports()"
```

**Judge it by the summary line, never the exit code.** Measured 2026-08-01 against
a deliberately broken contract, every invocation's exit code is wrong:

| Invocation | contracts clean | contracts broken |
| --- | --- | --- |
| `-m importlinter.cli lint` | exits 0, prints nothing | exits 0, prints nothing |
| `from importlinter.cli import …` | exits 0 | **exits 0** |
| `lint-imports.exe` | **exits 1** | exits 1 |

So read the last line of output. It must say:

```
Contracts: 4 kept, 0 broken.
```

No summary line at all means it crashed — treat that as a failure, not a pass.
(`hooks/pre-commit` parses this same line for the same reason.)

Then the suite. Run per-file rather than all at once — many files stub
`sys.modules` at import time and pollute each other, so a whole-suite run
reports failures that pass in isolation:

```bash
py -3.11 -m pytest tests/ --ignore=tests/test_pw_standalone.py -q --tb=no
```

Compare against the recorded baseline before treating anything as new breakage.

---

## Report format

Keep it short. Lead with the verdict.

```
TENKA smoke test — <PASS | PASS WITH WARNINGS | FAIL>

  Python env      ✓ 3.11.9, core deps present
  Config          ✓ 37 intents, no SQLite at import
  Module imports  ✓ pipeline + 39 handlers registered
  .env            ⚠ present, GEMINI_API_KEY missing (fallbacks active)
  Bridge ports    ✓ 7777/7778/7780 free
  Boundaries      ✓ lint-imports 4 kept, 0 broken
  Test suite      ✓ matches baseline

Blocking issues: <none | numbered list with the exact failing command>
Warnings:        <none | one line each>
```

Rules for the report:
- **Never claim a step passed without having run it.** If you skipped one, say so.
- Quote the real error text for failures — not a paraphrase.
- Distinguish *blocking* (TENKA won't boot) from *warning* (degraded but runs).
- If you stopped early, state which steps went unchecked.
