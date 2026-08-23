---
paths:
  - "tests/**"
  - "pyproject.toml"
---

# Testing

Tests live in `tests/`. Never at repo root, never inside `assistant/`. Name them
`tests/test_<feature>.py`.

## Never run the whole suite

```bash
py -3.11 -m pytest tests/test_<feature>.py -v     # per file
```

`pyproject.toml` sets `addopts = "-m 'not live_automation'"`, which protects only tests that
**carry the marker**. An unmarked desktop-driving test is still collected — on 2026-08-08 a
bare `pytest tests/` typed into someone's foreground window while they were working. The
suite is also too slow to run whole, which is how a route-completeness sweep sat red on
`main` for days.

Any new test that drives the desktop or launches a real browser gets
`@pytest.mark.live_automation`, and is run only on a machine nobody is using.

**The marker is a courtesy, not a guarantee.** `-m 'not live_automation'` filters *after*
collection, so a module's import-time side effects run whatever its markers say, and a file
can reach the desktop through something no signal list mentions. On 2026-08-23 a run of all
271 files -- serially, from a script written to be the safe way to do this -- seized the
mouse and keyboard while the operator was working. The marker audit had flagged one file
that day, and it was not the one that did it.

So the rule is not "audit, then run". It is: **run only files you have read, one at a time,
and never while someone is using the machine.** `scripts/baseline.py` now refuses a
whole-suite run unless given `--all-i-know-what-this-does`.

## The baseline

`tests/BASELINE.md` records the status of every test file, one pytest invocation each,
written by `scripts/baseline.py`. It exists so a failure during a change can be told apart
from one that was already there.

    py -3.11 scripts/baseline.py            # run everything, rewrite the ledger
    py -3.11 scripts/baseline.py --check    # compare, write nothing (exit 1 on a regression)
    py -3.11 scripts/baseline.py --files test_foo.py test_bar.py

Not theoretical: three changes in two days hit pre-existing failures
(`test_schema_versioning`, `test_verification_vision`, `test_repo_preference`) and each cost
a revert-and-rerun to attribute. **Check the ledger before assuming a red file is yours.**

It classifies six states, and the four unobvious ones are all shapes this project has been
bitten by: `HUNG` (a test that stalls on a mutant never goes red), `EMPTY` (a file that
collects nothing passes every assertion it does not contain), `SKIPPED` (not coverage), and
`UNKNOWN` (no parseable summary — fail closed, exactly as the `import-linter` hook does).

**Per-file is a safety requirement, not an equivalence claim.** Process-wide singletons —
one SQLite connection, `pending_registry`, `abort`, the contextvars — mean it cannot see a
test that only fails when another file ran first. `test_repo_preference` shows 1 failure
alone and 12 in company. A change that touches singleton lifetime must also run its own
affected files together in one pass, and say so.

It also audits markers: a file that imports something able to move the real mouse, without
`live_automation` and without mocking or a skip guard, is listed. `addopts` only protects
tests that carry the marker.

## A test that does not fail when its mechanism is removed is not a test

Before claiming a test covers something: **delete or invert the thing it is supposed to
catch and confirm it goes red.** Report which tests you checked this way and what you
deleted. Milestone 6b caught six tests passing vacuously — three were security tests, and
four had been authored in a plan rather than by an implementer.

Three failure shapes to watch for:

1. **Sibling refusal.** A test can pass because some *other* check happens to refuse the
   same input. Isolate the mechanism under test — assert the specific refusal's wording, or
   remove whatever else could answer. 6b's KI-17 containment test asserted a `Host` the
   pre-change code already refused: green before the feature existed, green after.
2. **A walk over nothing.** A structural test whose target set is empty passes forever. The
   route-completeness sweep iterated `app.routes`, which in this FastAPI version yields
   wrapper nodes with no paths, so `actual` was the empty set. Every AST or registry sweep
   needs an explicit `assert <collection>, "walked nothing"`.
3. **A hang is not a failure.** A test that stalls on a mutant never goes red. Bound
   anything that waits with `asyncio.wait_for`.

**A green mutant is investigated, not accepted.** It may be proof the test measures the
wrong thing. 6b's transport work found a test pinning the wrong depth of a cancellation
hazard exactly this way — chasing the green mutant found more than any review round.

## Unit tests are not feature tests

Type checks and unit tests verify code correctness, not feature correctness. Live-test
before claiming a feature is done — and live-test **the answer, not the refusal.** A control
that refuses correctly while silently corrupting what it permits passes every red-green
check there is.

## The routing differential harness

`tools/routing_differential.py` replays recorded regex-routed turns through the
classifier and prints the disagreements. `pre_route`'s failure mode is claiming too
much, and a fast path never asks a second opinion -- this asks it in bulk, after the
fact. About $0.002 for the whole local history on Flash-Lite; free on the free tier.

    py -3.11 tools/routing_differential.py            # cost estimate only
    py -3.11 tools/routing_differential.py --all --run

Three things it taught on its first run (2026-08-22), all of them still true:

- **It found a real over-claim.** `do you know who created you?` routed to
  `memory_query`, because `do you know` claims all of ordinary English.
- **The classifier is not an oracle.** It called `shut down` and `exit`
  `computer_task`. Acting on that would have broken the shutdown fast path.
  Disagreements get triaged; pairs where the regex is right go in `KNOWN_GOOD`.
- **Compare what the pipeline dispatches, not what resolved.** `main.py` overrides
  any intent to `planner` for a multi-step utterance, so two of the first seven
  "disagreements" were convergent paths.

It reads the local database and **never writes a file.** Its first version was a
committed fixture generated from real history, which pulled three live OAuth
credentials out of `interaction_events` and came within one `git add` of a public
repo -- KI-29. A case worth keeping is hand-copied, with inert content, into
`tests/test_routing_overclaim.py`.

## Existing structural sweeps

Do not break these, and move them if you move their target:

| Test | Walks |
| --- | --- |
| `test_6a5_predispatch_gate.py` | `main.py`'s pre-dispatch region — fails on a new unguarded `return` |
| `test_6b_principal.py` | every pending arming site in `main.py` — fails on one with no principal |
| `test_layering.py`, `test_api_layering.py` | import boundaries |
| `test_s5_config_no_sqlite.py` | `config.py` never reaches SQLite |

Both AST sweeps are bound to `main.py` by path and to their function by name. Renaming the
function errors loudly; **splitting** it shrinks the walked region and passes while
measuring almost nothing (KI-32). If you move that code, move the sweep and add a count
assertion.
