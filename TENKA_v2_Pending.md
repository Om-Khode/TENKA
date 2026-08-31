# TENKA v2 — what is left, and what to read first

**Written 2026-08-31, at the point work paused.** `TENKA-v2.md` is the specification and
records what shipped; this file is the resumption note for the three items that did not,
written on the assumption that whoever reads it — including me — has forgotten the context.
Each section says what is missing, why it stopped, the design decisions already made, and
the traps found while investigating.

Nothing here is a live vulnerability. Two of the three are dormant machinery; the third is a
containment guarantee that holds on one path and is absent on the others.

---

## Read this before touching anything

| Read | For |
| --- | --- |
| `TENKA-v2.md` header | what shipped, and the three deferrals named in one paragraph |
| `TENKA-v2.md` §17.P5, §17.P10, §17.P13 | the phase specs these items belong to |
| `.claude/rules/security.md` | the two enforcement points. **Item 1 nearly added a third** |
| `.claude/rules/testing.md` | the vacuous-test shapes. Item 1's first design tripped one |
| `TENKA_Known_Issues.md` KI-15, KI-35 | the two mitigated-not-closed containment entries |

`CLAUDE.md` process rule 10 is the one that matters most here: **changing or removing a
control means enumerating what the old mechanism was incidentally holding up.** All three
items below were re-scoped by that rule during investigation, not by argument about whether
the new design was correct in isolation.

---

## Item 1 — a Task is never created at runtime

### What is missing

`brain/task.py` (contracts), `brain/authority.py:create_task` (factory, five authority rules
A1–A5), `storage/repos/task.py` (persistence, schema v22) and `tests/test_task_persistence.py`
all exist and are tested. **Nothing calls the factory.** `grep -rn "create_task" assistant/`
returns the definition, two docstring references, and `asyncio.create_task` — no production
caller.

So `tasks` and `task_steps` stay empty, and `SUSPENDED_NEEDS_AUTHORITY` — a Task banked until
the operator raises a capability at the keyboard — is a state nothing can reach.

### Why it stopped

No phase owned it. P2 (`524722d`) was deliberately contracts-only: *"no coordinator, no
dispatch, nothing wired into a turn"*, so that `brain/task.py` depends on `core/` and `config`
alone and a Task can be built in a test with no database. That was the right call. P5's
deliverable was `Executor.run(step: TaskStep)` — step-level. **Task-creation-in-the-turn-path
fell between the two**, and `storage/db.py:852` is where that was first noticed: P14 declined
to add the Task-keyed telemetry columns rather than ship them always-NULL, and named P5 as
the gap.

### The trap, and it is the important part of this file

**Do not create the Task in `brain/turn.py`.** It is the obvious place — the single turn entry
all four sources pass through — and it would break a control while leaving that control's test
green.

`brain/turn.py`'s module docstring states it imports neither `authority` nor `task`, and
`tests/test_brain_authority.py:232` enumerates why: `scheduler.py` and
`automation/event_bus.py` import `run_turn`, and both install `LOCAL_GRANTS` on the argument
that whoever installed the trigger held `EXECUTE`. If `run_turn` imported `authority`, then
`from .brain.turn import run_turn` would bind a path to Task construction and resumption, and
a background runner could resume banked work with full local privilege.

That A5 test greps only the source of `scheduler.py` and `event_bus.py`. **It would not have
caught the transitive path.** A test that passes while its property breaks is the shape
`.claude/rules/testing.md` is mostly about.

Any new module in `brain/` that can construct or resume a Task goes in that test's
`_FORBIDDEN` tuple. Its own docstring says so; honour it.

### Where it goes instead

`main.py:_turn_pipeline`, immediately after `_tracker.intent_detected = intent_result.intent`
(~line 2030 as of `d06f73e`). All four intent sources — procedure, shortcut, regex, llm —
converge on that line. It sits after the pre-dispatch region and before dispatch.

`main.py` may import `brain/`; `actions/` may not, so `actions/__init__.py:execute()` is not
available even though it is the single dispatch site.

### The load-bearing decision: creation must never be a gate

`create_task` raises `PermissionError` when A2 refuses and `AuthorityMissing` when authority
is unset. Letting either become a refusal would add a **third enforcement point** beside the
two in `.claude/rules/security.md`, inside the region `test_6a5_predispatch_gate.py` sweeps —
and that sweep fails any new returning branch in the region that does not call `_gate` or
`capability_refusal` at the call site. Satisfying it would mean re-deriving the required
capability beside a predicate that already knows it, which `CLAUDE.md` forbids outright.

**So: on either exception, log and proceed with no Task.** `actions.execute()` refuses one
statement later with the same sentence — `create_task`'s docstring says the wording is
deliberately shared, so nothing about the refusal changes. `task_id` is NULL for refused
turns, which is honest rather than always-null.

State this in the code, and pin it: *creating a Task never refuses a turn that dispatch would
have permitted, and never permits one dispatch would refuse.*

### Which turns get a Task

Only turns that resolve a real `config.INTENTS` intent.

**Not** the pre-dispatch branches — slash commands, teaching sessions, batch teaching,
procedures, the speaker-verify toggles. Their `intent_detected` values (`"slash_command"`,
`"teaching_trigger"`, `"batch_teaching"`, `"speaker_verify"`) are branch labels, not intents,
so `Task.requires()` would fall through to `DEFAULT_REQUIRED` and A2 could newly refuse turns
`_gate` deliberately admits at a lower capability. This is a line drawn on purpose, not an
omission.

**Not** background turns either. `scheduler.py` and `event_bus.py` creating no Task is A5's
whole point, so those rows keep a NULL `task_id` by design.

### Telemetry: four columns, not five

V25 adds `task_id`, `affordance`, `operation`, `final_task_status` to the telemetry row.

`step_id` stays out. A turn row has no single step, and `task_steps` already carries
step-level data — the same rule P14 used when it declined all six: *a field that is always
meaningless is not observability.*

### The one open question

`action_outcome` → `TaskStatus` maps cleanly for three values: `success`→`SUCCEEDED`,
`failure`→`FAILED`, `uncertain`→`UNCERTAIN`. `refused` never occurs, because a refused turn
has no Task.

**`skipped` is undecided.** It is set at three sites in `main.py` (~2250, ~2295, ~2360) — the
KI-28 `security_skip` paths, where a turn was stopped by a security control and its reply must
not claim what was refused. `CANCELLED` is transition-legal from every non-terminal state but
means abort; `FAILED` is terminal and blunter. Pick one, write the argument in the docstring,
and give it a test.

Note also that `UNCERTAIN` and `NEEDS_CLARIFICATION` are **not** in `brain/task.py:_TERMINAL`.
A turn can legitimately end with a non-terminal Task. That is correct — do not "fix" it by
widening `_TERMINAL`.

### Files

| File | Change |
| --- | --- |
| `assistant/brain/task_store.py` | **new**, ~40 lines. Module-level `_repo`, `init(db)`, `save`, `load`. `telemetry.py` is the existing facade-over-repo pattern to copy — `TaskRepo` has no production constructor today |
| `tests/test_brain_authority.py` | add `task_store` to A5's `_FORBIDDEN` |
| `assistant/main.py` | create + persist at ~2030; `RUNNING` before dispatch; terminal status in the existing `finally` (~2876, beside `_tracker.save()`) |
| `assistant/storage/db.py` | V25, four columns |
| `assistant/telemetry.py`, `assistant/storage/repos/telemetry.py` | carry the four fields |

### Properties to pin

- creating a Task changes which turns are refused **not at all** — same turns dispatch, same
  turns are refused, same sentence
- `granted` and `principal` on the persisted Task equal the turn's installed contextvars, not
  the device's vault grants and not the transport ceiling
- a background turn creates no Task
- a pre-dispatch branch creates no Task
- `final_task_status` on the telemetry row matches the persisted Task's status
- A5 still holds: `run_turn`'s import graph reaches neither `authority` nor `task_store`

### Mutations that must red

Let A2's `PermissionError` propagate; pass `grants=` as a parameter instead of reading the
contextvar; import `task_store` in `brain/turn.py` (this one proves the A5 addition is not
decorative); map `skipped` to `SUCCEEDED`.

### Live test

**Required.** Schema migration against the live database, first write to new columns, and a
turn entry point — three of the triggers in `.claude/rules/testing.md`. Include the control
that a permitted turn still completes: a gate that refuses correctly while corrupting what it
permits passes every red-green check there is.

**Estimate:** 1–2 days. Branch `feat/task-per-turn`.

---

## Item 2 — five of six context profiles have no runtime consumer

### What is missing

`core/context.py` is built and correct: six profiles as field whitelists, an unknown profile
raises rather than defaults, every field TENKA did not author is fenced with a provenance
label, `redact_secrets_strict` runs before the bundle is built, and `context_bytes_by_profile`
(V24) measures the result.

**One caller.** `main.py:2400` builds the `interpretation` profile for the conversational
turn. `planning`, `execution`, `verification`, `response` and `self_knowledge` are defined and
unused, so the containment guarantee — whitelisted fields, fenced untrusted content, measured
bytes — holds on exactly one of the paths that feeds a model.

### Why it stopped

P10's deliverable was the Context Builder plus the path that most needed it. Wiring the
remaining five was never scoped, and each one is a different prompt-assembly site with its own
hand-rolled join.

### Start with `planning`

`actions/planner/planner.py:_generate_plan` (~1487–1530) assembles its prompt by hand:
`_PLAN_SYSTEM_PROMPT.format(...)`, then a `user_message` built as
`conv_context + date_ctx + goal`, then `prior_context` appended through
`render_untrusted_block`.

Two things follow, and the second is the reason to do this one first:

1. `prior_context` **is** fenced. The pattern is already in the file, so wiring the profile is
   a substitution rather than an invention.
2. `conv_context` — `memory.build_recent_context(limit=8)` — is **not** fenced, and
   `recent_conversation` is listed in `core/context.py:UNTRUSTED_FIELDS`. The planning path
   therefore fences one piece of content TENKA did not author and joins the other in raw. Same
   class as KI-15 and KI-35, both Medium and both **mitigated, not closed**; this is the third
   instance and the cheapest to close, because the mechanism already exists and is already
   used ten lines away.

`core/` sits at the bottom of the layering, so `actions/planner/` importing `core.context.build`
needs no contract change.

### Then the rest

| Profile | Site | Note |
| --- | --- | --- |
| `execution` | `brain/executor.py`, `automation/router.py:_APP_PLAN_PROMPT` / `_BROWSER_PLAN_PROMPT` | fields are structured parameters; `UNTRUSTED_FIELDS` deliberately omits them, so mostly a whitelist and a byte count |
| `verification` | `automation/verification.py:_VISION_PROMPT` | `observation` is untrusted; closest to KI-35's shape |
| `response` | the response layer / `personality.py` | `relevant_conversation` is untrusted |
| `self_knowledge` | `brain/selfknowledge.py` | one field, `metadata`, all TENKA's own words. Cheapest, lowest value |

### Properties to pin, per profile wired

- a field nobody listed on the profile does not arrive, and `dropped` is logged loudly — a
  silently dropped field is indistinguishable from one nobody passed
- every `UNTRUSTED_FIELDS` member present in that profile is fenced with its provenance label
- `context_bytes_by_profile` records a non-zero entry for the profile on a real turn
- the prompt the model receives is not *worse* than the hand-rolled join it replaced — compare
  on a real turn, not in a unit test

### Mutations that must red

Remove a field from the profile whitelist and confirm it stops arriving. Remove a field from
`UNTRUSTED_FIELDS` and confirm the fencing test reds. Assert the byte count is non-zero — an
empty bundle passes every assertion it does not contain.

### Live test

Required for `planning` and `verification`: both change what a model is asked, and the failure
mode is silent — a degraded prompt produces a worse plan, not an exception. Diff the assembled
prompt on a real turn, before and after.

**Estimate:** ~1 day for `planning`. The rest are independent of each other and of Item 1, so
they can go one per branch. Branch `feat/context-planning-profile` first.

---

## Item 3 — `main.py` still hosts the turn pipeline

**Deferred by decision, and the decision should hold until someone has a plan for the two
sweeps.** `main.py` is 4,139 lines. `TENKA-v2.md` §19 asks that it own process startup and
nothing a turn passes through.

The blocker is not size. `tests/test_6a5_predispatch_gate.py` and `tests/test_6b_principal.py`
are both bound to `main.py` by path and to their function by name. Renaming the function errors
loudly; **splitting** it shrinks the walked region and passes while measuring almost nothing.
That is KI-32's shape, and `_turn_pipeline`'s own docstring records that it was split out of a
closure specifically so the sweeps keep walking real statements.

So the order is: move the sweeps and add count assertions **first**, prove they still red on a
planted unguarded return, and only then move the code. Anyone starting from "main.py is too
long" will do it in the other order.

Not scheduled.

---

## What is deliberately not on this list

- **P15, the World Model.** Did not ship, correctly. Its own exit criterion deletes the
  Protocol when no consumer of the *absence* exists, and there is none. Do not revive it
  without one.
- **One-loop unification.** P13's deliverable was that the loops stop inventing private ways to
  say whether work succeeded, and three migrated to `Outcome` while keeping their own retry
  policy and budget. Collapsing six loops into one state machine rewrites ~40 files carrying a
  `goal` string, which §23 forbids in a single move. The vision agent's `todo_list` attaches to
  a goal, not a step, so `TaskStep` types it wrong — that argument is in `c6dd6de`'s message
  and should be read before anyone reopens it.
- **KI-15.** Cannot be closed by fencing. Closing it means not replaying model-written facts
  into the system prompt at all, which is a different feature.

---

## Order, if picking this up cold

1. **Item 2, `planning` profile.** Independent, ~1 day, closes a real containment gap, and the
   mechanism to copy is already in the file it edits.
2. **Item 1.** Needs this whole section read, especially the A5 trap and the never-a-gate rule.
   Live test required.
3. **Item 3.** Only after someone has moved the two sweeps and proved they still bite.
