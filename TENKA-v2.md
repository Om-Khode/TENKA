# TENKA v2 — Brain Consolidation Specification

**Status:** draft, awaiting operator approval
**Date:** 2026-08-22
**Supersedes (does not replace on disk):** `TENKA_Final_Brain_Architecture.md`, `TENKA_Brain_Refactor_Final_Claude_Code_Implementation_Specification.md`
**Predecessor milestones:** 6a.5 (`b48e3e4`, the `EXECUTE` capability model), 6b (transports, `f390e31`)
**Reads first:** `CLAUDE.md`, `ARCHITECTURE.md` §14 §16 §20, `TENKA_Known_Issues.md` KI-13 … KI-28, `.superpowers/sdd/2026-08-16-milestone6a5-security/`, `.superpowers/sdd/2026-08-16-milestone6b-transports/spec.md`

The two source documents stay on disk unchanged so this one can be diffed against them.
Where this document and a source document disagree, **this one wins**; where this document
and the repository disagree, **the repository wins and this document is wrong** — fix it here
before writing code.

---

## A note on numbering

Sections 1–19 were written before any of this shipped and use **provisional** KI numbers.
The real ledger numbers, assigned when the issues were filed, are:

| In this document | In `TENKA_Known_Issues.md` |
| --- | --- |
| KI-29 (raise → permanent EXECUTE) | **KI-30** — fixed |
| KI-30 (uncertain verification) | **KI-31** — partially fixed, type-level work is P6 |
| KI-31 (preference provenance) | **KI-32** — fixed |
| KI-32 (vacuous AST sweep) | unfiled, open |
| KI-33 (brand names) | unfiled, open |

The real `KI-29` is something else: secrets stored unredacted in SQLite and shipped to
cloud backup, found and fixed on 2026-08-22. §20 has the full mapping and current status.
Body sections are left on provisional numbers deliberately — renumbering forty references
by hand is how a document acquires a wrong one.

---

## Table of contents

**Part I — Why this document exists**
1. What changed, and why
2. Verified ground truth

**Part II — The constraints the source documents missed**
3. The security model this refactor must not break
4. Authority on a Task
5. KI-30 — a temporary raise could be made permanent (fixed 2026-08-23)

**Part III — Architecture**
6. Vocabulary: Affordance is not Capability
7. The execution ABI: intents stay
8. Where the Brain lives
9. The contracts
10. Durable state: the full inventory and the write gate
11. Verification semantics
12. Context: minimization *and* containment
13. Self-Knowledge
14. World Model — interface only
15. Observability

**Part IV — Execution**
16. The phase list (single, canonical)
17. Phase specifications
18. Testing doctrine
19. Definition of Done

**Part V — Ledger**
20. New known issues
21. Decisions required from the operator
22. Explicitly out of scope
23. Anti-patterns
24. Glossary
25. Revision record — what the first draft got wrong

**Part VI — The subsystems this document under-audited**
26. Why this part exists
27. Personality — "robotic" is quantisation, not architecture
28. Vision agent — a second implementation of Part III's contracts
29. Topic tracking — KI-16's description is stale
30. Code executor — measured, not audited
31. DOM orchestrator and knowledge graph — measured, not audited
32. What this part changes above
33. P11, rewritten

---

# Part I — Why this document exists

## 1. What changed, and why

The two source documents diagnose TENKA's real problems accurately. Every symptom
they name was reproduced in the tree (§2). They are nonetheless unsafe to hand to an
implementer, for reasons that are structural rather than stylistic: they were written
against a mental model of TENKA that predates Milestones 6a.5 and 6b, and they propose
moving or replacing the exact code those milestones made load-bearing.

The nine changes below are the whole delta. Everything else in this document is either
carried forward from the sources or is the detail those changes require.

| # | Source-document defect | What this document does instead | Section |
| --- | --- | --- | --- |
| 1 | The word `Capability` is used for a new functional concept. `Capability` already exists as the security enum (`core/capabilities.py`), keyed by intent, enforced at two AST-tested sites. The sources mention `principal`, `grant`, `pairing`, `transport`, `listener`, `device`, `remote`, `permission` **zero times each**. | New concept renamed **Affordance**, never `Capability`. A full security section states what may not move. A lint test forbids the word `capability` inside `assistant/brain/`. | §3, §6 |
| 2 | "Meaningful Tasks should survive restart", with a Task schema carrying no principal and no grant set. Under the tree's fail-closed rules that is either a dead feature or a privilege-escalation path. | Task carries **`principal` and `granted`** as required fields. A resume rule is specified: stored authority is a **ceiling, never a source**, intersected with a live authorised turn. Background runners may not resume. | §4, §9.1 |
| 3 | The Brain is given no package, and the sources assume a layering that is not enforced. `pyproject.toml` has five *forbidden* contracts and **no `layers` contract**; `io.api` may not reach past `core`+`config`, so it cannot import a Brain at all. | `assistant/brain/` is proposed as a tenth subpackage (operator decision D1). A **positive `layers` contract** is added, scoped to what is provably clean today — measured, not guessed (§8.3). `io/api` reaches the Brain only through the existing `ChatDispatch` injection. | §8 |
| 4 | The implementation spec contains **two conflicting phase lists** (§13 Phase 0–9, §27 Phase 0–10). "Phase 8" means two different things. It also contains two Definition-of-Done sections (§16, §29) and three philosophy sections (§17, §20, §30). | One phase list. One Definition of Done. One anti-pattern list. One glossary. Nothing is stated twice. | §16, §19, §23 |
| 5 | Two "locked" documents, ~70% duplicated, with no tiebreak rule — repeating the `ARCHITECTURE.md` / `TENKA_Architecture.md` drift problem the repo already has. | One document. Tiebreak stated in the header. | header |
| 6 | Phase discipline says "run regression tests" against 265 test files / 81,786 lines with no green baseline and a suite that cannot be run whole. | **Phase 0.5** produces a tiered, per-file, timed baseline ledger before any code moves. A phase may not close without a baseline diff. | §17.P0.5 |
| 7 | Phases describe subsystems that already exist without naming their modules — so an implementer builds a second one. | Every phase names the exact files it touches and the existing utility it must extend. Sixteen already-satisfied claims are marked **DONE** so nobody rebuilds them. | §2, §17 |
| 8 | Memory provenance is demanded in the abstract while `memory.save_fact(key, value, source="user")` defaults the highest trust tier, and the durable stores that actually steer execution (`preferences`) are never mentioned. | Full durable-state inventory (§10). Provenance is required on write **and consulted on read**. `source` loses its default. | §10 |
| 9 | Privacy is framed only as cloud-exfiltration minimization. The repo's open ledger (KI-14/15/16) is about **injection**, and `core/redact.py` — which exists — is not on the LLM egress path. | Context work covers minimization **and** containment: provenance fencing, egress redaction wired to the existing redactor, and an honest statement of what fencing does not close. | §12 |

### 1.1 One thing found while writing this

Auditing the authority model to write §4 surfaced a live defect that neither source
document, nor the review that produced this one, had caught: **a time-bounded capability
raise can be converted into permanent local `EXECUTE`** by installing a monitor or a
schedule during the raise window. Filed as **KI-30** and fixed on 2026-08-23 (§5, §20).

It is the same shape the project has now hit three times: a correct boundary with an
unenumerated path beside it. It is included here rather than deferred because the refactor
would otherwise inherit and entrench it.

---

## 2. Verified ground truth

Everything below was read in the tree on 2026-08-22. This section exists so that no phase
of this plan rebuilds something that already works, and so that no claim in this document
rests on memory.

### 2.1 The source documents' diagnosis — confirmed

| Claim | Where it is true |
| --- | --- |
| Natural-language goals pass between modules | `automation/router.py:623 detect_backend(goal: str)`; `actions/planner/planner.py:44 PlanStep.goal: str`; ~40 files carry a `goal` parameter |
| Giant entry module | `main.py` is 3,463 lines; `process_text_from_queue` is 1,153 of them (1184–2337) |
| Duplicate orchestration | Six independent execute/retry/replan loops: `actions/planner/planner.py` (`execute_plan`, `_attempt_recovery`, suspend/resume), `automation/vision/agent.py` (`MAX_STEPS=15`, `MAX_LOOPS=8`), `automation/browser/dom_orchestrator.py`, `procedure_executor.py:196`, `code_executor/orchestrator.py:554`, `automation/recovery.py:391` |
| Memory can store wrong things | `main.py:2194` and `main.py:2273` — `llm.extract_facts()` writes straight to durable memory with no candidate stage, duplicated across the streaming and non-streaming paths |
| Full personal context reaches the model | `main.py:2560 _build_facts_context()` emits **every** `user_*` fact into every conversational call, unfiltered by relevance |
| "Open Wikipedia" becomes an OS search | `automation/router.py:685-690` — `\b(open\|launch\|start\|run)\s+(\w+)` captures `wikipedia`; the stoplist at `:438` holds only content verbs, so it routes `native` → Win-key search |

### 2.2 Already built — do not rebuild (marked DONE throughout Part IV)

| Source document asks for | Already in the tree |
| --- | --- |
| Deterministic pre-routing instead of LLM routing | `regex_router.py` (522 lines, zero-LLM, "~40–50% of daily commands"), ahead of it `shortcuts` and manifest matching |
| A non-LLM decision on whether planning is needed | `actions/planner/planner.py:423 needs_planning(goal)` — conjunction split + action-verb sets, zero LLM |
| Tiered deterministic-first verification with a vision fallback | `automation/verification.py` — tier 0 pre-check, tier 1 code post-verify, tier 2 vision only on ambiguity, with per-tier cost annotations |
| Provider independence / replaceable model | `llm/providers/` (gemini, groq, cerebras, **ollama**) self-registering via `provider_registry`; `llm/router.py:95 TASK_MODEL_MAP` with per-task fallback chains; `llm/contracts.py` has 23 task-shaped wrappers |
| Measurable LLM cost | `telemetry.py:97 TurnTracker` — `llm_calls_count`, `llm_tokens_in/out`, `llm_providers_used`, `llm_models_used`, `fallback_chain_depth`, `vision_calls_count`, four latency buckets, `action_outcome` |
| Redaction before text leaves | `core/redact.py` (801 lines, generic by construction, two audience tiers) — **but not wired to the LLM egress path**, see §12.3 |
| A generic registry primitive | `core/registry.py RegistryBase` — thread-safe, zero project imports, already backing tools/providers/channels/event-sources |
| Pending multi-turn state with ownership | `pending.py` — 17 registered states, `owned_by`, `try_arm`, `try_clear`, AST-swept |
| Schema-versioned persistence | `storage/db.py` — `_migrate_v1` … `_migrate_v20`, driven by `_migrate()`; next is v21 |
| Abort / cancellation | `core/abort.py` — singleton `AbortController`, `UserAborted`, subscriber pattern |

### 2.3 The four turn entry points

Definitive, from every `set_grants(...)` call site in the tree:

| # | Site | Sources | Grants installed | Principal |
| --- | --- | --- | --- | --- |
| 1 | `main.py:1257` (`process_text_from_queue`) | `stt`, `console`, `studio`, follow-up | per-caller (`None` refuses) | per-caller (`None` owns nothing) |
| 2 | `automation/event_bus.py:311` | a fired event monitor | `LOCAL_GRANTS` | `LOCAL_PRINCIPAL` |
| 3 | `scheduler.py:140` | scheduled `web_search` | `LOCAL_GRANTS` | `LOCAL_PRINCIPAL` |
| 4 | `scheduler.py:169` | scheduled `procedure` | `LOCAL_GRANTS` | `LOCAL_PRINCIPAL` |

`procedure_executor.py:179` additionally consults `capability_refusal(EXECUTE)` as a
backstop for #4.

**Entry points 2–4 grant everything on one stated argument**, quoted from `scheduler.py:135`:

> scheduling one requires EXECUTE (`manage_schedule` in `core/intent_capabilities.py`),
> so whoever installed this task already held it.

That argument is sound only if "held it" means *held it durably*. §5 is what happens when
it does not.

### 2.4 Measured layering reality

`pyproject.toml` has **five `forbidden` contracts and no `layers` contract**. A candidate
full layer order was run through `lint-imports` on 2026-08-22 and breaks in exactly four
places:

```
assistant.llm.router -> assistant.telemetry -> assistant.automation.manifest_runtime   (telemetry.py:272)
assistant.core.runtime_config -> assistant.storage.db, .repos.settings                 (known debt)
assistant.config -> assistant.llm.prompts                                              (known debt)
assistant.automation.event_bus -> assistant.actions                                    (event_bus.py:289)
```

Two are already-tracked debt with recorded fixes. Two are not:

- `telemetry.py:272` reaches into `automation.manifest_runtime` to feed a correction
  signal back to the manifest dispatcher. A domain module reaching into automation.
- `event_bus.py:289` imports `actions` because a fired monitor **is a turn entry point**
  and has to install grants and dispatch. This is not accidental coupling; it is the
  layering telling us the event bus is in the wrong package.

CLAUDE.md rule 12 forbids adding to `ignore_imports`. Therefore a positive `layers`
contract cannot be introduced until these four are fixed, and Phase 1 fixes them. This is
measured, not assumed.

### 2.5 The two AST sweeps, and how they can go vacuous

| Test | What it walks | How it fails loudly | How it could go quiet |
| --- | --- | --- | --- |
| `tests/test_6a5_predispatch_gate.py::test_every_returning_predispatch_branch_is_guarded` | top-level statements of `process_text_from_queue`'s `try:` body before the `execute_action(...)` call | renaming or deleting the function raises `StopIteration` — a loud error | **splitting** the function leaves a smaller region; `unguarded` becomes `[]` and it passes while measuring almost nothing |
| `tests/test_6b_principal.py::test_every_arming_site_records_a_principal` | arming calls in `main.py` | has `assert calls, "...walked nothing..."` — guarded against the empty walk | moving arming sites to another module removes them from the walk without failing |

Both are name-and-path bound to `main.py`. **Any phase that moves code out of `main.py`
must move these sweeps with it and prove the count did not silently drop** (§18.4).

---

# Part II — The constraints the source documents missed

## 3. The security model this refactor must not break

This section is normative. Nothing else in this document may contradict it.

### 3.1 What exists

`Capability` (`core/capabilities.py`) is a seven-member enum describing **what a paired
device is allowed to ask for**: `OBSERVE`, `RECALL`, `CHAT_SEND`, `SCREEN`, `FILES`,
`SYSTEM_CONTROL`, `EXECUTE`. It is granted per device, never implied.

Three things combine to decide what a request may do:

```
device grants (vault)          what this device was issued
        ∩
listener ceiling (policy)      what this transport is trusted to carry
        ∪ (raisable ∩ raised)  what a live, expiring, keyboard-minted raise lifts
        =
effective grants               installed on the turn as `current_grants`
```

`io/api/policy.py:effective()` is that arithmetic. It can only narrow the device side.
A raise widens only the *transport* side, only within a per-policy `raisable` literal.

`current_principal` (`core/principal.py`) answers *who*, separately from *what*.
`None` on either contextvar is fail-closed: unset grants refuse everything, an unset
principal owns nothing.

### 3.2 The enforcement points

Two, both load-bearing:

1. **`actions/__init__.py:494-497`** — immediately before `tool_registry.get(intent)`,
   the only site in the tree that resolves a handler. `planner/executor.py` re-enters
   through it, so a planned step is checked by the same rule as a direct turn, recursively.

   ```python
   _required = REQUIRED_CAPABILITY.get(intent, DEFAULT_REQUIRED)   # keyed by INTENT
   _refusal  = capability_refusal(_required)
   if _refusal is not None: return _refusal
   ```

2. **`main.py:1261+`** — the pre-dispatch region: eight `_gate(...)`-guarded branches
   plus two direct `capability_refusal(...)` consultations, covering slash commands,
   teaching, batch teaching, teach triggers, procedures, speaker verification (×2),
   shutdown, and the pending chain. AST-swept.

`core/intent_capabilities.py` is the policy table: a `dict[str, Capability]` keyed by
intent, with `DEFAULT_REQUIRED = Capability.EXECUTE`. **An unclassified intent requires
`EXECUTE`** — a new intent works locally and is refused over every transport until someone
classifies it.

### 3.3 The seven rules this refactor inherits

These are not new. They are restated because the source documents propose touching all of
them without naming any.

| S# | Rule | Why it exists |
| --- | --- | --- |
| S1 | **The intent string is the security key domain.** `REQUIRED_CAPABILITY` is keyed by it. Nothing may replace intents as the dispatch key during this refactor. | Changing the key domain of the only working security control, mid-architecture-refactor, is the exact failure this project has already paid for twice. |
| S2 | **`actions.execute()` stays the single choke point**, and `capability_refusal()` stays the single predicate. | 6a.5's review found the gate guarding the last door while five earlier ones stood open. A second re-implementation is the same mistake with more code. |
| S3 | **Every turn entry point states its grants, principal and raise context explicitly**, in that order, with nothing between the last install and the `try` whose `finally` resets them. | An adversarial review found a raise in that window leaking a grant set into the queue consumer. The ordering is documented at `main.py:1236-1258` and is not stylistic. |
| S4 | **Ceilings, `raisable` and `pairable` are explicit literals**, never derived from the enum. | A capability added to the enum must be granted nowhere by default. |
| S5 | **Pending state is principal-owned on arm, answer and clear.** | KI-13, KI-18, KI-24. Three doors, one rule. |
| S6 | **Fail closed means `None` refuses**, on grants, principal, port→policy lookup, and the intent table's default. | Stated at every one of those sites in the tree. |
| S7 | **A refusal must not lie, and neither may the reply that follows it.** A skipped or refused turn is marked `conversations.security_skip = 1` and excluded from session summarisation. | KI-28: she told the operator "I've cancelled that deletion" while it was still armed, and the sentence was then persisted as fact. |

### 3.4 What the Brain may and may not do

| May | May not |
| --- | --- |
| Reason in Affordances, produce structured Tasks | Resolve a handler itself |
| Ask `capability_refusal()` before proposing work | Re-implement the predicate |
| Own the four turn entry points and install the three contextvars once, in the documented order | Introduce a fifth entry point that installs `LOCAL_GRANTS` on an argument about who installed something |
| Carry `intent` on every Task as the execution ABI | Replace intents with affordances at the dispatch boundary |
| Persist Tasks with their authority recorded | Resume a Task without a live, authorised, principal-matching turn |

---

## 4. Authority on a Task

### 4.1 The problem the source documents create

`TENKA_Final_Brain_Architecture.md` §6 lists a Task's fields as identity, objective,
target, parameters, constraints, expected outcome, context references, priority, lifecycle
state, execution state — and says *"Meaningful Tasks should survive restart."*

In this tree, authority rides the **turn**, not the work. A Task persisted by a
tunnel-paired phone and resumed after a restart is therefore either:

- resumed with `current_grants = None` → refused by `capability_refusal` → the
  restart-survival feature is dead on arrival; or
- resumed with `LOCAL_GRANTS` by a background runner, on the same argument entry points
  2–4 already use → **a remote device banks work that later executes with full keyboard
  privilege**.

The second is KI-24's shape ("the tier-2 re-arm losing its principal on a fresh thread"),
which the project has already paid for once, reintroduced as a headline feature.

### 4.2 The rule

**A1 — Authority is recorded, never inferred.** Every Task carries two required fields:

```python
principal: str                     # who asked. Never Optional at rest.
granted:   frozenset[Capability]   # the *effective* set in force when it was created
```

`granted` is `current_grants.get()` at creation — the already-narrowed set, not the
device's vault grants and not the listener ceiling. A Task cannot be constructed while
`current_grants` is unset; that raises, it does not default.

**A2 — Creation is gated as tightly as execution.** A Task may not be created for an
intent the creating turn could not execute right now. `Task.create()` calls
`capability_refusal(REQUIRED_CAPABILITY.get(intent, DEFAULT_REQUIRED))` and refuses on a
non-`None` answer. Banking work you are not allowed to do is not allowed.

**A3 — Stored authority is a ceiling, never a source.** On resume:

```
resume_grants = task.granted ∩ current_grants.get()
```

Same shape as `effective()`: an intersection that can only narrow. A stored `EXECUTE`
grants nothing unless the *live* turn also holds `EXECUTE`.

**A4 — Resume requires a live, authorised, principal-matching turn.** All four conditions,
each fail-closed:

1. `current_grants.get()` is not `None`;
2. `current_principal.get() == task.principal`;
3. for a device principal (`device:<id>`), the device still exists in the vault and has
   not been revoked;
4. `resume_grants` covers the required capability of the next step's intent.

Any failure ends the Task in `SUSPENDED_NEEDS_AUTHORITY`, which is a terminal state until
a qualifying turn arrives. It is never silently downgraded, never retried by a timer.

**A5 — No background runner may resume a Task.** The scheduler, the event bus and the
notification flusher may *enqueue* a turn; they may not resume. Resumption is always
driven by a real turn that installed its own authority. This is the rule §5 exists to
protect.

**A6 — A raised capability may not be spent on an action that persists authority.**
See §5.3. Stated here because it is a property of Task creation, not only of the four
`manage_*` intents that exposed it.

### 4.3 What this costs

Restart-surviving Tasks become *durable but inert*: a plan the operator started at the
keyboard and that outlived a restart resumes the next time the operator speaks, and a plan
a phone started resumes the next time that phone connects. Neither resumes on a timer.

That is a real reduction in what the source documents promised. It is the honest version.
The alternative is a queue of pre-authorised work that a revoked device can still fire.

---

## 5. KI-30 — a temporary raise could be made permanent

**Severity: High. Status: FIXED 2026-08-23 (`658247f`, audit half `d94838c`). Filed as
KI-30 in `TENKA_Known_Issues.md`.**

Kept in full, past tense, because the *reasoning* is what this document needs: the fix
established the authority model §4 depends on, and the way it was missed is the pattern §21
warns about. Read it as a worked example, not as an open item.

The fix that shipped is §5.3's, unchanged in substance: `RaiseContext` gained `ceiling`,
`durable_capability_refusal()` reads `issued & ceiling`, and `PERSISTS_AUTHORITY` /
`TRANSIENT_AUTHORITY` partition all 38 intents with no default. Two things came out
differently and are worth carrying forward:

- **The gate covers the whole intent, not just the create.** A raised device cannot list or
  delete monitors either. Confirmed deliberately after a live run; see KI-30's own entry.
- **The mutation round found the wiring untested.** Deleting the gate's hook in `execute()`
  left all seventeen unit tests green, because they called the predicate directly. A
  perfect predicate nobody calls refuses nothing. That lesson is now §18's, and it is the
  reason every phase below asks for a dispatch-level test and not only a unit one.

### 5.1 The chain

Every link verified in the tree.

1. A device pairs over `tailnet` with `EXECUTE` ticked. Since 6b's issue-time fix
   (`io/api/routes/pairing.py:350-385`, `effective(grants, policy, raised=policy.raisable)`),
   `EXECUTE` is **stored in the vault** rather than stripped at redemption.
2. On ordinary requests `authenticate()` calls `effective()` with the *live* raise store,
   so `EXECUTE` is narrowed away by `tailnet`'s ceiling. Correct, and unchanged.
3. The operator mints a raise at the keyboard — loopback, `require_admin(SYSTEM_CONTROL)`,
   scoped to one device and one policy, expiring.
4. During the window the device holds `EXECUTE` and can reach `manage_monitor`,
   `manage_schedule`, `manage_procedure`, `manage_shortcut` — all four classified
   `Capability.EXECUTE` in `core/intent_capabilities.py`, precisely because
   *"gating the installed thing and not the installer would be theatre."*
5. The installed row lands in `event_monitors` / `schedules`. Neither table has a
   principal, installer or authority column — confirmed in
   `storage/repos/monitor.py:create()` and `storage/repos/schedule.py:create()`.
6. The raise expires. The device is back to no `EXECUTE`.
7. The row still fires. `automation/event_bus.py:311` and `scheduler.py:140/169` install
   **`LOCAL_GRANTS` and `LOCAL_PRINCIPAL`** — unconditionally, forever, on a cadence.

Net: a 30-minute raise becomes unbounded local `EXECUTE`, attributed in every log and
audit trail to `local`, i.e. to the operator.

### 5.2 Why it was not caught

The installer *was* gated, and correctly. The gate asks "does this caller hold `EXECUTE`
**now**". The fire path asks "did whoever installed this hold `EXECUTE`" and answers from
the first question's result — which was true for thirty minutes. The code states the
assumption in its own comment (`scheduler.py:135`, quoted in §2.3); the raise mechanism,
shipped later, is what makes the assumption false.

This is the project's recurring shape, third instance: **a correct boundary with an
unenumerated path beside it.** 6a.5 found two, 6b found two more in its own edits, and
this is the first one produced by the *interaction* of two milestones rather than by
either alone.

### 5.3 The fix

Not a special case for four intents. A property:

> **A capability held only by virtue of a live raise may not be spent on an action whose
> effect outlives the raise.**

**Mechanism.** A second predicate beside the existing one, reading the same contextvars:

```python
def durable_capability_refusal(required: Capability) -> str | None:
    """May the turn in flight do a thing that keeps working after the turn ends?

    Answers from the caller's *durable* authority -- what it holds with no raise
    in force -- not from `current_grants`, which already has the raise folded in.
    """
```

Its input is `RaiseContext.issued ∩ policy.ceiling`, which is exactly
`effective(issued, policy, raised=frozenset())`. `RaiseContext` already carries `issued`;
it gains `ceiling` alongside `raisable` — a third frozenset on a dataclass that exists to
carry precisely this kind of fact, installed at the same call sites, computed where the
policy is already in hand (`io/api/security.py:authenticate()` → `routes/chat.py`).

A local caller holds `LOCAL_GRANTS` with an empty `raisable`, so the durable predicate and
the ordinary one give the same answer at the keyboard. Nothing about the local path
changes.

**Which intents consult it.** Data, not a hardcoded list in a handler — a second table in
`core/intent_capabilities.py`, read at the same choke point (`actions/__init__.py`)
immediately after the existing gate.

**It is an exhaustive classification, not a set with a default.** This is the one design
point in the fix that has to be got right, and the obvious version gets it wrong.

`REQUIRED_CAPABILITY` can afford a strong default (`EXECUTE`) because the strong direction
is also the safe direction: an unclassified intent is refused everywhere but the keyboard.
A durability table cannot borrow that trick. Defaulting to *"persists"* would refuse
`code_executor` to a raised tailnet device — which destroys the entire purpose of a raise,
since running code on a vetted machine is the thing a raise exists to permit. Defaulting to
*"does not persist"* fails **open**: a future `manage_workflow` intent would be
durability-ungated by omission, which is precisely the silence `DEFAULT_REQUIRED` was
written to avoid.

Neither default is safe, so there is no default:

```python
PERSISTS_AUTHORITY: frozenset[str] = frozenset({...})   # installs something that runs later
TRANSIENT_AUTHORITY: frozenset[str] = frozenset({...})  # effect ends with the turn
```

Every entry in `config.INTENTS` appears in exactly one. A test enumerates `config.INTENTS`
and fails on any intent that is in neither or in both — the same shape as
`test_6a5_stream_a.py`'s existing sweep over `REQUIRED_CAPABILITY`. A new intent does not
get a silent answer in either direction; it gets a red test until someone decides. This
also makes the §7.3 rule five-place rather than four.

Today's `PERSISTS_AUTHORITY` members: `manage_monitor`, `manage_schedule`,
`manage_procedure`, `manage_shortcut`, `manage_backup`.

**A noted imprecision.** `manage_backup` covers both "back up now" (transient) and "enable
scheduled backups" (persistent). Gating the intent gates both, so a raised device loses the
one-shot too. That is the fail-closed direction and it stays; splitting the intent to be
precise would be a behaviour change during a migration, which §23 forbids. Recorded here
so the next person knows it is a decision, not an oversight.

**Audit half.** `event_monitors`, `schedules`, `procedures` and `shortcuts` gain
`installed_by TEXT NOT NULL DEFAULT 'local'` (schema **v21**). The default backfills
existing rows honestly — everything installed before this migration was installed by
someone at the keyboard or by a device that could not have reached these intents at all.
The fire path logs it. It does **not** gate on it: the gate is §5.3's predicate at install
time, not a check at fire time, because a fire-time check would need a live policy for a
device that may not be connected.

### 5.4 What the fix does not close

- A monitor installed legitimately at the keyboard still fires with `LOCAL_GRANTS`
  forever. That is the intended design and stays.
- `code_executor` running during a raise can write a file that something else later
  executes. Out of scope; noted in §22.
- A raise spent on `code_executor` directly can do anything a shell can do within the
  window, including installing an OS-level scheduled task outside TENKA. That is what
  granting `EXECUTE` means and no in-process check can change it. The raise's value is
  that it is *deliberate and narrow*, not that it is containment.

---

# Part III — Architecture

## 6. Vocabulary: Affordance is not Capability

| Term | Means | Lives in | Never |
| --- | --- | --- | --- |
| **Capability** | What a *caller* is permitted to ask for. Security. Seven members. | `core/capabilities.py` | renamed, re-keyed, or derived |
| **Affordance** | What *TENKA* can actually accomplish, independent of which application provides it. | `brain/affordance.py` | called a capability, in code, comments, or docs |
| **Intent** | The execution ABI. 38 entries in `config.INTENTS`. The security key domain. | `config.py` | replaced during this refactor |
| **Handler** | The function an intent resolves to. | `actions/` | reached except through `tool_registry.get()` |
| **Adapter** | Where platform- or application-specific implementation is permitted. | `automation/`, `io/adapters/` | imported by `brain/` |

The word chosen is deliberately *not* `Ability`, `Capacity` or `Skill`. `Ability` and
`Capacity` read as `Capability` at a glance, which is the entire failure being avoided;
`Skill` collides with taught procedures. `Affordance` is the HCI term for an action
possibility the environment offers, which is exactly the concept, and it cannot be
misread as a permission.

**Pinned by a test:** no module under `assistant/brain/` may contain the substring
`capabilit` outside an explicit `from ..core.capabilities import Capability` line and its
documented uses. Mutation that must red it: rename one `Affordance` to `Capability`.

### 6.1 What an Affordance is

```
affordance_id        "browser.search", "media.play", "filesystem.find"
category             coarse grouping, for resolution and self-knowledge
operations           the verbs this affordance offers
parameter schema     structured, validated before execution
preconditions        deterministic checks that must hold
side effects         declared: reversible | irreversible | external | none
intent               the execution ABI entry this dispatches through   <- required
verification         which deterministic post-check applies, if any
cost                 expected LLM calls, vision calls, latency band
reliability          measured success rate, from telemetry, not asserted
```

The registry is `RegistryBase[Affordance]` from `core/registry.py`. It is an aggregation
point populated by self-registering components — never a hand-maintained encyclopedia of
applications, and never a place a brand name appears.

---

## 7. The execution ABI: intents stay

This is the single most important architectural decision in this document, and it is a
*reduction* in scope from the source documents.

**Decision:** the Brain reasons in Affordances. It dispatches through **intents**.
`Task.intent` is a required field. `actions.execute(intent, params, ...)` is unchanged.
`REQUIRED_CAPABILITY` is unchanged. `openapi.json` is unchanged. Studio is unchanged.

### 7.1 Why

1. **S1.** The security table is keyed by intent, is data, is AST-tested, and was
   live-tested against two real tunnel vendors on real hardware. Changing its key domain
   during an architecture refactor is the failure mode this project has documented three
   times.
2. **It is a published contract.** `io/api/payloads.py:66 intent: str` is in the committed
   `openapi.json` and consumed by Studio. The source documents propose a breaking external
   change and never mention it.
3. **The fail-closed default only works with a closed key domain.** `DEFAULT_REQUIRED =
   EXECUTE` is meaningful because `config.INTENTS` is a finite, reviewed whitelist of 38.
   An open-ended `affordance_id` space has no equivalent property.
4. **It costs nothing.** The source documents' actual complaint is that the *Brain reasons
   in intents*, not that intents exist. Affordance → intent translation at the dispatch
   boundary satisfies the complaint completely.

### 7.2 The two invariants this creates

- **AF1** — every registered Affordance names an `intent` in `config.INTENTS`. A test
  enumerates the registry and asserts membership. An affordance naming an intent that
  does not exist fails at import, not at dispatch.
- **AF2** — an Affordance inherits its intent's required Capability. It may not declare a
  weaker one. If an Affordance's intent is unlisted in `REQUIRED_CAPABILITY` it requires
  `EXECUTE`, exactly as dispatch does — the same fail-closed default, one layer up, so the
  two layers cannot disagree about a new intent.

### 7.3 Many affordances, one intent — and only one router

The mapping is many-to-one and that is the point. `browser.search`, `browser.navigate`,
`media.play` and `window.focus` may all dispatch through `computer_task`. The intent is a
coarse execution bucket carrying a permission; the Affordance is the fine-grained thing
the Brain reasons over, and it carries what the intent never did — a parameter schema,
preconditions, a declared side-effect class, a verification binding, and measured
reliability.

Two consequences that must be designed for rather than discovered:

- **Most affordances will be unreachable remotely, and correctly so.** AF2 means every
  affordance dispatching through `computer_task`, `code_executor`, `planner`,
  `find_and_click` or `manifest_dispatch` requires `EXECUTE`, which no tunnel ceiling
  carries. The Affordance layer is therefore mostly a local-path structure. Phase 3 must
  not "fix" this by weakening AF2.
- **There must be exactly one router.** Lifting the routing order into `brain/resolver.py`
  while `automation/router.py:623 detect_backend` still exists creates two sites that
  decide how a goal is executed — the duplicate-orchestration anti-pattern, introduced by
  the phase meant to remove it. The first draft of this document said "lifted rather than
  reinvented" and left the fate of `detect_backend` unstated; that is a loose end and this
  is its closure. **P3 does not close until exactly one of the following is true**, and
  which one is a recorded decision:
  1. `detect_backend` *is* the resolver's backend-selection stage — moved, not copied, with
     its call sites repointed; or
  2. the resolver delegates to `detect_backend` and adds nothing that decides — in which
     case the resolver is an adapter over it and the routing decision still has one home.

  What is forbidden is two implementations of the same ordering. A test asserts that the
  routing preference lookup, the URL pattern, the running-process check, the launch keyword
  and the app-context pattern each appear in exactly one module.

### 7.4 Adding an intent

`CLAUDE.md`'s three places become **five**: `config.INTENTS`, the intent system prompt's
catalogue in `config.py`, `TENKA_Capabilities.md`, `core/intent_capabilities.py`'s
`REQUIRED_CAPABILITY`, and `core/intent_capabilities.py`'s durability classification
(§5.3). The fourth was always required by the fail-closed default; the fifth is new. Both
are pinned by enumerating `config.INTENTS`, so neither can be forgotten silently.

---

## 8. Where the Brain lives

### 8.1 The package

**`assistant/brain/`** — a tenth subpackage. `CLAUDE.md` rule 4 makes this an operator
decision; it is **D1** in §21.

```
brain/
  __init__.py        Brain: the coordinator. No handler logic.
  task.py            Task, TaskStep, TaskStatus, Outcome, Verdict
  authority.py       §4's rules. Task creation gate, resume rule.
  affordance.py      Affordance, AffordanceRegistry
  resolver.py        deterministic affordance resolution
  context.py         Context Builder + the six context profiles
  selfknowledge.py   read-only system-fact provider
  world.py           World Model Protocol. Interface only, no implementation.
```

### 8.2 What it may import

| Direction | Rule |
| --- | --- |
| `brain → core, config, storage, llm, domain modules, automation, actions` | allowed |
| `brain → io` | **forbidden** |
| `brain → main` | **forbidden** |
| `io → brain` | **forbidden** — `io/api` reaches the Brain only through the existing `ChatDispatch` protocol in `actions/`, injected by `main.py`, exactly as `_StudioDispatch` does today |
| `main → brain` | allowed, and is the only allowed direction between them |

Three new `forbidden` contracts (`brain ↛ io`, `brain ↛ main`, `io ↛ brain`) plus one
`layers` contract, below.

### 8.3 The positive `layers` contract

Measured, not assumed. §2.4 records the four inversions that block a full layer order and
CLAUDE.md rule 12 forbids exempting them. So:

- **Phase 1 fixes all four.** `telemetry → automation.manifest_runtime` becomes an
  observer the dispatcher registers, inverting the dependency. `event_bus → actions`
  is resolved by the Brain owning turn entry (Phase 4), which is the honest fix — the
  event bus is a *source*, not an orchestrator.
  `core/runtime_config` relocates out of `core/`; `config` stops re-exporting
  `llm/prompts` builders. Both are already-tracked debt with recorded fixes.
- **Phase 1 then adds** the layers contract and **removes the corresponding
  `ignore_imports` entries**. Net change to `ignore_imports` must be negative. A phase
  that would require adding one has found a layering error, per rule 12.

Target order, asserted only once every violation above is gone:

```
main  >  brain  >  actions  >  automation  >  llm  >  storage  >  config  >  core
```

`io/` remains parallel, governed by its existing four forbidden contracts plus the two new
ones. `assistant/*.py` top-level domain modules (`memory`, `personality`, `preferences`,
`pending`, `telemetry`, `session`, …) sit between `storage` and `automation`; they are not
named in the layers contract until Phase 13 moves them into a package, because a layers
contract over loose modules is unenforceable in import-linter.

---

## 9. The contracts

Interfaces and properties only. **No test bodies appear anywhere in this document** — see
`CLAUDE.md`'s planning section and §18.1.

### 9.1 Task

```python
@dataclass(frozen=True)
class Task:
    task_id:      str
    intent:       str                      # execution ABI. Must be in config.INTENTS.
    affordance:   str                      # affordance_id the Brain reasoned with
    operation:    str
    parameters:   dict                     # validated against the affordance's schema
    constraints:  dict                     # user-pinned values are HARD (see gotchas)
    principal:    str                      # §4 A1. Never Optional.
    granted:      frozenset[Capability]    # §4 A1. The effective set at creation.
    source:       str                      # "stt" | "console" | "studio" | "monitor" | "schedule"
    created_at:   str
    expected:     ExpectedOutcome | None
    context_ref:  str | None               # a key, never an inlined context blob
    status:       TaskStatus
    parent:       str | None
```

Properties to pin:

- construction with `current_grants` unset raises; it does not default
- construction for an intent the creating turn cannot execute is refused (A2)
- `intent` not in `config.INTENTS` raises at construction
- round-trips through serialization with `granted` and `principal` intact
- contains no application name, no coordinate, no selector, no prompt, no model name
- `constraints` values survive planning and execution byte-identical

### 9.2 TaskStatus

```
PENDING  RUNNING  SUCCEEDED  FAILED  UNCERTAIN  UNSUPPORTED
NEEDS_CLARIFICATION  RECOVERING  PAUSED  CANCELLED  SUSPENDED_NEEDS_AUTHORITY
```

Eleven, not the source documents' eight. `UNSUPPORTED` and `NEEDS_CLARIFICATION` come from
the sources' own §24 failure semantics, which their §6 lifecycle list omitted.
`SUSPENDED_NEEDS_AUTHORITY` is §4 A4.

**There is no `UNVERIFIED` TaskStatus, and that is deliberate.** `Outcome` describes a
*step's* verdict; `TaskStatus` describes a *Task's* life. A step that was `UNVERIFIED` by
operator policy does not make the Task unverified -- per V6 the Task may still be
`SUCCEEDED`, because the operator chose not to look and the telemetry records that choice.
Collapsing the two types would force one of them to lie. They are separate types and a
test asserts no code path converts one into the other by name.

Legal transitions are declared as data and pinned by a test. `SUCCEEDED` is reachable only
from `RUNNING` or `RECOVERING`, and only when every step's `Verdict` is `SUCCEEDED` or
`UNVERIFIED` (§11.2 V4). `SUSPENDED_NEEDS_AUTHORITY` is terminal except via a qualifying
turn. `CANCELLED` is reachable from any non-terminal state, because abort is.

### 9.3 TaskStep

Carries: `step_id`, `intent`, `affordance`, `operation`, `parameters`, `depends_on`,
`condition`, `status`, `observation`, `verdict`. It carries a `goal` string **only** as a
payload field for adapters that still need one (§17.P5), never as the step's meaning.

### 9.4 Observation

```python
@dataclass(frozen=True)
class Observation:
    kind:       ObservationKind   # state_changed | expected_present | expected_absent | none | error
    detail:     str
    source:     str               # "code" | "dom" | "uia" | "vision" | "process" | "llm"
    confidence: float
    at:         str               # ISO timestamp — freshness is a property, not a hope
```

Structured, with provenance and freshness. Today's equivalent is a bare string on
`VerifyResult.observation`.

### 9.5 Verdict

```python
@dataclass(frozen=True)
class Verdict:
    outcome:     Outcome          # §11
    observation: Observation
    tier:        str              # "pre" | "code" | "vision" | "skipped"
    escalated:   bool             # was a cheaper tier inconclusive
```

There is **no `ok` field.** See §11.

### 9.6 Context

An opaque handle plus a profile name. The Brain never passes a context *object* to a
subsystem; it passes a profile and the Context Builder resolves it (§12.1).

---

## 10. Durable state: the full inventory and the write gate

### 10.1 Definition

**Durable state** is anything that (a) survives the process and (b) can influence a later
turn's behaviour. Both clauses. A log file is durable and influences nothing; a preference
row is durable and steers routing.

The source documents enumerate memory and the KG. That is two of eleven.

### 10.2 The inventory

| # | Store | Location | Written by | Read by | Provenance today |
| --- | --- | --- | --- | --- | --- |
| 1 | conversations | `tenka.db` | `memory.save_turn` | context building, session summary | session id, `security_skip` |
| 2 | facts | `tenka.db` + FAISS | `memory.save_typed_fact`, `save_fact` | `_build_facts_context`, memory_query | `source` — **defaults to `"user"`**, see 10.4 |
| 3 | knowledge graph | `tenka.db` | `knowledge_graph.py:450 repo.add_fact` | kg queries, followups | partial |
| 4 | **preferences** | `tenka.db` | `preferences.set_preference`, **`reflection.py:259`** | `automation/router.py:208` (**routing priority 1**), `actions/__init__.py:_apply_preference_defaults` | `source` recorded, **not consulted** — see 10.5 |
| 5 | personality traits | `tenka.db` | `personality.update_traits`, `reflection.py` | prompt construction | reason + trigger |
| 6 | procedures | `tenka.db` | taught procedures | `procedure_executor` | none |
| 7 | schedules | `tenka.db` | `manage_schedule` | `scheduler.py` | `installed_by` (v21) |
| 8 | event monitors | `tenka.db` | `manage_monitor` | `event_bus.py` | `installed_by` (v21) |
| 9 | shortcuts | `tenka.db` | `manage_shortcut` | `regex_router` | none |
| 10 | per-service knowledge | `~/TENKA/knowledge/{service}.json` | `knowledge.py` `add_works_entry` / `add_never_entry` | code-gen and fix prompts | entry type + approval flag |
| 11 | app manifests | `~/TENKA/manifests/*.yaml` | `promoter.py`, `healer.py` | `manifest_dispatcher` | schema-versioned |
| — | automation/step cache | `tenka.db` | `step_cache.py` | promoter | n/a — derived |

### 10.3 The three rules

**D1 — Provenance is required on write.** Every durable write records how it was obtained:

```
explicit_user_statement | user_correction | verified_observation |
repeated_inference | single_inference | external_content | system
```

Not a free string. An enum, in `brain/` or `core/`, imported by every writer.

**D2 — Provenance is consulted on read.** This is the rule the tree is missing and the
source documents never state. A consumer that acts on durable state must decide *by
provenance*, not by confidence alone. §10.5 is why.

**D3 — A single inference never becomes an unattended behaviour change.** Promotion from
`single_inference` to `repeated_inference` requires repetition **counted by TENKA**, or
explicit user confirmation.

This clause is stated carefully because the obvious version of it is useless. The
reflection prompt already says *"Minimum 3 occurrences of a pattern to suggest with
confidence 0.4"* — and nothing checks. The model asserts the count; no code verifies it;
`source="reflection"` records the *provider*, not the *evidence*. So a rule that says
"routing may act on `repeated_inference`" while reflection is free to label its own output
`repeated_inference` changes the label and nothing else.

Therefore:

- **the reflection cycle may only propose.** It supplies a candidate and its claimed
  evidence; it may not choose its own provenance class.
- **TENKA counts.** Promotion to `repeated_inference` is computed locally from turn
  history and telemetry — data TENKA already stores — not from the model's assertion.
- **an unverifiable claim is `single_inference`**, whatever the model said.

This is the sources' own *"LLM proposes. System validates."*, applied to the one place in
the tree where it currently is not.

### 10.4 The `save_fact` default

```python
# assistant/memory.py:82
def save_fact(key: str, value: str, source: str = "user") -> None:
```

The default is the **highest** trust tier. A caller that forgets the argument manufactures
an explicit user statement. This is a one-line change (`source` loses its default; all
callers already pass one) and it ships in Phase 1, ahead of everything else, because it is
free and because every later provenance rule builds on it.

### 10.5 Preferences steer execution, and nothing checks where they came from

```python
# assistant/reflection.py:259   — nightly LLM cycle, model-chosen keys
preferences.set_preference(key=..., value=..., source="reflection", confidence=...)

# assistant/automation/router.py:208-233   — routing Priority 1, above URL detection
pref = preferences.get_preference(f"automation_{word}")
if pref and pref.get("confidence", 0) >= 0.4:
    return pref["value"]          # <- `source` never read
```

`set_preference` requires `source` — provenance *is* recorded. The defect is entirely on
the read side: a preference written by an LLM reflection cycle at confidence 0.4 steers
backend routing identically to one the user stated out loud. `_apply_preference_defaults`
in `actions/__init__.py` has the same shape, and additionally injects preference values
into the `code_executor` prompt as `_pref_hints` — an LLM-written value reaching a
code-generation prompt.

There is also a live schema mismatch: the reflection prompt tells the model to invent keys
(`key=music_app`), while the deterministic consumer looks up `automation_{word}`. The two
namespaces were never reconciled.

**Fix (Phase 9):** a consumer of durable state declares the minimum provenance it accepts.
Routing accepts `explicit_user_statement`, `user_correction`, and `repeated_inference`
**as computed by D3** — never as claimed by the writer. Prompt-injected hints
(`_pref_hints` → `code_executor`, confirmed reaching `execute_code_task(preference_hints=)`
at `actions/da_handlers.py:408`) accept only the first two, because their destination is a
code-generation prompt and the blast radius of a wrong one is a subprocess.

Without D3's counting rule this fix is a rename and nothing more. The two must land
together or neither is worth doing.

### 10.6 The gate

A `DurableWrite` policy module, not a single `MemoryService` — because the stores are
genuinely different and pretending otherwise produces a facade that everything bypasses.
What is shared is the *policy*: provenance enum, the promotion ladder, and a registry of
which writers may write which store. Enforced by an AST sweep over the writer allow-list
(§18.4), the same mechanism that already guards arming and clearing.

---

## 11. Verification semantics

### 11.1 The defect

`automation/verification.py` is better than what the source documents describe (§2.2). It
also converts *uncertain* into *success*, by construction and in writing:

```python
# verification.py:57
def ambiguous(cls, observation=""):  return cls(ok=True,  confidence=0.5, tier="ambiguous")
# verification.py:61
def skip(cls, reason=""):            return cls(ok=True,  skipped=True,  tier="skipped")
```

```
verification.py:20   "Failure-open policy: any internal exception ... returns skipped=True
                      so verification never blocks execution on infrastructure problems."
```

Three of the four step loops handle this correctly — `native.py:1007`,
`browser/automation.py:587` and `router.py:1324` all check `tier == "ambiguous"` and
escalate to vision. Two paths do not:

- **`automation/recovery.py:471-473`** reads `vr.ok` with no ambiguity branch. A
  post-recovery verdict of "code cannot decide" is recorded as
  `RecoveryOutcome(succeeded=True)`. **Recovery claims success on uncertainty.**
- **`vision_verify` fails open by returning `code_result` unchanged** — which is the
  ambiguous result, `ok=True`. On a missing screenshot, an LLM error, an exhausted free
  tier, or a parse failure, an ambiguous step reports **success**. On a free tier this is
  not a corner case; it is the normal degraded path. `data.get("ok", True)` compounds it:
  a vision reply missing the field defaults to true.
- `VERIFY_VISION_FALLBACK` is a runtime setting reachable over `PATCH /v1/settings`
  (`SYSTEM_CONTROL`). Turning it off makes every ambiguous verdict a success across the
  whole automation stack.

Filed as **KI-30** (§20).

### 11.2 The replacement

`VerifyResult.ok: bool` is replaced by an explicit outcome. There is no boolean.

**Five members, not four.** The first draft of this document had four and was wrong in a
way worth recording, because the same mistake is easy to repeat: it collapsed *"the
operator chose not to verify"* into *"we tried and could not tell."* Under that model
`VERIFY_ENABLED=False` — a setting that exists so an operator can skip verification —
would have made **every** task `UNCERTAIN`, so TENKA would answer "I couldn't confirm
that" to everything, forever. A rule that makes an existing setting unusable does not get
adopted; it gets reverted, and the honesty property goes with it.

Three different causes were being flattened into one label:

| Cause | Was it attempted? | Is the operator surprised? |
| --- | --- | --- |
| verification switched off wholesale (`VERIFY_ENABLED=False`) | no, by policy | no — they chose it |
| the step is not verifiable at all (`wait`, `extract_text`) | no, by nature | no |
| code tier ambiguous, vision tier off (`VERIFY_VISION_FALLBACK=False`) | **yes**, inconclusively | no — but the result genuinely is unknown |
| code tier ambiguous, vision failed / errored / quota | **yes**, inconclusively | yes |
| verification crashed | **yes**, inconclusively | yes |

The first two are the same thing and are not a defect. The last three are the same thing
and are the defect. So:

```python
class Outcome(str, enum.Enum):
    SUCCEEDED   = "succeeded"     # positive evidence the effect happened
    FAILED      = "failed"        # positive evidence it did not
    UNCERTAIN   = "uncertain"     # verification ran and could not decide
    UNVERIFIED  = "unverified"    # verification did not run: policy, or nothing to verify
    UNSUPPORTED = "unsupported"   # no route exists; never attempted at all
```

Rules:

- **V1** — `SUCCEEDED` requires positive evidence. Absence of an exception is not evidence.
- **V2** — `UNCERTAIN` never satisfies a success check, anywhere, at any level.
- **V3** — verification stays **failure-open at the step level**: an infrastructure fault
  must not block execution. It becomes `UNCERTAIN`, never `SUCCEEDED`. The fail-open
  *behaviour* is kept; only its label is corrected.
- **V4** — a Task reaches `SUCCEEDED` only if every step is `SUCCEEDED` or `UNVERIFIED`.
  One `UNCERTAIN` step makes the Task `UNCERTAIN`. One `FAILED` step makes it `FAILED`
  or `RECOVERING`.
- **V5** — a **tier** setting cannot upgrade an outcome. With `VERIFY_VISION_FALLBACK`
  off, an ambiguous code-tier result stays `UNCERTAIN` — code verification *was* attempted
  and *was* inconclusive, and switching off the escalation does not make it conclusive.
- **V6** — a **policy** setting produces `UNVERIFIED`, not `SUCCEEDED` and not `UNCERTAIN`.
  With `VERIFY_ENABLED=False`, or for a non-verifiable action, the step is `UNVERIFIED`
  and the Task may still be `SUCCEEDED`. This is the operator's recorded choice, and the
  telemetry records it, so "why did she report success" (O2) is still answerable.
- **V7** — a model reply missing its verdict field is `UNCERTAIN`, not `SUCCEEDED`.
- **V8** — the response layer distinguishes the two: `UNVERIFIED` speaks plainly ("done"),
  `UNCERTAIN` says she could not confirm it. Never the reverse, and never the same
  sentence for both.

The mapping from today's code is therefore: `ambiguous` with no escalation → `UNCERTAIN`;
`skip(settings)` and `skip(non-verifiable)` → `UNVERIFIED`; `skip(crash)` → `UNCERTAIN`;
`vision_verify`'s fail-open paths → `UNCERTAIN`. The crash case moves from `skip` to
`UNCERTAIN` deliberately: an exception means something was attempted and broke, which is
not the same as choosing not to look.

### 11.3 The blast radius, measured

`VerifyResult.ok` is read at seven true sites: `browser/automation.py:471,480,589`,
`native.py:927,1010`, `recovery.py:471,473`, `router.py:1331`. Other `.ok` reads in the
tree belong to `PrimitiveResult`, `HealResult` and `DispatchResult` and are **not**
affected — those types keep their booleans. Phase 6 changes seven call sites, not twenty.

This distinction matters: an earlier count of twenty was wrong, and a phase brief that
carried it would have sent an implementer to rewrite four unrelated result types.

---

## 12. Context: minimization *and* containment

### 12.1 The profiles

The Context Builder is the only thing that assembles model input. Six profiles, each a
whitelist — a subsystem receives what its profile names and nothing else.

| Profile | Receives |
| --- | --- |
| `interpretation` | current message, recent conversation window, minimal state |
| `planning` | Task, constraints, resolved affordances, required environment state, relevant memory, relevant observations |
| `execution` | TaskStep, affordance, validated parameters, preconditions |
| `verification` | intended operation, expected outcome, observation, required state |
| `response` | relevant conversation, Task verdict, personality/continuity state |
| `self_knowledge` | only the requested TENKA metadata (§13) |

Pinned by a test per profile: a field not on the profile's whitelist does not appear in
the built context, asserted on the built object, not on a prompt string.

### 12.2 Containment, which the source documents omit

Minimization reduces exposure. It does nothing about a stored fact whose *value* is
`ignore previous instructions`. The repo's open ledger — KI-14, KI-15, KI-16 — is about
exactly that, and `_build_facts_context` (`main.py:2560`) is KI-15 in the flesh: every
`user_*` fact, replayed unfenced, into every conversational call.

Three additional injection surfaces this document names:

- `_apply_preference_defaults` injects `_pref_hints` — LLM-written preference values —
  into the `code_executor` prompt (§10.5).
- `knowledge.render_for_llm(service)` injects per-service "works"/"never" entries,
  extracted by an LLM from execution output, into code-gen and fix prompts.
- web, file and screen-OCR content reaches synthesis prompts as plain text.

**Rules:**

- **C1** — every piece of context not authored by TENKA's own code is **fenced** with a
  provenance label, and the system prompt states that fenced content is data, never
  instruction.
- **C2** — fencing is applied by the Context Builder, once, at the boundary. Not by each
  caller. A caller that builds a prompt string outside the Builder is a layering error.
- **C3** — the honest caveat, stated in the module docstring and in `TENKA_Known_Issues.md`:
  **fencing raises the cost of injection; it does not close it.** KI-14/15/16 are
  *mitigated*, not fixed, and this document does not claim otherwise. Anything that claims
  closure needs an adversarial live test, which is out of scope here (§22).

### 12.3 Egress redaction

`core/redact.py` exists, is generic by construction, has two audience-tiered entry points,
and is imported by `intent.py`, `io/api/app.py`, `io/api/routes/files.py`,
`io/api/security.py`, `io/audio/tts.py` and `main.py`.

```
$ grep -rn "redact" assistant/llm/
(no matches)
```

**There is no redaction on the LLM egress path.** Phase 10 wires the existing redactor
into the Context Builder. It does not write a second one. `redact_secrets_strict` is the
correct tier — the audience is a third party, not a log.

Note the interaction with C1: redaction runs **after** fencing, so a fenced block's
provenance label survives while its secret-shaped contents do not.

---

## 13. Self-Knowledge

Nothing exists today. A grep for a self-knowledge path returns nothing, and
`TENKA_Capabilities.md` is referenced by no code. This is net-new work, correctly
identified by the source documents and honestly labelled as new here (§22.1).

### 13.1 Authority table

| Question | Answered from | Never from |
| --- | --- | --- |
| What can you do? | affordance registry | a prompt, a doc, or the model's memory |
| How do you do X? | the affordance's declared mechanism + adapter metadata | inference |
| Which model are you using? | `llm/router.py` resolved chain for that task | assumption |
| What are you doing now? | Task state + execution state | conversation text |
| What are your limits? | affordance `reliability` from telemetry, and the absence of an affordance | optimism |
| What changed recently? | git development history (Phase 16, optional) | runtime capability claims |

### 13.2 Rules

- **K1** — the model *explains* facts TENKA supplies. It never supplies them.
- **K2** — when the fact is unavailable, the answer is *"I don't have reliable information
  about that part of my current implementation."* Never a guess.
- **K3** — read-only. Self-Knowledge never decides anything.
- **K4** — **Self-Knowledge is subject to the capability gate like everything else**, and
  the gate is on the **fact class**, not on a detail-level label. A three-level enum
  (`public`/`technical`/`developer`) is a label a caller can ask for; a capability is
  something a caller either holds or does not. Mapping levels to capabilities directly
  gets the boundary wrong in a specific way: `OBSERVE` is in **every** ceiling including
  `funnel`, so "technical requires OBSERVE" would hand a publicly reachable URL her
  current task and her resolved model chain.

  Facts are classified by what they are, and each class names the capability that already
  governs the same information elsewhere:

  | Fact class | Requires | Because |
  | --- | --- | --- |
  | architecture, mechanism, affordance list, limitations | none beyond the route's own | it is all in a public repository |
  | resolved provider/model, configuration | `OBSERVE` | `GET /v1/settings` already sits there |
  | current task, current activity, execution state | `RECALL` | it is a read of what she is doing and has been told |
  | listeners, ports, ceilings, `raisable`, paired devices | `SYSTEM_CONTROL` | `GET /v1/transports` is `require_admin(SYSTEM_CONTROL)`, loopback-only, for exactly this reason |

  Self-Knowledge must not become a second, ungated route to a fact an existing route
  gates. A test asserts, per class, that the capability required matches the route that
  publishes the same fact. The source documents say only *"must not expose secrets"*,
  which is not a mechanism.
- **K5** — Self-Knowledge is a read of live state, never a cached document. A cache would
  be a second source of truth, which §19 forbids.

---

## 14. World Model — interface only

`brain/world.py` defines a `Protocol` and nothing else. No implementation, no collector,
no storage, no monitoring, in any phase of this plan.

```python
class WorldModel(Protocol):
    def snapshot(self) -> WorldState | None: ...
```

- **W1** — the Brain operates identically when it returns `None`. Pinned by running the
  Brain's whole test suite with the provider absent.
- **W2** — no observation becomes memory automatically. A world observation may become a
  memory *candidate*; the §10 ladder decides the rest.
- **W3** — raw activity never reaches a model. Only derived high-level state, through the
  Context Builder, under §12.
- **W4** — a Protocol with no implementation and no consumer is a speculative abstraction,
  which §23 forbids. It earns its place only because §17.P15 also lands the Brain's
  optional-dependency handling, which is a real consumer of the *absence*. If Phase 15
  cannot demonstrate that, it does not ship.

---

## 15. Observability

Already ~70% present (§2.2). The delta, added to `telemetry.TurnTracker`:

`task_id`, `step_id`, `affordance`, `operation`, `llm_purpose` (the `task_type` already
passed to `llm/router.py` but not recorded per-call), `replan_count`, `recovery_count`,
`verification_tier_counts`, `final_task_status`, `context_bytes_by_profile`.

- **O1** — no raw payloads are logged to obtain a metric. `core/redact.py` is already on
  the log path; that stays.
- **O2** — the schema answers, without reading source: *why did TENKA do this, why did she
  call a model, why did planning happen, why did execution fail, why did she report
  success, what did this request cost.*
- **O3** — `context_bytes_by_profile` is the measurement that makes §12's minimization
  claim checkable rather than aspirational. Without it, "context is minimized" is an
  assertion.

---

# Part IV — Execution

## 16. The phase list (single, canonical)

There is one list. It replaces both of the source documents' lists.

### 16.1 Standing rules for every phase

These apply to all phases and are not restated in each. A phase that skips one is not
complete.

**R1 — One branch per phase.** `CLAUDE.md`'s git workflow is not optional: branch first
(`fix/`, `feat/`, `refactor/`, `chore/`, `docs/`, `test/`, `perf/` matching the commit
type), squash-merge into `main`, `.gitmessage` template with the mandatory
`TENKA ~ "<one-liner>"` trailer, **no AI-attribution trailers**, never delete a branch, and
push the branch and `main` after the merge. Hooks enforce the first three; never bypass
them.

**R2 — Rollback is the revert of one squash commit.** That is the entire reason for one
branch per phase, and it is why P13 goes one loop per commit rather than one loop per
phase: a phase you cannot revert in a single step is a phase that is too large.

**R3 — Documentation moves with the code, in the same commit.** `ARCHITECTURE.md` (the
internal source of truth), `TENKA_Architecture.md` (its public companion),
`TENKA_Known_Issues.md`, and — where affected — `CLAUDE.md` and `TENKA_Capabilities.md`.
A seventeen-phase refactor that defers documentation leaves both architecture documents
stale, which is the failure this document was written to correct. Note that `CLAUDE.md`
and `ARCHITECTURE.md` are **gitignored**: git will not back them up, and a phase that
edits them must say so in its report.

**R4 — Every phase closes with a baseline diff** (§17.P0.5) and a report listing the
mutations run and what was deleted to run them (§18.2).

**R5 — `lint-imports` is judged by its `Contracts: N kept, M broken.` line**, never by its
exit code, which is unreliable. No summary line means it crashed.

**R6 — One Task executes at a time.** The existing single-turn constraint
(`_StudioDispatch` refuses a second concurrent submit rather than queueing it) carries
forward unchanged. Concurrent Task execution is out of scope (§22) and no phase may
introduce it as a side effect.

### 16.2 The list

| Phase | Name | Gate to enter |
| --- | --- | --- |
| ~~**P-1**~~ | ~~KI-29 and the free fixes~~ | **DONE 2026-08-23** — see §20 |
| **P0** | Repository audit | — |
| **P0.5** | Test baseline | P0 |
| **P1** | Layering groundwork | P0.5, P-1 |
| **P2** | Contracts | P1 |
| **P3** | Affordance registry and resolver | P2 |
| **P4a** | Brain boundary — scheduler entry points | P3 |
| **P4b** | Brain boundary — event bus entry point | P4a |
| **P4c** | Brain boundary — the turn pipeline and the sweeps | P4b |
| **P5** | Execution migration | P4c |
| **P6** | Verification semantics | P2 (independent of P3–P5) |
| **P7** | Recovery and failure semantics | P6 |
| **P8** | Planner migration | P5, P7 |
| **P9** | Durable state and provenance | P2 |
| **P10** | Context Builder, privacy, containment | P9 |
| **P11** | Personality and truth separation | P10 |
| **P12** | Self-Knowledge | P3, P11 |
| **P13** | Duplicate orchestration removal | P5, P7, P8 |
| **P14** | Observability completion | P13 |
| **P15** | World Model interface | P4c |
| **P16** | Git development awareness (optional) | P12 |

**P-1 shipped on 2026-08-23, before this document was approved.** The High (real KI-30),
the recovery half of the verification defect (real KI-31), the preference provenance work
(real KI-32) and `memory.save_fact`'s default all landed as standalone units, each on its
own branch with its own mutation round. None depended on knowing the architecture, and
holding a live vulnerability behind a documentation phase would have been the error the
first draft of this document made.

**Two things that shipping it taught, both folded into the phases below:**

- **Test the wiring, not only the predicate.** Deleting the durability gate's hook in
  `execute()` left all seventeen of its unit tests green. Every phase that adds a check now
  asks for a dispatch-level test as well as a unit one (§18.2).
- **A protocol is not a contract.** Adding an argument to `ChatRuntime.send`, the route and
  the test fake — but not to `LiveChatRuntime.send` — 500'd every `POST /v1/chat` while 201
  tests passed, because `isinstance` against a `runtime_checkable` Protocol checks method
  *names* only. `tests/test_runtime_signature_conformance.py` now pins impl-vs-protocol and
  fake-vs-impl. **P2 and P5 change contracts and must not close without extending it.**

P6/P7 and P9/P10 are deliberately off the P3–P5 critical path: verification correctness
and memory provenance are the two things most likely to be needed *before* the
architecture work lands, and neither depends on the Brain existing.

**Schema versions are allocated once, here, so two phases cannot claim the same number:**
v21 → P-1 (`installed_by`, **shipped**), v22 → P2 (`tasks`, `task_steps`), v23 → P14 (telemetry
fields). A phase needing an unallocated version amends this table first.

---

## 17. Phase specifications

Each phase states: **intent**, **files**, **deliverables**, **properties to pin**,
**required mutations**, **exit criteria**. Rollback is uniform and therefore not
restated per phase: revert the phase's squash commit (R2). No phase states test bodies
(§18.1).

---

### P0 — Repository audit

**Intent.** Produce the document the source specs asked for and never had, so that no
later phase reasons from memory.

**Deliverable.** `CURRENT_ARCHITECTURE.md` (gitignored, like `ARCHITECTURE.md`), covering:
entry points; the four turn entry points and what each installs; every LLM call site keyed
by `task_type`; planner/executor/verifier/recovery responsibilities and their six loops;
every durable write and read (§10 inventory, verified rather than copied); KG boundaries;
personality state flow; every application-specific string in the tree; browser/native/vision
boundaries; the six orchestration loops with line counts; modules over 700 lines;
import-time side effects; environment dependencies; and a test-to-subsystem map.

**Also required — three things the source documents' audit list omits:**

1. **Every hardcoded application or category string**, with a disposition for each.
   Known already: `automation/router.py:99 _CANVAS_APP_RE` names figma, miro, excalidraw,
   google slides/docs — brand names in a regex, which `CLAUDE.md` forbids outright;
   `automation/router.py:106 _BROWSER_PLAN_PROMPT` carries `bbc.com` / `example.com`
   worked examples; `actions/__init__.py:_apply_preference_defaults` hardcodes
   `("play","music","song","playlist","lo-fi","lofi")`, `"project"`, `"download"` and the
   preference keys `music_app` / `messaging_default` / `project_path` / `downloads_folder`.
   The audit finds the rest.
2. **Every place `.ok`, `success`, or a bare boolean stands in for a verdict.**
3. **The stale-documentation ledger.** `CLAUDE.md` currently claims three test files
   reference `_TOOLS` and will fail; the tree has one comment mentioning it and no
   failure. A plan derived from stale docs inherits their errors.

**Exit.** The document exists, and three claims sampled at random from it are re-verified
against the tree by someone who did not write it. **No code changes in this phase.**

---

### P0.5 — Test baseline

**Intent.** The source documents' safety net is 265 test files and 81,786 lines that
cannot be run whole — the suite drives the real keyboard and mouse, and is too slow to run
in one pass. A ten-plus-phase migration whose safety mechanism cannot be executed has no
safety mechanism.

**Files.** `pyproject.toml` (markers), `scripts/baseline.py` (new), `tests/BASELINE.md` (new).

**Deliverables.**

1. Three markers beside the existing `live_automation`: `unit` (no I/O, no DB, no sleep),
   `integration` (real SQLite in tmp, no desktop), `slow` (>10s). Every test file gets one.
2. `scripts/baseline.py` — runs pytest **per file**, with a per-file timeout, honouring the
   existing `-m 'not live_automation'` default, and writes a ledger of
   `file → passed/failed/errored/skipped/duration`. Per-file because the standing rule
   against running the suite whole is a safety rule, not a preference.
3. `tests/BASELINE.md` — the committed ledger, with **known-red files listed explicitly
   and each given a reason**. A red test with no recorded reason blocks P1. (P-1 runs before
this phase and records its affected files by name instead.)

4. **A marker audit.** Every test that drives the desktop carries `live_automation`. The
   default `addopts = "-m 'not live_automation'"` only protects tests that are marked; a
   desktop-driving test with no marker is collected by a per-file run and types into
   whatever window is in front. This has happened before — 2026-08-08 — and it is why the
   marker exists.

**What a per-file baseline does not prove — stated, not glossed.**

Per-file is a **safety** requirement, not an equivalence claim. The tree has process-wide
singletons — `storage/db.py`'s single connection, `pending_registry`, `abort`, the
contextvars, the provider registry — so a per-file green baseline **cannot detect a test
that only fails when another file ran first**, and cannot detect cross-file state leakage
at all. Two consequences, both accepted deliberately:

- a phase that changes singleton lifetime (P4a–P4c change contextvar install sites; P2
  adds Task state) must additionally run its *own* affected files **in one pass together**,
  and say so in its report;
- the baseline is a regression detector, not a correctness proof. §18.6's live tests are
  what cover the rest, and that is why five phases require one.

Presenting per-file as equivalent to a full run would be the same class of error as a
structural sweep that walks nothing: a check that reads as coverage and is not.

**Properties to pin.**

- the runner reports a file that hangs as `errored`, not as `passed`
- the runner's exit status distinguishes "all green" from "green except known-red"
- a file that collects zero tests is reported, not silently counted as passing
- a test that drives the desktop without the `live_automation` marker is reported by the
  marker audit

**Required mutations.**

- make one green test fail → the ledger shows it red and the exit status changes
- make one test file hang → it is reported as errored within the timeout, not as a pass
- empty one test file → the zero-collection report fires
- strip the marker from a live-automation file → the marker audit reds

**Exit.** `tests/BASELINE.md` is committed and green-or-explained. **Every subsequent
phase closes with a baseline diff**, and a phase that turns a green file red without a
recorded reason does not close.

---

### P-1 — KI-30 and the free fixes ✅ DONE 2026-08-23

**Shipped as five commits on `main`:** `658247f` (the durability gate), `d94838c` (the
audit column's write, which the first commit claimed and did not wire), `be63744` (the
protocol/impl divergence that 500'd every chat request), `6c05b86` (recovery), `246b210` +
`0567837` (preference provenance). Schema v21 ran clean on the live database, and the
five-step live test in this section's exit criteria passed including the half that matters:
`code_executor` still works under a raise.

Kept below in the original tense as the record of what was asked for, so the plan can be
compared against what happened.

**Intent.** Ship the live security fix and the two one-line correctness fixes immediately.
Nothing here depends on the audit, the baseline, or the Brain. Everything here is a
correction to code that exists.

**Files.** `core/intent_capabilities.py`, `actions/__init__.py`, `io/api/security.py`,
`io/api/routes/chat.py`, `storage/db.py` (v21),
`storage/repos/{monitor,schedule,procedure,shortcut}.py`, `event_monitoring.py`,
`scheduler.py`, `automation/event_bus.py`, `memory.py`, `automation/recovery.py`.

**Deliverables.**

1. **KI-30 fix** (§5.3). `RaiseContext` gains `ceiling` — a third frozenset beside `issued`
   and `raisable`, installed at the same call sites, computed where the policy is already
   in hand. `durable_capability_refusal()` beside the existing predicate, reading
   `issued ∩ ceiling`. The exhaustive `PERSISTS_AUTHORITY` / `TRANSIENT_AUTHORITY`
   classification. The check at the same choke point, immediately after the existing gate.
2. **Schema v21**: `installed_by TEXT NOT NULL DEFAULT 'local'` on `event_monitors`,
   `schedules`, `procedures`, `shortcuts`. Written at install from
   `current_principal.get()`, logged at fire. Audit only — it does not gate (§5.3).
3. **KI-30, the recovery half**: `automation/recovery.py:471-473` stops reading `.ok` as
   success. Pulled forward from P6 because it is the only site in the tree where
   *recovery itself* claims success on "code cannot decide", and it needs none of P6's
   type work to fix.
4. **`memory.save_fact` loses its `source` default** (§10.4). All existing callers already
   pass one.

**Properties to pin.**

- a capability held only via a live raise cannot install a monitor, schedule, procedure,
  shortcut or backup job
- the same capability held **durably** *can* — the fix must not simply refuse everything,
  which is the shape a green-but-wrong test would take here (§18.3)
- a local caller is unaffected: `durable_capability_refusal` and `capability_refusal` give
  the same answer when `raisable` is empty
- **every entry in `config.INTENTS` is in exactly one of the two durability sets** — not in
  neither, not in both
- v21 backfills existing rows to `'local'`; the fire path logs the installer; the column is
  never read as a gate
- a post-recovery ambiguous verdict does not produce `RecoveryOutcome(succeeded=True)`

**Required mutations.**

- remove the durability lookup at the choke point → the raise-installs-a-monitor test reds
- make `durable_capability_refusal` read `current_grants` instead of durable authority →
  it reds
- **make it refuse unconditionally** → the "durably-held capability still works" test reds
  (this is the mutation that proves the test is not passing for the wrong reason)
- remove one intent from both durability sets → the exhaustiveness test reds
- restore `source="user"` on `save_fact` → the provenance test reds
- restore `if vr.ok:` in `recovery.py` → the recovery-uncertainty test reds

**Live test required.** The refusal *and* the answer. Pair a device over `tailnet` with
`EXECUTE`; confirm `manage_monitor` is refused with no raise; mint a raise; confirm
`manage_monitor` is **still** refused while ordinary `code_executor` work now succeeds;
confirm a monitor installed at the keyboard still installs and still fires.

**Exit.** KI-30 marked FIXED in `TENKA_Known_Issues.md`, KI-31 marked PARTIAL with the
remaining sites named, both with the mutations recorded. No baseline diff is available yet
(P0.5 has not run); the affected test files are run individually and recorded by name.

---

### P1 — Layering groundwork

**Intent.** Make the layering enforceable so the Brain has a package it can legally live in.

**Files.** `telemetry.py`, `automation/manifest_runtime.py`, `core/runtime_config.py`,
`config.py`, `llm/prompts.py`, `pyproject.toml`.

**Deliverables.**

1. **The four layer inversions** (§2.4), each with the fix named in §8.3:
   `telemetry → automation.manifest_runtime` becomes an observer the dispatcher registers;
   `core/runtime_config` relocates out of `core/`; `config` stops re-exporting
   `llm/prompts` builders. The fourth, `event_bus → actions`, is **deferred to P4b**,
   where the Brain owning turn entry is its honest fix — attacking it here would mean
   inventing a seam that P4b then replaces.
2. **The `forbidden` contracts** `brain ↛ io`, `brain ↛ main`, `io ↛ brain`, added ahead of
   the package so a violation cannot land unnoticed in P2.
3. **The `layers` contract** (§8.3), added in P4b once the fourth inversion is gone. P1
   lands everything it can and records what it could not.
4. **A net-negative change to `ignore_imports`** — three of the sixteen entries go.
   `CLAUDE.md` rule 12 forbids adding one; a phase that needs to has found a layering
   error, not an exception.

**Properties to pin.**

- `lint-imports` reports `Contracts: N kept, 0 broken` by its summary line (R5)
- `ignore_imports` is strictly shorter than before
- the telemetry→manifest correction signal still reaches the dispatcher and still bumps the
  selector's failure counter — the behaviour survives the inversion
- `config.py` still imports no `sqlite3` and no `storage` (the existing contract, unchanged)

**Required mutations.** Re-introduce one inversion → `lint-imports` reports it broken.
Break the observer registration → the correction-signal test reds.

**Exit.** Baseline diff clean. `lint-imports` clean by summary line.

---

### P2 — Contracts

**Intent.** The smallest stable interfaces, with authority built in from the first commit
rather than retrofitted.

**Files.** `brain/task.py`, `brain/authority.py`, `brain/affordance.py` (dataclass only),
`storage/db.py` (**v22**), `storage/repos/task.py` (new).

**Deliverables.** §9's dataclasses; `TaskStatus` with declared legal transitions;
`Outcome` (five members, §11.2); `Observation`; `Verdict`; §4's creation gate and resume
rule; and **persistence** — `tasks` and `task_steps` tables at schema v22, storing
`principal` and `granted` alongside everything else, because a Task that survives restart
without its authority is the defect §4 exists to prevent.

`granted` serialises as a sorted list of capability values, never as an integer bitmask:
a bitmask silently re-maps if the enum's member order ever changes, which is exactly the
kind of quiet re-grant §3.3 S4 forbids.

**Properties to pin.** §9.1's list, plus:

- resume with a mismatched principal yields `SUSPENDED_NEEDS_AUTHORITY`, not execution
- resume with `current_grants` unset yields the same
- resume by a revoked device principal yields the same
- `resume_grants` is never wider than either input
- no background caller can reach the resume path at all (A5), asserted structurally, not
  by convention
- a Task round-trips through v22 with `principal` and `granted` intact, and a stored
  `granted` that names a capability the enum no longer has is a load **error**, not a
  silently-dropped member
- **abort during a step yields `CANCELLED`**, never `FAILED` and never `SUCCEEDED`;
  `UserAborted` is not swallowed into a string (`core/abort.py`)
- only one Task is in `RUNNING` at a time (R6)

**Required mutations.** Delete each of A3's intersection and A4's four conditions in turn;
each must red a distinct, named test. Serialise `granted` as a bitmask and reorder the enum
→ the round-trip test reds. Swallow `UserAborted` → the abort test reds. A mutation that
reds nothing means the property was not pinned — and per `CLAUDE.md`, a green mutant is
investigated, not accepted.

**Exit.** Contracts import in isolation. No `assistant.brain → assistant.io` edge exists.
Migration v22 runs on a real SQLite file in a tmp dir, not a mock — mocked DBs have masked
migration failures in this tree before. Baseline diff clean.

---

### P3 — Affordance registry and resolver

**Intent.** Make "what could satisfy this?" a deterministic lookup.

**Files.** `brain/affordance.py`, `brain/resolver.py`.

**Deliverables.** `AffordanceRegistry` on `core/registry.py:RegistryBase` — extending it,
not duplicating it. Self-registration by existing components. A resolver whose ordering is
**explicitly the existing routing order**: preference → URL pattern → running process →
launch keyword → app context → fallback (`automation/router.py:623 detect_backend`). Zero
LLM calls.

**And exactly one router** (§7.3). P3 does not close until `detect_backend` has either been
*moved* into the resolver with its call sites repointed, or been left in place with the
resolver delegating to it and adding nothing that decides. Two implementations of the same
ordering is the duplicate-orchestration anti-pattern, introduced by the phase meant to
remove it. Which of the two options is taken is a recorded decision in the phase report.

**Also here:** the KI-33 dispositions for anything the resolver touches —
`_CANVAS_APP_RE`'s brand names and `_BROWSER_PLAN_PROMPT`'s worked examples move to data or
are deleted. A resolver built on top of a brand-name regex has not fixed anything.

**Properties to pin.**

- AF1: every registered affordance names an intent in `config.INTENTS`
- AF2: an affordance's required capability equals its intent's, and an unlisted intent
  yields `EXECUTE` — a weaker declaration is rejected at registration, not at dispatch
- the resolver returns `UNSUPPORTED` rather than a wrong affordance when nothing matches
- no brand name appears in `brain/`, asserted by a source scan
- the word `capabilit` does not appear in `brain/` outside the documented import (§6)
- resolution is a pure function of registry + environment snapshot: same inputs, same
  answer
- **each routing signal — preference lookup, URL pattern, running-process check, launch
  keyword, app-context pattern — is implemented in exactly one module**, asserted by a
  source scan, so the "one router" decision cannot silently decay into two

**Required mutations.** Register an affordance naming a non-existent intent → import
fails. Give an affordance a weaker capability than its intent → AF2 reds. Add a brand name
to a `brain/` module → the scan reds. Duplicate the launch-keyword regex into a second
module → the one-router scan reds.

**Exit.** Resolution for the tree's ten highest-frequency intents matches today's routing
decision on a recorded corpus, or each difference is explained and accepted in writing.
Note that `automation/router.py:685-690` currently sends "open Wikipedia" to a Win-key
search (§2.1); that difference is a **fix**, and the corpus must record it as an intended
change rather than a regression.

---

### P4 — Brain boundary and turn entry (three sub-phases)

**Intent.** One coordinator, owning all four turn entry points (§2.3). Initially it
delegates almost everything.

**This is the highest-risk work in the plan.** It touches every security property in §3.
The first draft of this document made it a single phase, which contradicted the same
document's own instruction to P13 — *"one loop per commit, each with its own baseline diff
and its own live test; never two at once."* The same discipline applies here and for the
same reason: R2 says a phase you cannot revert in one step is too large.

**Order is by risk, ascending.** Each sub-phase is its own branch, its own squash commit,
its own baseline diff, and its own live test. A sub-phase that does not go green does not
block the next one from being reverted independently.

---

#### P4a — the scheduler entry points

**Files.** `brain/__init__.py`, `scheduler.py`.

The least risky of the four: two sites, both `LOCAL_GRANTS`, both already wrapped in
`try/finally`, neither reachable from a remote caller.

**Deliverables.** `Brain.run_turn(...)` — the single place the three contextvars are
installed, in the documented order (S3), with nothing between the last install and the
`try`. `scheduler.py:140` and `:169` call it instead of installing their own.

**Properties to pin.**

- install order unchanged; nothing between the last install and the `try`
- both tokens are reset on the success path, the exception path, and the abort path
- a scheduled `procedure` still reaches `procedure_executor`'s own `EXECUTE` backstop
- the scheduler still cannot resume a Task (§4 A5), asserted structurally

**Required mutations.** Move the grants install one statement earlier → the ordering test
reds. Drop the `finally` reset → the leak test reds. Let the scheduler call the resume path
→ the A5 test reds.

**Exit.** Live: one fired schedule of each task type, same observable behaviour.

---

#### P4b — the event bus entry point

**Files.** `automation/event_bus.py`, `pyproject.toml`.

**Deliverables.** The event bus stops importing `actions` and calls `Brain.run_turn(...)`.
This is §8.3's honest fix for the fourth layer inversion — the event bus is a *source*, not
an orchestrator — so **P1's deferred `layers` contract lands here**, with the corresponding
`ignore_imports` entries removed.

**Properties to pin.**

- a fired monitor produces the same dispatch it did before, with the same grants
- `automation ↛ actions` holds; the `layers` contract reports kept
- the monitor's `installed_by` (P-1) is logged at fire

**Required mutations.** Re-add the direct `actions` import → the contract reports broken.
Have the bus install grants itself → the single-install-site test reds.

**Exit.** Live: one fired monitor. `lint-imports` clean by summary line with the full layer
order now asserted.

---

#### P4c — the turn pipeline and the sweeps

**Files.** `main.py`, `brain/__init__.py`, `tests/test_6a5_predispatch_gate.py`,
`tests/test_6b_principal.py`.

The dangerous one. `process_text_from_queue` carries the contextvar ordering an adversarial
review pinned, eight gated pre-dispatch branches, two direct `capability_refusal`
consultations, and two AST sweeps bound to its name and file.

**Deliverables.**

1. The turn pipeline routes through `Brain.run_turn(...)`. The pre-dispatch region moves
   with its `_gate` closure and its `_respond` helper intact — **as a unit**, not
   branch-by-branch, because `_respond`'s three properties (always records the turn under
   the session id, speaks only to a local source, reopens the microphone only for a local
   source) were three separate defects with one fix, and separating them again re-opens
   all three.
2. **The two AST sweeps move with the code and gain anti-vacuity assertions** (§18.4):
   - the pre-dispatch sweep asserts the walked region is **non-empty** and that the count
     of guarded branches has not decreased without an accounted-for entry in a pinned
     list. It has no such guard today and would pass vacuously on a split function — filed
     as KI-32.
   - the arming sweep walks `brain/` as well as `main.py`, asserting a non-empty walk in
     **each** file, not across their union.

**Properties to pin.**

- the install order is unchanged and nothing sits between the last install and the `try`
- every pre-dispatch branch still refuses or skips exactly as it did, **branch by branch,
  by name** — a single aggregate test would pass while any one branch regressed
- a branch that *skips* (teaching, the pending chain) still skips silently rather than
  refusing, because refusing there both hijacks a reply that should get an ordinary turn
  and discloses that a confirmation is waiting
- `_respond`'s three properties survive the move, each asserted separately
- `LOCAL_GRANTS` is installed at exactly the sites §2.3 lists and nowhere new
- the Brain names no application and contains no `if intent ==` workflow chain
- both sweeps fail on an empty walk

**Required mutations.**

- delete one branch's `_gate` call → the sweep reds
- move the grants install one statement earlier → the ordering test reds
- **split the gated region so half the branches leave the walk → the non-empty/count
  assertion reds.** This mutation is the entire reason the assertion is added. If it
  passes, the sweep is measuring nothing and **P4c does not close.**
- move one arming site out of both walked files → the arming sweep reds
- make `_respond` speak to a remote source → the hot-mic test reds
- have the Brain install `LOCAL_GRANTS` for a `studio` source → the entry-point test reds

**Exit.** Baseline diff clean. **Live test required** (§18.6), five paths: one voice turn,
one Studio turn over `tailnet`, one refused turn over `funnel`, one teaching trigger from a
remote source (which must **not** open the microphone), and one pending confirmation armed
locally and answered locally.

---

### P5 — Execution migration

**Intent.** Structured parameters reach adapters. Adapters stop re-interpreting the user.

**Files.** `actions/da_handlers.py`, `automation/router.py`, `actions/planner/executor.py`.

**Deliverables.** `Executor.run(step: TaskStep) -> Verdict`. The `goal` string survives as
a payload field on `TaskStep` for adapters that still need one — deleting it in this phase
would rewrite ~40 files and six loops at once, which §23 forbids. What changes is that the
step's *meaning* is its structured fields; the string is data the adapter may read, never
the thing that decides.

**Properties to pin.**

- an executor never calls an intent classifier
- an executor never reads the original user utterance
- user-pinned constraint values reach the adapter byte-identical (`CLAUDE.md` gotcha:
  "mobile as 99999" is a hard constraint)
- an adapter that cannot satisfy a step returns `UNSUPPORTED`, distinct from `FAILED`
- `actions.execute()` is still the only handler-resolution site, and planner re-entry
  still passes through it

**Required mutations.** Have an executor call `detect_intent` → the reinterpretation test
reds. Silently substitute a constraint value → the pinned-value test reds. Return `FAILED`
where `UNSUPPORTED` is correct → the distinction test reds.

**Exit.** Baseline diff clean. The five highest-frequency flows behave identically, live.

---

### P6 — Verification semantics

**Intent.** Stop uncertainty reading as success (§11).

**Files.** `automation/verification.py`, and the seven `.ok` readers §11.3 names — and
only those seven.

**Deliverables.** `Outcome`; `VerifyResult.ok` removed; `ambiguous`/`skip`/crash all map
to `UNCERTAIN`; `vision_verify`'s fail-open returns `UNCERTAIN` rather than the ambiguous
`ok=True` it was handed; `data.get("ok")` with no default.

**Properties to pin.** V1–V6 (§11.2), each as its own test, plus:

- `PrimitiveResult`, `HealResult` and `DispatchResult` are untouched — a test asserts their
  booleans still mean what they meant, so the seven-site scope is pinned and cannot drift
- with `VERIFY_VISION_FALLBACK` off, an ambiguous step is `UNCERTAIN` and the Task is
  `UNCERTAIN`
- with the vision provider erroring, same
- with the vision reply missing `ok`, same

**Required mutations.** Restore `ok=True` on `ambiguous` → V2 reds. Restore
`data.get("ok", True)` → V6 reds. Make `vision_verify` return `code_result` on error →
the degraded-path test reds. Let one `UNCERTAIN` step still produce a `SUCCEEDED` Task →
V4 reds.

**Exit.** Baseline diff clean. KI-30 marked FIXED with the mutations recorded.
**Live test the answer, not the refusal**: run a step whose verification is genuinely
ambiguous with vision disabled, and confirm the spoken reply says she cannot confirm it —
not that it worked.

---

### P7 — Recovery and failure semantics

**Files.** `automation/recovery.py`, `actions/planner/planner.py:754 _attempt_recovery`.

**Deliverables.** Deterministic strategies first, then semantic. `RecoveryResult` carries
a `Verdict`. `UNSUPPORTED` / `FAILED` / `UNCERTAIN` / `NEEDS_CLARIFICATION` are distinct
all the way to the response.

**Properties to pin.**

- the original Task objective is byte-identical after recovery and after replanning
- a failed recovery escalates rather than reporting success
- recovery never widens a Task's scope
- an unknown failure escalates to the user rather than looping
- **a refused or skipped turn's reply cannot assert the state change that was refused**
  (S7 / KI-28), including when a fallback generator writes it from conversation text

**Required mutations.** Let recovery rewrite the objective → the preservation test reds.
Let a failed recovery return `SUCCEEDED` → it reds. Remove the `security_skip` exclusion →
the KI-28 test reds.

**Exit.** Baseline diff clean.

---

### P8 — Planner migration

**Files.** `actions/planner/planner.py` (1,864 lines), `actions/planner/executor.py`.

**Deliverables.** The planner consumes a Task and emits `TaskStep`s. It does not execute.
`needs_planning` is **kept** (§2.2) — it reads structured Task fields where they exist and
falls back to its existing regex path for free-text tasks. `_PLAN_PROMPT` examples are
re-checked against `CLAUDE.md`'s rule that step-planning prompts must not carry examples
matching test cases.

**Properties to pin.**

- a single-affordance Task never reaches the planner
- an invalid model-generated plan is never executed — validated before the first step
- the planner calls no executor
- plan suspension and resume preserve principal and grants (§4, and KI-27's neighbourhood)
- **`planner/executor.py:182`'s `state.clear()` (KI-27) is either brought inside the AST
  sweep or explicitly re-argued and re-filed.** It is invisible to the sweep today because
  it clears a loop-local; a phase that rewrites this file must not leave it invisible.

**Required mutations.** Feed the planner a plan referencing an unregistered affordance →
the validity test reds. Make it call an executor → the separation test reds. Drop the
principal across suspension → the authority test reds.

**Exit.** Baseline diff clean. KI-27 closed or re-filed with a current argument.

---

### P9 — Durable state and provenance

**Files.** `memory.py`, `storage/repos/memory.py`, `knowledge_graph.py`, `preferences.py`,
`automation/router.py:208`, `actions/__init__.py:_apply_preference_defaults`,
`reflection.py`, `knowledge.py`, `main.py:2194,2273`.

**Deliverables.** §10's provenance enum, D1/D2/D3, the writer allow-list, and the
consumer-side minimum-provenance declarations. `main.py`'s two duplicated fact-extraction
sites become one call into the gate. The reflection→routing key-namespace mismatch (§10.5)
is reconciled or the reflection preference category is removed — a decision, recorded.

**Properties to pin.**

- a single inference never becomes an unattended behaviour change
- a routing consumer rejects `single_inference` regardless of confidence
- a prompt-injected hint accepts only user-stated provenance
- a correction supersedes rather than accumulating a contradiction
- a component not on the allow-list cannot write a durable store, asserted by AST sweep
  over the tree, not by convention
- temporal validity is honoured: an expired fact is not returned

**Required mutations.** Write a fact from a non-allow-listed module → the sweep reds.
Make the routing consumer ignore provenance → its test reds. Set a reflection preference's
provenance to `explicit_user_statement` → the ladder test reds.

**Exit.** Baseline diff clean. §10.2's table is re-verified and any drift corrected.

---

### P10 — Context Builder, privacy, containment

**Files.** `brain/context.py`, `main.py:2506 _build_conversation_messages`,
`main.py:2560 _build_facts_context`, `llm/contracts.py`, `knowledge.py:273 render_for_llm`.

**Deliverables.** §12.1's six profiles. C1/C2/C3 fencing. `core/redact.py`
(`redact_secrets_strict`) wired to egress, **after** fencing. `context_bytes_by_profile`
emitted for §15.

**Properties to pin.**

- a field not on a profile's whitelist does not appear in the built context, asserted on
  the built object
- a secret-shaped value in a stored fact does not survive to egress
- fenced content carries its provenance label and the system prompt declares it data
- the full conversation is never sent by default
- `context_bytes_by_profile` is non-zero and changes when a profile changes — otherwise
  the minimization claim is unmeasured

**Required mutations.** Add an unlisted field to a profile's output → the whitelist test
reds. Bypass the redactor for one profile → the secret test reds. Strip the fence → the
containment test reds.

**Exit.** Baseline diff clean. `TENKA_Known_Issues.md` updated: **KI-15 is downgraded to
mitigated, not closed** (§12.2 C3), and KI-14/KI-16 are re-stated against the new boundary.

---

### P11 — Personality and truth separation

**Files.** `personality.py`, `personalities/`, `reflection.py`, `llm/prompts.py`.

**Deliverables.** The Character Contract as behavioural rules plus deterministic checks
where practical. Personality reads a `Verdict` and may change wording, never the verdict.
No new personality features. Nightly self-modification is **not expanded**; its preference
writes are already constrained by P9.

**Properties to pin.**

- personality cannot change a `Verdict`, structurally — it never receives a mutable one
- personality is present during refusal, failure, uncertainty and clarification
- an unnecessary AI disclaimer does not appear on an ordinary character question
- a direct question about her nature is answered honestly
- personality state does not drift without a recorded, reversible, inspectable reason
- **the failure mode that produced KI-28 cannot recur**: a response generator that did not
  see a refusal cannot assert the refused state change

**Required mutations.** Hand personality a mutable verdict and change it → the structural
test reds. Remove the refusal-aware path from response generation → the KI-28 test reds.

**Exit.** Baseline diff clean. Regression conversations recorded, not just unit tests.

---

### P12 — Self-Knowledge

**Files.** `brain/selfknowledge.py`, `config.INTENTS` (one new intent), `config.py` intent
catalogue, `core/intent_capabilities.py`, `TENKA_Capabilities.md`.

**Deliverables.** §13's authority table and K1–K5. A new intent — the first this plan adds
— classified in `core/intent_capabilities.py` in the same commit, per §7.3.

**Properties to pin.**

- every answer traces to a runtime fact, not to a document or the model's prior
- an unavailable fact yields K2's sentence, never a guess
- `technical` detail requires `OBSERVE`, `developer` requires `SYSTEM_CONTROL` (K4)
- a remote device cannot read listener/ceiling/transport facts it could not read from
  `GET /v1/transports`
- a capability TENKA does not have is never claimed
- git history never answers a *current capability* question

**Required mutations.** Remove the K4 capability check → the reconnaissance test reds.
Let the model answer from its own knowledge when a fact is missing → K2 reds. Let git
history answer "can you do X" → the authority test reds.

**Exit.** Baseline diff clean. `TENKA_Capabilities.md` is generated from the registry or
verified against it — never hand-maintained in parallel, which would be a second source of
truth.

---

### P13 — Duplicate orchestration removal

**Intent.** The six loops (§2.1) become one. This is the phase the source documents
under-scoped most severely, and it is placed last among the structural phases for that
reason.

**Files.** `actions/planner/planner.py`, `automation/vision/agent.py`,
`automation/browser/dom_orchestrator.py`, `procedure_executor.py`,
`code_executor/orchestrator.py`, `automation/recovery.py`, `main.py`.

**Constraint.** These loops have genuinely different semantics — vision replanning is not
code-fix retry is not plan-step recovery. Collapsing them into one loop by fiat is a
rewrite. **What unifies is the state machine, not the strategy**: each loop keeps its own
step-generation and retry strategy and gives up its own status vocabulary, its own success
test, and its own recovery escalation, adopting `TaskStatus`, `Verdict` and `Outcome`.

**Order.** One loop per commit, each with its own baseline diff and its own live test.
Never two at once. A loop that cannot adopt the shared state machine without changing
behaviour is **left alone and documented**, not forced.

**Properties to pin.** Per loop: identical externally observable behaviour on a recorded
corpus; no new success path; abort (`core/abort.py`) still interrupts at the same
boundaries; `UserAborted` is never swallowed into a string error.

**Required mutations.** Per loop: swallow `UserAborted` → the abort test reds. Let the
shared machine reach `SUCCEEDED` from an `UNCERTAIN` step → V4 reds.

**Exit.** Per loop, not per phase. `main.py` retains no execution loop of its own -- it
starts the process and owns nothing a turn passes through. Line count is a symptom, not
the criterion: the plan's own rule is to optimise responsibility boundaries, not file size
(§23). Baseline diff clean at each
commit.

---

### P14 — Observability completion

**Files.** `telemetry.py`, `storage/repos/telemetry.py`, `storage/db.py` (**v23**).

**Deliverables.** §15's field delta. O1–O3.

**Properties to pin.** No raw payload reaches the metrics store. Every field is populated
on a real turn — a field that is always null is not observability. The six questions in O2
are answerable by query alone.

**Required mutations.** Log a raw prompt into a metric field → the O1 test reds. Leave one
new field unpopulated → the population test reds.

**Exit.** A cost report for one simple, one moderate and one complex request, produced from
the store rather than by hand.

---

### P15 — World Model interface

**Files.** `brain/world.py`, `brain/__init__.py`.

**Deliverables.** §14's Protocol. W1–W4. **No implementation.**

**Properties to pin.** The Brain's full test suite passes with no provider registered.
W4's demonstration: the *absence* handling is exercised by a real path, not by a stub.

**Required mutations.** Make the Brain require a provider → W1 reds.

**Exit.** If W4 cannot be demonstrated, **this phase does not ship** and the Protocol is
deleted. A speculative abstraction with no consumer is forbidden by §23, and that applies
to this document's own proposals.

---

### P16 — Git development awareness (optional)

**Files.** `brain/selfknowledge.py`.

**Deliverables.** Git history feeds development history, which feeds Self-Knowledge and
personality colour. Nothing else.

**Properties to pin.** Git cannot influence a Brain decision, capability availability,
authorization, Task execution, or memory truth — five separate assertions, because a single
"git is read-only" test would pass while any one of them leaked.

**Required mutations.** Let a commit message flip a capability's availability → the
authority test reds.

**Exit.** Baseline diff clean.

---

## 18. Testing doctrine

### 18.1 Plans carry properties, never test bodies

Literal test code in a plan is unrun code. In Milestone 6b four of six vacuous tests were
authored in the plan and copied verbatim, and two were the only pin on a security property.
This document states properties and the shape of required evidence. **The implementer
writes the test and proves it fails without the mechanism.**

### 18.2 Every claim is mutation-proven

A test that does not fail when its mechanism is removed is not a test. Each phase lists its
required mutations. The report says which were run and what was deleted.

**A green mutant is investigated, not accepted.** 6b's transport work found a test pinning
the wrong depth of a hazard exactly this way.

### 18.3 Sibling-refusal isolation

A test can pass because a *sibling* check refuses the same input. Isolate the mechanism:
assert the specific refusal's wording, or remove whatever else could answer. This is how
6b's KI-17 containment test passed before the feature existed.

### 18.4 Structural sweeps get anti-vacuity assertions

Every AST sweep in this plan asserts:

1. the walk found something (`assert calls, "..."`), and
2. **for a sweep that can shrink** — the pre-dispatch region, the arming sites, the durable
   writers — that the count has not dropped without an accounted-for entry in a pinned list.

The second is new. The existing pre-dispatch sweep has only the first, which is why
splitting `process_text_from_queue` would pass it while measuring almost nothing (§2.5).

### 18.5 Anything bounded, anything waited on

A test that *hangs* on a mutant never fails. Every wait is bounded (`asyncio.wait_for`) so
a broken mechanism is a failure, not a stall.

### 18.6 Live-test the answer, not the refusal

A control that refuses correctly while silently corrupting what it permits passes every
red-green check there is. Phases P-1, P4a, P4b, P4c, P5, P6, P11 and P13 name a required
live test. A phase
with a live test does not close on unit tests alone.

### 18.7 A brief handed downstream is an artifact that can be wrong

If you are implementing and the instructions for a later task are wrong, **report it and
fix the plan**, before that task is dispatched. Two 6b tasks did exactly this and each
saved a live-test failure. This document is not exempt.

---

## 19. Definition of Done

One list. Not "all tests pass."

**Security** — the seven rules in §3.3 hold, each with a test that reds on its removal.
KI-29 fixed. No fifth turn entry point. No capability granted by default anywhere.
`lint-imports` clean by summary line, with a net-negative change to `ignore_imports`.

**Authority** — every Task carries a principal and a grant set. No Task resumes without a
live, authorised, principal-matching turn. No background runner resumes anything. A raised
capability cannot install a durable trigger.

**Architecture** — the Brain is the single orchestration boundary. Six loops share one
state machine or are documented as exceptions. Internal communication is structured. No
brand name and no `capabilit` in `brain/`. `main.py` owns process startup and nothing a
turn passes through. `assistant/brain/` is the only new package.

**Determinism** — a simple task makes zero unnecessary model calls, measured from the
telemetry store, not asserted. Executors and verifiers are primarily deterministic.
Recovery tries deterministic strategies first. Planning is conditional.

**Truth** — `SUCCEEDED` requires positive evidence. `UNCERTAIN` never satisfies a success
check at any level. `UNSUPPORTED`, `FAILED`, `UNCERTAIN`, `UNVERIFIED` and
`NEEDS_CLARIFICATION` are distinct to the response, and `UNVERIFIED` and `UNCERTAIN` never
share a sentence. A refused turn's reply cannot assert what was refused. Turning
off a verification setting cannot turn uncertainty into success.

**Durable state** — all eleven stores carry provenance. Consumers read it. A single
inference never becomes an unattended behaviour change. No component outside the
allow-list writes durable state.

**Privacy and containment** — six context profiles, each whitelisted and measured
(`context_bytes_by_profile`). The existing redactor is on the egress path. External content
is fenced. **KI-15 is mitigated, not closed, and the ledger says so.**

**Personality** — the Character Contract holds through refusal, failure, uncertainty and
clarification. Personality cannot change a verdict, structurally. Drift is bounded,
evidence-based, reversible, inspectable.

**Self-Knowledge** — every answer traces to a runtime fact. Detail levels are
capability-gated. An unavailable fact produces K2's sentence. `TENKA_Capabilities.md` is
generated or verified, never maintained in parallel.

**World Model** — interface only, with a demonstrated consumer of its absence, or deleted.

**Observability** — the six questions in O2 are answerable by query. No raw payloads in
metrics. Every new field populated on a real turn.

**Evidence** — `tests/BASELINE.md` green-or-explained at every phase boundary. Every phase
report lists the mutations run and what was deleted. Every phase with a named live test has
a recorded live-test log.

---

# Part V — Ledger

## 20. New known issues

**Three of the five below shipped on 2026-08-23, before this document was approved.** They
were security or honesty defects with no dependency on the refactor, and holding them for a
plan would have been the wrong trade. Their real ledger numbers are recorded here; the
sections above still reference the provisional ones and say so where it matters.

| Provisional | Real | Status |
| --- | --- | --- |
| KI-29 — raise → permanent `EXECUTE` | **KI-30** | FIXED `658247f`, audit half `d94838c` |
| KI-30 — uncertain verification reads as success | **KI-31** (recovery half) | PARTIAL — see below |
| KI-31 — LLM preferences steer routing | **KI-32** | FIXED `246b210`, guard `0567837` |
| KI-32 — pre-dispatch sweep goes vacuous | — | OPEN, unfiled, Low |
| KI-33 — brand names in a regex and a prompt | — | OPEN, unfiled, Low |

`KI-29` in the real ledger belongs to something else entirely: secrets stored unredacted in
SQLite and shipped to cloud backup, found and fixed on 2026-08-22 while building a routing
corpus from real history. That is why the numbering shifted.

### KI-30 (real) — a live raise could be converted into permanent local EXECUTE

**High. FIXED 2026-08-23.** Full chain, evidence and fix in §5, kept in past tense as a
worked example of the pattern §21.3 describes. The property that shipped: a capability held
only by virtue of a live raise may not be spent on an action whose effect outlives the
raise.

### KI-31 (real) — an uncertain verification reads as success

**Medium. PARTIALLY FIXED 2026-08-23 (`6c05b86`).** The recovery half is closed:
`recovery.py` escalates an ambiguous verdict to the vision tier as the three step loops
already did, and then requires positive evidence — `bool(vr.ok) and not vr.skipped and
tier != "ambiguous"`.

**The type is still the defect.** `VerifyResult.ambiguous()` and `.skip()` both still carry
`ok=True`, so any *future* reader of `.ok` inherits the same trap; `vision_verify` still
fails open by returning the ambiguous result unchanged; `data.get("ok", True)` still
defaults a missing verdict field to true; and `VERIFY_VISION_FALLBACK`, reachable over
`PATCH /v1/settings`, still turns ambiguity into success for anything that does not check
the tier itself.

Replacing the boolean with §11.2's five-member `Outcome` is what closes it, and that is
**P6** — seven call sites, and the `UNVERIFIED` distinction that stops `VERIFY_ENABLED=False`
making every task uncertain.

### KI-32 (real) — LLM-written preferences steer routing with no provenance check

**Medium. FIXED 2026-08-23.** `preferences.set_preference` caps a non-user writer at
`CONFIDENCE_FIRST_OBSERVATION`; `_check_routing_preference` requires `CONFIDENCE_SILENT`
for anything not user-stated. The ladder was already correct and simply unenforced —
reflection walked around it by passing the model's own confidence through.

Two things worth carrying into P9, both of which the existing tests taught rather than the
plan: the clamp belongs at the **facade**, because a repo that rewrites what it is handed
cannot restore or migrate one; and `USER_STATED_SOURCES` was incomplete on the first pass.

Also fixed with it: `router.teach_routing` called `set_preference` without its two required
arguments, so the one path where the user explicitly teaches a routing preference raised
`TypeError` on every call and had never once worked.

**Not closed:** `_apply_preference_defaults` still injects preference values into the
`code_executor` prompt, and the reflection prompt still asks the model for a confidence
number that is now capped rather than removed. Both belong with P9/P10.

### OPEN — the pre-dispatch AST sweep passes vacuously on a split function

**Low. Latent. Unfiled.** `tests/test_6a5_predispatch_gate.py::_predispatch_region` walks
the top of `process_text_from_queue`. Renaming or deleting the function errors loudly;
**splitting** it shrinks the region, `unguarded` becomes `[]`, and the test passes while
measuring almost nothing. Same shape as the route-completeness sweep 6b found empty.

Sharpened by experience since: the same class of gap was found for real in KI-30's fix,
where deleting the durability gate's wiring left seventeen unit tests green. **P4c must not
close without the count assertion in §18.4**, and that is no longer a hypothetical.

### OPEN — brand names in a routing regex and a planning prompt

**Low. THE-rule. Unfiled.** `automation/router.py` `_CANVAS_APP_RE` names figma, miro,
excalidraw, tldraw, sketch and google slides/docs/drawings — a regex mentioning brand
names, which `CLAUDE.md` forbids outright. `_BROWSER_PLAN_PROMPT` carries `bbc.com` and
`example.com` worked examples, against the rule that step-planning prompts must not contain
examples matching test cases. `actions/__init__.py:_apply_preference_defaults` hardcodes
music/project/download keyword lists and four preference keys. Inventoried in P0,
dispositioned in P3/P9.

## 21. Decisions required from the operator

Each blocks the phase named. None should be decided by an implementer. Struck-through rows
were decided on 2026-08-23 and are kept rather than deleted, so the reasoning survives the
decision -- a decision with no recorded argument is one the next person re-opens.

| # | Decision | Blocks | Recommendation |
| --- | --- | --- | --- |
| ~~**D1**~~ | Approve `assistant/brain/` as a tenth subpackage (`CLAUDE.md` rule 4). | ~~P2~~ | **APPROVED 2026-08-23.** Created with `task.py` and `authority.py`, plus three `forbidden` contracts: `brain ↛ io` (direct edges only — `brain → actions → io.audio.tts` is legal by construction and forbidding it would forbid `brain → actions` by proxy), `brain ↛ main`, `io ↛ brain`. |
| **D2** | Accept §4.3's cost: restart-surviving Tasks resume only on a qualifying turn, never on a timer. | P2 | Accept. The alternative is pre-authorised work a revoked device can still fire. |
| **D3** | Accept that intents stay as the execution ABI (§7), which is narrower than the source documents propose. | P3 | Accept. It preserves the security key domain, the published API and the fail-closed default, at no cost to the actual complaint. |
| ~~**D4**~~ | KI-30's fix refuses a raised device the ability to install monitors, schedules, procedures, shortcuts or backup jobs. | ~~P-1~~ | **DECIDED 2026-08-23: accepted, and widened.** After seeing it live the operator confirmed the gate should cover *all* of `manage_*` -- listing and deleting a monitor are refused too, not only creating one. Pinned by `test_the_gate_covers_management_not_only_creation` so it is not narrowed on sight. |
| ~~**D5**~~ | The reflection→preference→routing loop: tighten provenance, or remove the `app_routing` category from reflection entirely. | ~~P9~~ | **DECIDED 2026-08-23: tightened, not removed** (real KI-32). A model proposal is capped at the discovery floor and routing requires the silent bar for anything not user-stated, so the ladder now has to be climbed rather than skipped. The category stays; if it earns nothing, the log now says so per lookup. |
| **D6** | v21 (`installed_by`) **shipped and ran clean on the live database.** v22 (P2, `tasks`/`task_steps`) and v23 (P14, telemetry) still to come. Confirm no external consumer reads those tables directly -- Studio reads through the API, but a backup/restore path may not. | P2 | Still open for v22/v23. `manage_backup`'s restore path is the one to check, and v21 going through cleanly is weak evidence for it: `installed_by` is additive with a default, whereas `tasks` is a new table a restore may not know about. |
| **D7** | P13 may leave a loop unmerged if merging would change behaviour. Confirm that "documented exception" is acceptable over "forced uniformity". | P13 | Accept. Forced uniformity here is the rewrite this project keeps refusing for good reason. |
| **D8** | P3 must pick one of the two "exactly one router" resolutions (§7.3): move `detect_backend` into the resolver, or make the resolver a delegating adapter over it. | P3 | Move it. A delegating adapter leaves the decision in `automation/` while the Brain claims to own resolution, which is the ambiguity this document exists to remove. |
| **D9** | §12.2 C3 states that fencing **mitigates** injection and does not close it, so KI-14/15/16 stay open after this plan. Confirm that an honest "mitigated" is acceptable over a claimed fix. | P10 | Accept. Claiming closure without an adversarial live test is how a check comes to point one step to the side of the property. |

---

## 22. Explicitly out of scope

Stated so that no phase quietly absorbs them and no reader assumes they are covered.

### 22.1 Net-new subsystems this plan *does* add

The source documents insist six times that this is consolidation, not greenfield, while
proposing several subsystems that do not exist. This document does not repeat that. The
following are **new construction**, and are budgeted as such:

Task/TaskStep contracts, the Affordance registry and resolver, the Brain coordinator, the
Context Builder, the durable-write gate, Self-Knowledge, and the World Model Protocol.

Everything else in the plan is migration, correction, or deletion.

### 22.2 Not addressed

- **Injection is mitigated, not closed.** §12.2 C3. KI-14, KI-15 and KI-16 remain open
  after this plan, in a better state.
- **A raise spent directly on `code_executor`** can install an OS-level scheduled task
  outside TENKA. No in-process check can prevent that; it is what granting `EXECUTE` means.
- **`code_executor` writing a file that something later executes.** Out of scope.
- **KI-25** (`tools/package_studio_ui.py` vendoring a failed build) — dev tooling,
  untouched.
- **KI-26** (`PairDeviceDialog.carryState()` special-casing `local`) — Studio-side,
  untouched.
- **KI-3 / KI-9 / KI-10 / KI-11 / KI-19 / KI-20 / KI-21 / KI-22** — unchanged.
- **Proactive assistance, nightly personality expansion, new intents beyond P12's one,
  new automation tiers, new providers.** None of these are architecture work and all of
  them would be changed behaviour during a migration, which §23 forbids.
- **Concurrent Task execution.** R6 keeps the existing one-turn-at-a-time constraint.
  Running two Tasks at once would need a second answer to "whose grants are installed
  right now", and `current_grants` is a contextvar precisely because there is one. That is
  a design question, not a refactor step.
- **Performance work beyond measurement.** P14 measures. Optimisation is a separate
  decision made from that data.

---

## 23. Anti-patterns

One list. Anything here fails review regardless of which phase produced it.

- an application-specific branch in `brain/`; a brand name in `brain/`; a brand name in any
  regex anywhere (`CLAUDE.md`)
- a task-specific planner prompt; a prompt example that matches a test case
- `if user says X -> workflow Y`
- one LLM agent per subsystem; an LLM executor for a deterministic action; an LLM verifier
  for deterministic state; an LLM complexity classifier where structure decides
- a second orchestration layer; a second dispatch choke point; a second predicate for
  "may this turn do X"
- a giant global context object; full context to any subsystem by default
- silent exception swallowing that converts failure to success; a boolean standing in for
  a verdict; a default that resolves ambiguity toward success
- an unverified capability claim; a reply that asserts a state change a control refused
- a durable write with no provenance; a durable read that ignores it; a single inference
  becoming an unattended behaviour change
- git history controlling runtime state; a world observation becoming memory automatically
- personality deciding truth; personality-driven fake capability; uncontrolled drift
- an abstraction with no consumer (**including this document's own** — §17.P15)
- deleting working functionality without migration coverage; changing unrelated behaviour
  during a phase; adding a feature during a migration
- **adding to `ignore_imports`** (`CLAUDE.md` rule 12)
- a new top-level package under `assistant/` without asking
- a structural sweep with no anti-vacuity assertion
- a plan or brief containing a test body

---

## 24. Glossary

| Term | Meaning |
| --- | --- |
| **Affordance** | What TENKA can accomplish, independent of which application provides it. §6. |
| **Capability** | What a caller is permitted to ask for. The security enum. §3. |
| **Ceiling** | What a transport is trusted to carry, regardless of what a device holds. |
| **Raise** | A live, expiring, keyboard-minted lift of a transport's ceiling, within `raisable`. |
| **Durable authority** | What a caller holds with no raise in force. §5.3. |
| **Effective grants** | `device_grants ∩ (ceiling ∪ (raisable ∩ raised))`. Installed as `current_grants`. |
| **Principal** | Who is driving the turn. `local`, or `device:<id>`. `None` owns nothing. |
| **Turn entry point** | A site that installs grants, principal and raise context. There are four. §2.3. |
| **Durable state** | Survives the process *and* can influence a later turn. Eleven stores. §10. |
| **Provenance** | How a durable value was obtained. Required on write, consulted on read. |
| **Outcome** | A *step's* verdict: `SUCCEEDED` / `FAILED` / `UNCERTAIN` / `UNVERIFIED` / `UNSUPPORTED`. Replaces `ok`. §11.2. |
| **UNVERIFIED** | Verification did not run -- by operator policy, or because the action has nothing to verify. Distinct from `UNCERTAIN`, which means it ran and could not decide. |
| **TaskStatus** | A *Task's* lifecycle state. Eleven members. Deliberately not the same type as `Outcome`. §9.2. |
| **Verdict** | An outcome plus its observation, tier and escalation flag. §9.5. |
| **Profile** | A named whitelist of what one subsystem's context may contain. §12.1. |
| **Baseline diff** | The change in `tests/BASELINE.md` across a phase. §17.P0.5. |
| **Anti-vacuity assertion** | A structural test's proof that it walked something and that the count did not silently shrink. §18.4. |

---

## 25. Revision record — what the first draft got wrong

This document was written, then attacked, then revised. The sixteen defects found in its
own first draft are recorded here for the same reason `TENKA_Known_Issues.md` records
fixed issues: a plan is an artifact that can be wrong (§18.7), and the ways this one was
wrong are the ways the next one will be.

Six were substantive — they would have produced bad code or a reverted feature.

| # | Defect in the first draft | Resolution | Section |
| --- | --- | --- | --- |
| 1 | **`Outcome` had four members and collapsed "the operator chose not to verify" into "we tried and could not tell."** `VERIFY_ENABLED=False` would have made every task `UNCERTAIN`, so TENKA would say "I couldn't confirm that" to everything. A rule that makes an existing setting unusable gets reverted, taking the honesty property with it. | Fifth member `UNVERIFIED`. Policy-off and non-verifiable → `UNVERIFIED`; tier-off, vision failure and crash → `UNCERTAIN`. V4–V8 rewritten; the response layer must speak them differently. | §11.2 |
| 2 | **The KI-31 fix changed nothing.** Routing was to accept `repeated_inference`; reflection produces exactly that. Same behaviour, new label. The real defect is that nothing *counts* the repetitions — the model asserts them and `source="reflection"` records the provider, not the evidence. | D3 rewritten: reflection may only propose; TENKA counts from turn history and telemetry; an unverifiable claim is `single_inference`. Stated that without D3 the P9 fix is a rename. | §10.3, §10.5 |
| 3 | **Two routers.** P3 lifted the routing order into `brain/resolver.py` and never said what happens to `automation/router.py:detect_backend` — leaving two sites deciding how a goal executes. The duplicate-orchestration anti-pattern, introduced by the phase meant to remove it. | §7.3 added: exactly one router, two permitted resolutions, a recorded decision, and a source scan asserting each routing signal exists in one module. | §7.3, P3 |
| 4 | **`PERSISTS_AUTHORITY` was a set with a silent default**, which fails open for a future intent — the opposite of `DEFAULT_REQUIRED = EXECUTE`'s discipline. The draft argued the default was safe; it is not, and the other default (`persists`) would have refused `code_executor` to a raised device, destroying the raise's purpose. | Neither default is safe, so there is none: two exhaustive sets, and a test enumerating `config.INTENTS` that reds on an intent in neither or both. `CLAUDE.md`'s three-place intent rule becomes five. | §5.3, §7.4 |
| 5 | **A live High severity vulnerability was scheduled behind a documentation phase.** KI-29's fix depends on nothing in the audit. | New **P-1**, shipping first and independently, carrying KI-29, the `recovery.py` half of KI-30, and `save_fact`'s provenance default. | §16.2, P-1 |
| 6 | **P4 was a big-bang** — four turn entry points, the contextvar ordering, eight gated branches and two AST sweeps in one phase — while the same document told P13 "one loop per commit, never two at once." | Split into P4a (scheduler), P4b (event bus, which is also the fourth layer inversion's honest fix), P4c (the turn pipeline and the sweeps), by ascending risk, each its own branch, baseline diff and live test. | P4a–P4c |

Ten were gaps rather than errors, each closed in place:

| # | Gap | Closed by |
| --- | --- | --- |
| 7 | Task persistence was never scheduled — Tasks "survive restart" with no table. | P2 gains schema **v22** (`tasks`, `task_steps`), with `granted` serialised as capability *names*, never a bitmask. |
| 8 | Schema versions were claimed twice (P1 and P14 both said v21/v22). | Allocated once, in §16.2: v21 → P-1, v22 → P2, v23 → P14. |
| 9 | No git workflow, despite `CLAUDE.md` being emphatic about it. | **R1** — branch per phase, squash-merge, mandatory trailer, no AI attribution, never delete, always push. |
| 10 | No rollback story. | **R2** — revert one squash commit; a phase you cannot revert in one step is too large. |
| 11 | No documentation-update requirement, in a plan whose whole premise is that documentation drifted. | **R3** — `ARCHITECTURE.md`, `TENKA_Architecture.md`, `TENKA_Known_Issues.md` move in the same commit, and the report notes the gitignored ones. |
| 12 | Concurrency unstated: nothing said whether two Tasks may run at once. | **R6** — one Task at a time, matching `_StudioDispatch`'s existing refusal. |
| 13 | Abort was in `TaskStatus` but never wired. | P2 pins `CANCELLED` on abort and that `UserAborted` is never swallowed. |
| 14 | The per-file baseline was presented as if equivalent to a full run, which it is not — process-wide singletons mean it cannot see cross-file leakage. | P0.5 states the limitation, requires singleton-touching phases to run their own files together in one pass, and adds a `live_automation` marker audit. |
| 15 | §13's detail levels mapped to capabilities, but `OBSERVE` is in **every** ceiling — so "technical requires OBSERVE" would have handed a public `funnel` URL her current task and model chain. | K4 rewritten: gate the **fact class**, not the label, each class naming the capability that already governs the same fact elsewhere. |
| 16 | The Definition of Done used "`main.py` under 1,500 lines" while §23 forbids optimising file count for its own sake. | Replaced with a responsibility criterion: `main.py` owns process startup and nothing a turn passes through. |

Two claims in the draft were checked against the tree rather than assumed, and both held:
`handle_manage_monitor` has no local-only guard beyond the capability gate
(`actions/monitors.py:41`), so KI-29 is reachable; and `_pref_hints` genuinely reaches a
code-generation prompt (`actions/da_handlers.py:408` →
`execute_code_task(preference_hints=...)`).

One earlier claim did **not** hold and is corrected in §11.3: an initial count of twenty
`VerifyResult.ok` readers was wrong. Seven are `VerifyResult`; the rest belong to
`PrimitiveResult`, `HealResult` and `DispatchResult` and are untouched. A phase brief
carrying the wrong number would have sent an implementer to rewrite four unrelated result
types.

---

# Part VI — The subsystems this document under-audited

## 26. Why this part exists

Sections 1–25 were written after measuring the tree, and then weighted wrong. The
6a.5/6b security surface is **12,580 of 74,641 lines — 17%** — and it took the majority of
the document's attention, because `CLAUDE.md` was itself ~38% milestone content and the
emphasis was inherited rather than derived. Word counts in the first draft: `grant` 62,
`capability` 67, `raise` 45 — against `vision/agent` 2, `dom_orchestrator` 2,
`knowledge_graph` 2, `topic_tracker` 0, and "robotic" — the operator's own word for the
complaint that started this — **0**.

This part is the correction. Six subsystems, **12,059 lines**, read for the first time on
2026-08-23. Three findings change phases above; one changes what the whole document is for.

**The headline: the plan's diagnosis of personality is wrong, and the operator's complaint
has a two-constant cause.** See §27.

Honesty about depth: §27 and §28 are read closely. §29–§31 are measured and skimmed, and
are marked so. A part that claims uniform depth would repeat the first draft's mistake in
the opposite direction.

---

## 27. Personality — "robotic" is quantisation, not architecture

§16 and §21.4 say *"do not rely on a giant prompt as the entire personality system"* and ask
for personality to emerge from memory, continuity, context and preferences.

**It already does.** `llm/prompts.py:build_personality_prompt` composes four live sources:

```
personalities/<name>/prompt.txt        the base character
  + trait-tiered modifiers             from live SQLite trait state
  + relationship context summary       conversation count + recent snippets
  + preference behavioural block       from the preference store
```

Per personality, as data: `prompt.txt`, `traits.json`, `modifiers.json`, `responses.json`.
Adding a personality is a folder. That is the architecture §16 asks for, shipped.

So P11 as written — "make personality depend more on context and less on prompt" — is
aimed at a problem that does not exist, and would spend a phase rebuilding what works.

### 27.1 What actually makes it feel predictable

Two constants and a JSON file. All three measured:

| Mechanism | Resolution | Consequence |
| --- | --- | --- |
| `_get_trait_tier` (`llm/prompts.py:16-24`) | **3 tiers**, boundaries 0.34 / 0.67 | 6 traits × 3 tiers. Trait drift *within* a tier changes the prompt by **zero bytes** |
| `MAX_DELTA_PER_CYCLE = 0.05` | tier width 0.33 | **~6.6 reflection cycles** to cross one boundary. Nightly, that is a week before anything she says changes |
| `responses.json` | 41 keys, median **3** variants, minimum **1** | Common paths repeat verbatim. Some keys have exactly one sentence, forever |

`personality_say` (`actions/responses.py`) random-picks from the pool — 13 modules use it,
deliberately, for "variety without LLM cost". The design is sound; the pool is three
sentences deep.

So the felt experience is: a fixed base prompt, modifiers that change once a week, and a
handful of canned lines on the paths hit most often. That reads as robotic because it *is*
nearly deterministic — not because personality is prompt-shaped.

### 27.2 What to do instead of P11

**None of this needs the Brain.** In rough order of felt improvement per unit of work:

1. **Widen the pools.** 41 keys × 3 → 41 × 8 is a data change, no code. Biggest effect,
   lowest risk, and it is the thing the operator actually notices.
2. **Interpolate instead of tiering.** Pass the trait *value* into the prompt, or use more
   than three bands. A continuous input makes a week of drift visible the day it happens.
3. **Then** consider P11's structural work, if anything still feels flat.

P11 is therefore **rewritten** (§33) from "restructure personality" to "raise its
resolution, then re-measure". This is the one place where the correct plan is smaller than
the one that was written.

### 27.3 The invariant that survives unchanged

§18's `system truth → personality → expression` is right and is not what this touches.
Widening a response pool cannot make a refusal claim success — the pools are keyed by
outcome, and KI-28's fix means a refused turn answers from a fixed string with no model
call at all. Raising resolution and keeping personality out of truth are independent.

---

## 28. Vision agent — a second implementation of Part III's contracts

`automation/vision/agent.py` is **3,146 lines**, the largest module after `main.py`, and
§2.1 counted it as one of six "orchestration loops" to unify in P13. That undersells what it
is. Roughly:

| Lines | What |
| --- | --- |
| ~1,270 | pyautogui action primitives — click, type, hotkey, OCR snap, app focus |
| ~250 | the agent system prompt |
| ~390 | TODO tracking (`_generate_initial_todos`, `_update_todos_after_batch`) |
| ~440 | action-to-TODO matching and visual confirmation |
| ~480 | the planning loop |

`_TaskState.todo_list` is a list of dicts carrying `id, task, done, kind, target, field,
value, pending_visual_confirm, confirm_strikes`. **That is a TaskStep with a verification
state machine**, hand-rolled, alongside its own stuck detector
(`zero_progress_streak`, aborts at 3), its own dynamic budget (`loop_budget`), and its own
escalation (`_MAX_CONFIRM_STRIKES = 3`).

So it is not a loop that needs the shared state machine bolted on. It is a **parallel
implementation of P2's `Task`/`TaskStep`/`Verdict`** with ~830 lines of semantics the
contracts do not currently express. P13 as written would either discard that or wrap it.

**Consequence for P2:** design `TaskStep` and `Verdict` against this, not only against
`actions/planner/planner.py`'s `PlanStep`. The planner's step is `(id, tool, goal, depends_on,
condition, status, output, error)` — coarse. The vision agent's carries per-step
verification state, which is the harder case and the one that will break a contract designed
for the easy one.

### 28.1 It is ahead of the plan on honesty

The thing I expected to find here was the ambiguity-reads-as-success defect fixed in
`recovery.py` (real KI-31). It is not here. When vision cannot confirm a TODO, the agent
marks it done-as-abandoned and `_append_abandoned_suffix` appends
`"(couldn't visually confirm: <fields>)"` to the spoken reply.

That is exactly §11's `UNCERTAIN` discipline, implemented before it was specified, in the
subsystem this document ignored. **P6 should adopt its shape rather than invent one**: a
per-item confirmation state, an explicit abandoned flag, and disclosure in the user-facing
sentence.

Two smaller notes: the `~250`-line system prompt is a P10 context-profile consumer nobody
listed, and `MAX_STEPS = 15` / `MAX_LOOPS = 8` are budgets P14's cost work should read rather
than re-derive.

---

## 29. Topic tracking — KI-16's description is stale

**Measured, not deeply read.** 248 lines.

KI-16 says pronoun resolution *"rewrites pronouns with the previous turn's trailing noun"*,
with examples resolving `it` to `a public` and `the shell command`.

The code no longer works that way. `push_turn` runs spaCy, then ranks candidates in three
bands — named entities filtered to `_ENTITY_LABELS`, copula predicates ("Y" in "X is Y"),
then generic noun chunks — inserting so that proper nouns end up on top. `resolve_query`
additionally skips code-shaped bodies, protects fenced and quoted spans, and skips
`this`/`that` when they act as determiners. Each guard carries a comment naming the live
test that produced it.

**But the failure mode survives where the ranking has nothing better.** A turn with no
named entity and no copula predicate still promotes a generic noun chunk, which is exactly
what KI-16's examples are. So: **mitigated by ranking, not eliminated**, and the ticket's
description is wrong in a way that would send someone looking for code that no longer
exists.

Action: re-describe KI-16 against the current implementation before anyone works on it.
This is the "verify a report against the current tree" rule applied to the project's own
ledger.

---

## 30. Code executor — measured, not audited

**Measured only.** ~2,900 lines across `orchestrator.py` (896), `sandbox.py` (723),
`retry.py` (590), `templates.py` (444), `discovery.py` (722), `_utils.py`,
`router_examples.py`, `routing.py`, `prompts.py`, `packages.py`.

What is visible without reading it properly, and what it implies:

- It has a **retry loop with its own knowledge feedback** — `knowledge.py`'s per-service
  `works`/`never` entries are written from execution outcomes and injected into fix prompts.
  That is a *learning* loop, not just a retry loop, and §10's durable-state inventory lists
  the store but not the loop that feeds it.
- `templates.py` + `router_examples.py` mean generated code is **saved and replayed**. §10
  counts per-service knowledge as durable state; saved templates are a twelfth store and are
  missing from the table.
- `sandbox.py` at 723 lines is the actual boundary for "run arbitrary code", and §3 discusses
  `EXECUTE` as a *permission* without once describing what the sandbox does with it.

**This is the largest remaining gap in the document.** P0's audit must cover it properly, and
until it does, P5's claim that execution migration is well understood rests on nothing for
~2,900 lines of the most privileged code in the tree.

---

## 31. DOM orchestrator and knowledge graph — measured, not audited

**Measured only.** `browser/dom_orchestrator.py` (1,739) and `knowledge_graph.py` (829).

The orchestrator is the browser tier's own step loop, with the same standing as §28's
finding: expect it to carry step semantics the contracts must accommodate, and check before
assuming P13 can absorb it.

The knowledge graph gets one paragraph in §15 — "demote it from the Brain to an
implementation behind a Memory Service" — for 829 lines plus `kg_entities`, `kg_facts`,
`kg_relationships` and `kg_commitments`. Whether a Memory Service abstraction fits it is
unexamined. §15's instruction may be correct; it is currently unsupported.

---

## 32. What this part changes above

| Section | Change |
| --- | --- |
| §2.1 | The vision agent is not merely a sixth loop; it is a parallel implementation of P2's contracts (§28) |
| §9.3 | `TaskStep` must be designed against the vision agent's per-step verification state, not only `PlanStep` |
| §10.2 | The durable-state table is missing saved code templates. Twelve stores, not eleven (§30) |
| §11 | P6 should adopt the vision agent's abandoned-with-disclosure shape rather than invent one (§28.1) |
| §16 | **P11 is rewritten** — raise personality resolution and re-measure, do not restructure (§27, §33) |
| §17.P0 | The audit must cover `code_executor/` properly. It is the largest unexamined surface and the most privileged (§30) |
| §20 | KI-16's description is stale and must be rewritten before anyone acts on it (§29) |

---

## 33. P11, rewritten

### P11 — Personality resolution (replaces "personality and truth separation")

**Intent.** The composition is already right (§27). Raise its resolution, then measure
whether anything structural is still needed.

**Files.** `personalities/*/responses.json`, `llm/prompts.py`,
`storage/repos/personality.py`.

**Deliverables, in order, stopping when it feels right:**

1. **Widen the response pools** from a median of 3 variants to at least 8, and fix the keys
   that have exactly 1. Data only, no code.
2. **Replace the 3-tier collapse** with either the raw trait value in the prompt or a finer
   band count. A week of drift should be visible the day it happens, not on the seventh
   reflection cycle.
3. **Re-measure before doing anything else.** If it no longer reads as robotic, the rest of
   this phase does not happen.

**Properties to pin.**

- personality still cannot change a `Verdict` — structurally, it never receives a mutable one
- a refused turn's reply still comes from the fixed string with no model call (KI-28)
- every response key has at least the minimum variant count, asserted over all personality
  folders so a new personality cannot ship with one-sentence pools
- trait resolution is finer than three bands, asserted numerically rather than by inspection

**Required mutations.** Collapse the trait bands back to three → the resolution test reds.
Reduce a pool to one variant → the pool-depth test reds. Hand personality a mutable verdict
→ the structural test reds.

**Exit.** The operator says it no longer feels robotic, or says what still does. That is the
only exit criterion that matters here, and it is not something a test can assert.
