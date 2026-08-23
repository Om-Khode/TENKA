---
paths:
  - "assistant/io/api/**"
  - "assistant/core/capabilities.py"
  - "assistant/core/intent_capabilities.py"
  - "assistant/core/principal.py"
  - "assistant/actions/__init__.py"
  - "assistant/pending.py"
  - "assistant/main.py"
  - "assistant/scheduler.py"
  - "assistant/automation/event_bus.py"
  - "assistant/procedure_executor.py"
  - "tests/test_6a5_*.py"
  - "tests/test_6b_*.py"
  - "tests/test_api_*.py"
---

# The capability and authority model

Full architecture: `ARCHITECTURE.md` §20. This file is the rules — what may not change,
and why. Milestones 6a.5 (`b48e3e4`) and 6b built it; both were live-tested against real
tunnels on real hardware.

## The model in four lines

```
device grants (vault)              what this device was issued
  ∩ listener ceiling (policy)      what this transport is trusted to carry
  ∪ (raisable ∩ raised)            what a live, expiring, keyboard-minted raise lifts
  = effective grants               installed on the turn as `current_grants`
```

`io/api/policy.py:effective()` is that arithmetic. It can only **narrow** the device side.
A raise widens only the *transport* side, only within a per-policy `raisable` literal.

`current_grants` answers *what*. `current_principal` (`core/principal.py`) answers *who*.
They are different questions: 6a.5 asked only the first, which is why a device legitimately
holding `FILES` could answer a file confirmation the operator had armed (KI-13).

## The two enforcement points — both load-bearing

1. **`actions/__init__.py`**, immediately before `tool_registry.get(intent)` — the only site
   in the tree that resolves a handler. `planner/executor.py` re-enters through it, so a
   planned step is checked by the same rule as a direct turn, recursively.
2. **`main.py`'s pre-dispatch region** — branches that produce an effect and `return` before
   dispatch. Guarded by `actions.capability_refusal()`, AST-walked by
   `tests/test_6a5_predispatch_gate.py`, which fails on any new unguarded `return`.

`capability_refusal()` is the **only** predicate that answers "may this turn do X". Never
re-implement it at a call site. 6a.5's review found the gate guarding the last door while
five earlier ones stood open, precisely because the check lived *inside* `execute()`
instead of beside it as something a caller could ask.

## Rules that do not bend

- **Fail closed means `None` refuses.** Unset grants refuse everything; an unset principal
  owns nothing; an unregistered port grants nothing; an unlisted intent requires `EXECUTE`.
  The absence of a decision is never a decision to allow.
- **Policy is keyed on the local port a connection was accepted on** — read from the ASGI
  scope's own server address. Never the peer address (tunnels connect from 127.0.0.1 and are
  indistinguishable from a local caller), never the `Host` header (attacker-controlled).
- **Listener ports are fixed, one per policy, declared as data in `io/api/listeners.py`.**
  Never kernel-assigned. A policy added to `policy.py` without a matching port entry must
  fail loudly, never inherit one.
- **`ceiling`, `raisable` and `pairable` are explicit literals on every policy.** Never
  `frozenset(Capability)`, never a subtraction. The enum growing must never grant anything
  on any transport by default.
- **The `local` listener never trusts a published hostname.** Its `HostGate` accepts
  loopback names only (`127.0.0.1`, `localhost`, `[::1]`), regardless of `PublishedHosts`.
  This is KI-17's load-bearing layer 3 — it holds against a tunnel TENKA never launched.
- **A raise is ceiling-only, minted at the keyboard, expiring, and cannot manufacture a
  capability.** It widens the transport side within a fixed `raisable`, never the device
  side. Minting is `require_admin(SYSTEM_CONTROL)`, loopback-only.
- **Three doors, one rule.** `pending.try_arm` and `pending.try_clear` apply the same
  ownership condition the answer side uses — 18 arm sites and every clear site, each with an
  AST sweep that fails when an unguarded site appears (KI-13, KI-18, KI-24).
- **A refusal must not lie, and neither may the reply after it.** A skipped or refused turn
  sets `conversations.security_skip = 1` and is excluded from session summarisation (KI-28:
  she said "I've cancelled that deletion" while it was still armed, and the sentence was
  persisted as fact).
- **`DANGEROUS_PATTERNS` stays deleted.** Do not reintroduce a word deny-list. If a new
  class of harm needs blocking, the boundary is a capability enforced at dispatch.

## Turn entry points — there are four

Every site that installs grants, principal and raise context, in that order, with **nothing
between the last install and the `try` whose `finally` resets them**. An adversarial review
found a raise in that window leaking a grant set into the queue consumer.

| Site | Sources | Grants |
| --- | --- | --- |
| `main.py:~1257` `process_text_from_queue` | stt, console, studio, follow-up | per-caller |
| `automation/event_bus.py:~311` | a fired event monitor | `LOCAL_GRANTS` |
| `scheduler.py:~140` | scheduled `web_search` | `LOCAL_GRANTS` |
| `scheduler.py:~169` | scheduled `procedure` | `LOCAL_GRANTS` |

The last three grant everything on one argument: *"whoever installed this already held
`EXECUTE`."* True when written, false once a raise exists -- "held" has to mean *durably*.
See KI-30. Do not add a fourth site that reasons this way, and do not widen the three that
do.

**Two predicates, two questions.** `capability_refusal` asks what the caller may do *now*,
which includes anything a live raise lifted. `durable_capability_refusal` asks what it
holds with **no raise in force** (`issued & ceiling`), and gates the intents in
`PERSISTS_AUTHORITY` -- the ones that install something which runs later. A raise expires;
a monitor does not. The gate covers the **whole intent**, so a raised device cannot list or
delete monitors either -- decided deliberately (KI-30), because only the handler knows which
calls create, and moving the check there makes every handler responsible for remembering it. The two durability sets are exhaustive over `config.INTENTS` with no
default, because "persists" would refuse `code_executor` to a raised device and
"transient" fails open for the next intent nobody classified.

## When you touch this

- Read `ARCHITECTURE.md` §20 first, then `io/api/policy.py`'s module docstring — it argues
  the rejected alternatives at length so they are not rediscovered.
- Reviewing pieces in isolation systematically passes things that do not compose. 6b
  produced this four times, and two were found only by a person using the product. Live-test
  the **answer**, not the refusal.
- Open items live in `TENKA_Known_Issues.md` — the only ledger. Read it before changing
  anything in this area; several closed issues are closed *because* of a specific mechanism
  here, and removing the mechanism reopens them silently.
