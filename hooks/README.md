# Git hooks

Tracked in git and shared by everyone, unlike `.git/hooks/` — which is local-only,
never pushed, and was silently lost in a disk failure along with every rule it
enforced. That is why these live here.

## Enable (one time, per clone)

```bash
git config core.hooksPath hooks
git config commit.template .gitmessage
```

`core.hooksPath` is per-clone config, so a fresh clone has hooks **off** until
you run that line. There is no way to make a repo auto-enable hooks — git
deliberately forbids it, since running repo-supplied scripts on clone would be
a security hole.

Verify:

```bash
git config core.hooksPath   # -> hooks
```

## What runs

| Hook | Enforces |
| --- | --- |
| `pre-commit` | No direct commits to `main`; no non-squash merges into `main`; no `task/` branch squashed into `main`; every branch named `<type>/<slug>`; `import-linter` contracts hold whenever a `.py` **or `pyproject.toml`** is staged |
| `commit-msg` | The `.gitmessage` template — type prefix, ≤72-char subject, and the mandatory `TENKA ~ "…"` trailer |

`_common.sh` holds shared helpers and is sourced, not executed.

### The squash-merge exception

`pre-commit` blocks commits on `main`, but the intended merge flow *does* commit
there:

```bash
git switch main
git merge --squash feat/my-thing
git commit                       # allowed
```

`git merge --squash` leaves a `SQUASH_MSG` file behind, and the hook treats that
as the signal for a legitimate squash commit. A plain `git merge` leaves
`MERGE_HEAD` instead and is rejected, because it would replay the branch's whole
history onto `main`.

### One commit on `main` per unit of work

`pre-commit` used to ask only *"is this a squash?"*, never *"a squash of what?"*.
Milestone 6b satisfied it twenty-plus times: every task branch squashed into `main`
separately, each commit individually legitimate, and `main` ended up with the whole
milestone smeared across it.

Two namespaces fix the part a hook can fix:

```
feat/milestone-7               integration branch  ──squash──▶ main   (ONCE)
task/milestone-7/1-listeners   task branch         ──squash──▶ feat/milestone-7
task/milestone-7/2-verifier    task branch         ──squash──▶ feat/milestone-7
```

The full flow:

```bash
git switch -c feat/milestone-7                    # integration branch, once
git switch -c task/milestone-7/1-listeners        # a step inside it
#   ... work, commit freely ...
git switch feat/milestone-7
git merge --squash task/milestone-7/1-listeners && git commit

#   ... repeat per task ...

git switch main                                   # once, at the end
git merge --squash feat/milestone-7 && git commit
```

`main` gets one commit. `feat/milestone-7` keeps one commit per task. Each
`task/…` branch keeps every individual step. Nothing is lost, and nothing is
deleted — the never-delete-a-branch rule covers `task/` too.

**Why `task/` is a separate top level, not a child of the integration branch.**
Git stores refs as files, so `refs/heads/feat/milestone-7` cannot also be a
directory. Creating `feat/milestone-7/1-listeners` while `feat/milestone-7`
exists fails in git itself:

```
fatal: cannot lock ref 'refs/heads/feat/milestone-7/1-listeners':
       'refs/heads/feat/milestone-7' exists; cannot create ...
```

Measured 2026-08-22. The nested-under-the-integration-branch layout is
impossible, not merely discouraged — which is why the task namespace is
`task/<unit>/<slug>`.

**How the hook knows the source branch.** `git merge --squash` writes no
`MERGE_HEAD` and records the source branch nowhere. But `SQUASH_MSG` lists the
squashed commits, so the hook takes the newest SHA and runs
`git branch --contains` on it to recover which branches hold it. `@{-1}` and the
reflog were both rejected: each depends on how you happened to arrive at `main`.

**What this does *not* enforce.** No hook can tell that twenty separate
`feat/thing-1` … `feat/thing-20` branches were one unit of work. The hook rejects
the obvious mistake (`task/` reaching `main`) and makes the intended path the easy
one. Deciding what counts as one unit is still a judgement call.

## Notes for anyone editing these

- **POSIX `sh`, not bash.** Git for Windows runs hooks through its bundled `sh`.
- **Don't trust `import-linter`'s exit code.** All three obvious invocations
  report the wrong status — the entry-point call exits 0 even with broken
  contracts. `pre-commit` parses the `Contracts: N kept, M broken.` line and
  fails closed when that line is missing. See the comment in the hook.
- **Test changes by running the hook directly**, e.g.
  `sh hooks/commit-msg /path/to/msgfile`, rather than by making real commits.

## Bypassing

`git commit --no-verify` skips both hooks. Don't. If a hook fires, fix the cause —
that is the whole point of it existing.
