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
| `pre-commit` | No direct commits to `main`; no non-squash merges into `main`; `import-linter` contracts hold whenever a `.py` file is staged |
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
