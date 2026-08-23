"""assistant/brain/ — the coordination boundary.

Approved as a tenth subpackage (`CLAUDE.md` rule 4) on 2026-08-23. There is no
existing package whose responsibility this is: putting it in `actions/` would
make the handler package an orchestrator, and `core/` imports nothing by design.

**Contracts only, for now.** P2 lands the types; the coordinator that uses them
is P4. Nothing here dispatches, and nothing here is wired into a turn yet.

Layering, enforced by `pyproject.toml` and by `tests/test_brain_layering.py`:

    brain -> core, config, storage, llm, domain, automation, actions   allowed
    brain -> io                                                       FORBIDDEN
    brain -> main                                                     FORBIDDEN
    io    -> brain                                                    FORBIDDEN

`io/api` reaches the Brain the way it already reaches the pipeline: through the
`ChatDispatch` protocol in `actions/`, injected by `main.py`. That indirection
is not ceremony -- `io.api` may not import past `core`+`config`, so a direct
import would be a contract violation and a re-litigation of a boundary that was
argued once already.

One vocabulary note, pinned by a test: the word "capability" in this package
means `core/capabilities.py`'s security enum and nothing else. What TENKA can
*do* is an **affordance**. The two were the same word in the source documents,
and that collision is what made the plan look implementable while it quietly
proposed re-keying the only working security control in the tree.
"""
