"""Replay regex-routed turns through the classifier and diff the answers.

`regex_router.pre_route` is a cost optimisation: it answers ~40-50% of daily
commands with zero LLM calls. Its failure mode is not being wrong about what it
claims -- it is claiming too much. A pattern that matches beyond its intent
silently steals the request from the classifier that would have got it right,
and the misroute is invisible because the fast path never asks a second
opinion.

This asks the second opinion, in bulk, after the fact. For every recorded turn
the regex claimed, it runs the classifier on the same utterance and prints the
disagreements. Each disagreement is either a regex over-claim or a case worth
labelling as a deliberate exception.

Why a tool and not a test: the useful input is real history, and real history
cannot live in the repository. The first version of this was a committed
fixture generated from `interaction_events`, which pulled three live OAuth
credentials out of the database and came within one `git add` of a public repo
-- see KI-29. So this reads the local database, prints to the terminal, and
**never writes a file**. A disagreement worth keeping gets hand-copied, with
inert content, into `tests/test_routing_overclaim.py`.

    py -3.11 tools/routing_differential.py                 # cost estimate only
    py -3.11 tools/routing_differential.py --run           # 20 utterances
    py -3.11 tools/routing_differential.py --run --all     # everything

Roughly 2,500 input tokens per utterance (the intent system prompt dominates).
On Gemini Flash-Lite that is about $0.01 for 40 utterances, and free on the
free tier -- the real limit there is requests per day, not cost.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import pathlib
import sqlite3
import sys

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from assistant.core.redact import redact_secrets  # noqa: E402

# Roughly the intent system prompt plus a short utterance. Used only to print
# an estimate before spending anything.
_TOKENS_PER_CALL = 2500
_USD_PER_1M_IN = 0.10          # Gemini 2.5 Flash-Lite, paid tier


def _load(db: pathlib.Path, only_regex: bool, limit: int | None):
    """Distinct utterances the regex claimed, newest first."""
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    where = "intent_source = 'regex'" if only_regex else "1=1"
    rows = conn.execute(f"""
        SELECT transcript, intent_detected, intent_source
        FROM interaction_events
        WHERE transcript IS NOT NULL AND TRIM(transcript) <> ''
          AND intent_detected IS NOT NULL AND {where}
        ORDER BY id DESC
    """).fetchall()
    conn.close()

    seen, out = set(), []
    for r in rows:
        t = (r["transcript"] or "").strip()
        if not t or t.lower() in seen:
            continue
        seen.add(t.lower())
        out.append((t, r["intent_detected"], r["intent_source"]))
        if limit and len(out) >= limit:
            break
    return out


# main.py overrides ANY resolved intent to `planner` when the utterance is
# multi-step (main.py:1981-1987), whatever produced it. Applying the same
# override to both sides is what stops "open wikipedia and find X" reading as
# a disagreement: the regex says computer_task, the classifier says planner,
# and the pipeline sends both to the planner anyway. Without this the harness
# reports convergent paths as divergent -- two of the seven disagreements in
# its first run were exactly that.
_PLANNER_EXEMPT = ("manage_shortcut", "manage_procedure",
                   "manage_schedule", "manage_monitor")

# (regex intent, classifier intent) pairs where the REGEX is the correct one.
# Keyed by intent pair, never by utterance, so nothing from real history is
# recorded here. Each entry is a judgement someone made once; the reason is
# the whole value of the entry.
#
# Without this the harness reads as "7 disagreements, tighten 7 patterns",
# and acting on the first two would have broken the shutdown fast path.
KNOWN_GOOD = {
    ("shutdown", "computer_task"):
        "bare shutdown phrases ('exit', 'shut down') read as generic desktop "
        "control to the classifier. The regex is right, and this is exactly "
        "the sort of short, unambiguous command the fast path exists for.",
}


def _as_pipeline_would(intent_name: str, text: str) -> str:
    """The intent the pipeline actually dispatches, not the one resolved."""
    from assistant.actions.planner import planner as planner_mod
    if intent_name in _PLANNER_EXEMPT:
        return intent_name
    return "planner" if planner_mod.needs_planning(text) else intent_name


async def _classify(text: str) -> str:
    from assistant import intent as intent_mod
    try:
        return (await intent_mod.detect_intent(text)).intent
    except Exception as e:  # noqa: BLE001 -- a provider failure is a skip, not a verdict
        return f"<error: {type(e).__name__}>"


async def _run(cases) -> int:
    from assistant.regex_router import pre_route

    agree, differ, skipped, known = 0, [], 0, 0
    for i, (text, _observed, _src) in enumerate(cases, 1):
        claimed = pre_route(text)
        if claimed is None:
            # No longer claimed -- the pattern was tightened since. Not a
            # differential case, but worth counting: it is the fix landing.
            skipped += 1
            continue

        llm = await _classify(text)
        print(f"  [{i}/{len(cases)}] {redact_secrets(text)[:64]}")
        if llm.startswith("<error"):
            print(f"        classifier unavailable: {llm}")
            skipped += 1
            continue
        regex_final = _as_pipeline_would(claimed.intent, text)
        llm_final = _as_pipeline_would(llm, text)
        if llm_final == regex_final:
            agree += 1
        elif (regex_final, llm_final) in KNOWN_GOOD:
            known += 1
            print(f"        regex={regex_final}  classifier={llm_final}   "
                  f"(known good, regex wins)")
        else:
            differ.append((text, regex_final, llm_final))
            print(f"        regex={regex_final}  classifier={llm_final}   <-- DIFFERS")

    print()
    print(f"agree {agree}   known-good {known}   differ {len(differ)}   skipped {skipped}")

    if differ:
        print("\nDisagreements -- each is an over-claim or a deliberate exception:")
        for text, r, l in differ:
            print(f"  {redact_secrets(text)}")
            print(f"      regex says {r!r}, classifier says {l!r}")
        print("\nThe classifier is NOT an oracle. Triage; do not act blindly.")
        print("Measured 2026-08-22: it called 'shut down' and 'exit' computer_task,")
        print("where the regex was right. A disagreement means one of the two is")
        print("wrong -- reading it as 'the regex is wrong' would have broken the")
        print("shutdown fast path.")
        print("\nFor any that is a real over-claim: tighten the pattern, and add a")
        print("hand-written case with inert content to")
        print("tests/test_routing_overclaim.py. Do not paste real history into the repo.")
    else:
        print("\nNo disagreements. The regex path answered every claimed turn the")
        print("way the classifier would have.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", type=pathlib.Path,
                    default=pathlib.Path.home() / "TENKA" / "memory" / "tenka.db")
    ap.add_argument("--run", action="store_true",
                    help="actually call the classifier. Without this, only estimates.")
    ap.add_argument("--all", action="store_true", help="every recorded turn, not a sample")
    ap.add_argument("--sample", type=int, default=20)
    ap.add_argument("--include-llm-routed", action="store_true",
                    help="also replay turns the regex declined. Circular for the "
                         "regex-routed set, but these show a pattern that has WIDENED.")
    args = ap.parse_args()

    if not args.db.exists():
        print(f"no database at {args.db}")
        return 1

    limit = None if args.all else args.sample
    cases = _load(args.db, only_regex=not args.include_llm_routed, limit=limit)
    if not cases:
        print("nothing to replay -- no regex-routed turns recorded yet.")
        return 0

    from assistant.regex_router import pre_route
    still_claimed = [c for c in cases if pre_route(c[0]) is not None]

    tokens = len(still_claimed) * _TOKENS_PER_CALL
    print(f"{len(cases)} distinct utterance(s); {len(still_claimed)} still claimed "
          f"by the regex path")
    print(f"estimate: {tokens:,} input tokens "
          f"~ ${tokens / 1_000_000 * _USD_PER_1M_IN:.4f} on Flash-Lite paid, "
          f"free on the free tier")

    if not args.run:
        print("\nDry run. Re-run with --run to call the classifier.")
        return 0
    if not still_claimed:
        print("\nNothing left to compare.")
        return 0

    if not (os.getenv("GEMINI_API_KEY") or os.getenv("GROQ_API_KEY")
            or os.getenv("CEREBRAS_API_KEY")):
        print("\nno provider key in the environment -- nothing to ask.")
        return 1

    print()
    return asyncio.run(_run(still_claimed))


if __name__ == "__main__":
    raise SystemExit(main())
