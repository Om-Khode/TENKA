"""A reply may not claim an effect the turn never produced.

Live test, 2026-08-25. Two false claims in one exchange, both composed by a
model narrating actions nothing had taken:

    21:37:02  web_search   "I've made a note for you that says
                            'groceries: milk and' -- did you want to add the
                            weather to that note, or create a new one?"
    21:37:22  'create a new one'
    21:37:24  Intent: unknown        <- no handler dispatched, at all
    21:37:34  Response: "Okay, I've created a new note called 'Pune Weather'
                         with those details."

`C:\\Users\\omkho\\TENKA\\Notes\\` held one file, `untitled.txt`. No note was
made in either turn. She read an existing note's contents, said she had written
it, and then said she had written a second one.

**Why the existing guard did not catch it.** `_SECURITY_SKIP_FALLBACK` covers
the turn a security control *skipped*: that branch never reaches the LLM at
all, precisely so nothing can compose a claim about what just happened. This is
the other half. `small_talk` and `unknown` reach the LLM by design and dispatch
no handler by design, so every completed-action claim they produce is false by
construction -- not sometimes, not probably. There is no tool behind that
branch for the claim to be true about.

**The shape of the fix, and why it is not a prompt change.** `CLAUDE.md` says
fix at code level unless the model misunderstands the task. The model does not
misunderstand anything here; it is doing what a fluent conversational model
does, and asking it more nicely is a probability, not a property. This is
deterministic: the same sentence is removed every time, and the rest of the
reply survives.

**Deliberately narrow, in three ways**, because the cost of a false positive is
mangling an ordinary sentence:

1. Only the **perfect/past first person**. "I've created", "I made", "I saved".
   Not "I can create", not "I'll create", not "creating" -- an offer or a plan
   is not a claim.
2. Only verbs that name an **effect TENKA actually has**, paired with an object
   that names an artifact. "I made a note" is a claim; "I made a mistake" and
   "I've made up my mind" are not, and the object list is what separates them.
3. Only where **no handler ran**. A `create_note` turn that really did write a
   file should say so, and this filter never sees that turn.

Generic by construction: no application, service or brand appears here. The
verbs are TENKA's own capability vocabulary and the objects are the artifacts
her handlers produce.
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger("claims")

# Sentence splitter, shared shape with `core/identity.py`: keep the delimiter
# so a rebuilt reply reads the way it was written.
_SENTENCE = re.compile(r"[^.!?\n]*[.!?\n]|[^.!?\n]+$")

# Verbs that name an effect on the world outside the conversation. Deliberately
# excludes speech-act verbs ("told", "said", "explained") -- she really did do
# those, in the very sentence making the claim.
_EFFECT_VERBS = (
    "created", "made", "saved", "written", "wrote", "added", "put",
    "deleted", "removed", "cleared", "cancelled", "canceled",
    "opened", "closed", "launched", "started", "stopped", "paused", "resumed",
    "sent", "posted", "shared", "emailed", "messaged",
    "scheduled", "set up", "set", "installed", "downloaded", "uploaded",
    "updated", "changed", "renamed", "moved", "copied",
    "played", "skipped", "searched",
)

# Artifacts her handlers produce. The object is what separates "I made a note"
# from "I made a mistake", and it is why this list is not optional.
_ARTIFACTS = (
    "note", "notes", "file", "files", "folder", "directory", "document",
    "reminder", "reminders", "alarm", "schedule", "schedules", "monitor",
    "monitors", "shortcut", "shortcuts", "procedure", "procedures",
    "backup", "backups", "task", "tasks", "event", "appointment",
    "message", "messages", "email", "emails", "post",
    "tab", "window", "app", "application", "program", "browser",
    "entry", "record", "list", "memory", "fact",
)

_CLAIM = re.compile(
    r"\b(?:i|i'?ve|i\s+have)\s+"
    r"(?:just\s+|already\s+|now\s+|also\s+|gone\s+ahead\s+and\s+)*"
    r"(?:" + "|".join(_EFFECT_VERBS) + r")\b"
    r"[^.!?\n]{0,60}?"
    r"\b(?:" + "|".join(_ARTIFACTS) + r")\b",
    re.IGNORECASE,
)

# There is deliberately no "but it was phrased as an offer" escape hatch.
#
# One existed, and a mutation proved it did nothing: removing it changed no
# test result, because an offer cannot satisfy the triple above -- it has no
# perfect-tense verb. What it *could* do was worse than nothing. It matched on
# the whole sentence, so a reply that made a false claim and then asked a
# question got a free pass, and that is exactly the shape of the sentence that
# shipped:
#
#     I've made a note for you that says "..." - did you want to add the
#     weather to that note, or create a new one?
#
# That one escaped only because the hatch spelled it "do you want" and the
# model wrote "did you want". Tense, verb and object are what do the work here;
# a second predicate that can only ever *unblock* is a hole waiting for
# a phrasing.

_FALLBACK = (
    "I haven't actually done that yet -- ask me directly and I will."
)


def claims_an_effect(text: str) -> bool:
    """Does `text` assert, in the perfect or past tense, that TENKA changed
    something outside the conversation?"""
    if not text:
        return False
    return any(_CLAIM.search(s) for s in _SENTENCE.findall(text))


def strip_effect_claims(text: str) -> str:
    """Remove sentences claiming an effect, and keep everything else.

    Called only for a turn that dispatched no handler, so there is no effect
    for such a sentence to be true about. Returns the text unchanged when there
    is nothing to remove, which is the overwhelmingly common case.
    """
    if not text or not text.strip():
        return text

    kept: list[str] = []
    dropped: list[str] = []
    for match in _SENTENCE.finditer(text):
        sentence = match.group(0)
        if not sentence.strip():
            continue
        if _CLAIM.search(sentence):
            dropped.append(sentence.strip())
        else:
            kept.append(sentence)

    if not dropped:
        return text

    logger.info(f"[CLAIMS] Dropped unbacked effect claim: {dropped}")

    rebuilt = "".join(kept).strip()
    if not rebuilt:
        # The whole reply was the false claim. Say the true thing rather than
        # nothing, and rather than the original.
        logger.info("[CLAIMS] Whole reply was an unbacked claim — substituting")
        return _FALLBACK
    return rebuilt
