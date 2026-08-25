"""What came of a step: the outcome vocabulary, and nothing else.

**In `core/` because the layers say so.** These four types are read by
`automation/` (five modules, for `Outcome` alone) and by `storage/repos/task.py`,
both of which sit *below* `brain/` in the documented order
`core -> config -> storage/llm -> domain -> automation -> actions -> brain ->
main`. They lived in `brain/task.py` and put five wrong-way imports into the
tree, which nothing caught because there was no contract to catch them -- P4b's
measurement is what found them, and they arrived in P6.

The split is between **vocabulary** and **coordination**. What a step concluded
is a word every layer needs to say; what a Task *is*, who owns it and whether it
may resume are decisions, and those stay in `brain/`. `brain/task.py` re-exports
these names, so nothing that already imports them from there breaks.

Deliberately dependency-free: stdlib only, no `Capability`, no config, no
storage. A vocabulary that drags a dependency behind it is not one.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass


# ─── What came of a step ─────────────────────────────────────────────────────

class Outcome(str, enum.Enum):
    """A step's verdict. Replaces the boolean that made three states look like
    one (`VerifyResult.ok` is True for success, ambiguity and skip alike).

    The distinction between `UNCERTAIN` and `UNVERIFIED` is the whole reason
    this is an enum: one is a failure of knowledge, the other is a choice not
    to acquire it, and only the first is a problem.
    """

    SUCCEEDED = "succeeded"      # positive evidence the effect happened
    FAILED = "failed"            # positive evidence it did not
    UNCERTAIN = "uncertain"      # verification ran and could not decide
    UNVERIFIED = "unverified"    # verification did not run: policy, or nothing to verify
    UNSUPPORTED = "unsupported"  # no route exists; never attempted at all

    @property
    def is_evidence_of_success(self) -> bool:
        """Only `SUCCEEDED` is. Absence of an exception is not evidence.

        A property rather than a bare comparison so there is one place that
        answers it — the bug this type replaces was six call sites each
        deciding for themselves what `ok` meant.
        """
        return self is Outcome.SUCCEEDED


class ObservationKind(str, enum.Enum):
    STATE_CHANGED = "state_changed"
    EXPECTED_PRESENT = "expected_present"
    EXPECTED_ABSENT = "expected_absent"
    NOTHING_CHANGED = "nothing_changed"
    ERROR = "error"
    # P5. Every member above reports something that was *seen*, and
    # `NOTHING_CHANGED` is a claim as strong as any of them -- someone looked,
    # and the state was the same. There was no way to say the weaker and much
    # more common thing: nobody looked, so there is nothing to report. An
    # `UNVERIFIED` outcome needs exactly that, and forcing it into
    # `NOTHING_CHANGED` would turn "no evidence" into "evidence of no effect",
    # which is the inversion §6 exists to prevent.
    NOT_OBSERVED = "not_observed"


@dataclass(frozen=True)
class Observation:
    """What was seen, by what, when, and how sure.

    Freshness and provenance are fields rather than hopes: an observation of
    the desktop is stale the moment it is taken, and a caller that cannot tell
    a code-tier check from a vision guess cannot weigh them differently. The
    thing this replaces is a bare string.
    """

    kind: ObservationKind
    detail: str = ""
    source: str = "code"          # code | dom | uia | vision | process | llm
    confidence: float = 1.0
    at: str = ""                  # ISO 8601; set by the observer, never inferred


@dataclass(frozen=True)
class Verdict:
    """An outcome plus the evidence for it. **There is no `ok` field.**

    `escalated` records that a cheaper tier was inconclusive and a dearer one
    ran, which is what makes the cost of a verification legible after the fact.
    """

    outcome: Outcome
    observation: Observation
    tier: str = "code"            # pre | code | vision | skipped
    escalated: bool = False
