"""Run one step. Never decide what the step meant.

P5. Today a plan step's *meaning* is a string: `actions/planner/executor.py`
builds `params = {param_key: resolved_goal}` from a manifest-chosen key, hands
that to `actions.execute()`, and every adapter downstream re-reads the sentence
to work out what was wanted. `automation/router.py` does it with regexes --
`\\b(open|launch|start|run)\\s+(\\w+)`, `\\bsearch\\s+(for|on|the)`, and a dozen
more -- so the user's phrasing is parsed at least twice, by two components with
no shared vocabulary, and the second one can disagree with the first.

That is the whole defect class this phase is aimed at. A step that says
`operation="open", parameters={"target": "notepad"}` cannot be re-read into
something else; a step that says `"open notepad"` can, and has been.

**What this does not do**, deliberately. `goal` survives on `TaskStep` as a
payload field, and the adapters keep reading it. Deleting it would rewrite
roughly forty files and six loops at once, which is the big-bang §23 forbids
and which P5's own text rules out. What changes here is that there is now one
place where a step's structured fields are what decide, `constraints` are
carried where they cannot be quietly widened, and the string is data an adapter
may read rather than the thing that chose the adapter.

**The three properties worth stating out loud**, because each is a thing this
tree has done wrong before:

1. *An executor never classifies.* If a step arrives ambiguous, the answer is a
   worse plan, not a second opinion invented at execution time. A classifier
   call here would mean the authority-checked intent and the executed intent
   could differ, which is the shape of every confused-deputy bug in the file.
2. *An executor never reads the utterance to decide.* It may pass `goal`
   through to an adapter that still needs one. Branching on it is the
   re-interpretation being removed.
3. *`UNSUPPORTED` is not `FAILED`.* "No route exists, nothing was attempted" and
   "it was attempted and did not work" want different responses from the
   planner, from recovery, and from what TENKA says out loud. Collapsing them
   is how a missing adapter becomes an apology for a failure that never
   happened.

Dispatch still goes through `actions.execute()` -- the single
handler-resolution site and the EXECUTE enforcement point -- so nothing here
widens what a turn may do. The Executor decides *how to describe* what
happened, never *whether it may happen*.
"""
from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, Optional

from ..core.verdict import Observation, ObservationKind, Outcome, Verdict

logger = logging.getLogger("brain.executor")

Dispatch = Callable[..., Awaitable[Any]]


def _default_dispatch() -> Dispatch:
    """`actions.execute`, resolved late.

    Deferred for the reason every `actions` import in this package is deferred,
    and for one more: injecting it is what lets the tests below assert that no
    classifier is reachable from here without stubbing half the tree.
    """
    from ..actions import execute
    return execute


class Executor:
    """Runs a `TaskStep` and reports a `Verdict`.

    `dispatch` is injected rather than imported at call time so a caller can
    supply a narrower one, and so the no-classifier property can be tested
    against a real run rather than only against the source.
    """

    def __init__(self, dispatch: Optional[Dispatch] = None) -> None:
        self._dispatch = dispatch

    # ── parameters ──

    @staticmethod
    def build_params(step, constraints: Optional[dict] = None,
                     *, goal_key: str = "goal") -> dict:
        """The step's structured fields, with hard constraints laid over them.

        Delegates to `core/step_params.py`. The merge rule is shared with the
        planner's own step runner, which cannot import this module: `brain`
        sits above `actions`, there is no `actions -> brain` import in the
        tree, and adding one would be the fifth layer inversion. Putting the
        pure part below both is what lets the rule about pinned values have one
        implementation instead of one per layer.
        """
        from ..core.step_params import build_step_params

        return build_step_params(
            getattr(step, "parameters", None),
            getattr(step, "constraints", None),
            getattr(step, "goal", "") or "",
            extra_constraints=constraints,
            goal_key=goal_key,
        )

    # ── running ──

    async def run(
        self,
        step,
        *,
        constraints: Optional[dict] = None,
        bridge: Any = None,
        llm_response: str = "",
    ) -> Verdict:
        """Execute `step` and describe the result.

        Returns a `Verdict` and raises nothing it can describe instead --
        except an abort, which is the operator's decision and belongs to
        whoever is running the loop.
        """
        intent = getattr(step, "intent", "") or ""

        if not self.supports(intent):
            # Never attempted. The distinction from FAILED is the point: a
            # planner that reads "no route exists" can pick another tool, while
            # one that reads "it failed" retries the same one.
            return _verdict(
                Outcome.UNSUPPORTED,
                ObservationKind.EXPECTED_ABSENT,
                f"no handler is registered for intent {intent!r}",
            )

        params = self.build_params(step, constraints)
        dispatch = self._dispatch or _default_dispatch()

        try:
            result = await dispatch(
                intent=intent,
                params=params,
                llm_response=llm_response,
                bridge=bridge,
                _from_planner=True,
            )
        except Exception as e:
            from ..core.abort import UserAborted
            if isinstance(e, UserAborted):
                # The operator pressed ESC. Not an outcome to report -- a
                # decision to obey, and swallowing it here is what made a
                # second ESC necessary once already.
                raise
            logger.error("[BRAIN] step %s raised: %s",
                         getattr(step, "step_id", "?"), e)
            return _verdict(Outcome.FAILED, ObservationKind.ERROR, str(e))

        return self.describe(result)

    # ── classification of the result ──

    @staticmethod
    def supports(intent: str) -> bool:
        """Is there a handler for this intent at all?

        Asked of the registry, not by attempting the call and reading the
        wreckage. `tool_registry.has` does not resolve or run anything, so this
        cannot become a way around the gate in `actions.execute()`.
        """
        if not intent:
            return False
        from ..actions.registry import tool_registry
        return tool_registry.has(intent)

    @staticmethod
    def describe(result: Any) -> Verdict:
        """Turn a handler's return value into a `Verdict`.

        Three outcomes, and the middle one is the one that matters. A handler
        returning nothing has not told us it worked -- `UNVERIFIED` says
        exactly that, where `SUCCEEDED` would be a claim nobody made. §6's
        whole argument is that silence is not evidence.
        """
        if result is None:
            return _verdict(
                Outcome.UNVERIFIED, ObservationKind.NOT_OBSERVED,
                "the handler returned nothing, so nothing confirms the effect",
            )

        text = str(result)
        if not text.strip():
            return _verdict(
                Outcome.UNVERIFIED, ObservationKind.NOT_OBSERVED,
                "the handler returned an empty string",
            )

        if _reads_as_failure(text):
            return _verdict(Outcome.FAILED, ObservationKind.ERROR, text[:300])

        return _verdict(Outcome.SUCCEEDED,
                        ObservationKind.STATE_CHANGED, text[:300])


def _reads_as_failure(text: str) -> bool:
    """Delegates to the planner's existing detector.

    Deliberately not a second opinion. `_step_failed` is what decides today,
    it has been tuned against real handler output, and a parallel copy here
    would be one more thing to keep in sync -- which is how the pre-dispatch
    hole opened. If its judgement is wrong, it is wrong in one place.
    """
    try:
        from ..actions.planner.planner import _step_failed
    except Exception:  # pragma: no cover - planner import is not optional
        return False
    return bool(_step_failed(text))


def _verdict(outcome: Outcome, kind: ObservationKind, detail: str) -> Verdict:
    return Verdict(
        outcome=outcome,
        observation=Observation(kind=kind, detail=detail, source="code"),
        tier="code",
    )
