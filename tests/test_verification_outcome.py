"""Uncertainty is not success, and not looking is not uncertainty.

`VerifyResult.ok` was `True` for three different things — confident success,
`ambiguous()` (the code tier could not decide) and `skip()` (nothing was
checked). Six call sites each decided for themselves what that meant, and one
decided wrong: `recovery.py` reported a recovery as succeeded when nothing had
confirmed it (KI-31).

The field is gone rather than kept as a derived property. A reader that has not
been updated should raise `AttributeError` loudly, not quietly receive a
boolean whose meaning changed underneath it.

`UNVERIFIED` is the member that earns its place, and the reason is a rule that
would otherwise be unusable: fold it into `UNCERTAIN` and `VERIFY_ENABLED=False`
— a setting that exists so the operator can skip verification — makes every task
uncertain, so TENKA answers "I couldn't confirm that" to everything. That rule
gets reverted, and the honesty property goes with it.

Both directions throughout. A type that reported nothing as confirmed would
satisfy every "does not claim success" assertion while halting every step loop.

Run with:  py -3.11 -m pytest tests/test_verification_outcome.py -v
"""
import pathlib
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from assistant.automation.verification import VerifyResult  # noqa: E402
from assistant.brain.task import Outcome  # noqa: E402


def _stub_screen_and_llm(monkeypatch, screen, llm):
    """Stub the screen and the model **on the package attribute**, not in
    `sys.modules`.

    `verification.py` does `from ..io import screen`, which reads the attribute
    on the already-imported `assistant.io` package. Patching `sys.modules` does
    nothing once that package exists, so the first version of these tests
    called the real `mss.mss()` and took an actual screenshot. Harmless here --
    it reads the screen, it does not drive it -- but a test of mine reaching
    real hardware at all is the thing to design out, not to notice.

    The same trap bit `assistant.preferences` earlier: the fix is always to
    patch where the importing module will look.
    """
    import assistant.io as io_pkg
    monkeypatch.setattr(io_pkg, "screen", screen, raising=False)
    if llm is not None:
        import assistant as pkg
        monkeypatch.setattr(pkg, "llm", llm, raising=False)


# ─── V1 — SUCCEEDED requires positive evidence ───────────────────────────────

def test_only_success_is_evidence_of_success():
    assert Outcome.SUCCEEDED.is_evidence_of_success
    for o in Outcome:
        if o is Outcome.SUCCEEDED:
            continue
        assert not o.is_evidence_of_success, (
            f"{o.value} reads as evidence of success. Absence of an exception "
            f"is not evidence."
        )


def test_there_is_no_ok_attribute():
    """Removed, not derived. A missed reader must raise, not silently receive a
    boolean whose meaning changed."""
    r = VerifyResult.ambiguous("cannot tell")
    with pytest.raises(AttributeError):
        _ = r.ok


# ─── V2 / V6 — the constructors mean what they say ───────────────────────────

@pytest.mark.parametrize("factory,expected,confirmed", [
    (lambda: VerifyResult.ok_(), Outcome.SUCCEEDED, True),
    (lambda: VerifyResult.fail("gone"), Outcome.FAILED, False),
    (lambda: VerifyResult.ambiguous("cannot tell"), Outcome.UNCERTAIN, False),
    (lambda: VerifyResult.skip("verify disabled"), Outcome.UNVERIFIED, True),
    (lambda: VerifyResult.crashed("locator timeout"), Outcome.UNCERTAIN, False),
])
def test_each_constructor_maps_to_one_outcome(factory, expected, confirmed):
    r = factory()
    assert r.outcome is expected
    assert r.confirmed is confirmed, (
        f"{expected.value} reads as confirmed={r.confirmed}. `UNVERIFIED` "
        f"counts (the operator opted out, or there is nothing to check); "
        f"`UNCERTAIN` does not (something was checked and could not be "
        f"established)."
    )


def test_a_crash_is_uncertain_not_unverified():
    """The distinction the old code lost by using `skip()` for both. An
    exception means something was attempted and the answer is unknown -- a fact
    about the world. Choosing not to look is a fact about the configuration."""
    assert VerifyResult.crashed("boom").outcome is Outcome.UNCERTAIN
    assert VerifyResult.skip("disabled").outcome is Outcome.UNVERIFIED


def test_skipped_still_means_unverified():
    """`skipped` is kept as a property because, unlike `ok`, its meaning did
    not change -- several loops read it to mean "do not treat this as a
    failure"."""
    assert VerifyResult.skip("x").skipped is True
    for r in (VerifyResult.ok_(), VerifyResult.fail("x"),
              VerifyResult.ambiguous("x"), VerifyResult.crashed("x")):
        assert r.skipped is False, f"{r.outcome.value} reports itself skipped"


# ─── V7 — a missing verdict field is not a verdict ───────────────────────────

@pytest.mark.asyncio
async def test_a_vision_reply_without_an_ok_field_is_uncertain(monkeypatch):
    """`data.get("ok", True)` defaulted a MISSING field to success, so a model
    answering in an unexpected shape was read as confirmation."""
    from assistant.automation import verification as v

    monkeypatch.setattr(v.config, "VERIFY_VISION_FALLBACK", True, raising=False)
    monkeypatch.setattr(v, "_capture_for_test", None, raising=False)

    class _Screen:
        @staticmethod
        def capture_screenshot_base64():
            return "Zm9v"

    class _Resp:
        text = '{"observation": "a dialog is open", "confidence": 0.8}'

    class _LLM:
        @staticmethod
        async def get_vision_response(**_):
            return _Resp()

    _stub_screen_and_llm(monkeypatch, _Screen, _LLM)

    out = await v.vision_verify(
        {"type": "app", "action": "click", "params": {}},
        VerifyResult.ambiguous("code could not tell"),
    )
    assert out.outcome is Outcome.UNCERTAIN, (
        f"a vision reply with no verdict field produced {out.outcome.value}. "
        f"The absence of a verdict is the absence of one."
    )


@pytest.mark.asyncio
async def test_a_vision_reply_with_a_verdict_is_honoured(monkeypatch):
    """The other direction: the escalation has to be able to resolve an
    ambiguous code verdict, or paying for the vision call buys nothing."""
    from assistant.automation import verification as v

    monkeypatch.setattr(v.config, "VERIFY_VISION_FALLBACK", True, raising=False)

    class _Screen:
        @staticmethod
        def capture_screenshot_base64():
            return "Zm9v"

    class _Resp:
        text = '{"ok": true, "observation": "dialog closed", "confidence": 0.9}'

    class _LLM:
        @staticmethod
        async def get_vision_response(**_):
            return _Resp()

    _stub_screen_and_llm(monkeypatch, _Screen, _LLM)

    out = await v.vision_verify(
        {"type": "app", "action": "click", "params": {}},
        VerifyResult.ambiguous("code could not tell"),
    )
    assert out.outcome is Outcome.SUCCEEDED
    assert out.tier == "vision"


# ─── V5 — a degraded vision tier cannot upgrade uncertainty ──────────────────

@pytest.mark.asyncio
async def test_vision_switched_off_leaves_the_code_verdict_alone(monkeypatch):
    """Turning off a *tier* cannot make an inconclusive code check conclusive.
    `VERIFY_VISION_FALLBACK` is reachable over `PATCH /v1/settings`, so this is
    a settings flip away."""
    from assistant.automation import verification as v

    monkeypatch.setattr(v.config, "VERIFY_VISION_FALLBACK", False, raising=False)
    code = VerifyResult.ambiguous("cannot tell")
    out = await v.vision_verify({"type": "app", "action": "click", "params": {}}, code)
    assert out.outcome is Outcome.UNCERTAIN


@pytest.mark.asyncio
async def test_a_vision_failure_leaves_the_code_verdict_alone(monkeypatch):
    """Fail-open, which is safe now in a way it was not before: what is handed
    back is whatever the code tier concluded, and an ambiguous code verdict is
    UNCERTAIN. It used to carry `ok=True`, so every degraded path -- no
    screenshot, LLM down, exhausted free tier, bad JSON -- returned something a
    caller read as confirmation. On a free tier that is the normal path."""
    from assistant.automation import verification as v

    monkeypatch.setattr(v.config, "VERIFY_VISION_FALLBACK", True, raising=False)

    class _Screen:
        @staticmethod
        def capture_screenshot_base64():
            return None          # capture failed

    _stub_screen_and_llm(monkeypatch, _Screen, None)

    out = await v.vision_verify(
        {"type": "app", "action": "click", "params": {}},
        VerifyResult.ambiguous("cannot tell"),
    )
    assert out.outcome is Outcome.UNCERTAIN, (
        "a degraded vision tier upgraded an ambiguous verdict"
    )


# ─── the readers were all updated ────────────────────────────────────────────

def test_no_module_still_reads_ok_off_a_verify_result():
    """Source-level sweep. `.ok` is gone, so a stale reader would raise at
    runtime -- but only on the path that reaches it, which for a verification
    branch can be rare. This finds it without executing anything.

    The three other result types in `automation/` (`PrimitiveResult`,
    `HealResult`, `DispatchResult`) keep their booleans and are excluded by
    name: narrowing the sweep to the files that actually handle a
    `VerifyResult` is what keeps it honest rather than noisy.
    """
    import re

    handlers = [
        "automation/verification.py", "automation/recovery.py",
        "automation/native.py", "automation/router.py",
        "automation/browser/automation.py",
    ]
    offenders = []
    for rel in handlers:
        for i, line in enumerate(
            (_ROOT / "assistant" / rel).read_text(encoding="utf-8").splitlines(), 1
        ):
            code = line.split("#", 1)[0]
            if re.search(r"\b(?:vr|pre|post|result|code_result)\.ok\b", code):
                offenders.append(f"{rel}:{i} {line.strip()}")
    assert not offenders, (
        f"a VerifyResult `.ok` read survives: {offenders}. There is no such "
        f"attribute -- this would raise on whichever branch reaches it."
    )


def test_the_other_result_types_keep_their_booleans():
    """Scope, pinned. P6 changed one type. An earlier count of twenty `.ok`
    readers was wrong -- seven were `VerifyResult` and the rest belong to these
    three, so a brief carrying that number would have sent someone to rewrite
    four unrelated types."""
    from assistant.automation.healer import HealResult
    from assistant.automation.manifest_dispatcher import DispatchResult

    assert HealResult(ok=True, tier=1).ok is True
    assert DispatchResult(ok=False).ok is False


# ═════════════════════════════════════════════════════════════════════════
# V4 — a task is only as confirmed as its least confirmed step
#
# The rule existed in §11.2 and in no code. Every `Outcome` in the tree was a
# *step's*, and nothing combined them, so what a whole task reported depended
# on whoever summarised it. `roll_up` is that one place.
# ═════════════════════════════════════════════════════════════════════════

from assistant.core.verdict import roll_up, speaks_as_done  # noqa: E402


def _v(outcome):
    from assistant.core.verdict import Observation, ObservationKind, Verdict
    return Verdict(outcome=outcome,
                   observation=Observation(kind=ObservationKind.STATE_CHANGED))


def _task(*outcomes):
    from assistant.brain.task import Task, TaskStep
    return Task(
        task_id="t", intent="planner", principal="local", granted=frozenset(),
        steps=tuple(
            TaskStep(step_id=i, intent="x",
                     verdict=None if o is None else _v(o))
            for i, o in enumerate(outcomes, 1)
        ),
    )


class TestV4TaskRollUp:

    def test_one_uncertain_step_makes_the_task_uncertain(self):
        """**The headline.** A task that reports success while one of its steps
        could not be confirmed is the behaviour §11 exists to remove."""
        assert roll_up([Outcome.SUCCEEDED, Outcome.UNCERTAIN,
                        Outcome.SUCCEEDED]) is Outcome.UNCERTAIN
        assert _task(Outcome.SUCCEEDED, Outcome.UNCERTAIN).outcome() \
            is Outcome.UNCERTAIN

    def test_unverified_steps_do_not_lower_a_task(self):
        """V6, and the reason `UNVERIFIED` is a member at all. Treating the
        operator's own choice not to verify as doubt would make
        `VERIFY_ENABLED=False` apologise about everything, forever."""
        assert roll_up([Outcome.SUCCEEDED,
                        Outcome.UNVERIFIED]) is Outcome.SUCCEEDED
        assert _task(Outcome.SUCCEEDED, Outcome.UNVERIFIED).outcome() \
            is Outcome.SUCCEEDED

    def test_all_unverified_is_still_not_uncertain(self):
        assert roll_up([Outcome.UNVERIFIED,
                        Outcome.UNVERIFIED]) is Outcome.SUCCEEDED

    def test_a_failed_step_outranks_an_uncertain_one(self):
        """Positive evidence against beats no evidence either way. There is
        nothing to hedge about a step that demonstrably did not work."""
        assert roll_up([Outcome.UNCERTAIN,
                        Outcome.FAILED]) is Outcome.FAILED

    def test_an_unsupported_step_is_not_reported_as_failure(self):
        """The distinction P5 built: no route existed, so nothing was
        attempted. A planner reading FAILED retries; one reading UNSUPPORTED
        picks another tool."""
        assert roll_up([Outcome.SUCCEEDED,
                        Outcome.UNSUPPORTED]) is Outcome.UNSUPPORTED

    def test_every_step_succeeding_succeeds(self):
        """The control. A rollup that never returns SUCCEEDED satisfies every
        test above and makes her incapable of reporting a finished job."""
        assert roll_up([Outcome.SUCCEEDED] * 3) is Outcome.SUCCEEDED
        assert _task(Outcome.SUCCEEDED, Outcome.SUCCEEDED).outcome() \
            is Outcome.SUCCEEDED

    def test_a_task_with_no_steps_is_unverified_not_succeeded(self):
        """Doing nothing is not success. Reporting it as such is the same false
        claim in a smaller package."""
        assert roll_up([]) is Outcome.UNVERIFIED
        assert _task().outcome() is Outcome.UNVERIFIED

    def test_a_step_that_has_not_run_does_not_count_as_success(self):
        """A half-finished task must not report a finished one. `verdict=None`
        means the step has no evidence yet, which is `UNVERIFIED` -- honest,
        and it keeps `roll_up` from promoting."""
        # **From a green mutant.** The two cases below hold whether an unrun
        # step reads as UNVERIFIED or as SUCCEEDED, because another step
        # decides the answer in both. The distinguishing case is a task where
        # the unrun step is the only evidence there is.
        assert _task(None).outcome() is Outcome.UNCERTAIN, (
            "a task whose only step never ran reported success")
        assert _task(None, None).outcome() is Outcome.UNCERTAIN

        # **This assertion found a bug in the first draft of `outcome()`.** It
        # mapped an unrun step to UNVERIFIED, so a task with one finished step
        # and one that had not started reported SUCCEEDED -- done, while half
        # of it had not begun. `UNVERIFIED` is "nobody looked, and that was the
        # plan"; an unrun step is not a decision about verification.
        assert _task(Outcome.SUCCEEDED, None).outcome() is Outcome.UNCERTAIN, (
            "a half-finished task reported success")
        assert _task(Outcome.UNCERTAIN, None).outcome() is Outcome.UNCERTAIN
        assert _task(Outcome.FAILED, None).outcome() is Outcome.FAILED

    def test_the_task_outcome_is_computed_not_stored(self):
        """A stored answer goes stale against the steps it came from, and the
        one thing a task's outcome must never be is out of date with what
        happened."""
        from assistant.brain.task import Task
        assert callable(Task.outcome)
        assert "outcome" not in {f.name for f in
                                 __import__("dataclasses").fields(Task)}


# ═════════════════════════════════════════════════════════════════════════
# V8 — "done" and "I couldn't confirm that" are different sentences
# ═════════════════════════════════════════════════════════════════════════

class TestV8SpokenDistinction:

    def test_uncertain_never_speaks_as_done(self):
        """The whole point of the phase, at the last layer. Everything else is
        bookkeeping if this sentence comes out wrong."""
        assert not speaks_as_done(Outcome.UNCERTAIN)

    def test_unverified_speaks_plainly(self):
        """V6 again at the response layer: the operator switched verification
        off, and being told "I couldn't confirm that" about every single turn
        is how the setting gets switched back on and the honesty lost."""
        assert speaks_as_done(Outcome.UNVERIFIED)

    def test_succeeded_speaks_plainly(self):
        assert speaks_as_done(Outcome.SUCCEEDED)

    @pytest.mark.parametrize("outcome", [Outcome.FAILED, Outcome.UNSUPPORTED])
    def test_nothing_else_speaks_as_done(self, outcome):
        assert not speaks_as_done(outcome)

    def test_the_two_quiet_outcomes_are_not_the_same_answer(self):
        """`UNVERIFIED` and `UNCERTAIN` both mean "no confirmation", and they
        must never produce the same sentence -- one is a choice, the other is a
        failure to find out."""
        assert speaks_as_done(Outcome.UNVERIFIED)
        assert not speaks_as_done(Outcome.UNCERTAIN)


# ═════════════════════════════════════════════════════════════════════════
# V3 — verification fails open at the step level, never upward
# ═════════════════════════════════════════════════════════════════════════

class TestV3FailOpenNotUpward:

    def test_a_crashed_verification_does_not_block_the_task(self):
        """The behaviour is kept and only the label is corrected: an
        infrastructure fault must not stop execution. It becomes UNCERTAIN --
        never SUCCEEDED, and never FAILED either, because nothing observed the
        step failing."""
        crashed = VerifyResult.crashed("boom")
        assert crashed.outcome is Outcome.UNCERTAIN
        assert crashed.outcome is not Outcome.FAILED
        assert crashed.outcome is not Outcome.SUCCEEDED

    def test_a_crash_makes_the_whole_task_uncertain(self):
        """Failing open at the step must not become failing open at the task.
        The doubt has to survive the rollup or the fail-open is just the old
        `ok=True` with more steps."""
        assert _task(Outcome.SUCCEEDED,
                     VerifyResult.crashed("boom").outcome).outcome() \
            is Outcome.UNCERTAIN
