"""Recovery reports success only on positive evidence.

`VerifyResult.ok` is `True` for three different things: confident success,
`ambiguous()` (the code tier could not decide), and `skip()` (nothing was
checked). `automation/recovery.py` read it bare, so all three produced
`RecoveryOutcome(succeeded=True)` -- "the code cannot tell whether the recovery
worked" reported as "recovered", and the browser step loop continued on an
unverified screen.

It was the only site with this shape. The three loops that call `post_verify`
directly -- `native.py`, `browser/automation.py`, `router.py` -- all escalate
an ambiguous verdict to the vision tier first. Recovery did not.

The failure mode this belongs to is KI-28's: a control behaving correctly
while the report about it lies. Nothing here changes what recovery *does*;
it changes what it is willing to claim.

Both directions are pinned. A fix that stopped reporting success at all would
pass every "does not claim uncertainty" test while making recovery useless,
so the confident-success path is asserted just as hard.

Run with:  py -3.11 -m pytest tests/test_recovery_needs_evidence.py -v
"""
import pathlib
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from assistant.automation.verification import VerifyResult
from assistant.brain.task import Outcome  # noqa: E402

STEP = {"type": "app", "action": "click", "params": {"name": "OK"}}


@pytest.fixture()
def recovered(monkeypatch):
    """Drive `attempt_recovery` to the point of its post-recovery verify, with
    that verify returning whatever the test asks for.

    Everything before it is stubbed: diagnosis and the strategies are what
    recovery *does*, and this file is about what it *claims*.
    """
    from assistant.automation import recovery

    async def run(verdicts, *, vision=None, vision_enabled=True):
        seen = list(verdicts)

        async def _fake_post_verify(step, **kw):
            return seen.pop(0) if seen else VerifyResult.fail("exhausted")

        async def _fake_vision(step, code_result, **kw):
            # Fail-open by contract: returns the code result unchanged when it
            # cannot do better. That default is exactly how an ambiguous
            # verdict used to reach the caller wearing ok=True.
            return vision if vision is not None else code_result

        async def _fake_diagnose(goal, verify_observation, step):
            # `_diagnose` returns the dict the loop destructures, not a tuple.
            #
            # `overlay_appeared`, not `no_change`: the latter is in
            # `_UNIMPLEMENTED_STRATEGIES`, so the loop now stops on the
            # diagnosis and never reaches the re-verification these tests are
            # about. The class is incidental here -- what is under test is what
            # a verdict means once a strategy HAS run.
            return {"class": "overlay_appeared", "detail": "a dialog is open",
                    "recovery_target": "close button"}

        async def _fake_strategy(*args, **kwargs):
            return True, 1

        # `recovery.py` imports `verification` inside the function, so it is
        # not an attribute of the recovery module -- patch the verification
        # module itself, which the deferred import resolves to.
        from assistant.automation import verification as verification_mod
        monkeypatch.setattr(verification_mod, "post_verify", _fake_post_verify)
        monkeypatch.setattr(verification_mod, "vision_verify", _fake_vision)
        monkeypatch.setattr(recovery, "_diagnose", _fake_diagnose)
        monkeypatch.setattr(recovery, "_recover_overlay", _fake_strategy)

        from assistant import config
        monkeypatch.setattr(config, "VERIFY_VISION_FALLBACK", vision_enabled,
                            raising=False)

        return await recovery.attempt_recovery(
            step=STEP, goal="press OK",
            verify_result=VerifyResult.fail("button still there"),
            max_attempts=1,
        )

    return run


# ─── uncertainty is not success ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_an_ambiguous_verdict_is_not_a_recovery(recovered):
    """`ambiguous()` carries ok=True. With the vision tier unable to improve
    on it, the honest answer is that nothing was confirmed."""
    outcome = await recovered([VerifyResult.ambiguous("cannot tell")])

    assert not outcome.succeeded, (
        "recovery claimed success on a verdict that means 'the code cannot "
        "decide'. The step loop then continues on an unverified screen."
    )
    # UNCERTAIN, not FAILED: a strategy ran and nothing could confirm what it
    # did. Reporting failure would assert positive evidence the step did not
    # work, which is the same overclaim in the opposite direction.
    assert outcome.outcome is Outcome.UNCERTAIN, (
        f"an unconfirmed recovery reported {outcome.outcome.value}")


@pytest.mark.asyncio
async def test_a_skipped_verdict_is_not_a_recovery(recovered):
    """`skip()` carries ok=True and means nothing was checked at all."""
    outcome = await recovered([VerifyResult.skip("verify disabled")])
    assert not outcome.succeeded, (
        "recovery claimed success on a verdict that means nothing was checked"
    )


@pytest.mark.asyncio
async def test_vision_being_switched_off_cannot_upgrade_uncertainty(recovered):
    """A settings flip must not turn 'cannot tell' into 'recovered'.
    `VERIFY_VISION_FALLBACK` is reachable over `PATCH /v1/settings`."""
    outcome = await recovered([VerifyResult.ambiguous("cannot tell")],
                              vision_enabled=False)
    assert not outcome.succeeded, (
        "turning the vision tier off made an ambiguous verdict read as success"
    )


@pytest.mark.asyncio
async def test_the_attempt_record_matches_the_verdict(recovered):
    """`RecoveryAttempt.outcome` feeds the operator-facing summary. It used to
    be a bool recording `bool(vr.ok)`, so the log agreed with the lie."""
    outcome = await recovered([VerifyResult.ambiguous("cannot tell")])
    assert outcome.attempts, "no attempt recorded -- test would pass vacuously"
    assert not outcome.attempts[-1].outcome.is_evidence_of_success, (
        "the attempt log records the recovery as verified while the outcome "
        "says it was not"
    )


@pytest.mark.asyncio
async def test_an_unconfirmed_attempt_says_so_rather_than_repeating_the_old_error(
        recovered):
    """The observation reaches the user. Replaying the original failure text
    would say 'the button is still there' when what actually happened is that
    nobody looked."""
    outcome = await recovered([VerifyResult.ambiguous("")])
    assert "button still there" not in outcome.final_observation, (
        f"the pre-recovery failure was replayed as the outcome: "
        f"{outcome.final_observation!r}"
    )
    assert outcome.final_observation, "no observation at all"


# ─── evidence still counts ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_a_confident_verdict_is_still_a_recovery(recovered):
    """The half that stops this becoming 'recovery never works'. A fix that
    refused to report success would pass every test above."""
    outcome = await recovered([VerifyResult.ok_(tier="code", observation="gone")])
    assert outcome.succeeded, (
        "a confident code-tier success was not reported as a recovery -- "
        "recovery is now unable to ever succeed"
    )
    assert outcome.outcome is Outcome.SUCCEEDED
    assert outcome.attempts[-1].outcome is Outcome.SUCCEEDED


@pytest.mark.asyncio
async def test_vision_resolving_an_ambiguous_verdict_counts(recovered):
    """Escalation is the point: ambiguous at the code tier, confident at the
    vision tier, is a real recovery."""
    outcome = await recovered(
        [VerifyResult.ambiguous("cannot tell")],
        vision=VerifyResult.ok_(tier="vision", observation="dialog closed"),
    )
    assert outcome.succeeded, (
        "the vision tier confirmed the recovery and it was still reported as "
        "unconfirmed -- the escalation is not wired"
    )


@pytest.mark.asyncio
async def test_vision_confirming_a_failure_is_still_a_failure(recovered):
    outcome = await recovered(
        [VerifyResult.ambiguous("cannot tell")],
        vision=VerifyResult.fail("still on screen", tier="vision"),
    )
    assert not outcome.succeeded
