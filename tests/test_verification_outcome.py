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
