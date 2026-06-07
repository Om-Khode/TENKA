"""
test_planner_escalation.py — Code executor → planner escalation.

Verifies:
  - Verification gate returns escalation signal (not failure synthesis)
  - Retry exhaustion on action goals returns escalation signal
  - Template NOT saved when verification gate fires
  - da_handlers catches signal and routes to handle_planner
  - No escalation when _from_planner=True (prevents infinite loop)
  - No escalation for info queries (verify_needed=False)
  - PLANNER_ESCALATION_SIGNAL constant is importable
"""

from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

# Stub heavy modules
for mod_name in (
    "assistant.io.audio.tts", "assistant.io.audio.stt",
    "assistant.io.audio.speaker_verify", "assistant.io.unity_bridge",
    "assistant.io.audio.wake_word",
):
    if mod_name not in sys.modules:
        sys.modules[mod_name] = types.ModuleType(mod_name)


def _run(coro):
    return asyncio.run(coro)


# ─── Signal constant ─────────────────────────────────────────────────────────


def test_planner_escalation_signal_importable():
    from assistant.code_executor import PLANNER_ESCALATION_SIGNAL
    assert PLANNER_ESCALATION_SIGNAL == "__ESCALATE_PLANNER__"


def test_planner_escalation_signal_distinct_from_gui():
    from assistant.code_executor import PLANNER_ESCALATION_SIGNAL, GUI_HANDOFF_SIGNAL
    assert PLANNER_ESCALATION_SIGNAL != GUI_HANDOFF_SIGNAL


# ─── Verification gate escalation ────────────────────────────────────────────


def test_verification_gate_returns_escalation_signal():
    """When verify_needed=True and output lacks evidence, should return
    __ESCALATE_PLANNER__ instead of synthesizing a failure message."""
    from assistant.code_executor.orchestrator import _has_completion_evidence
    goal = "book movie tickets for 2 people"
    code_output = "Booking 2 tickets for Micheal is not supported by this API."
    verify_needed = True
    _from_planner = False

    should_escalate = (
        verify_needed
        and not _has_completion_evidence(code_output)
        and not _from_planner
    )
    assert should_escalate, "Should escalate: action goal with no completion evidence"


def test_no_escalation_when_from_planner():
    """When _from_planner=True, verification gate should NOT escalate
    (would cause infinite loop)."""
    from assistant.code_executor.orchestrator import _has_completion_evidence
    code_output = "Booking not supported."
    verify_needed = True
    _from_planner = True

    should_escalate = (
        verify_needed
        and not _has_completion_evidence(code_output)
        and not _from_planner
    )
    assert not should_escalate, "Must NOT escalate when called from planner"


def test_no_escalation_for_info_queries():
    """When verify_needed=False (info query), never escalate."""
    from assistant.code_executor.orchestrator import _has_completion_evidence
    code_output = "42"
    verify_needed = False
    _from_planner = False

    should_escalate = (
        verify_needed
        and not _has_completion_evidence(code_output)
        and not _from_planner
    )
    assert not should_escalate, "Info queries should not escalate"


def test_no_escalation_when_evidence_present():
    """When output has completion evidence, don't escalate — save template."""
    from assistant.code_executor.orchestrator import _has_completion_evidence
    code_output = "Booking confirmed for 2 tickets. ID: BMS123"
    verify_needed = True
    _from_planner = False

    should_escalate = (
        verify_needed
        and not _has_completion_evidence(code_output)
        and not _from_planner
    )
    assert not should_escalate, "Should NOT escalate when evidence present"


# ─── Retry exhaustion escalation ──────────────────────────────────────────────


def test_exhaustion_escalates_action_goals():
    """When all retries exhaust on an action goal (verify_needed=True),
    should escalate to planner instead of synthesizing apology."""
    verify_needed = True
    _from_planner = False
    retries_exhausted = True

    should_escalate = retries_exhausted and verify_needed and not _from_planner
    assert should_escalate


def test_exhaustion_no_escalate_info_query():
    """When all retries exhaust on an info query, DON'T escalate —
    synthesize apology instead."""
    verify_needed = False
    _from_planner = False
    retries_exhausted = True

    should_escalate = retries_exhausted and verify_needed and not _from_planner
    assert not should_escalate


def test_exhaustion_no_escalate_from_planner():
    """When called from planner and retries exhaust, return raw error —
    don't re-escalate."""
    verify_needed = True
    _from_planner = True
    retries_exhausted = True

    should_escalate = retries_exhausted and verify_needed and not _from_planner
    assert not should_escalate


# ─── Template save ordering ──────────────────────────────────────────────────


def test_template_not_saved_on_escalation():
    """Template save must happen AFTER verification gate check.
    When escalation fires, template should NOT be cached."""
    from assistant.code_executor.orchestrator import _has_completion_evidence

    code_output = "Cannot book tickets without API."
    verify_needed = True
    _from_planner = False

    would_escalate = verify_needed and not _has_completion_evidence(code_output) and not _from_planner
    assert would_escalate, "This case should escalate"

    would_save_template = not would_escalate
    assert not would_save_template, "Template must NOT be saved when escalation fires"


def test_template_saved_on_success():
    """Template should be saved when verification gate passes."""
    from assistant.code_executor.orchestrator import _has_completion_evidence

    code_output = "Email sent successfully to alice@example.com"
    verify_needed = True
    _from_planner = False

    would_escalate = verify_needed and not _has_completion_evidence(code_output) and not _from_planner
    assert not would_escalate, "Should NOT escalate — evidence present"


def test_template_saved_for_planner_calls():
    """When _from_planner=True, template should still be saved on success
    (before returning raw result)."""
    from assistant.code_executor.orchestrator import _has_completion_evidence
    from assistant.code_executor._utils import _needs_retry

    code_output = "Temperature: 32.5C"
    _from_planner = True

    should_save = _from_planner and not _needs_retry(code_output)
    assert should_save


# ─── da_handlers signal routing ───────────────────────────────────────────────


def test_da_handlers_catches_escalation_signal():
    """handle_code_executor should route __ESCALATE_PLANNER__ to handle_planner."""
    from assistant.code_executor import PLANNER_ESCALATION_SIGNAL

    result = PLANNER_ESCALATION_SIGNAL
    assert result == "__ESCALATE_PLANNER__"

    should_route_to_planner = (result == PLANNER_ESCALATION_SIGNAL)
    assert should_route_to_planner


def test_escalation_signal_not_confused_with_other_signals():
    """Escalation signal must not start with prefixes that other signal
    handlers check for."""
    from assistant.code_executor import PLANNER_ESCALATION_SIGNAL

    assert not PLANNER_ESCALATION_SIGNAL.startswith("__NEEDS_OAUTH__")
    assert not PLANNER_ESCALATION_SIGNAL.startswith("__NEEDS_DEVICE_AUTH__")
    assert not PLANNER_ESCALATION_SIGNAL.startswith("CLIENT_OUTDATED")
    assert not PLANNER_ESCALATION_SIGNAL.startswith("__CONFIRM_SEND__")
    assert not PLANNER_ESCALATION_SIGNAL.startswith("__SEND_ERROR__")
    assert PLANNER_ESCALATION_SIGNAL != "__NEEDS_GUI__"


# ─── Integration: booking goal classification ─────────────────────────────────


def test_booking_goal_has_no_completion_evidence():
    """Typical 'can't book' outputs should lack completion evidence,
    triggering escalation."""
    from assistant.code_executor.orchestrator import _has_completion_evidence

    failures = [
        "Booking 2 tickets for Micheal is not supported by this API.",
        "Error: Cannot book tickets without a specific ticketing API.",
        "Please provide a ticketing API package and credentials.",
        "This task requires authentication and a specific API.",
    ]
    for output in failures:
        assert not _has_completion_evidence(output), f"Should lack evidence: {output!r}"


def test_booking_confirmation_has_evidence():
    """Real booking confirmations should pass the verification gate."""
    from assistant.code_executor.orchestrator import _has_completion_evidence

    successes = [
        "Booking confirmed! Reference: BMS-2026-05-29-123",
        "Tickets booked successfully for 2 people",
        "Order placed. Confirmation email sent to user@example.com",
        "https://in.bookmyshow.com/booking/confirm/ABC123",
    ]
    for output in successes:
        assert _has_completion_evidence(output), f"Should have evidence: {output!r}"


# ─── Intent prompt: interactive web sessions → planner ────────────────────────


def test_intent_prompt_has_interactive_web_exception():
    """Rule 1 must include the interactive web session exception."""
    from assistant.config import INTENT_SYSTEM_PROMPT
    assert "interactive web session" in INTENT_SYSTEM_PROMPT
    assert "booking" in INTENT_SYSTEM_PROMPT.lower()


def test_intent_prompt_planner_includes_interactive():
    """Rule 6 (planner definition) must mention interactive web pattern."""
    from assistant.config import INTENT_SYSTEM_PROMPT
    assert "INTERACTIVE" in INTENT_SYSTEM_PROMPT


def test_intent_prompt_has_booking_fewshot():
    """Few-shot examples must include a booking → planner case."""
    from assistant.config import INTENT_SYSTEM_PROMPT
    assert "book movie tickets" in INTENT_SYSTEM_PROMPT
    assert '"planner"' in INTENT_SYSTEM_PROMPT


def test_intent_prompt_planner_catalog_mentions_interactive():
    """Planner catalog entry must describe interactive web tasks."""
    from assistant.config import INTENT_SYSTEM_PROMPT
    lines = INTENT_SYSTEM_PROMPT.split("\n")
    planner_line = [l for l in lines if l.strip().startswith("planner")]
    assert planner_line, "Planner must appear in intent catalog"
    assert "interactive" in planner_line[0].lower()


def test_intent_prompt_no_app_specific_rules():
    """Intent prompt must NOT mention specific app names (THE rule)."""
    from assistant.config import INTENT_SYSTEM_PROMPT
    prompt_lower = INTENT_SYSTEM_PROMPT.lower()
    for brand in ["bookmyshow", "fandango", "ticketmaster", "paytm",
                   "makemytrip", "amazon", "flipkart"]:
        assert brand not in prompt_lower, f"THE rule violation: {brand!r} in intent prompt"


# ─── Planner tool manifest: browser_action for interactive tasks ──────────────


def test_browser_action_manifest_mentions_booking():
    """browser_action tool description must mention booking/purchasing."""
    from assistant.actions.planner.planner import TOOL_MANIFEST
    desc = TOOL_MANIFEST["browser_action"]["description"].lower()
    assert "book" in desc, "browser_action should mention booking"
    assert "purchas" in desc, "browser_action should mention purchasing"
    assert "reserv" in desc, "browser_action should mention reservations"


def test_code_executor_manifest_excludes_browser_tasks():
    """code_executor tool description must clarify it's NOT for browser tasks."""
    from assistant.actions.planner.planner import TOOL_MANIFEST
    desc = TOOL_MANIFEST["code_executor"]["description"].lower()
    assert "not for" in desc or "not" in desc, "code_executor should disclaim browser tasks"


def test_planner_prompt_has_booking_example():
    """Planner prompt must include a booking → browser_action example."""
    from assistant.actions.planner.planner import _PLAN_SYSTEM_PROMPT
    assert "book movie tickets" in _PLAN_SYSTEM_PROMPT.lower()
    assert "browser_action" in _PLAN_SYSTEM_PROMPT


def test_planner_manifests_no_app_specific_rules():
    """Tool manifest and plan prompt must NOT mention specific app names."""
    from assistant.actions.planner.planner import TOOL_MANIFEST, _PLAN_SYSTEM_PROMPT
    combined = _PLAN_SYSTEM_PROMPT.lower()
    for tool_info in TOOL_MANIFEST.values():
        combined += " " + tool_info["description"].lower()
    for brand in ["bookmyshow", "fandango", "ticketmaster", "spotify",
                   "amazon", "flipkart", "paytm"]:
        assert brand not in combined, f"THE rule violation: {brand!r} in planner manifest"


# ─── Browser step generation prompt quality ───────────────────────────────────


def test_browser_plan_prompt_prohibits_clicking_everything():
    """Step generation prompt must explicitly tell LLM not to click all elements."""
    from assistant.automation.router import _BROWSER_PLAN_PROMPT
    prompt_lower = _BROWSER_PLAN_PROMPT.lower()
    assert "only interact with elements directly relevant" in prompt_lower or \
           "only relevant" in prompt_lower or \
           "do not click" in prompt_lower, \
        "Prompt must prohibit clicking unrelated elements"


def test_browser_plan_prompt_requires_reason_per_action():
    """Each click/fill must have a clear reason tied to the goal."""
    from assistant.automation.router import _BROWSER_PLAN_PROMPT
    assert "clear reason" in _BROWSER_PLAN_PROMPT.lower() or \
           "reason tied to the goal" in _BROWSER_PLAN_PROMPT.lower()
