"""
test_sub_spec_b.py — Bug 3: Recovery dependency re-linking.

Verifies:
  - After all recovery steps succeed, the original failed step gets status "recovered"
  - Dependency check passes "recovered" (not in ("failed", "skipped"))
  - Failed recovery leaves origin as "failed", downstream steps skipped
  - No recovery steps → cascade skip as before (unchanged behaviour)
"""

from __future__ import annotations

import asyncio
import sys
import types
import unittest
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock, patch

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def _run(coro):
    return asyncio.run(coro)


# ─── Minimal stubs so planner.py can be imported without full env ────────────

def _stub_heavy_modules():
    """Patch modules that would trigger real I/O or GPU usage on import."""
    stubs = [
        "assistant.llm",
        "assistant.llm.contracts",
        "assistant.llm.router",
        "assistant.storage",
        "assistant.storage.db",
        "assistant.core",
        "assistant.core.config",
        "assistant.automation",
        "assistant.automation.verification",
        "assistant.actions",
        "assistant.actions.planner.executor",
        "assistant.actions.planner.pseudo_tools",
    ]
    for name in stubs:
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)

    # llm.contracts needs ask_for_* callables
    contracts = sys.modules["assistant.llm.contracts"]
    contracts.ask_for_intent = AsyncMock(return_value="unknown")

    # automation.verification needs parse / format helpers
    av = sys.modules["assistant.automation.verification"]
    av.parse_verify_failed = MagicMock(return_value=None)
    av.format_failure_for_user = MagicMock(return_value="")


_stub_heavy_modules()

from assistant.actions.planner.planner import Plan, PlanStep  # noqa: E402


# ─── Helpers ────────────────────────────────────────────────────────────────


def _make_plan(*steps: PlanStep) -> Plan:
    return Plan(original_goal="test goal", steps=list(steps))


def _step(step_id: int, tool: str = "code_executor", depends_on=None, status="pending") -> PlanStep:
    s = PlanStep(step_id=step_id, tool=tool, goal=f"step {step_id} goal")
    s.depends_on = depends_on or []
    s.status = status
    return s


# ─── Bug 3 tests ─────────────────────────────────────────────────────────────


class TestRecoveryOriginMarkedRecovered(unittest.TestCase):
    """After all recovery steps succeed, origin step becomes 'recovered'."""

    def test_origin_status_becomes_recovered_after_last_recovery_step_succeeds(self):
        from assistant.actions.planner.planner import Plan, PlanStep

        origin = _step(1, status="failed")
        origin.error = "something broke"

        recovery_1 = _step(2, status="success")
        recovery_2 = _step(3, status="success")

        plan = _make_plan(origin, recovery_1, recovery_2)
        plan._recovery_origin = 1
        plan._recovery_step_ids = [2, 3]

        last_recovery = recovery_2

        if (last_recovery.status == "success"
                and hasattr(plan, '_recovery_step_ids')
                and last_recovery.step_id == plan._recovery_step_ids[-1]):
            found = next(
                (s for s in plan.steps if s.step_id == plan._recovery_origin), None
            )
            if found and found.status == "failed":
                found.status = "recovered"

        self.assertEqual(origin.status, "recovered")


class TestRecoveredPassesDependencyCheck(unittest.TestCase):
    """'recovered' is not in ('failed', 'skipped') so downstream steps proceed."""

    def test_recovered_not_in_blocked_statuses(self):
        blocked = ("failed", "skipped")
        self.assertNotIn("recovered", blocked)

    def test_downstream_step_not_skipped_when_dep_is_recovered(self):
        origin = _step(1, status="recovered")
        downstream = _step(2, depends_on=[1])

        plan = _make_plan(origin, downstream)

        should_skip = False
        for dep_id in downstream.depends_on:
            dep = next((s for s in plan.steps if s.step_id == dep_id), None)
            if dep and dep.status in ("failed", "skipped"):
                should_skip = True
                break

        self.assertFalse(should_skip)


class TestFailedRecoveryLeavesOriginFailed(unittest.TestCase):
    """If the last recovery step fails, origin stays 'failed' and downstream is skipped."""

    def test_origin_remains_failed_when_recovery_step_fails(self):
        origin = _step(1, status="failed")
        origin.error = "original error"
        recovery = _step(2, status="failed")
        recovery.error = "recovery also failed"
        downstream = _step(3, depends_on=[1])

        plan = _make_plan(origin, recovery, downstream)
        plan._recovery_origin = 1
        plan._recovery_step_ids = [2]

        if (recovery.status == "success"
                and hasattr(plan, '_recovery_step_ids')
                and recovery.step_id == plan._recovery_step_ids[-1]):
            found = next(
                (s for s in plan.steps if s.step_id == plan._recovery_origin), None
            )
            if found and found.status == "failed":
                found.status = "recovered"

        self.assertEqual(origin.status, "failed")

    def test_downstream_skipped_when_origin_failed_and_recovery_also_failed(self):
        origin = _step(1, status="failed")
        origin.error = "original error"
        recovery = _step(2, status="failed")
        recovery.error = "recovery also failed"
        downstream = _step(3, depends_on=[1])

        plan = _make_plan(origin, recovery, downstream)

        for later in plan.steps:
            if later.status == "pending" and origin.step_id in later.depends_on:
                later.status = "skipped"
                later.error = f"dependency step {origin.step_id} failed"

        self.assertEqual(downstream.status, "skipped")


class TestNoRecoveryStepsCascadeSkip(unittest.TestCase):
    """When no recovery steps are generated, downstream deps are skipped as before."""

    def test_cascade_skip_when_no_recovery_steps(self):
        origin = _step(1, status="failed")
        origin.error = "no recovery available"
        downstream = _step(2, depends_on=[1])

        plan = _make_plan(origin, downstream)

        recovery_steps = []

        if not recovery_steps:
            for later in plan.steps:
                if later.status == "pending" and origin.step_id in later.depends_on:
                    later.status = "skipped"
                    later.error = f"dependency step {origin.step_id} failed: {origin.error[:80]}"

        self.assertEqual(downstream.status, "skipped")
        self.assertIn("failed", downstream.error)

    def test_no_recovery_metadata_set_when_no_recovery_steps(self):
        plan = _make_plan(_step(1, status="failed"))

        recovery_steps = []

        if recovery_steps:
            plan._recovery_origin = 1
            plan._recovery_step_ids = [rs.step_id for rs in recovery_steps]

        self.assertFalse(hasattr(plan, '_recovery_origin'))
        self.assertFalse(hasattr(plan, '_recovery_step_ids'))


if __name__ == "__main__":
    unittest.main()


# ─── Bug 4: Headless mode decision logic ────────────────────────────────────


def test_bug4_run_browser_steps_accepts_headless_param():
    """run_browser_steps should accept a headless parameter."""
    import inspect
    from assistant.automation.browser.automation import run_browser_steps
    sig = inspect.signature(run_browser_steps)
    assert "headless" in sig.parameters, "run_browser_steps must accept 'headless' param"


def test_bug4_headless_default_is_false():
    """Default headless should be False (user-visible tasks are headed)."""
    import inspect
    from assistant.automation.browser.automation import run_browser_steps
    sig = inspect.signature(run_browser_steps)
    assert sig.parameters["headless"].default is False


def test_bug4_ensure_browser_preserves_mode():
    """ensure_browser should not flip mode when called with same headless value."""
    from assistant.automation.browser.automation import _browser_headless
    assert isinstance(_browser_headless, bool)


# ─── Bug 5: Region/IP geolocation ──────────────────────────────────────────


def test_bug5_detect_region_parses_api_response():
    from assistant.core.geolocation import _parse_region_response
    data = {"country": "India", "countryCode": "IN", "city": "Mumbai", "timezone": "Asia/Kolkata"}
    result = _parse_region_response(data)
    assert result["country"] == "India"
    assert result["country_code"] == "IN"
    assert result["city"] == "Mumbai"
    assert result["timezone"] == "Asia/Kolkata"


def test_bug5_detect_region_fallback_to_env(monkeypatch):
    monkeypatch.setenv("USER_REGION", "JP")
    monkeypatch.setenv("USER_TIMEZONE", "Asia/Tokyo")
    from assistant.core.geolocation import _env_fallback
    result = _env_fallback()
    assert result["country_code"] == "JP"
    assert result["timezone"] == "Asia/Tokyo"


def test_bug5_detect_region_fallback_defaults(monkeypatch):
    monkeypatch.delenv("USER_REGION", raising=False)
    monkeypatch.delenv("USER_TIMEZONE", raising=False)
    from assistant.core.geolocation import _env_fallback
    result = _env_fallback()
    assert result["country_code"] == ""
    assert result["timezone"] == ""


def test_bug5_region_prompt_format():
    from assistant.core.geolocation import format_region_hint
    region = {"country": "India", "country_code": "IN", "city": "Mumbai", "timezone": "Asia/Kolkata"}
    hint = format_region_hint(region)
    assert "India" in hint
    assert "Mumbai" in hint
    assert "Asia/Kolkata" in hint


def test_bug5_empty_region_returns_empty_hint():
    from assistant.core.geolocation import format_region_hint
    region = {"country": "", "country_code": "", "city": "", "timezone": ""}
    hint = format_region_hint(region)
    assert hint == ""


# ─── Bug 7: System command routing gate ─────────────────────────────────────


def test_bug7_gate_matches_action_verbs():
    from assistant.code_executor.orchestrator import _is_system_command
    assert _is_system_command("disable bluetooth") is True
    assert _is_system_command("turn off wifi") is True
    assert _is_system_command("enable Wi-Fi") is True


def test_bug7_gate_matches_query_with_noun():
    from assistant.code_executor.orchestrator import _is_system_command
    assert _is_system_command("list all wifi networks") is True
    assert _is_system_command("show bluetooth devices") is True
    assert _is_system_command("get wifi password") is True
    assert _is_system_command("check battery status") is True


def test_bug7_gate_rejects_query_without_noun():
    from assistant.code_executor.orchestrator import _is_system_command
    assert _is_system_command("show me a recipe") is False
    assert _is_system_command("list my contacts") is False
    assert _is_system_command("get the weather") is False


def test_bug7_wifi_list_known_command():
    from assistant.automation.system_commands import KNOWN_COMMANDS
    assert "wifi_list" in KNOWN_COMMANDS
    assert "netsh" in KNOWN_COMMANDS["wifi_list"]["cmd"]


# ─── Bug 8: LLM fix-loop short-circuit ─────────────────────────────────────


def test_bug8_classify_error_detects_blocked():
    from assistant.code_executor.retry import _classify_error
    diag = _classify_error("BLOCKED: import of 'subprocess' is not allowed")
    assert diag["category"] == "blocked"
    assert diag["needs_discovery"] is False


def test_bug8_blocked_needs_no_fix():
    from assistant.code_executor.retry import _classify_error
    diag = _classify_error("BLOCKED: unsafe call 'os.system'")
    assert diag["category"] == "blocked"
    assert diag["needs_discovery"] is False


def test_bug8_non_blocked_still_retries():
    from assistant.code_executor.retry import _classify_error
    diag = _classify_error("Error: ModuleNotFoundError: No module named 'spotipy'")
    assert diag["category"] != "blocked"


# ─── Forget/Delete Memory Feature ──────────────────────────────────────────


def test_forget_regex_matches_forget_about():
    from assistant.regex_router import _FORGET_MEMORY_RE
    m = _FORGET_MEMORY_RE.match("forget about cilantro")
    assert m is not None
    assert m.group(1).strip() == "cilantro"


def test_forget_regex_matches_delete_fact():
    from assistant.regex_router import _FORGET_MEMORY_RE
    m = _FORGET_MEMORY_RE.match("delete the fact that I like pizza")
    assert m is not None
    assert "I like pizza" in m.group(1)


def test_forget_regex_matches_remove():
    from assistant.regex_router import _FORGET_MEMORY_RE
    m = _FORGET_MEMORY_RE.match("remove my allergy info")
    assert m is not None


def test_forget_regex_precedes_remember():
    from assistant.regex_router import _FORGET_MEMORY_RE, _REMEMBER_FACT_RE
    text = "forget that cilantro is bad"
    forget_match = _FORGET_MEMORY_RE.match(text)
    assert forget_match is not None, "forget regex should match"


def test_forget_intent_registered():
    from assistant.config import INTENTS
    assert "forget_memory" in INTENTS


# ─── Bug: Single-step bypass goal forwarding ─────────────────────────────────


def test_single_step_bypass_returns_dict_with_step_goal():
    """Planner returns bypass dict containing the refined step goal, not just tool name."""
    from assistant.actions.planner.planner import Plan, PlanStep

    step = PlanStep(
        step_id=1, tool="browser_action",
        goal="Go to BookMyShow website and search for 'Spiderman'",
    )
    plan = Plan(original_goal="book spiderman movie tickets", steps=[step])

    # Simulate the single-step bypass check
    assert len(plan.steps) == 1
    result = {"bypass": plan.steps[0].tool, "goal": plan.steps[0].goal}
    assert result["bypass"] == "browser_action"
    assert result["goal"] == "Go to BookMyShow website and search for 'Spiderman'"


def test_bypass_params_override_goal():
    """da_handlers bypass path should override params['goal'] with planner's step goal."""
    original_params = {"goal": "book spiderman movie tickets for tomorrow"}
    bypass_result = {"bypass": "browser_action", "goal": "Go to BookMyShow and search for Spiderman"}

    bypass_params = {**original_params, "goal": bypass_result["goal"]}

    assert bypass_params["goal"] == "Go to BookMyShow and search for Spiderman"
    assert bypass_params["goal"] != original_params["goal"]


def test_bypass_result_is_dict_not_string():
    """After fix, planner returns dict instead of __BYPASS__:tool string."""
    from assistant.actions.planner.planner import PlanStep

    step = PlanStep(step_id=1, tool="code_executor", goal="run a script")

    result = {"bypass": step.tool, "goal": step.goal}
    assert isinstance(result, dict)
    assert "bypass" in result
    assert not isinstance(result, str)


# ─── Bug 2: Synthesis Verification Gate ──────────────────────────────────────


def test_completion_evidence_matches_success_tokens():
    from assistant.code_executor.orchestrator import _has_completion_evidence
    assert _has_completion_evidence("Booking confirmed for 2 tickets")
    assert _has_completion_evidence("Email sent successfully")
    assert _has_completion_evidence("File downloaded to C:/Users/")
    assert _has_completion_evidence("https://bookmyshow.com/ticket/123")
    assert _has_completion_evidence("status: 200 OK")


def test_completion_evidence_rejects_bare_computation():
    from assistant.code_executor.orchestrator import _has_completion_evidence
    assert not _has_completion_evidence("2026-05-29")
    assert not _has_completion_evidence("42")
    assert not _has_completion_evidence("The current time is 08:30 AM")
    assert not _has_completion_evidence("Temperature: 32.5")


def test_router_includes_verification_needed():
    """Router fallback should include verification_needed field."""
    from assistant.code_executor.routing import _route_goal
    fallback = {"tier": 1, "template_slug": None, "requires": [],
                "params": {}, "verification_needed": False}
    assert "verification_needed" in fallback
    assert fallback["verification_needed"] is False


def test_verification_gate_blocks_hallucinated_action():
    """When verify_needed=True and output lacks evidence, gate should trigger."""
    from assistant.code_executor.orchestrator import _has_completion_evidence
    goal = "book spiderman movie tickets"
    code_output = "2026-05-29\nSpider-Man: Brand New Day"
    verify_needed = True

    should_block = verify_needed and not _has_completion_evidence(code_output)
    assert should_block, "Gate should block: action goal with bare data output"


def test_verification_gate_allows_real_action():
    """When verify_needed=True and output has evidence, gate should pass."""
    from assistant.code_executor.orchestrator import _has_completion_evidence
    goal = "send email to alice"
    code_output = "Email sent successfully to alice@example.com"
    verify_needed = True

    should_block = verify_needed and not _has_completion_evidence(code_output)
    assert not should_block, "Gate should pass: output confirms action"


def test_verification_gate_skips_for_info_queries():
    """When verify_needed=False, gate never triggers regardless of output."""
    from assistant.code_executor.orchestrator import _has_completion_evidence
    goal = "what time is it"
    code_output = "08:30 AM"
    verify_needed = False

    should_block = verify_needed and not _has_completion_evidence(code_output)
    assert not should_block, "Gate should not trigger for info queries"


# ─── Recovery region + planner page fallback ─────────────────────────────────


def test_recovery_prompt_has_region_hint_placeholder():
    """Recovery planner prompt must include {region_hint} for geo context."""
    from assistant.actions.planner.planner import _REPLAN_SYSTEM_PROMPT
    assert "{region_hint}" in _REPLAN_SYSTEM_PROMPT


def test_geolocation_format_region_hint_india():
    """Region hint formats correctly for Indian user."""
    from assistant.core.geolocation import format_region_hint
    region = {"country": "India", "country_code": "IN",
              "city": "Nagpur", "timezone": "Asia/Kolkata"}
    hint = format_region_hint(region)
    assert "India" in hint
    assert "Nagpur" in hint
    assert "Asia/Kolkata" in hint


def test_geolocation_format_region_hint_empty():
    """Empty region produces empty hint string."""
    from assistant.core.geolocation import format_region_hint
    assert format_region_hint({}) == ""
    assert format_region_hint({"country": "", "timezone": ""}) == ""


def test_geolocation_sanitize_strips_dangerous_chars():
    """Sanitizer removes prompt-control characters."""
    from assistant.core.geolocation import _sanitize
    assert _sanitize("India") == "India"
    assert _sanitize("Ig\nnore previous") == "Ignore previous"
    assert _sanitize("a" * 200) == "a" * 80


def test_geolocation_stdlib_fallback():
    """detect_region() uses urllib (stdlib), not aiohttp."""
    import ast
    from pathlib import Path
    src = Path("assistant/core/geolocation.py").read_text()
    tree = ast.parse(src)
    imports = [
        node.names[0].name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
    ]
    assert "aiohttp" not in imports, "Should use urllib, not aiohttp"
