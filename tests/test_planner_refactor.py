"""
Tests for S10 — planner move to actions/planner/ package.

Verifies:
  - Package structure and imports work
  - Public API accessible from __init__.py
  - Pseudo-tools importable and callable
  - Executor importable
  - needs_planning() still works correctly
  - _step_failed() still works correctly
  - _extract_json_array() still works correctly
  - _evaluate_condition() still works correctly
  - _resolve_references() still works correctly
  - _extract_note_params() still works correctly
  - Old import path (assistant.planner) is gone
"""

import sys
import os
import unittest
import importlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestPlannerPackageStructure(unittest.TestCase):
    """Verify the planner package is properly structured."""

    def test_package_importable(self):
        import assistant.actions.planner
        self.assertTrue(hasattr(assistant.actions.planner, 'execute_plan'))
        self.assertTrue(hasattr(assistant.actions.planner, 'resume_plan'))
        self.assertTrue(hasattr(assistant.actions.planner, 'needs_planning'))
        self.assertTrue(hasattr(assistant.actions.planner, 'has_suspended_plan'))
        self.assertTrue(hasattr(assistant.actions.planner, 'clear_suspended_plan'))
        self.assertTrue(hasattr(assistant.actions.planner, 'PlanStep'))
        self.assertTrue(hasattr(assistant.actions.planner, 'Plan'))
        self.assertTrue(hasattr(assistant.actions.planner, 'TOOL_MANIFEST'))

    def test_planner_module_importable(self):
        from assistant.actions.planner import planner
        self.assertTrue(hasattr(planner, 'execute_plan'))
        self.assertTrue(hasattr(planner, '_generate_plan'))
        self.assertTrue(hasattr(planner, '_synthesize_result'))
        self.assertTrue(hasattr(planner, '_attempt_recovery'))

    def test_executor_module_importable(self):
        from assistant.actions.planner import executor
        self.assertTrue(hasattr(executor, 'execute_step'))
        self.assertTrue(hasattr(executor, '_snapshot_pending_states'))
        self.assertTrue(hasattr(executor, '_pending_state_changed'))

    def test_pseudo_tools_module_importable(self):
        from assistant.actions.planner import pseudo_tools
        self.assertTrue(hasattr(pseudo_tools, 'run_synthesize_step'))
        self.assertTrue(hasattr(pseudo_tools, 'run_vision_analyze_step'))
        self.assertTrue(hasattr(pseudo_tools, 'run_prompt_user_step'))
        self.assertTrue(hasattr(pseudo_tools, 'run_camera_preview_step'))
        self.assertTrue(hasattr(pseudo_tools, '_parse_overlay_type'))
        self.assertTrue(hasattr(pseudo_tools, '_draw_overlay'))
        self.assertTrue(hasattr(pseudo_tools, '_camera_preview_blocking'))

    def test_old_import_path_gone(self):
        if 'assistant.planner' in sys.modules:
            del sys.modules['assistant.planner']
        with self.assertRaises((ImportError, ModuleNotFoundError)):
            import importlib
            importlib.import_module('assistant.planner')


class TestToolManifest(unittest.TestCase):
    """Verify TOOL_MANIFEST is complete and well-formed."""

    def test_manifest_has_expected_tools(self):
        from assistant.actions.planner import TOOL_MANIFEST
        expected = {
            "code_executor", "computer_task", "browser_action", "app_action",
            "web_search", "browse_url", "file_task", "camera_look",
            "read_screen", "memory_query", "create_note", "open_browser",
            "set_reminder", "recognize_face",
            "synthesize", "vision_analyze", "camera_preview", "prompt_user",
            "store_memory",
        }
        self.assertEqual(set(TOOL_MANIFEST.keys()), expected)

    def test_manifest_entries_have_required_keys(self):
        from assistant.actions.planner import TOOL_MANIFEST
        for name, info in TOOL_MANIFEST.items():
            self.assertIn("description", info, f"{name} missing description")
            self.assertIn("param_key", info, f"{name} missing param_key")
            self.assertIn("interactive", info, f"{name} missing interactive")


class TestNeedsPlanning(unittest.TestCase):
    """Verify needs_planning() still works after the move."""

    def test_single_step_goals_return_false(self):
        from assistant.actions.planner import needs_planning
        self.assertFalse(needs_planning("what's the weather"))
        self.assertFalse(needs_planning("play music"))
        self.assertFalse(needs_planning("hello"))

    def test_multi_step_goals_return_true(self):
        from assistant.actions.planner import needs_planning
        self.assertTrue(needs_planning("check the weather and play some music"))
        self.assertTrue(needs_planning("read my emails and then send a reply"))

    def test_and_in_title_not_multi_step(self):
        """'and' inside song/app names must not trigger multi-step."""
        from assistant.actions.planner import needs_planning
        self.assertFalse(needs_planning("play Beauty and a Beat on Spotify"))
        self.assertFalse(needs_planning("play Rock and Roll All Nite on Spotify"))
        self.assertFalse(needs_planning("play Romeo and Juliet by Dire Straits"))

    def test_and_with_verbs_both_sides_is_multi_step(self):
        """'and' separating two verb-led clauses is genuinely multi-step."""
        from assistant.actions.planner import needs_planning
        self.assertTrue(needs_planning("play some music and open the browser"))
        self.assertTrue(needs_planning("check the weather and send an email"))

    def test_then_with_nouns_still_multi_step(self):
        """'then' / 'also' / 'plus' still trigger on nouns."""
        from assistant.actions.planner import needs_planning
        self.assertTrue(needs_planning("check the weather then spotify playlist"))
        self.assertTrue(needs_planning("read my emails and then send a reply"))

    def test_short_goals_always_false(self):
        from assistant.actions.planner import needs_planning
        self.assertFalse(needs_planning("hi"))
        self.assertFalse(needs_planning("stop"))
        self.assertFalse(needs_planning("play it"))


class TestStepFailed(unittest.TestCase):
    """Verify _step_failed() still works after the move."""

    def test_success_output(self):
        from assistant.actions.planner.planner import _step_failed
        self.assertFalse(_step_failed("The weather is sunny and 25°C"))
        self.assertFalse(_step_failed("Playing Bohemian Rhapsody on Spotify"))

    def test_failure_output(self):
        from assistant.actions.planner.planner import _step_failed
        self.assertTrue(_step_failed("ERROR: Camera is currently disabled"))
        self.assertTrue(_step_failed("couldn't find the file"))
        self.assertTrue(_step_failed(""))
        self.assertTrue(_step_failed("(no output)"))

    def test_verify_failed_prefix(self):
        from assistant.actions.planner.planner import _step_failed
        self.assertTrue(_step_failed("VERIFY_FAILED|goal:fill form|obs:field empty"))


class TestExtractJsonArray(unittest.TestCase):
    """Verify JSON sanitization still works (formerly _extract_json_array).

    After S10's planner refactor, the string-level sanitization moved to
    assistant.core.json_utils.sanitize_json. The planner now uses the parsed
    helper extract_json_array which returns a list, but the pre-parse
    cleanup invariants still live in sanitize_json.
    """

    def test_plain_json(self):
        from assistant.core.json_utils import sanitize_json
        raw = '[{"step_id": 1, "tool": "web_search"}]'
        self.assertEqual(sanitize_json(raw), raw)

    def test_markdown_fenced(self):
        from assistant.core.json_utils import sanitize_json
        raw = '```json\n[{"step_id": 1}]\n```'
        result = sanitize_json(raw)
        self.assertIn('"step_id"', result)

    def test_think_tags_stripped(self):
        from assistant.core.json_utils import sanitize_json
        raw = '<think>reasoning here</think>[{"step_id": 1}]'
        result = sanitize_json(raw)
        self.assertNotIn('think', result)
        self.assertIn('"step_id"', result)

    def test_trailing_comma_fixed(self):
        from assistant.core.json_utils import sanitize_json
        raw = '[{"a": 1,}]'
        result = sanitize_json(raw)
        self.assertNotIn(',}', result)


class TestEvaluateCondition(unittest.TestCase):
    """Verify _evaluate_condition() still works after the move."""

    def test_no_condition(self):
        from assistant.actions.planner.planner import _evaluate_condition, Plan
        plan = Plan(original_goal="test", steps=[])
        self.assertTrue(_evaluate_condition(None, plan))
        self.assertTrue(_evaluate_condition("", plan))

    def test_contains_match(self):
        from assistant.actions.planner.planner import (
            _evaluate_condition, Plan, PlanStep,
        )
        step1 = PlanStep(step_id=1, tool="test", goal="", status="success",
                         output="Message from Mom: hello")
        plan = Plan(original_goal="test", steps=[step1],
                    context={"step_1": "Message from Mom: hello"})
        self.assertTrue(
            _evaluate_condition("if $step_1 contains 'Mom'", plan)
        )
        self.assertFalse(
            _evaluate_condition("if $step_1 contains 'Dad'", plan)
        )

    def test_does_not_contain(self):
        from assistant.actions.planner.planner import (
            _evaluate_condition, Plan, PlanStep,
        )
        step1 = PlanStep(step_id=1, tool="test", goal="", status="success",
                         output="No messages")
        plan = Plan(original_goal="test", steps=[step1],
                    context={"step_1": "No messages"})
        self.assertTrue(
            _evaluate_condition("if $step_1 does not contain 'Mom'", plan)
        )


class TestResolveReferences(unittest.TestCase):
    """Verify _resolve_references() still works after the move."""

    def test_replaces_step_ref(self):
        from assistant.actions.planner.planner import (
            _resolve_references, Plan, PlanStep,
        )
        step1 = PlanStep(step_id=1, tool="web_search", goal="weather",
                         status="success", output="Sunny, 25°C")
        plan = Plan(original_goal="test", steps=[step1],
                    context={"step_1": "Sunny, 25°C"})
        result = _resolve_references("Tell user: $step_1", plan)
        self.assertIn("Sunny", result)
        self.assertNotIn("$step_1", result)

    def test_unresolved_ref_kept(self):
        from assistant.actions.planner.planner import _resolve_references, Plan
        plan = Plan(original_goal="test", steps=[])
        result = _resolve_references("Check $step_99", plan)
        self.assertIn("$step_99", result)


class TestExtractNoteParams(unittest.TestCase):
    """Verify _extract_note_params() still works after the move."""

    def test_structured_format(self):
        from assistant.actions.planner.planner import _extract_note_params
        result = _extract_note_params("title: Shopping List, content: Milk, eggs, bread")
        self.assertEqual(result["title"], "Shopping List")
        self.assertIn("Milk", result["content"])

    def test_fallback_format(self):
        from assistant.actions.planner.planner import _extract_note_params
        result = _extract_note_params("hello")
        self.assertEqual(result["title"], "Plan Note")


class TestOverlayParsing(unittest.TestCase):
    """Verify pseudo_tools overlay parsing still works."""

    def test_grid_3x3(self):
        from assistant.actions.planner.pseudo_tools import _parse_overlay_type
        self.assertEqual(_parse_overlay_type("Show 3x3 grid"), "grid_3x3")

    def test_crosshair(self):
        from assistant.actions.planner.pseudo_tools import _parse_overlay_type
        self.assertEqual(_parse_overlay_type("Show crosshair"), "crosshair")

    def test_rectangle(self):
        from assistant.actions.planner.pseudo_tools import _parse_overlay_type
        self.assertEqual(_parse_overlay_type("Document alignment"), "rectangle")

    def test_none_default(self):
        from assistant.actions.planner.pseudo_tools import _parse_overlay_type
        self.assertEqual(_parse_overlay_type("just open the camera"), "none")

    def test_grid_fallback(self):
        from assistant.actions.planner.pseudo_tools import _parse_overlay_type
        self.assertEqual(_parse_overlay_type("show grid overlay"), "grid_3x3")


class TestPendingStateHelpers(unittest.TestCase):
    """Verify executor pending state helpers work."""

    def test_pending_state_changed_detects_activation(self):
        from assistant.actions.planner.executor import _pending_state_changed
        before = {"oauth_setup": False, "device_auth": False}
        after = {"oauth_setup": True, "device_auth": False}
        self.assertTrue(_pending_state_changed(before, after))

    def test_pending_state_changed_no_change(self):
        from assistant.actions.planner.executor import _pending_state_changed
        before = {"oauth_setup": False}
        after = {"oauth_setup": False}
        self.assertFalse(_pending_state_changed(before, after))

    def test_pending_state_changed_deactivation_ignored(self):
        from assistant.actions.planner.executor import _pending_state_changed
        before = {"oauth_setup": True}
        after = {"oauth_setup": False}
        self.assertFalse(_pending_state_changed(before, after))


class TestSuspensionAPI(unittest.TestCase):
    """Verify plan suspension API works."""

    def test_no_suspended_plan_initially(self):
        from assistant.actions.planner import has_suspended_plan, clear_suspended_plan
        clear_suspended_plan()
        self.assertFalse(has_suspended_plan())

    def test_suspend_and_clear(self):
        from assistant.actions.planner.planner import (
            _suspend_plan, has_suspended_plan, clear_suspended_plan, Plan,
        )
        plan = Plan(original_goal="test", steps=[])
        _suspend_plan(plan, 0, None, None, None)
        self.assertTrue(has_suspended_plan())
        clear_suspended_plan()
        self.assertFalse(has_suspended_plan())


# --- Bug fix tests (S10 live-test findings) ---


class TestBug1CodeExecutorPlannerGuard(unittest.TestCase):
    """Bug 1: code_executor must return raw error when called from planner.

    When all retries are exhausted and _from_planner=True, code_executor
    should return the raw result (truncated to 200 chars), NOT run
    personality synthesis. Without this guard, planner's _step_failed()
    can't detect failures wrapped in personality text like
    "[angry] Ugh, seriously?!..."
    """

    def test_from_planner_guard_exists_in_exhausted_path(self):
        """Verify the _from_planner early-return exists after retry exhaustion."""
        import ast
        code_executor_path = os.path.join(
            os.path.dirname(__file__), "..", "assistant", "code_executor", "orchestrator.py"
        )
        with open(code_executor_path, "r", encoding="utf-8") as f:
            source = f.read()

        # The pattern: after "All {_MAX_RETRIES} retries exhausted" log,
        # there must be an `if _from_planner:` guard before the synthesis call.
        exhausted_idx = source.find("retries exhausted")
        self.assertNotEqual(exhausted_idx, -1, "retries exhausted marker not found")

        after_exhausted = source[exhausted_idx:exhausted_idx + 1200]
        planner_guard_idx = after_exhausted.find("if _from_planner:")
        synthesis_idx = after_exhausted.find("task_type=\"synthesis\"")

        self.assertNotEqual(planner_guard_idx, -1,
                            "_from_planner guard missing after retry exhaustion")
        self.assertLess(planner_guard_idx, synthesis_idx,
                        "_from_planner guard must come BEFORE synthesis call")


class TestBug2AuthSentinelTightening(unittest.TestCase):
    """Bug 2: auth sentinel check must not false-positive on web content.

    Generic phrases like "developer app" and "Do you already have a"
    appeared in web search results about Gmail API setup, causing
    recovery steps (web_search) to be incorrectly flagged as auth-required.
    Sentinels are now machine-readable prefixes only.
    """

    def test_sentinels_are_machine_readable_only(self):
        """Verify _AUTH_SENTINELS doesn't contain generic phrases."""
        import assistant.actions.planner.executor as executor_mod

        source_path = os.path.join(
            os.path.dirname(__file__), "..",
            "assistant", "actions", "planner", "executor.py"
        )
        with open(source_path, "r", encoding="utf-8") as f:
            source = f.read()

        # Extract the _AUTH_SENTINELS tuple from source
        sentinel_start = source.find("_AUTH_SENTINELS = (")
        self.assertNotEqual(sentinel_start, -1)
        sentinel_end = source.find(")", sentinel_start) + 1
        sentinel_block = source[sentinel_start:sentinel_end]

        # These generic phrases must NOT be in the sentinels (Bug 2 cause)
        dangerous_phrases = [
            "developer app",
            "Do you already have a",
            "client_id",
            "client_secret",
        ]
        for phrase in dangerous_phrases:
            self.assertNotIn(phrase, sentinel_block,
                             f"Generic phrase '{phrase}' in _AUTH_SENTINELS causes false positives")

    def test_machine_sentinels_still_detected(self):
        """Verify that real auth sentinels still trigger detection."""
        source_path = os.path.join(
            os.path.dirname(__file__), "..",
            "assistant", "actions", "planner", "executor.py"
        )
        with open(source_path, "r", encoding="utf-8") as f:
            source = f.read()

        sentinel_block = source[source.find("_AUTH_SENTINELS = ("):
                                source.find(")", source.find("_AUTH_SENTINELS = (")) + 1]

        for sentinel in ("__NEEDS_OAUTH__", "NEEDS_OAUTH|",
                         "__NEEDS_DEVICE_AUTH__", "NEEDS_DEVICE_AUTH|"):
            self.assertIn(sentinel, sentinel_block,
                          f"Machine sentinel '{sentinel}' missing from _AUTH_SENTINELS")

    def test_web_search_content_not_flagged(self):
        """Simulate web search result about Gmail API — must not trigger auth."""
        web_result = (
            "To use Gmail API, create a developer app in Google Cloud Console. "
            "Do you already have a project? You'll need a client_id and "
            "client_secret from the OAuth consent screen."
        )
        sentinels = (
            "__NEEDS_OAUTH__", "NEEDS_OAUTH|",
            "__NEEDS_DEVICE_AUTH__", "NEEDS_DEVICE_AUTH|",
            "I need to set up",
        )
        self.assertFalse(
            any(s in web_result for s in sentinels),
            "Web search content about OAuth falsely triggers auth sentinel"
        )

    def test_real_auth_response_flagged(self):
        """Real auth sentinel results must still be detected."""
        real_results = [
            "NEEDS_OAUTH|spotify|https://accounts.spotify.com/authorize?...",
            "__NEEDS_OAUTH__ Please set up Spotify first.",
            "__NEEDS_DEVICE_AUTH__ Visit https://microsoft.com/devicelogin",
            "NEEDS_DEVICE_AUTH|xbox|ABC123",
            "I need to set up Gmail API access first.",
        ]
        sentinels = (
            "__NEEDS_OAUTH__", "NEEDS_OAUTH|",
            "__NEEDS_DEVICE_AUTH__", "NEEDS_DEVICE_AUTH|",
            "I need to set up",
        )
        for result in real_results:
            self.assertTrue(
                any(s in result for s in sentinels),
                f"Real auth result not detected: {result[:60]}"
            )


class TestBug3WhatsAppPhoneDetection(unittest.TestCase):
    """Bug 3: WhatsApp adapter must recognize phone numbers in contact_name.

    When code_executor passes contact_name="9764280339", the adapter must
    detect it's a phone number and use it directly instead of trying
    _resolve_contact() which fails and says "Try using a phone number instead."
    """

    def test_phone_number_detection_in_source(self):
        """Verify the phone number detection guard exists in _send_message."""
        wa_path = os.path.join(
            os.path.dirname(__file__), "..",
            "assistant", "io", "adapters", "whatsapp.py"
        )
        with open(wa_path, "r", encoding="utf-8") as f:
            source = f.read()

        # The phone detection must come before _resolve_contact
        send_msg_idx = source.find("def _send_message")
        self.assertNotEqual(send_msg_idx, -1)

        after_method = source[send_msg_idx:]
        phone_detect_idx = after_method.find("stripped.isdigit()")
        resolve_idx = after_method.find("_resolve_contact")

        self.assertNotEqual(phone_detect_idx, -1,
                            "Phone number detection missing in _send_message")
        self.assertLess(phone_detect_idx, resolve_idx,
                        "Phone detection must come BEFORE _resolve_contact call")

    def test_pure_digits_detected(self):
        """Pure digit strings 7-15 chars should be detected as phone numbers."""
        test_cases = [
            ("9764280339", True),      # 10-digit Indian mobile
            ("+919764280339", True),    # with country code
            ("1234567", True),          # minimum 7 digits
            ("123456789012345", True),  # maximum 15 digits
            ("123456", False),          # too short (6 digits)
            ("1234567890123456", False),  # too long (16 digits)
            ("John", False),            # name
            ("Mom", False),             # short name
            ("John Smith", False),      # name with space
        ]
        for contact_name, expected_is_phone in test_cases:
            stripped = contact_name.replace("+", "").replace("-", "").replace(" ", "")
            is_phone = stripped.isdigit() and 7 <= len(stripped) <= 15
            self.assertEqual(
                is_phone, expected_is_phone,
                f"contact_name='{contact_name}': expected is_phone={expected_is_phone}, got {is_phone}"
            )

    def test_formatted_phone_numbers_detected(self):
        """Phone numbers with formatting chars should still be detected."""
        formatted = [
            "+91-976-428-0339",
            "976 428 0339",
            "+1-555-123-4567",
            "91 9764280339",
        ]
        for contact_name in formatted:
            stripped = contact_name.replace("+", "").replace("-", "").replace(" ", "")
            self.assertTrue(
                stripped.isdigit() and 7 <= len(stripped) <= 15,
                f"Formatted phone '{contact_name}' not detected (stripped='{stripped}')"
            )


class TestBug4SpotifyPollLoop(unittest.TestCase):
    """Bug 4: Post-launch poll loop must not exit early on APP_NOT_READY.

    _needs_retry("APP_NOT_READY|spotify") returns False because the sentinel
    isn't in any failure pattern. The old poll loop checked _needs_retry first
    and broke on "success" before reaching the _still_device check.
    Fix: check _still_device first, continue if app not ready.
    """

    def test_poll_loop_checks_still_device_before_break(self):
        """Verify the poll loop uses continue-on-device, break-otherwise pattern."""
        ce_path = os.path.join(
            os.path.dirname(__file__), "..", "assistant", "code_executor", "orchestrator.py"
        )
        with open(ce_path, "r", encoding="utf-8") as f:
            source = f.read()

        poll_start = source.find("for _poll in range(_polls):")
        self.assertNotEqual(poll_start, -1)
        poll_block = source[poll_start:poll_start + 1500]

        # The _still_device check must come BEFORE any break
        still_device_idx = poll_block.find("_still_device")
        self.assertNotEqual(still_device_idx, -1, "_still_device check missing from poll loop")

        first_break = poll_block.find("break")
        self.assertGreater(first_break, still_device_idx,
                           "_still_device check must come BEFORE any break in poll loop")

        # Must use continue (keep polling) not break on device error
        self.assertIn("continue", poll_block[:first_break],
                       "Poll loop should 'continue' on device error, not fall through")


class TestBug5TemplateSaveGuard(unittest.TestCase):
    """Bug 5: Template must NOT be saved when result is APP_NOT_READY."""

    def test_template_save_guards_app_not_ready(self):
        """Verify APP_NOT_READY-class results don't reach the template save.

        The original explicit `APP_NOT_READY` check on the save guard was
        replaced by the structural `_needs_retry(result)` predicate, which
        also catches APP_NOT_READY (see _needs_retry in code_executor/_utils).
        This test now validates the equivalent invariant on the new guard.
        """
        ce_path = os.path.join(
            os.path.dirname(__file__), "..", "assistant", "code_executor", "orchestrator.py"
        )
        with open(ce_path, "r", encoding="utf-8") as f:
            source = f.read()

        save_idx = source.find("_save_template(slug")
        self.assertGreater(save_idx, 0)
        guard_region = source[max(0, save_idx - 300):save_idx]
        self.assertIn("_needs_retry", guard_region,
                       "Template save must be guarded by _needs_retry (catches APP_NOT_READY)")

        from assistant.code_executor import _needs_retry
        self.assertTrue(_needs_retry("APP_NOT_READY|spotify"),
                        "_needs_retry must catch APP_NOT_READY (used as save guard)")


class TestBug6PlannerAppNotReady(unittest.TestCase):
    """Bug 6: Planner _step_failed must detect APP_NOT_READY sentinel."""

    def test_app_not_ready_in_failure_prefixes(self):
        from assistant.actions.planner.planner import _FAILURE_PREFIXES
        self.assertTrue(
            any("APP_NOT_READY" in p for p in _FAILURE_PREFIXES),
            "APP_NOT_READY| missing from _FAILURE_PREFIXES"
        )

    def test_step_failed_catches_app_not_ready(self):
        from assistant.actions.planner.planner import _step_failed
        self.assertTrue(_step_failed("APP_NOT_READY|spotify"))
        self.assertTrue(_step_failed("APP_NOT_READY|discord"))

    def test_step_failed_still_passes_normal_output(self):
        from assistant.actions.planner.planner import _step_failed
        self.assertFalse(_step_failed("Playing music on Spotify."))
        self.assertFalse(_step_failed("33.1°C  wind 17.1 km/h"))


class TestBug7OAuthResumeSkip(unittest.TestCase):
    """Bug 7: OAuth setup must skip to auth_code when credentials exist.

    When client_id and client_secret are already saved in credential_store,
    the "has_app" step should detect them and skip directly to the
    authorization URL step instead of asking the user again.
    """

    def test_has_app_step_checks_existing_credentials(self):
        """Verify the has_app step checks credential_store before asking."""
        ph_path = os.path.join(
            os.path.dirname(__file__), "..",
            "assistant", "actions", "pending_handlers.py"
        )
        with open(ph_path, "r", encoding="utf-8") as f:
            source = f.read()

        has_app_idx = source.find('if step == "has_app":')
        self.assertNotEqual(has_app_idx, -1)

        has_app_block = source[has_app_idx:has_app_idx + 2000]

        # Must check for existing credentials before yes/no parsing
        cred_check_idx = has_app_block.find("get_credential")
        yes_check_idx = has_app_block.find("is_yes")

        self.assertNotEqual(cred_check_idx, -1,
                            "has_app step must check credential_store for existing credentials")
        self.assertNotEqual(yes_check_idx, -1,
                            "has_app step must still parse yes/no for the no-creds branch")
        self.assertLess(cred_check_idx, yes_check_idx,
                        "Credential check must come BEFORE yes/no parsing")

    def test_skip_sets_auth_code_step(self):
        """Verify skip path sets step to 'auth_code'."""
        ph_path = os.path.join(
            os.path.dirname(__file__), "..",
            "assistant", "actions", "pending_handlers.py"
        )
        with open(ph_path, "r", encoding="utf-8") as f:
            source = f.read()

        has_app_idx = source.find('if step == "has_app":')
        has_app_block = source[has_app_idx:has_app_idx + 2000]

        cred_skip_region = has_app_block[:has_app_block.find("is_yes")]
        self.assertIn('"auth_code"', cred_skip_region,
                      "Credential-exists skip must set step to 'auth_code'")


class TestBug8AppNotReadyRetryable(unittest.TestCase):
    """Bug 11: After poll exhaustion with APP_NOT_READY and app confirmed open,
    result must be converted to a retryable error so the retry loop can
    regenerate code with relaxed device checking."""

    def _read_ce_source(self):
        ce_path = os.path.join(
            os.path.dirname(__file__), "..",
            "assistant", "code_executor", "orchestrator.py"
        )
        with open(ce_path, "r", encoding="utf-8") as f:
            return f.read()

    def test_app_opened_flag_exists(self):
        """The poll block must track whether open_app succeeded."""
        src = self._read_ce_source()
        self.assertIn("_app_opened = False", src)
        self.assertIn("_app_opened = True", src)

    def test_conversion_gated_on_app_opened(self):
        """Conversion to retryable error only happens when app confirmed open."""
        src = self._read_ce_source()
        self.assertIn("if _app_opened and", src)

    def test_converted_result_starts_with_error(self):
        """Converted result must start with 'Error:' to trigger _needs_retry."""
        src = self._read_ce_source()
        idx = src.find("APP_NOT_READY→retryable")
        self.assertGreater(idx, 0, "Retryable conversion log line must exist")
        block = src[idx - 1000:idx]
        self.assertIn('f"Error:', block,
                      "Converted result must start with 'Error:' for _needs_retry")

    def test_hint_mentions_any_available_device(self):
        """The retryable error must hint to use any available device/session/item."""
        src = self._read_ce_source()
        idx = src.find("APP_NOT_READY→retryable")
        block = src[idx - 500:idx]
        # Wording was broadened from "ANY available device" to "any available item"
        # to also cover sessions/instances. Match the broader hint, case-insensitive.
        self.assertIn("any available", block.lower(),
                      "Retryable hint must mention using any available device/session/item")

    def test_needs_retry_catches_error_prefix(self):
        """Sanity: _needs_retry returns True for 'Error:' prefix."""
        from assistant.code_executor import _needs_retry
        self.assertTrue(_needs_retry("Error: app is running but no active device"))


class TestBug9OAuthErrorDetail(unittest.TestCase):
    """Bug 12: OAuth exchange_code_for_tokens must return error detail
    so pending_handlers can give actionable advice."""

    def test_exchange_returns_tuple(self):
        """Return type must be tuple[bool, str]."""
        import inspect
        from assistant import oauth_helper
        sig = inspect.signature(oauth_helper.exchange_code_for_tokens)
        src_path = os.path.join(
            os.path.dirname(__file__), "..",
            "assistant", "oauth_helper.py"
        )
        with open(src_path, "r", encoding="utf-8") as f:
            src = f.read()
        self.assertIn("tuple[bool, str]", src,
                      "Return annotation must be tuple[bool, str]")

    def test_response_body_parsed_on_failure(self):
        """On non-200, response body must be parsed for error detail."""
        src_path = os.path.join(
            os.path.dirname(__file__), "..",
            "assistant", "oauth_helper.py"
        )
        with open(src_path, "r", encoding="utf-8") as f:
            src = f.read()
        fn_start = src.find("def exchange_code_for_tokens")
        fn_block = src[fn_start:fn_start + 2000]
        self.assertIn("response.json()", fn_block)
        self.assertIn('.get("error"', fn_block)
        self.assertIn('.get("error_description"', fn_block)

    def test_pending_handler_uses_error_detail(self):
        """pending_handlers must destructure (success, error_detail)."""
        ph_path = os.path.join(
            os.path.dirname(__file__), "..",
            "assistant", "actions", "pending_handlers.py"
        )
        with open(ph_path, "r", encoding="utf-8") as f:
            src = f.read()
        self.assertIn("success, error_detail", src)

    def test_redirect_uri_mismatch_keeps_pending(self):
        """On redirect_uri_mismatch, pending state stays at auth_code step."""
        ph_path = os.path.join(
            os.path.dirname(__file__), "..",
            "assistant", "actions", "pending_handlers.py"
        )
        with open(ph_path, "r", encoding="utf-8") as f:
            src = f.read()
        idx = src.find('redirect_uri_mismatch')
        self.assertGreater(idx, 0)
        block = src[idx:idx + 300]
        self.assertIn('"auth_code"', block,
                      "redirect_uri_mismatch must keep step at auth_code for retry")

    def test_invalid_client_clears_pending(self):
        """On invalid_client, pending state must be cleared."""
        ph_path = os.path.join(
            os.path.dirname(__file__), "..",
            "assistant", "actions", "pending_handlers.py"
        )
        with open(ph_path, "r", encoding="utf-8") as f:
            src = f.read()
        idx = src.find('invalid_client')
        self.assertGreater(idx, 0)
        block = src[idx:idx + 300]
        self.assertIn(".clear()", block,
                      "invalid_client must clear pending state")


class TestStructuralFailureDetection(unittest.TestCase):
    """Structural fix: failure detection must be consistent across all 4 systems."""

    def test_needs_retry_catches_unexpected_error(self):
        """LLM-generated code wraps errors as 'An unexpected error occurred: ...'"""
        from assistant.code_executor import _needs_retry
        self.assertTrue(_needs_retry("An unexpected error occurred: Connection timed out"))

    def test_needs_retry_catches_app_not_ready(self):
        """APP_NOT_READY must be retryable inside the retry loop."""
        from assistant.code_executor import _needs_retry
        self.assertTrue(_needs_retry("APP_NOT_READY|spotify"))

    def test_needs_retry_still_skips_oauth(self):
        """NEEDS_OAUTH is NOT retryable — it triggers the setup flow."""
        from assistant.code_executor import _needs_retry
        self.assertFalse(_needs_retry("NEEDS_OAUTH|gmail|..."))

    def test_step_failed_catches_unexpected_error(self):
        """Planner _step_failed must detect LLM-wrapped errors."""
        from assistant.actions.planner.planner import _step_failed
        self.assertTrue(_step_failed("An unexpected error occurred: timed out"))

    def test_step_failed_catches_timeout_phrases(self):
        """Planner _step_failed must detect timeout-related failures."""
        from assistant.actions.planner.planner import _step_failed
        self.assertTrue(_step_failed("connection timed out"))
        self.assertTrue(_step_failed("read timed out"))

    def test_step_failed_catches_app_not_ready(self):
        """Planner _step_failed must detect APP_NOT_READY sentinel."""
        from assistant.actions.planner.planner import _step_failed
        self.assertTrue(_step_failed("APP_NOT_READY|spotify"))

    def test_template_save_uses_needs_retry(self):
        """Template save guard must use _needs_retry, not ad-hoc prefix checks."""
        src_path = os.path.join(
            os.path.dirname(__file__), "..",
            "assistant", "code_executor", "orchestrator.py"
        )
        with open(src_path, "r", encoding="utf-8") as f:
            src = f.read()
        save_idx = src.find("_save_template(slug, code, goal=goal")
        self.assertGreater(save_idx, 0)
        guard_block = src[save_idx - 200:save_idx]
        self.assertIn("_needs_retry(result)", guard_block,
                      "Template save must use _needs_retry for consistent failure detection")
        self.assertNotIn('result.startswith("APP_NOT_READY', guard_block,
                         "Template save must NOT use ad-hoc APP_NOT_READY check")

    def test_knowledge_proposal_not_in_tts(self):
        """Knowledge proposal must NOT be appended to spoken synthesis output."""
        src_path = os.path.join(
            os.path.dirname(__file__), "..",
            "assistant", "code_executor", "orchestrator.py"
        )
        with open(src_path, "r", encoding="utf-8") as f:
            src = f.read()
        synth_section = src[src.find("Synthesize result for spoken output"):]
        synth_section = synth_section[:1000]
        self.assertNotIn("_knowledge_proposal", synth_section,
                         "Knowledge proposal must not be appended to TTS output")

    def test_invalid_client_clears_credentials(self):
        """On invalid_client, stale credentials must be deleted from credential_store."""
        ph_path = os.path.join(
            os.path.dirname(__file__), "..",
            "assistant", "actions", "pending_handlers.py"
        )
        with open(ph_path, "r", encoding="utf-8") as f:
            src = f.read()
        idx = src.find('invalid_client')
        self.assertGreater(idx, 0)
        block = src[idx:idx + 400]
        self.assertIn("delete_credential", block,
                      "invalid_client must delete stale credentials from credential_store")


# ─── Bug: _step_failed must catch Playwright network errors ───────────────


class TestStepFailedPlaywrightErrors(unittest.TestCase):
    """Planner _step_failed must detect Playwright/Chromium network errors
    that were previously slipping through as 'success'."""

    def test_dns_resolution_error(self):
        from assistant.actions.planner.planner import _step_failed
        self.assertTrue(_step_failed(
            "Error extracting text: Page.goto: net::ERR_NAME_NOT_RESOLVED at https://www.amctheatres.com)"
        ))

    def test_connection_refused(self):
        from assistant.actions.planner.planner import _step_failed
        self.assertTrue(_step_failed(
            "Page.goto: net::ERR_CONNECTION_REFUSED at https://localhost:9999"
        ))

    def test_ssl_error(self):
        from assistant.actions.planner.planner import _step_failed
        self.assertTrue(_step_failed(
            "Page.goto: net::ERR_SSL_PROTOCOL_ERROR at https://expired.badssl.com"
        ))

    def test_generic_net_err(self):
        from assistant.actions.planner.planner import _step_failed
        self.assertTrue(_step_failed(
            "net::ERR_INTERNET_DISCONNECTED"
        ))

    def test_404_page_not_found(self):
        from assistant.actions.planner.planner import _step_failed
        self.assertTrue(_step_failed(
            "ERROR 404: PAGE NOT FOUND\nHOLD UP\nLooks like the page you are looking for cannot be found"
        ))

    def test_error_extracting_text(self):
        from assistant.actions.planner.planner import _step_failed
        self.assertTrue(_step_failed(
            "Error extracting text: Page.goto: net::ERR_CONNECTION_RESET"
        ))

    def test_locator_click_timeout(self):
        from assistant.actions.planner.planner import _step_failed
        self.assertTrue(_step_failed(
            "Navigated to https://www.fandango.com\n"
            "Error running steps: Locator.click: Timeout 10000ms exceeded."
        ))

    def test_overlay_intercepts_pointer(self):
        from assistant.actions.planner.planner import _step_failed
        self.assertTrue(_step_failed(
            "onetrust-pc-dark-filter subtree intercepts pointer events"
        ))

    def test_error_running_steps(self):
        from assistant.actions.planner.planner import _step_failed
        self.assertTrue(_step_failed(
            "Error running steps: some playwright error"
        ))

    def test_normal_output_still_passes(self):
        from assistant.actions.planner.planner import _step_failed
        self.assertFalse(_step_failed("Navigated to https://www.fandango.com\nWaited for 2000ms"))
        self.assertFalse(_step_failed("Found 3 showtimes for Spider-Man at PVR Phoenix"))


if __name__ == "__main__":
    unittest.main()
