"""
test_computer_task_integration.py — Live integration tests for computer_task Polish.

These tests actually open apps, click buttons, and call LLM APIs.
Run with: python test_computer_task_integration.py

Tests:
1. Terminator API: Open Calculator, click 7×5=, read result
2. Prompt construction: Verify "ALREADY OPEN" hint when app is running
3. Routing detection: Verify detect_backend returns correct backends
4. Intent classification: Verify "open settings" ->computer_task (needs API key)
"""

import asyncio
import sys
import os
import time
import logging

# Add project root to path so we can import assistant modules
sys.path.insert(0, os.path.dirname(__file__))

# Load .env
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
logger = logging.getLogger("test_computer_task_integration")

PASS = 0
FAIL = 0
SKIP = 0


def report(name, passed, detail=""):
    global PASS, FAIL
    status = "PASS" if passed else "FAIL"
    if passed:
        PASS += 1
    else:
        FAIL += 1
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))


def skip(name, reason=""):
    global SKIP
    SKIP += 1
    print(f"  [SKIP] {name}" + (f" — {reason}" if reason else ""))


# ─── Test 1: Terminator Calculator ─────────────────────────────────────────

async def test_terminator_calculator():
    """Open Calculator, click 7 × 5 =, read result."""
    print("\n--- Test 1: Terminator Calculator (7 × 5 = 35) ---")

    try:
        import terminator
    except ImportError:
        skip("Terminator import", "terminator not installed")
        return

    desktop = terminator.Desktop()

    # Open Calculator
    try:
        desktop.open_application("calc")
        await asyncio.sleep(2.0)
        report("Open Calculator", True)
    except Exception as e:
        report("Open Calculator", False, str(e))
        return

    # Wait for Calculator window
    try:
        calc = await desktop.locator("name:Calculator").first()
        report("Find Calculator window", True, f"name={calc.name()}")
    except Exception as e:
        report("Find Calculator window", False, str(e))
        return

    # Click Clear first to ensure clean state
    try:
        clear_btn = await desktop.locator("name:Clear").first()
        clear_btn.click()
        await asyncio.sleep(0.3)
    except Exception:
        pass  # Clear button may not exist if already 0

    # Click 7
    try:
        seven = await desktop.locator("name:Seven").first()
        seven.click()
        await asyncio.sleep(0.3)
        report("Click Seven", True)
    except Exception as e:
        report("Click Seven", False, str(e))
        return

    # Click Multiply
    try:
        multiply = await desktop.locator("name:Multiply by").first()
        multiply.click()
        await asyncio.sleep(0.3)
        report("Click Multiply", True)
    except Exception as e:
        report("Click Multiply", False, str(e))
        return

    # Click 5
    try:
        five = await desktop.locator("name:Five").first()
        five.click()
        await asyncio.sleep(0.3)
        report("Click Five", True)
    except Exception as e:
        report("Click Five", False, str(e))
        return

    # Click Equals
    try:
        equals = await desktop.locator("name:Equals").first()
        equals.click()
        await asyncio.sleep(0.5)
        report("Click Equals", True)
    except Exception as e:
        report("Click Equals", False, str(e))
        return

    # Read result via app_automation.get_text with name: selector
    # (Terminator doesn't support automationid:, uses tree search fallback)
    try:
        from assistant.automation import native as app_automation
        app_automation._backend = "terminator"
        app_automation._desktop = desktop
        result_text = await app_automation.get_text("name:Display is 35", "Calculator")
        has_35 = "35" in result_text
        report("Read result via name: selector", has_35, f"got: '{result_text}'")
    except Exception as e:
        report("Read result via name: selector", False, str(e))

    # Also test get_text() from app_automation module
    try:
        from assistant.automation import native as app_automation
        app_automation._backend = "terminator"
        app_automation._desktop = desktop
        result = await app_automation.get_text("automationid:CalculatorResults", "Calculator")
        has_35 = "35" in result
        report("app_automation.get_text()", has_35, f"got: '{result}'")
    except Exception as e:
        report("app_automation.get_text()", False, str(e))

    # Close Calculator
    try:
        import pyautogui
        calc_elem = await desktop.locator("name:Calculator").first()
        calc_elem.click()
        pyautogui.hotkey("alt", "F4")
        await asyncio.sleep(0.5)
        report("Close Calculator", True)
    except Exception as e:
        report("Close Calculator", False, str(e))


# ─── Test 2: list_elements via get_window_tree ─────────────────────────────

async def test_list_elements():
    """Open Calculator, call list_elements, verify it returns button names."""
    print("\n--- Test 2: list_elements (Terminator tree walk) ---")

    try:
        import terminator
    except ImportError:
        skip("list_elements", "terminator not installed")
        return

    desktop = terminator.Desktop()

    # Open Calculator
    try:
        desktop.open_application("calc")
        await asyncio.sleep(2.0)
    except Exception as e:
        skip("list_elements", f"Can't open calc: {e}")
        return

    try:
        from assistant.automation import native as app_automation
        app_automation._backend = "terminator"
        app_automation._desktop = desktop

        elements = await app_automation.list_elements("Calculator")
        has_content = len(elements) > 50 and "Error" not in elements
        report("list_elements returns content", has_content, f"{len(elements)} chars")

        has_buttons = "Button" in elements
        report("list_elements shows buttons", has_buttons)

        # Check if it shows element names useful for the LLM
        has_names = any(word in elements for word in ["Seven", "Plus", "Equals", "Number pad"])
        report("list_elements shows usable names", has_names,
               elements[:300] if has_names else "no known button names found")
    except Exception as e:
        report("list_elements", False, str(e))

    # Close
    try:
        import pyautogui
        calc_elem = await desktop.locator("name:Calculator").first()
        calc_elem.click()
        pyautogui.hotkey("alt", "F4")
        await asyncio.sleep(0.5)
    except Exception:
        pass


# ─── Test 3: Routing Detection ─────────────────────────────────────────────

async def test_routing_detection():
    """Test detect_backend returns correct backends for various goals."""
    print("\n--- Test 3: Routing Detection ---")

    from assistant.automation.router import detect_backend

    # URL ->browser
    backend, meta = detect_backend("go to google.com")
    report("URL ->browser", backend == "browser", f"backend={backend}, meta={meta}")

    # Open app ->native
    backend, meta = detect_backend("open calculator")
    report("'open calculator' ->native", backend == "native", f"backend={backend}, meta={meta}")

    # Open settings ->native
    backend, meta = detect_backend("open settings")
    report("'open settings' ->native", backend == "native", f"backend={backend}, meta={meta}")

    # Visit website ->browser
    backend, meta = detect_backend("visit amazon.com and search for keyboards")
    report("'visit amazon' ->browser", backend == "browser", f"backend={backend}, meta={meta}")

    # Browse URL ->browser
    backend, meta = detect_backend("browse https://github.com")
    report("'browse https://...' ->browser", backend == "browser", f"backend={backend}, meta={meta}")

    # Fill form ->browser
    backend, meta = detect_backend("fill out the form on the registration page")
    report("'fill out form' ->browser", backend == "browser", f"backend={backend}, meta={meta}")


# ─── Test 4: Prompt Construction (mock LLM) ────────────────────────────────

async def test_prompt_construction():
    """Verify _execute_native_task injects 'ALREADY OPEN' when app is running."""
    print("\n--- Test 4: Prompt Construction (mock LLM) ---")

    from assistant.automation import router as desktop_automation

    captured_prompts = []

    def mock_llm(prompt, task_type=None):
        captured_prompts.append(prompt)
        # Return a minimal valid step array
        return '[{"action": "get_text", "params": {"selector": "name:Display", "window": "Calculator"}}]'

    # Open Calculator so it's detected as running
    try:
        import terminator
        desktop = terminator.Desktop()
        desktop.open_application("calc")
        await asyncio.sleep(2.0)
    except Exception as e:
        skip("Prompt construction", f"Can't open calc: {e}")
        return

    try:
        # Call _execute_native_task with a goal that references calculator
        result = await desktop_automation._execute_native_task(
            "compute 7 times 5 in calculator", mock_llm
        )

        if captured_prompts:
            prompt = captured_prompts[0]
            has_already_open = "ALREADY OPEN" in prompt
            report("Prompt contains 'ALREADY OPEN'", has_already_open,
                   prompt[:200] if not has_already_open else "")

            has_no_open_step = "Do NOT include an 'open' step" in prompt
            report("Prompt says no 'open' step", has_no_open_step)

            has_elements = "role:" in prompt or "name:" in prompt or "Button" in prompt
            report("Prompt contains available elements", has_elements)
        else:
            # The simple_match shortcut may have fired
            report("LLM was called (not shortcut path)", False,
                   "Goal may have matched simple 'open X' pattern — try a more complex goal")
    except Exception as e:
        report("Prompt construction", False, str(e))

    # Close Calculator
    try:
        import pyautogui
        calc_elem = await desktop.locator("name:Calculator").first()
        calc_elem.click()
        pyautogui.hotkey("alt", "F4")
        await asyncio.sleep(0.5)
    except Exception:
        pass


# ─── Test 5: Intent Classification (needs API key) ─────────────────────────

async def test_intent_classification():
    """Test that 'open settings' routes to computer_task, not find_and_click."""
    print("\n--- Test 5: Intent Classification (needs Groq API key) ---")

    groq_key = os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API_KEY_1")
    if not groq_key:
        skip("Intent classification", "No GROQ_API_KEY in .env")
        return

    try:
        from assistant import intent, config
        # Test "open settings"
        result = await intent.detect_intent("open settings")
        intent_name = result.get("intent") if isinstance(result, dict) else getattr(result, "intent", None)
        is_computer_task = intent_name == "computer_task"
        report("'open settings' ->computer_task", is_computer_task, f"got intent={intent_name}")

        # Test "click the submit button"
        result2 = await intent.detect_intent("click the submit button")
        intent_name2 = result2.get("intent") if isinstance(result2, dict) else getattr(result2, "intent", None)
        is_find_click = intent_name2 == "find_and_click"
        report("'click submit button' ->find_and_click", is_find_click, f"got intent={intent_name2}")

    except Exception as e:
        report("Intent classification", False, str(e))


# ─── Main ──────────────────────────────────────────────────────────────────

async def main():
    print("=" * 60)
    print("computer_task Polish — Live Integration Tests")
    print("=" * 60)

    await test_terminator_calculator()
    await test_list_elements()
    await test_routing_detection()
    await test_prompt_construction()
    await test_intent_classification()

    print("\n" + "=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed, {SKIP} skipped")
    print("=" * 60)

    if FAIL > 0:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
