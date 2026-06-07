"""Tests for S12: vision.py → automation/vision/ package split.

Verifies:
  - Package structure: all submodules importable
  - Backward compat: attribute access via package __getattr__
  - No circular imports
  - Re-exported symbols match original public surface
  - Submodule isolation: todo_classifier and _parsing have no project deps
"""

import importlib
import sys
import types


# --- Package structure ---

def test_vision_is_package():
    import assistant.automation.vision as v
    assert hasattr(v, "__path__"), "vision should be a package (directory), not a module"


def test_submodules_importable():
    import assistant.automation.vision.agent
    import assistant.automation.vision.verifier
    import assistant.automation.vision.todo_classifier
    import assistant.automation.vision._parsing


# --- Backward compat via __getattr__ ---

def test_run_computer_task_accessible():
    import assistant.automation.vision as v
    assert callable(v.run_computer_task)


def test_agent_typing_flag_accessible():
    import assistant.automation.vision as v
    val = v._agent_typing
    assert isinstance(val, bool)


def test_task_state_accessible():
    import assistant.automation.vision as v
    ts = v._task_state
    assert hasattr(ts, "reset")
    assert hasattr(ts, "todo_list")
    assert hasattr(ts, "all_todos_done")


def test_task_state_class_accessible():
    import assistant.automation.vision as v
    assert hasattr(v, "_TaskState")
    assert v._TaskState.TODO_MAX == 15


def test_recover_truncated_json_accessible():
    import assistant.automation.vision as v
    assert callable(v._recover_truncated_json)
    assert v._recover_truncated_json('{"a": 1') == '{"a": 1}'  # already balanced? no
    assert v._recover_truncated_json('{"a": 1') == '{"a": 1}'  # test idempotence
    # Actually test truncation recovery
    result = v._recover_truncated_json('{"a": "hello')
    assert result.endswith('}')


def test_parse_plan_accessible():
    import assistant.automation.vision as v
    assert callable(v._parse_plan)
    result = v._parse_plan('{"achieved": true}')
    assert result == {"achieved": True}


def test_is_yes_answer_accessible():
    import assistant.automation.vision as v
    assert callable(v._is_yes_answer)
    assert v._is_yes_answer("YES") is True
    assert v._is_yes_answer("NO") is False


def test_action_failed_accessible():
    import assistant.automation.vision as v
    assert callable(v._action_failed)


def test_texts_overlap_accessible():
    import assistant.automation.vision as v
    assert callable(v._texts_overlap)


def test_snap_to_ocr_accessible():
    import assistant.automation.vision as v
    assert callable(v._snap_to_ocr)


def test_constants_accessible():
    import assistant.automation.vision as v
    assert isinstance(v.MAX_STEPS, int)
    assert isinstance(v.MAX_LOOPS, int)
    assert isinstance(v.SAFE_MODE, bool)


# --- Submodule isolation ---

def test_todo_classifier_no_project_imports():
    """todo_classifier.py should only import `re` — no project dependencies."""
    from assistant.automation.vision import todo_classifier
    source_file = todo_classifier.__file__
    with open(source_file, "r", encoding="utf-8") as f:
        content = f.read()
    # Should not import from assistant.*
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("from assistant") or stripped.startswith("import assistant"):
            raise AssertionError(f"todo_classifier has project import: {stripped}")


def test_parsing_only_core_imports():
    """_parsing.py may import from assistant.core (foundation layer) but nothing else."""
    from assistant.automation.vision import _parsing
    source_file = _parsing.__file__
    with open(source_file, "r", encoding="utf-8") as f:
        content = f.read()
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith(("from assistant", "import assistant")):
            if "assistant.core" not in stripped and "core.json_utils" not in stripped:
                raise AssertionError(f"_parsing has non-core project import: {stripped}")


# --- TODO classifier functionality ---

def test_classify_todo_type():
    from assistant.automation.vision.todo_classifier import _classify_todo
    result = _classify_todo("Type 'John' in First Name")
    assert result["kind"] == "type"
    assert result["value"] == "John"
    assert result["field"] == "First Name"


def test_classify_todo_select():
    from assistant.automation.vision.todo_classifier import _classify_todo
    result = _classify_todo("Select 'Male' from Gender dropdown")
    assert result["kind"] == "select"
    assert result["value"] == "Male"
    assert "Gender" in result["field"]


def test_classify_todo_click():
    from assistant.automation.vision.todo_classifier import _classify_todo
    result = _classify_todo("Click the Submit button")
    assert result["kind"] == "click"
    assert result["target"] == "Submit"


def test_classify_todo_other():
    from assistant.automation.vision.todo_classifier import _classify_todo
    result = _classify_todo("Navigate to the settings page")
    assert result["kind"] == "other"


def test_make_todo_dict():
    from assistant.automation.vision.todo_classifier import _make_todo_dict
    d = _make_todo_dict(1, "Type 'hello' in Name")
    assert d["id"] == 1
    assert d["task"] == "Type 'hello' in Name"
    assert d["done"] is False
    assert d["kind"] == "type"
    assert d["value"] == "hello"
    assert d["field"] == "Name"
    assert d["pending_visual_confirm"] is False
    assert d["confirm_abandoned"] is False


# --- Verifier functionality ---

def test_is_yes_answer():
    from assistant.automation.vision.verifier import _is_yes_answer
    assert _is_yes_answer("YES") is True
    assert _is_yes_answer("Yes.") is True
    assert _is_yes_answer("Yes — the field shows the value") is True
    assert _is_yes_answer("Y") is True
    assert _is_yes_answer("NO") is False
    assert _is_yes_answer("No") is False
    assert _is_yes_answer("") is False
    assert _is_yes_answer(None) is False


def test_quick_verify_returns_none_for_non_music():
    from assistant.automation.vision.verifier import _quick_verify_from_window_title
    result = _quick_verify_from_window_title("open notepad", "Untitled - Notepad")
    assert result is None


def test_quick_verify_detects_music_in_title():
    from assistant.automation.vision.verifier import _quick_verify_from_window_title
    result = _quick_verify_from_window_title("play perfect by ed sheeran", "Ed Sheeran - Perfect")
    assert result is True


# --- Parsing functionality ---

def test_parse_plan_basic():
    from assistant.automation.vision._parsing import _parse_plan
    result = _parse_plan('{"actions": [{"type": "click"}], "done": false}')
    assert result is not None
    assert "actions" in result


def test_parse_plan_code_fence():
    from assistant.automation.vision._parsing import _parse_plan
    result = _parse_plan('```json\n{"done": true}\n```')
    assert result == {"done": True}


def test_parse_plan_truncated():
    from assistant.automation.vision._parsing import _parse_plan
    result = _parse_plan('{"thinking": "need to click')
    assert result is not None
    assert "thinking" in result


def test_recover_truncated_json():
    from assistant.automation.vision._parsing import _recover_truncated_json
    assert _recover_truncated_json('{"a": 1}') == '{"a": 1}'
    recovered = _recover_truncated_json('{"a": "hello')
    assert '"hello"' in recovered
    assert recovered.endswith("}")


# --- No circular imports ---

def test_no_circular_imports():
    """Fresh import of each submodule should succeed without ImportError."""
    mods_to_check = [
        "assistant.automation.vision",
        "assistant.automation.vision.todo_classifier",
        "assistant.automation.vision._parsing",
        "assistant.automation.vision.verifier",
        "assistant.automation.vision.agent",
    ]
    for mod_name in mods_to_check:
        if mod_name in sys.modules:
            continue
        try:
            importlib.import_module(mod_name)
        except ImportError as e:
            raise AssertionError(f"Circular import detected in {mod_name}: {e}")
