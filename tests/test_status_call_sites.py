# tests/test_status_call_sites.py
"""Static check: every handler given overlay status surface MUST call
status.set(StatusPhase.IDLE) inside a finally block, AND
status.set(StatusPhase.THINKING) at entry."""
import ast
from pathlib import Path

HANDLERS = [
    ("assistant/actions/da_handlers.py",
     ["handle_computer_task", "handle_find_and_click", "handle_planner",
      "handle_code_executor", "handle_browser_action", "handle_app_action"]),
    ("assistant/actions/manifest_dispatch.py", ["handle_manifest_dispatch"]),
]


def _function_calls_status_set(fn: ast.FunctionDef, phase_name: str) -> bool:
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr == "set":
                if isinstance(node.func.value, ast.Name) and node.func.value.id == "status":
                    for arg in node.args:
                        if isinstance(arg, ast.Attribute) and arg.attr == phase_name:
                            return True
    return False


def _function_has_idle_in_finally(fn: ast.FunctionDef) -> bool:
    for node in ast.walk(fn):
        if isinstance(node, ast.Try):
            for f_node in node.finalbody:
                for sub in ast.walk(f_node):
                    if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute) \
                       and sub.func.attr == "set":
                        for arg in sub.args:
                            if isinstance(arg, ast.Attribute) and arg.attr == "IDLE":
                                return True
    return False


def test_all_handlers_call_status_thinking_at_entry_and_idle_in_finally():
    failures = []
    for path, names in HANDLERS:
        tree = ast.parse(Path(path).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
                if not _function_calls_status_set(node, "THINKING"):
                    failures.append(f"{path}::{node.name} missing status.set(StatusPhase.THINKING)")
                if not _function_has_idle_in_finally(node):
                    failures.append(f"{path}::{node.name} missing status.set(StatusPhase.IDLE) in finally")
    assert not failures, "\n".join(failures)
