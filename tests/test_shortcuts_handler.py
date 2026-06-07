"""Tests for P7 fix: shortcuts handler variable shadowing bug.

The bug: `shortcuts = shortcuts.list_shortcuts()` reassigned the module
import to a list, causing AttributeError on subsequent `shortcuts.delete_shortcut()`
calls — silently swallowed by the outer try/except.
"""

import ast


def test_no_variable_shadows_module_import():
    """P7 regression: no assignment to 'shortcuts' that shadows the module."""
    import assistant.actions.shortcuts as mod

    source = open(mod.__file__).read()
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "shortcuts":
                    if isinstance(node.value, ast.Call):
                        func = node.value.func
                        if isinstance(func, ast.Attribute) and func.attr == "list_shortcuts":
                            raise AssertionError(
                                f"Line {node.lineno}: `shortcuts = shortcuts.list_shortcuts()` "
                                "shadows the module import. Use a different variable name."
                            )


def test_list_branch_uses_shortcut_list_variable():
    """The list result must be stored in 'shortcut_list', not 'shortcuts'."""
    import inspect
    from assistant.actions.shortcuts import handle_manage_shortcut

    source = inspect.getsource(handle_manage_shortcut)
    assert "shortcut_list = shortcuts.list_shortcuts()" in source
    assert "shortcuts = shortcuts.list_shortcuts()" not in source
