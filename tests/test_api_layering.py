"""io/api may reach core and config only."""
import ast
import pathlib

FORBIDDEN = ("assistant.storage", "assistant.actions", "assistant.automation",
             "assistant.llm", "assistant.main")


def _imported_modules(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module)
        elif isinstance(node, ast.ImportFrom) and node.level:
            # relative: ".." from io/api/x.py is assistant.io; "..." is assistant
            found.add(f"relative:{node.level}:{node.module or ''}")
    return found


def test_no_module_under_io_api_imports_a_forbidden_package():
    paths = list(pathlib.Path("assistant/io/api").rglob("*.py"))
    # A relative "assistant/io/api" resolves against the process's cwd, not
    # this file's location -- run from anywhere else and rglob() silently
    # walks zero files, and every assertion below passes vacuously whether or
    # not a forbidden import actually exists. This is the guard the whole
    # test exists to be: without it, a broken layering contract and a broken
    # working directory look identical -- both are a clean pytest run.
    assert paths, "no files found under assistant/io/api -- run pytest from the repo root"
    for path in paths:
        for module in _imported_modules(path):
            for banned in FORBIDDEN:
                assert not module.startswith(banned), f"{path}: imports {module}"


def test_no_module_under_io_api_climbs_out_relatively():
    """`from ...actions import x` is the same violation spelled differently."""
    paths = list(pathlib.Path("assistant/io/api").rglob("*.py"))
    assert paths, "no files found under assistant/io/api -- run pytest from the repo root"
    for path in paths:
        source = path.read_text(encoding="utf-8")
        for banned in ("actions", "storage", "automation", "llm"):
            assert f"from ...{banned}" not in source, f"{path}: relative import of {banned}"
            assert f"from ....{banned}" not in source, f"{path}: relative import of {banned}"


def test_pyproject_declares_the_contract():
    text = pathlib.Path("pyproject.toml").read_text(encoding="utf-8")
    assert "assistant.io.api" in text, "the io.api contract is missing from pyproject.toml"
