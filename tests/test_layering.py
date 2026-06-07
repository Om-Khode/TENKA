# tests/test_layering.py
import subprocess


def test_lint_imports_passes():
    result = subprocess.run(
        ["lint-imports"], capture_output=True, text=True, cwd="."
    )
    assert result.returncode == 0, f"import-linter failed:\n{result.stdout}\n{result.stderr}"
