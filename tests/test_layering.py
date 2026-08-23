# tests/test_layering.py
"""`import-linter` reports the layering contracts as kept.

Judged by the summary line, not the exit code. `import-linter`'s exit code is
unreliable in this project -- the pre-commit hook has read the
`Contracts: N kept, M broken.` line since the day a contract edit sailed
through a zero exit -- and a test that trusts the code while the hook does not
is a test that disagrees with the thing actually gating commits.

Fail-closed on a missing summary, exactly as the hook does: no summary means the
linter crashed, and a crash is not a pass.
"""
import re
import subprocess

_SUMMARY = re.compile(r"Contracts:\s*(\d+)\s*kept,\s*(\d+)\s*broken\.")


def _run():
    return subprocess.run(
        ["lint-imports"], capture_output=True, text=True, cwd="."
    )


def test_every_contract_is_kept():
    result = _run()
    combined = f"{result.stdout}\n{result.stderr}"
    match = _SUMMARY.search(combined)
    assert match, (
        "import-linter printed no `Contracts: N kept, M broken.` line, so it "
        f"crashed rather than reporting. Exit code was {result.returncode}, "
        f"which is exactly what must not be trusted here.\n{combined}"
    )
    kept, broken = int(match.group(1)), int(match.group(2))
    assert broken == 0, f"{broken} contract(s) broken:\n{combined}"
    assert kept > 0, "the linter reported zero contracts -- nothing is enforced"


def test_the_contracts_the_tree_relies_on_are_present():
    """A contract deleted from `pyproject.toml` makes the run above greener,
    not safer. These four are the ones other tests and rules argue from, so
    their disappearance has to be loud."""
    import pathlib

    text = (pathlib.Path(__file__).resolve().parent.parent
            / "pyproject.toml").read_text(encoding="utf-8")
    for name in (
        "IO never imports actions",
        "IO never imports main",
        "Brain never imports io directly",
        "Automation never imports actions or brain",
    ):
        assert f'name = "{name}"' in text, (
            f"the `{name}` contract is gone from pyproject.toml"
        )
