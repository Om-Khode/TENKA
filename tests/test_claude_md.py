"""
test_claude_md.py — Guards the gitignored development docs.

CLAUDE.md and ARCHITECTURE.md are NOT tracked in git (.gitignore:74-75), so
nothing else notices when they go missing or rot. These tests are the only
safety net.

Scope is deliberately narrow: presence, and the handful of facts that silently
go stale and then actively mislead (intent list, subpackage list, layering).
Prose is not asserted — a doc test that pins wording just forces docs to
describe whatever the test was written against, which is how the previous
version of this file ended up demanding a module layout that no longer existed.
"""

import os
import re

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
CLAUDE_MD_PATH = os.path.join(REPO_ROOT, "CLAUDE.md")
ARCHITECTURE_MD_PATH = os.path.join(REPO_ROOT, "ARCHITECTURE.md")


@pytest.fixture
def claude_md():
    with open(CLAUDE_MD_PATH, "r", encoding="utf-8") as f:
        return f.read()


@pytest.fixture
def architecture_md():
    with open(ARCHITECTURE_MD_PATH, "r", encoding="utf-8") as f:
        return f.read()


class TestDocsExist:
    def test_claude_md_exists(self):
        assert os.path.isfile(CLAUDE_MD_PATH), (
            "CLAUDE.md must exist at repo root. It is gitignored — if it is "
            "missing, rebuild it from the code plus TENKA_Architecture.md."
        )

    def test_architecture_md_exists(self):
        assert os.path.isfile(ARCHITECTURE_MD_PATH), (
            "ARCHITECTURE.md must exist at repo root. It is gitignored — if it "
            "is missing, rebuild it from the code plus TENKA_Architecture.md."
        )

    def test_docs_not_stubs(self):
        assert os.path.getsize(CLAUDE_MD_PATH) > 2000
        assert os.path.getsize(ARCHITECTURE_MD_PATH) > 5000


class TestTheRule:
    """THE-rule is the one non-negotiable. If it's not in CLAUDE.md, the doc is broken."""

    def test_claude_md_states_the_rule(self, claude_md):
        assert "No hardcoded app-specific rules" in claude_md

    def test_architecture_md_states_the_rule(self, architecture_md):
        assert "No hardcoded app-specific rules" in architecture_md


class TestIntentsInSync:
    """The intent list rots the moment an intent is added. Pin it to config.INTENTS."""

    def test_every_intent_is_documented(self, claude_md):
        from assistant.config import INTENTS

        missing = [i for i in INTENTS if i not in claude_md]
        assert not missing, (
            f"Intents in config.INTENTS are absent from CLAUDE.md: {missing}. "
            f"Update the intent list in CLAUDE.md."
        )

    def test_no_phantom_intents_documented(self, claude_md):
        """Catch intents deleted from config but left in the doc."""
        from assistant.config import INTENTS

        # The doc's intent block is the fenced section following the "## Intents" heading.
        block = re.search(r"## Intents.*?```\n(.*?)```", claude_md, re.S)
        assert block, "CLAUDE.md must carry a fenced intent list under '## Intents'"

        documented = {t for t in re.split(r"[,\s]+", block.group(1)) if t}
        phantom = documented - set(INTENTS)
        assert not phantom, (
            f"CLAUDE.md documents intents that no longer exist in config.INTENTS: "
            f"{sorted(phantom)}."
        )


class TestSubpackagesInSync:
    """The nine-subpackage contract is enforced by process rule; keep the doc honest."""

    def test_documented_subpackages_match_disk(self, claude_md):
        import assistant

        pkg_root = os.path.dirname(assistant.__file__)
        on_disk = {
            name for name in os.listdir(pkg_root)
            if os.path.isdir(os.path.join(pkg_root, name))
            and not name.startswith(("_", "."))
        }

        for pkg in on_disk:
            assert f"`{pkg}/`" in claude_md, (
                f"Subpackage '{pkg}/' exists on disk but is not documented in "
                f"CLAUDE.md's approved-subpackage list."
            )


class TestLayeringDocumented:
    def test_io_boundary_stated(self, claude_md):
        assert "io/" in claude_md
        assert "NEVER" in claude_md

    def test_contract_count_matches_pyproject(self, claude_md):
        """CLAUDE.md claims a specific number of import-linter contracts."""
        import tomllib

        with open(os.path.join(REPO_ROOT, "pyproject.toml"), "rb") as f:
            pyproject = tomllib.load(f)

        actual = len(pyproject["tool"]["importlinter"]["contracts"])
        words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
        accepted = {str(actual), words.get(actual, str(actual))}

        assert any(f"{form} contracts" in claude_md for form in accepted), (
            f"pyproject.toml defines {actual} import-linter contracts; CLAUDE.md "
            f"does not say so. Keep the count in sync "
            f"(accepted: {sorted(f'{f} contracts' for f in accepted)})."
        )


class TestGitignoreWarning:
    """The docs were lost once because they are gitignored. Keep the warning."""

    def test_claude_md_notes_it_is_gitignored(self, claude_md):
        assert "gitignored" in claude_md.lower()

    def test_still_actually_gitignored(self):
        """If someone starts tracking these, the warning becomes a lie."""
        with open(os.path.join(REPO_ROOT, ".gitignore"), encoding="utf-8") as f:
            ignore = f.read().splitlines()

        assert "CLAUDE.md" in ignore
        assert "ARCHITECTURE.md" in ignore


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
