"""The intent policy layer after milestone 6a.5 removed its deny-list.

This file used to verify that `DANGEROUS_PATTERNS` used word boundaries, so
that "form with" no longer matched `rm `. That list is gone; `policy.py`'s
module docstring records why at length. The short version: it judged a string
`main.py` overwrote immediately afterwards, it was evadable with a single
Cyrillic character, and it refused ordinary English.

Two things are pinned here now. First, that the deny-list stays gone --
"add a pattern" is a tempting answer to the next scare and it is the wrong
one. Second, and far more important, **that removing it did not remove the
protection**: the goals the old list refused are still refused, by the
capability gate, which a caller cannot evade by rephrasing.

The old file was a `run()` script with no test functions, so pytest collected
nothing from it. These are real tests.

Run: py -3.11 -m pytest tests/test_policy_patterns.py -v
"""

import asyncio

import pytest

from assistant import actions, config, policy
from assistant.core.capabilities import Capability
from assistant.intent import IntentResult


def _evaluate(goal: str, intent: str = "computer_task"):
    return policy.evaluate(
        IntentResult(intent=intent, response="", params={"goal": goal})
    )


# ─── The deny-list is gone, and must stay gone ───────────────────────────

def test_the_dangerous_pattern_list_no_longer_exists():
    """Reintroducing it would restore all three of its defects at once."""
    assert not hasattr(config, "DANGEROUS_PATTERNS")


def test_policy_no_longer_scans_parameters_for_words():
    """The scan judged `params["goal"]`, which main.py overwrites with the raw
    transcription for six intents afterwards -- so the string it approved was
    not the string that ran."""
    import inspect
    src = inspect.getsource(policy.evaluate)
    assert "_DANGEROUS_REGEXES" not in src


# ─── Ordinary English is no longer refused ───────────────────────────────

@pytest.mark.parametrize("goal", [
    "format this as a table",
    "restart the music",
    "kill the timer",
    "what command should I run",
    "the root of the problem",
    "terminate my subscription",
    "fill out the form with random test data",
    "run it in the shell",
    "warm up the model",
    "show me a skillet recipe",
])
def test_ordinary_english_is_allowed(goal):
    """Every one of these was refused by the old list. Three live tests during
    6a.5 were eaten before reaching what they meant to test, and a refusal a
    user hits daily teaches them to rephrase until something works -- which is
    the opposite of a control."""
    assert _evaluate(goal).allowed, goal


# ─── The protection did not go with it ───────────────────────────────────

@pytest.mark.parametrize("goal", [
    "rm -rf /",
    "sudo apt install foo",
    "format c:",
    "shutdown -h now",
    "taskkill /im chrome.exe",
    "mkfs.ext4 /dev/sda1",
])
def test_the_goals_the_old_list_refused_are_still_refused(goal):
    """This is the test that matters. The old list refused these by spelling,
    which a Cyrillic character defeated. They are refused now by capability --
    `computer_task` requires EXECUTE, which no transport carries -- and a
    caller cannot rephrase their way past that."""
    assert _evaluate(goal).allowed, "policy is no longer the layer that decides"

    token = actions.set_grants(frozenset({Capability.CHAT_SEND}))
    try:
        result = asyncio.run(actions.execute("computer_task", {"goal": goal}, ""))
    finally:
        actions.current_grants.reset(token)
    assert "permission" in result.lower(), goal


# ─── What the layer still does, and does correctly ───────────────────────

def test_an_intent_off_the_whitelist_is_refused():
    """The whitelist is a positive check -- "is this one of the things we
    permit" -- which is a question a caller cannot argue with. It survives."""
    assert not _evaluate("anything", intent="not_a_real_intent").allowed


def test_every_configured_intent_passes_the_whitelist():
    """The converse: the whitelist must not have drifted from config.INTENTS."""
    for intent in config.INTENTS:
        assert intent in config.ALLOWED_INTENTS, intent


def test_a_non_http_url_is_still_refused():
    res = policy.evaluate(IntentResult(intent="open_browser", response="",
                                       params={"url": "javascript:alert(1)"}))
    assert not res.allowed


def test_an_http_url_is_still_allowed():
    res = policy.evaluate(IntentResult(intent="open_browser", response="",
                                       params={"url": "https://example.com"}))
    assert res.allowed


# ─── The path check, with the three bugs a review found ──────────────────

def test_a_sibling_directory_sharing_the_prefix_is_refused():
    """`startswith` is not ancestry: a sandbox at `.../sandbox` accepted
    `.../sandbox_evil` because the string matched."""
    assert not policy._validate_path("../sandbox_evil/loot.txt")


def test_a_drive_relative_path_is_refused():
    """`C:evil.txt` has no separator, so the old "simple filename" test
    admitted it -- and it resolves against that drive's current directory."""
    assert not policy._validate_path("C:evil.txt")


def test_an_alternate_data_stream_is_refused():
    """`note.txt:hidden` is an NTFS ADS, not a filename."""
    assert not policy._validate_path("note.txt:hidden")


def test_a_nul_byte_is_refused():
    """Windows truncates at NUL, so `a.txt\\x00.png` writes `a.txt`."""
    assert not policy._validate_path("a.txt\x00.png")


def test_traversal_out_of_the_sandbox_is_refused():
    assert not policy._validate_path("../../Windows/System32/drivers/etc/hosts")


def test_an_ordinary_filename_is_still_allowed():
    """The repair must not refuse the thing the check exists to permit."""
    assert policy._validate_path("notes.txt")


def test_a_subdirectory_inside_the_sandbox_is_still_allowed():
    assert policy._validate_path("subdir/notes.txt")


def test_an_empty_path_is_refused():
    assert not policy._validate_path("")
