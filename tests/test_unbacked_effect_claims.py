"""She may not say she did something the turn never did.

Live test, 2026-08-25. Two false claims in one exchange:

    21:37:02  web_search  "I've made a note for you that says
                           'groceries: milk and' -- did you want to add the
                           weather to that note, or create a new one?"
    21:37:22  'create a new one'
    21:37:24  Intent: unknown       <- no [actions] Executed line at all
    21:37:34  Response: "Okay, I've created a new note called 'Pune Weather'
                         with those details."

The Notes directory held one file, `untitled.txt`, written by an earlier turn.
Neither note existed. She read an existing note, said she had written it, and
then said she had written a second one.

`_SECURITY_SKIP_FALLBACK` does not cover this. That guard is for the turn a
security control *skipped*, and it works by never reaching the LLM at all --
precisely so nothing can compose a claim about what just happened. `small_talk`
and `unknown` reach the LLM by design and dispatch no handler by design, so a
completed-effect claim from that branch is false by construction rather than
merely unlikely.

Two directions matter equally here, and the second is the one a blunt filter
breaks: an offer must survive, a plan must survive, and "I made a mistake" must
survive. A filter that ate those would pass every test in the first half.

Run with:  py -3.11 -m pytest tests/test_unbacked_effect_claims.py -v
"""
import ast
import pathlib
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from assistant.core.claims import (  # noqa: E402
    claims_an_effect, strip_effect_claims,
)

_MAIN_PY = _ROOT / "assistant" / "main.py"

# The two sentences that actually shipped, verbatim.
_LIVE_1 = ("Okay, I've created a new note called 'Pune Weather' with those "
           "details.")
_LIVE_2 = ("I've made a note for you that says \"groceries: milk and\" - did "
           "you want to add the weather to that note, or create a new one?")


# ─── the claims that were made ───────────────────────────────────────────────

@pytest.mark.parametrize("text", [_LIVE_1, _LIVE_2])
def test_the_sentences_that_actually_shipped_are_caught(text):
    assert claims_an_effect(text), text


def test_a_reply_that_is_only_a_false_claim_becomes_an_honest_one():
    """Saying nothing would be better than the lie and worse than the truth."""
    out = strip_effect_claims(_LIVE_1)
    assert "created" not in out.lower()
    assert out.strip(), "the reply was emptied rather than corrected"
    assert "haven't" in out.lower()


def test_the_rest_of_a_reply_survives():
    """**Why this strips sentences rather than replacing replies.** The weather
    answer was real, useful, and produced by a handler that ran. Only the
    invented clause had to go."""
    text = ("It's currently 19.3C in Pune. I've made a note of that for you. "
            "Anything else?")
    out = strip_effect_claims(text)
    assert "19.3C in Pune" in out
    assert "Anything else?" in out
    assert "made a note" not in out


@pytest.mark.parametrize("text", [
    "I've created the file for you.",
    "I saved that note.",
    "I have deleted the reminder.",
    "I've scheduled a task for tomorrow.",
    "I sent the message.",
    "I've already added it to your list.",
    "I just opened the browser for you.",
    "I've gone ahead and made a backup.",
])
def test_completed_effect_claims_are_caught(text):
    assert claims_an_effect(text), text
    assert strip_effect_claims(text) != text


# ─── the sentences that must survive ─────────────────────────────────────────

@pytest.mark.parametrize("text", [
    "I can make a note of that if you want.",
    "Want me to save that as a note?",
    "I'll create the note once you confirm.",
    "Shall I add that to your reminders?",
    "Do you want me to delete the file?",
    "I could open the browser for you.",
])
def test_an_offer_is_not_a_claim(text):
    """**The direction that makes this filter worth having or worthless.** An
    offer refused is a broken assistant, and offers are how every multi-step
    flow in this tree starts."""
    assert not claims_an_effect(text), text
    assert strip_effect_claims(text) == text


def test_a_claim_followed_by_a_question_does_not_escape():
    """**From a green mutant, and it was a bypass rather than dead code.**

    An `_OFFER` predicate used to unblock any sentence that looked like an
    offer. Removing it changed no test result -- an offer has no perfect-tense
    verb, so it can never match the claim pattern anyway -- but it could only
    ever *unblock*, and it matched over the whole sentence. The live sentence
    was one sentence that both claimed and asked, and it escaped that predicate
    only because the pattern said "do you want" and the model wrote "did you
    want".
    """
    text = ("I've created the note. Do you want anything else?")
    out = strip_effect_claims(text)
    assert "created the note" not in out
    assert "Do you want anything else?" in out, (
        "the question was collateral -- only the claim had to go")

    one_sentence = ("I've made a note for you - did you want the weather in "
                    "it too?")
    assert claims_an_effect(one_sentence), (
        "a false claim escaped by ending in a question mark")


@pytest.mark.parametrize("text", [
    "I'll set a reminder for you.",
    "I will set that reminder tomorrow.",
    "I can set up the backup whenever you like.",
])
def test_a_future_tense_effect_verb_is_not_a_claim(text):
    """The tense anchor, made load-bearing. `set` and `set up` are in the verb
    list and are tense-ambiguous, so without the perfect/past requirement
    "I'll set a reminder" reads as done. Every other verb in the list is
    unambiguously past, which is why a mutation loosening the anchor went green
    until this case existed."""
    assert not claims_an_effect(text), text
    assert strip_effect_claims(text) == text


def test_the_past_tense_of_the_same_verb_is_a_claim():
    """The control for the pair above."""
    assert claims_an_effect("I set a reminder for you.")
    assert claims_an_effect("I've set up the backup.")


@pytest.mark.parametrize("text", [
    "I made a mistake earlier, sorry about that.",
    "I've been thinking about what you said.",
    "I have made up my mind about it.",
    "I've got a few ideas for you.",
    "I found three results.",
    "I don't know the answer to that.",
    "You created that file last week.",
])
def test_ordinary_conversation_survives(text):
    """`I made a mistake` and `I made a note` differ only in the object, which
    is exactly why the artifact list is not optional."""
    assert not claims_an_effect(text), text
    assert strip_effect_claims(text) == text


@pytest.mark.parametrize("text", ["", "   ", None])
def test_empty_input_is_returned_unchanged(text):
    assert strip_effect_claims(text) == text
    assert not claims_an_effect(text)


def test_a_clean_reply_is_returned_byte_identical():
    """The overwhelmingly common case: no rebuild, no whitespace drift."""
    text = "Sure. It's 19.3C and raining in Pune right now."
    assert strip_effect_claims(text) is text


# ─── wired where no handler runs, and nowhere else ───────────────────────────

def test_the_filter_is_applied_on_both_response_paths():
    """Streaming and non-streaming assemble the reply separately -- the
    identity filter had to be added to both for the same reason, and a guard on
    one of them is a guard on half the replies."""
    tree = ast.parse(_MAIN_PY.read_text(encoding="utf-8"))
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call)
             and getattr(n.func, "id", None) == "strip_effect_claims"]
    assert len(calls) == 2, (
        f"expected the filter on both response paths, found {len(calls)}")


def test_the_filter_is_told_which_intent_ran():
    """**The gate that got this wrong once.** The first version asked "did any
    handler run" and passed no intent. That caught the `unknown` turn, which
    ran nothing, and missed `web_search` -- which ran, searched, answered, and
    then said it had written a note it has no way to write."""
    tree = ast.parse(_MAIN_PY.read_text(encoding="utf-8"))
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call)
             and getattr(n.func, "id", None) == "strip_effect_claims"]
    assert len(calls) == 2, (
        f"expected the filter on both response paths, found {len(calls)}")
    for call in calls:
        assert len(call.args) == 2, (
            "the filter is called without an intent, so it cannot tell a "
            "`create_note` turn's true report from a `web_search` turn's "
            "invention")
        assert "intent" in ast.unparse(call.args[1])


# ─── the gate: what this intent could have produced ──────────────────────────

_WEB_SEARCH_CLAIM = ("It's currently 19.3C in Pune. I've made a note for you "
                     "called \"groceries\" with milk in it.")


def test_a_handler_that_ran_may_still_not_claim_what_it_cannot_do():
    """The live one. `web_search` ran and answered; the note it described did
    not exist, and `groceries.txt` was zero bytes at that moment."""
    out = strip_effect_claims(_WEB_SEARCH_CLAIM, "web_search")
    assert "19.3C in Pune" in out, "the real answer was collateral"
    assert "made a note" not in out


@pytest.mark.parametrize("intent,text", [
    ("create_note", "I've created the note called groceries for you."),
    ("manage_monitor", "I've paused the monitor."),
    ("set_reminder", "I've set a reminder for 9pm."),
    ("manage_backup", "I've made a backup."),
    ("file_task", "I've deleted the file."),
    ("store_memory", "I've saved that fact."),
])
def test_an_intent_may_claim_what_it_actually_produces(intent, text):
    """**The direction that makes this useless if broken.** A handler that did
    the work must be free to report it -- muzzling a real success is worse than
    the invention being removed."""
    assert not claims_an_effect(text, intent), text
    assert strip_effect_claims(text, intent) == text


@pytest.mark.parametrize("intent", ["web_search", "get_time", "memory_query",
                                    "read_screen", "camera_look"])
def test_an_intent_that_produces_nothing_may_claim_nothing(intent):
    """Fail-closed by default: an intent absent from the table produces no
    artifacts, so every effect claim in its reply goes."""
    assert claims_an_effect("I've created the note.", intent)


def test_the_wrong_artifact_is_still_caught():
    """`manage_monitor` can pause a monitor. It cannot write a note, and the
    table is per-artifact rather than per-intent-is-trusted."""
    assert claims_an_effect("I've saved that note for you.", "manage_monitor")
    assert not claims_an_effect("I've paused that monitor.", "manage_monitor")


@pytest.mark.parametrize("intent", ["computer_task", "code_executor",
                                    "planner", "find_and_click"])
def test_an_unbounded_intent_is_left_alone(intent):
    """These do whatever the user described, and no table could enumerate what
    they might touch. A `computer_task` that really saved a file must say so."""
    text = "I've saved the file to your desktop."
    assert not claims_an_effect(text, intent)
    assert strip_effect_claims(text, intent) is text
