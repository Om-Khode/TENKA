"""URL recon picks a host deliberately, and only localises when location matters.

Both defects here had the shape of an unvalidated guess that silently won.

`Open Wikipedia and find Alan Turing's birth year.` opened
**hif.wikipedia.org** -- Fiji Hindi -- on 2026-08-22. Two causes compounded:

1. `_url_recon` appended the user's city to *every* recon query, so the text
   that left the machine for the search provider was "... Alan Turing <city>",
   which biased ranking toward a regional-language edition. Alan Turing's
   birth year is the same in every city.

2. The host was then chosen by `if hint in host` over the results **in search
   order**, returning the first match and discarding the rest. `"wikipedia" in
   "hif.wikipedia.org"` is true, so ranking luck decided it. The log printed
   only the winner, so there was no way to see that `en.wikipedia.org` had
   matched too. Ninety seconds earlier in the same session the identical code
   path returned `en.wikipedia.org` for a different query.

Neither needed a network call to reproduce once the decisions were separable,
which is why both are pure functions now. These tests make no API calls.

Run with:  py -3.11 -m pytest tests/test_url_recon_choice.py -v
"""
import pathlib
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from assistant.automation.router import (  # noqa: E402
    _choose_recon_url, _host_specificity, _is_locality_query,
)


def _r(*urls):
    return [{"url": u} for u in urls]


# ─── host specificity ───────────────────────────────────────────────────────

@pytest.mark.parametrize("host,expected", [
    ("wikipedia.org", 0),
    ("en.wikipedia.org", 1),
    ("hif.wikipedia.org", 1),
    ("a.b.wikipedia.org", 2),
    ("example.co.uk", 1),        # two-label public suffix: not worth a PSL dep
    ("", 0),
])
def test_host_specificity_counts_labels_before_the_domain(host, expected):
    assert _host_specificity(host) == expected


# ─── host choice ────────────────────────────────────────────────────────────

def test_the_canonical_host_wins_over_a_regional_subdomain():
    """The reported bug. Search order put the regional edition first."""
    url, reason = _choose_recon_url(
        _r("https://hif.wikipedia.org/wiki/Alan_Turing",
            "https://wikipedia.org/wiki/Alan_Turing"),
        ["wikipedia"],
    )
    assert url == "https://wikipedia.org/wiki/Alan_Turing", (
        f"picked {url!r} -- a deeper subdomain won on search rank alone"
    )
    assert "2 hosts" in reason, (
        f"the reason must disclose that the choice was ambiguous, got {reason!r}. "
        "A log line naming only the winner is why this bug was invisible."
    )


def test_the_ambiguity_is_named_in_the_reason():
    """Mechanism, not cosmetics: the old code discarded the alternatives
    silently, so the log could not show that a choice had even been made."""
    _, reason = _choose_recon_url(
        _r("https://hif.wikipedia.org/x", "https://en.wikipedia.org/x"),
        ["wikipedia"],
    )
    assert "hif.wikipedia.org" in reason and "en.wikipedia.org" in reason, (
        f"both candidates must appear in the reason, got {reason!r}"
    )


def test_a_tie_keeps_search_order():
    """`en.` and `hif.` are equally specific. With nothing to separate them,
    the provider's ranking is the only signal there is -- and inventing a
    language preference would need a config surface nobody has decided on."""
    url, _ = _choose_recon_url(
        _r("https://en.wikipedia.org/x", "https://hif.wikipedia.org/x"),
        ["wikipedia"],
    )
    assert url == "https://en.wikipedia.org/x"


def test_a_single_match_is_returned_without_claiming_ambiguity():
    url, reason = _choose_recon_url(_r("https://en.wikipedia.org/x"), ["wikipedia"])
    assert url == "https://en.wikipedia.org/x"
    assert "hosts" not in reason, f"one match must not read as several: {reason!r}"


def test_hints_are_tried_in_order():
    """The first hint that matches anything decides. A later hint must not
    override an earlier one -- the step goal lists them by significance."""
    url, _ = _choose_recon_url(
        _r("https://maps.example.com/x", "https://wikipedia.org/y"),
        ["wikipedia", "maps"],
    )
    assert url == "https://wikipedia.org/y"


def test_no_hint_match_falls_back_to_the_first_result():
    url, reason = _choose_recon_url(_r("https://somewhere.test/x"), ["wikipedia"])
    assert url == "https://somewhere.test/x"
    assert "no domain hint match" in reason


def test_no_results_returns_none():
    url, reason = _choose_recon_url([], ["wikipedia"])
    assert url is None and reason == "no results"


def test_results_without_a_url_are_ignored_not_crashed_on():
    url, _ = _choose_recon_url(
        [{"title": "no url here"}, {"url": "https://wikipedia.org/x"}],
        ["wikipedia"],
    )
    assert url == "https://wikipedia.org/x"


# ─── locality detection ─────────────────────────────────────────────────────

@pytest.mark.parametrize("goal", [
    "coffee near me",
    "nearest pharmacy",
    "is the library still open",
    "directions to the station",
    "restaurants around here",
    "how long is the commute",
])
def test_a_location_dependent_goal_is_localised(goal):
    assert _is_locality_query(goal), f"{goal!r} needs the city to be answerable"


@pytest.mark.parametrize("goal", [
    "open wikipedia and search Alan Turing",          # the reported bug
    "Open Wikipedia and find Alan Turing's birth year.",
    "what is the boiling point of water",
    "convert a webp to png",
    "who wrote the second world war",
    "search for the best sorting algorithm",
])
def test_a_location_independent_goal_is_not_localised(goal):
    """The half that fixes the bug. A city appended here is noise that steers
    ranking -- and the answer is the same everywhere."""
    assert not _is_locality_query(goal), (
        f"{goal!r} would have the city appended, biasing the search provider "
        "toward a regional result for a question that has no regional answer"
    )


def test_the_locality_vocabulary_names_no_product_or_topic():
    """THE-rule. Cues are requests for a location ('near me'), never topics
    that merely often want one ('restaurant', 'hotel') -- a topic list would
    be the app-specific table the project forbids, and it would never end."""
    from assistant.automation.router import _LOCALITY_CUES, _LOCALITY_PHRASES

    assert _LOCALITY_CUES, "empty cue set -- these tests would pass vacuously"
    banned = {"restaurant", "hotel", "cafe", "pharmacy", "wikipedia", "google",
              "maps", "youtube", "spotify", "shop", "store", "food"}
    overlap = _LOCALITY_CUES & banned
    assert not overlap, f"topic/product words in the cue set: {overlap}"
    for phrase in _LOCALITY_PHRASES:
        assert not (set(phrase.split()) & banned), f"topic word in {phrase!r}"
