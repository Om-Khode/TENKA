"""Making a note must not empty a note that already exists.

Live test, 2026-08-25, a hundred seconds apart:

    21:58:48  'make a note called groceries saying milk and vegetables'
              Note created: Notes\\groceries.txt

    22:00:28  'make a note called groceries'
              Detected intent: create_note | params: {'title': 'groceries'}
              Note created: Notes\\groceries.txt
              "Done -- 'groceries' is saved."

`groceries.txt` was then zero bytes. `handle_create_note` did
`file_path.write_text(content)` with `content=""`, which is a truncate, and the
reply called it saving.

Two failures in one line and they are worth separating, because only the first
one destroys anything:

1. The write was destructive. A request that names a note and gives it no body
   is not permission to empty that note.
2. The sentence describing it was false. That half is `core/claims.py`'s.

The classifier is not at fault and this is not fixed there. "make a note called
groceries" really does name a note and give it no body; deciding what that
means is the handler's job.

Run with:  py -3.11 -m pytest tests/test_create_note_never_truncates.py -v
"""
import pathlib
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


@pytest.fixture
def notes(tmp_path, monkeypatch):
    """Point NOTES_DIR at a temporary directory.

    Patched on the `config` module object, which is where the handler resolves
    it at call time -- not on a copy taken at import.
    """
    from assistant import config
    monkeypatch.setattr(config, "NOTES_DIR", tmp_path)
    return tmp_path


def _make(**params):
    from assistant.actions.simple import handle_create_note
    return handle_create_note(params, "")


# ─── the sequence that lost a note ───────────────────────────────────────────

def test_a_titled_note_with_no_body_does_not_empty_the_existing_one(notes):
    """**The live failure, exactly.**"""
    _make(title="groceries", content="milk and vegetables")
    assert (notes / "groceries.txt").read_text() == "milk and vegetables"

    reply = _make(title="groceries")

    assert (notes / "groceries.txt").read_text() == "milk and vegetables", (
        "the second request emptied the note the first one wrote")
    assert "left it alone" in reply.lower(), reply


def test_the_reply_does_not_call_it_saving(notes):
    """It said "Done -- 'groceries' is saved." about a truncate. Whatever the
    handler does here, it has to describe it."""
    _make(title="groceries", content="milk")
    reply = _make(title="groceries")

    assert "saved" not in reply.lower(), reply
    assert "groceries" in reply


def test_new_content_for_a_taken_name_goes_beside_it(notes):
    """Overwriting is the one thing that cannot be undone from here. A second
    file can be deleted in a breath, so the ambiguity resolves toward keeping
    both -- and the reply says which file it went to, because a note the user
    cannot find is barely better than one that was destroyed."""
    _make(title="groceries", content="milk and vegetables")
    reply = _make(title="groceries", content="eggs")

    assert (notes / "groceries.txt").read_text() == "milk and vegetables"
    assert (notes / "groceries (2).txt").read_text() == "eggs"
    assert "groceries (2)" in reply, reply


def test_a_third_note_of_the_same_name_keeps_counting(notes):
    for content in ("one", "two", "three"):
        _make(title="n", content=content)

    assert (notes / "n.txt").read_text() == "one"
    assert (notes / "n (2).txt").read_text() == "two"
    assert (notes / "n (3).txt").read_text() == "three"


# ─── the ordinary paths still work ───────────────────────────────────────────

def test_a_new_note_is_written(notes):
    """The control. A handler that refused to write anything would pass every
    test above."""
    reply = _make(title="shopping", content="bread")

    assert (notes / "shopping.txt").read_text() == "bread"
    assert "shopping" in reply


def test_an_untitled_note_still_works(notes):
    _make(content="the wifi password is hunter2")
    assert (notes / "untitled.txt").read_text() == "the wifi password is hunter2"


def test_a_new_empty_note_is_created_and_described_as_empty(notes):
    """Naming a note and filling it later is a real thing to want. Creating it
    is fine; implying something was written into it is not."""
    reply = _make(title="ideas")

    assert (notes / "ideas.txt").exists()
    assert (notes / "ideas.txt").read_text() == ""
    assert "empty" in reply.lower(), reply


@pytest.mark.parametrize("content", [None, "", "   "])
def test_whitespace_is_treated_as_no_content(notes, content):
    """A body of three spaces is not a body.

    **From a green mutant.** Checking `if not content` instead of
    `if not content.strip()` left this test passing: three spaces took the
    other branch and wrote `g (2).txt` rather than truncating `g.txt`, so the
    assertion about the original file held while the handler did the wrong
    thing. Asserting the *absence* of a second file is what catches it -- an
    empty request should produce no file at all, not a spare one.
    """
    _make(title="g", content="real content")
    _make(title="g", content=content)

    assert (notes / "g.txt").read_text() == "real content"
    assert sorted(p.name for p in notes.glob("*.txt")) == ["g.txt"], (
        "an empty request created a second file instead of doing nothing")


def test_the_directory_is_created_when_missing(tmp_path, monkeypatch):
    from assistant import config
    target = tmp_path / "does" / "not" / "exist"
    monkeypatch.setattr(config, "NOTES_DIR", target)

    _make(title="first", content="x")
    assert (target / "first.txt").read_text() == "x"


def test_a_path_traversal_title_stays_in_the_notes_directory(notes):
    """`_sanitize_filename` already handled this; pinned here because the new
    branch computes a second path (`name (2).txt`) from the same title, and a
    sanitiser applied to one and not the other would be a hole."""
    _make(title="../../escape", content="a")
    _make(title="../../escape", content="b")

    written = sorted(p.name for p in notes.glob("*.txt"))
    assert written, "nothing was written at all"
    for name in written:
        assert ".." not in name, name
    assert not list(notes.parent.glob("escape*.txt")), (
        "a note escaped the notes directory")
