"""A search reports what it searched. TENKA-v2 §17.P8, live-test fallout.

Live testing a resumed plan turned up a file the assistant could not find:
`D:\\VR Model\\model.vroid`, two directories below a drive root, reported as
absent by both the fast and the deep search. It was 2.7 seconds away.

Three separate bugs, all of the same kind -- claiming an action that did not
happen:

1. **One deadline for every drive, walked alphabetically.** The system drive is
   both first and by far the largest; it took ~37s at tier 2's depth against a
   15s budget, so it consumed the entire clock and the next drive was never
   opened. Every drive now gets an equal share of the time remaining, and one
   that finishes early hands the rest to those behind it.

2. **Depth-first with an unlimited depth, so tier 3 found *less* than tier 2.**
   The first large subtree in alphabetical order ate the drive's whole share
   before its sibling was reached. The walk is breadth-first now, which makes
   a shallow file cheap to find and makes each tier a superset of the one
   below it by construction.

3. **`[]` meant both "not there" and "gave up".** The caller reported the
   first while meaning the second: *"I did a thorough search of your entire
   computer and couldn't find any file called X"* after covering part of one
   drive out of three. `find_files` returns a `SearchResult` that says whether
   it finished, and the caller now declines to claim what it cannot support.

The property under all three: **a search may fail to find something; it may
not claim to have looked where it did not.**

Run with:  py -3.11 -m pytest tests/test_search_tells_the_truth.py -v
"""
import pathlib
import sys
import time

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ─── helpers ─────────────────────────────────────────────────────────────────

def _slow_tree(root: pathlib.Path, breadth: int = 12, depth: int = 6) -> None:
    """A directory that is expensive to walk and holds nothing."""
    node = root
    for level in range(depth):
        for i in range(breadth):
            (node / f"d{level}_{i}").mkdir(parents=True, exist_ok=True)
            (node / f"d{level}_{i}" / "filler.bin").write_bytes(b"x")
        node = node / f"d{level}_0"


def _wide(root: pathlib.Path, count: int) -> None:
    """`count` sibling directories — expensive to get past, holding nothing."""
    for i in range(count):
        (root / f"w{i:04d}").mkdir(parents=True, exist_ok=True)


def _deep_chain(root: pathlib.Path, levels: int) -> None:
    """A narrow chain — cheap breadth-first, ruinous depth-first."""
    node = root
    for i in range(levels):
        node = node / f"level{i:03d}"
        node.mkdir(parents=True, exist_ok=True)
        (node / "filler.bin").write_bytes(b"x")


def _fake_clock(monkeypatch, tick: float = 0.01):
    """Make the search's budget count *directory visits*, not wall time.

    Every test below that is about running out of time uses this, because the
    alternative is asserting that a real directory walk outruns a real
    stopwatch — which passes or fails according to how busy the machine is.
    A timing test that flakes gets deleted rather than fixed, and these are
    the tests standing between a user and "I searched your entire computer"
    said about a drive that was never opened.
    """
    ticks = iter([1000.0 + i * tick for i in range(2_000_000)])
    monkeypatch.setattr(time, "time", lambda: next(ticks))


@pytest.fixture
def fm(monkeypatch):
    """`find_files` with its two roots pointed at temp directories.

    Tier 1 is emptied so it cannot answer, and the drive list is injectable so
    the alphabetical, first-is-largest shape can be reproduced without real
    drive letters.
    """
    from assistant import file_manager
    monkeypatch.setattr(file_manager, "_get_tier1_folders", lambda: [])
    return file_manager


# ─── 1. one drive cannot spend everyone's budget ─────────────────────────────

def test_a_slow_first_drive_does_not_starve_the_rest(fm, tmp_path,
                                                     monkeypatch):
    """The original bug, reproduced without real drives.

    `a_big` is too wide to get through on the budget and holds nothing;
    `z_small` holds the file, one directory down. Under a single shared
    deadline `a_big` spends all of it and `z_small` is never opened -- which
    is precisely what happened to `D:\\VR Model\\model.vroid`, twice.
    """
    big = tmp_path / "a_big"
    small = tmp_path / "z_small"
    big.mkdir()
    small.mkdir()
    _wide(big, 400)
    (small / "model.vroid").write_bytes(b"found me")

    monkeypatch.setattr(fm, "_get_drives", lambda: [big, small])
    _fake_clock(monkeypatch)

    # 200 ticks of budget; `a_big`'s top level alone is 400 entries.
    result = fm.find_files("model.vroid", tier=2, timeout_seconds=2.0)

    monkeypatch.undo()
    assert [p.name for p in result] == ["model.vroid"], (
        "the second location was never searched -- the first one spent the "
        "whole budget")


def test_the_starved_location_is_reported_as_unsearched(fm, tmp_path,
                                                        monkeypatch):
    """Running out of time on one drive is a fact the caller has to be able to
    state. Finding the file elsewhere does not make the unsearched drive
    searched."""
    big = tmp_path / "a_big"
    small = tmp_path / "z_small"
    big.mkdir()
    small.mkdir()
    _wide(big, 400)
    (small / "model.vroid").write_bytes(b"found me")

    monkeypatch.setattr(fm, "_get_drives", lambda: [big, small])
    _fake_clock(monkeypatch)

    result = fm.find_files("model.vroid", tier=2, timeout_seconds=2.0)

    monkeypatch.undo()
    assert list(result), "setup failed -- the file was not found at all"
    assert not result.exhaustive
    assert any("a_big" in u for u in result.unsearched), (
        f"the starved location was not reported: {result.unsearched}")


# ─── 2. deeper must not mean worse ───────────────────────────────────────────

def test_a_shallow_file_is_found_behind_a_deep_subtree(fm, tmp_path,
                                                       monkeypatch):
    """Why the walk is breadth-first.

    `a_deep` is a narrow, very deep chain; `z.txt`'s directory is a sibling one
    level down. Depth-first descends the chain to exhaustion first. The file is
    two steps from the root and must not be expensive to find.
    """
    root = tmp_path / "drive"
    root.mkdir()

    # One deep chain either side of the target, so *any* order that descends
    # before finishing a level burns the budget -- the test does not depend on
    # which sibling a depth-first walk happens to take first.
    _deep_chain(root / "a_deep", 300)
    _deep_chain(root / "z_deep", 300)
    shallow = root / "m_shallow"
    shallow.mkdir()
    (shallow / "model.vroid").write_bytes(b"found me")

    monkeypatch.setattr(fm, "_get_drives", lambda: [root])
    _fake_clock(monkeypatch)

    # Breadth-first reaches the file in a couple of dozen visits. Descending
    # either chain costs hundreds.
    result = fm.find_files("model.vroid", tier=3, timeout_seconds=1.5)

    monkeypatch.undo()
    assert [p.name for p in result] == ["model.vroid"], (
        "a file two directories down was lost behind a deep sibling -- the "
        "walk is depth-first again")


def test_the_deep_tier_finds_what_the_fast_tier_finds(fm, tmp_path,
                                                      monkeypatch):
    """Tier 3 returning less than tier 2 is the shape of the original bug, and
    the one a user would never think to check."""
    root = tmp_path / "drive"
    (root / "one" / "two").mkdir(parents=True)
    (root / "one" / "two" / "model.vroid").write_bytes(b"x")
    _slow_tree(root / "a_noise")

    monkeypatch.setattr(fm, "_get_drives", lambda: [root])

    fast = fm.find_files("model.vroid", tier=2, timeout_seconds=6.0)
    deep = fm.find_files("model.vroid", tier=3, timeout_seconds=6.0)

    assert [p.name for p in fast] == ["model.vroid"]
    assert [p.name for p in deep] == ["model.vroid"], (
        "the deep search found less than the fast one")


# ─── 3. the result says whether it finished ──────────────────────────────────

def test_a_completed_search_says_so(fm, tmp_path, monkeypatch):
    root = tmp_path / "drive"
    root.mkdir()
    (root / "a.txt").write_bytes(b"x")
    monkeypatch.setattr(fm, "_get_drives", lambda: [root])

    result = fm.find_files("absent", tier=2, timeout_seconds=30.0)

    assert list(result) == []
    assert result.exhaustive, (
        "a search that finished reported itself as cut short")
    assert result.unsearched == ()


def test_a_search_cut_short_does_not_claim_to_have_finished(fm, tmp_path,
                                                            monkeypatch):
    """The fact the honest message depends on. Without it `[]` means both
    "it is not there" and "I ran out of time", and only one of those was ever
    said out loud.

    The clock is faked rather than the tree made slow: a test that depends on
    a directory walk outrunning a real stopwatch passes or fails by how busy
    the machine is, and a timing test that flakes gets deleted rather than
    fixed.
    """
    big = tmp_path / "a_big"
    big.mkdir()
    _slow_tree(big, breadth=6, depth=3)
    monkeypatch.setattr(fm, "_get_drives", lambda: [big])

    ticks = iter([1000.0 + i * 0.05 for i in range(10_000)])
    monkeypatch.setattr(time, "time", lambda: next(ticks))

    result = fm.find_files("absent", tier=3, timeout_seconds=1.0)

    monkeypatch.undo()
    assert list(result) == []
    assert not result.exhaustive, (
        "a search that ran out of time reported itself as exhaustive")
    assert result.unsearched, "it did not say what it failed to reach"


def test_tier_one_is_exhaustive_because_it_has_no_deadline(fm, tmp_path,
                                                           monkeypatch):
    monkeypatch.setattr(fm, "_get_tier1_folders", lambda: [tmp_path])
    (tmp_path / "a.txt").write_bytes(b"x")

    result = fm.find_files("absent", tier=1)

    assert result.exhaustive
    assert isinstance(result, fm.SearchResult), (
        "tier 1 returns a bare list, so the caller cannot ask whether it "
        "finished")


# ─── 4. the caller does not claim what the result cannot support ─────────────

def _search_message(monkeypatch, result, tier):
    """Drive `handle_pending_file_search`'s background thread and capture what
    it would say."""
    import asyncio
    import queue

    import assistant.actions as actions
    import assistant.file_manager as file_manager
    from assistant.actions import file_ops

    # `_run_search` does `from .. import file_manager` at call time, on a
    # fresh thread, so the name is read off the real module -- patch it there.
    monkeypatch.setattr(file_manager, "find_files", lambda *a, **k: result)

    actions.pending_file_search.set({"name": "model.vroid", "tier": tier - 1})
    answer = "deep" if tier == 3 else "fast"
    asyncio.run(file_ops.handle_pending_file_search(answer))

    # The search runs on a daemon thread and posts to this queue; the real
    # consumer is main.py's turn epilogue.
    try:
        msg = actions._search_result_queue.get(timeout=5.0)
    except queue.Empty:
        msg = ""
    actions.pending_file_search.clear()
    return msg


def test_a_timed_out_deep_search_does_not_claim_the_whole_computer(
        monkeypatch):
    """The sentence that started this: *"I did a thorough search of your entire
    computer and couldn't find any file called model.vroid."* It had searched
    part of one drive out of three."""
    from assistant.file_manager import SearchResult

    msg = _search_message(
        monkeypatch,
        SearchResult([], exhaustive=False, unsearched=["D:\\", "E:\\"]),
        tier=3,
    )

    assert msg, "the background search said nothing at all"
    lowered = msg.lower()
    assert "entire computer" not in lowered, (
        f"claimed an exhaustive search it did not run: {msg}")
    assert "ran out of time" in lowered or "didn't get to" in lowered, (
        f"did not say the search was cut short: {msg}")


def test_a_completed_deep_search_may_still_say_it_is_not_there(monkeypatch):
    """The control. A truthfulness fix that makes every answer hedge is not a
    fix -- when the search really did finish, the confident answer is the
    correct one."""
    from assistant.file_manager import SearchResult

    msg = _search_message(monkeypatch, SearchResult([], exhaustive=True),
                          tier=3)

    assert msg
    assert "ran out of time" not in msg.lower(), (
        f"hedged about a search that actually finished: {msg}")
