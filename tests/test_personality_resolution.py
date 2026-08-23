"""Personality has enough resolution to show movement, and no authority over truth.

TENKA-v2 §27 found the composition already correct -- `build_personality_prompt`
composes four live sources, and "robotic" was not a missing mechanism. It was
**quantisation**, in two places, and both are arithmetic rather than opinion:

1. **Trait tiers.** Six traits collapsed to three bands at 0.34 / 0.67, so a
   band is a third of the range wide. The reflection cycle moves a trait by at
   most `MAX_DELTA_PER_CYCLE = 0.05`, so crossing one takes about 6.6 cycles --
   nearly a week in which a trait drifts steadily and the prompt is
   byte-identical every day. Evolution that cannot be observed until the seventh
   night is indistinguishable from none.
2. **Response pools.** 41 keys per voiced personality, median 3 variants, and
   one key with exactly 1.

The fix for (1) is `_trait_intensity`: the personality's own sentence still
carries the voice, and a 0-100 figure carries the movement. A number rather than
more bands because more bands need more sentences from every personality, while
this is resolution the tree already had and was throwing away.

The pool floor here asserts what is true **now**, not the target. §P11 wants a
median of 8 and says to re-measure before writing more; a test asserting 8 today
would just be red. What this floor does catch is a regression, and a new
personality shipping one-sentence pools.

Run with:  py -3.11 -m pytest tests/test_personality_resolution.py -v
"""
import json
import pathlib
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from assistant.llm.prompts import _get_trait_tier, _trait_intensity  # noqa: E402
from assistant.storage.repos.personality import MAX_DELTA_PER_CYCLE  # noqa: E402

_PERSONALITIES = _ROOT / "assistant" / "personalities"

# What is true today. Raise it as pools are widened -- that is the point of it
# being a named constant rather than a literal buried in an assertion.
_MIN_VARIANTS = 2


def _folders():
    return [d for d in sorted(_PERSONALITIES.iterdir())
            if (d / "responses.json").exists()]


def _base_pools(folder):
    """Response pools, excluding the `_casual_extras` overlay.

    That key is not a pool: it is an opt-in set of extra variants merged into
    the pools of the same name when `VOCAL_CASUAL_LANGUAGE` is on. Counting it
    as one made an earlier measurement of this tree report a minimum of 1 for
    tsundere when its real minimum was 1 for a different reason entirely.
    """
    data = json.loads((folder / "responses.json").read_text(encoding="utf-8"))
    data.pop("_casual_extras", None)
    return {k: v for k, v in data.items() if isinstance(v, list)}


# ─── trait resolution ────────────────────────────────────────────────────────

def test_one_reflection_cycle_of_drift_is_visible_immediately():
    """The property the phase actually asks for, stated as arithmetic.

    A single cycle's movement must change what the model sees. Under the tier
    alone it did not: 0.50 and 0.55 are both "mid", so a week of steady drift
    produced an identical prompt every day.
    """
    for start in (0.10, 0.30, 0.50, 0.70, 0.90):
        after = start + MAX_DELTA_PER_CYCLE
        assert _trait_intensity(start) != _trait_intensity(after), (
            f"a full reflection cycle of drift ({start} -> {after:.2f}) left "
            f"the prompt unchanged"
        )


def test_resolution_is_finer_than_the_three_tiers():
    """Numerically, not by inspection. Counting distinct outputs across the
    range is what makes "finer" checkable -- three tiers give three."""
    tiers = {_get_trait_tier(i / 1000) for i in range(1001)}
    intensities = {_trait_intensity(i / 1000) for i in range(1001)}
    assert len(tiers) == 3, f"the tier count changed: {sorted(tiers)}"
    assert len(intensities) > 3 * 5, (
        f"intensity yields only {len(intensities)} distinct values -- barely "
        f"more resolution than the tiers it was added to fix"
    )


def test_intensity_is_an_integer_and_stays_in_range():
    """A float in a prompt renders as `0.5700000000000001`, which is noise the
    model then has to ignore. Out-of-range values are clamped rather than
    trusted: trait floors and ceilings are per-personality data, and a bad row
    should degrade the wording, not emit `intensity 140/100`."""
    for value in (-1.0, 0.0, 0.333, 0.5, 1.0, 2.0):
        got = _trait_intensity(value)
        assert isinstance(got, int) and not isinstance(got, bool)
        assert 0 <= got <= 100, f"{value} -> {got}"


def test_the_tier_still_selects_a_personality_sentence():
    """The tier is kept, not replaced. Each personality supplies exactly three
    modifier sentences per trait, so removing tiers would mean rewriting every
    personality's data -- which is why the resolution was added beside them
    rather than instead of them."""
    assert _get_trait_tier(0.0) == "low"
    assert _get_trait_tier(0.5) == "mid"
    assert _get_trait_tier(1.0) == "high"


def test_the_rendered_modifier_line_carries_both(monkeypatch):
    """On the real builder, because the two pieces are joined at the call site
    and a test of the helpers alone would not notice them being joined wrongly.
    """
    from assistant.llm import prompts as p

    class _Loader:
        @staticmethod
        def get_modifiers():
            return {"warmth": {"mid": "Be moderately warm."}}

        @staticmethod
        def get_prompt_base():
            return "BASE"

    # `build_personality_prompt` does `from ..personalities import
    # get_active_loader` INSIDE the function, so the name resolves against
    # `assistant.personalities` -- patching it on `prompts` binds something
    # nothing reads. Patch where the importing code will look; the first draft
    # of this test did not, and the modifier never appeared.
    import assistant.personalities as personalities_mod
    monkeypatch.setattr(personalities_mod, "get_active_loader", lambda: _Loader)
    monkeypatch.setattr(personalities_mod, "consume_switch_flag", lambda: False)
    monkeypatch.setattr(p, "_get_personality_traits", lambda: {"warmth": 0.57})
    monkeypatch.setattr(p, "_get_personality_base", lambda: "BASE")
    monkeypatch.setattr(p, "_build_personality_context_summary", lambda: "")
    monkeypatch.setattr(p, "_build_preference_prompt_block", lambda: "")

    out = p.build_personality_prompt()
    assert "Be moderately warm." in out, "the personality's own sentence was lost"
    assert "57" in out, "the trait's position was not rendered"


# ─── response pools ──────────────────────────────────────────────────────────

def test_every_personality_folder_is_walked():
    """A sweep over an empty set passes forever."""
    folders = _folders()
    assert len(folders) >= 3, f"only found {[f.name for f in folders]}"


@pytest.mark.parametrize("folder", _folders(), ids=lambda f: f.name)
def test_no_response_pool_is_below_the_floor(folder):
    """A one-variant pool means the same sentence every single time, which is
    the most literal form of "robotic" there is.

    **`terse` is declared data, not a folder name.** `minimal`'s entire identity
    is saying things exactly one way, so a floor that applied to it would be
    wrong -- but naming the folder in this test would repeat the mistake just
    removed from `personalities/__init__.py`, which special-cased `tsundere` by
    name. A personality declares its own terseness in `traits.json`; the test
    reads the declaration.
    """
    traits = json.loads((folder / "traits.json").read_text(encoding="utf-8"))
    if traits.get("terse"):
        pytest.skip(f"{folder.name} declares terse=true")

    pools = _base_pools(folder)
    assert pools, f"{folder.name} has no response pools -- nothing walked"
    thin = {k: len(v) for k, v in pools.items() if len(v) < _MIN_VARIANTS}
    assert not thin, (
        f"{folder.name} has pools below {_MIN_VARIANTS} variants: {thin}. "
        f"A voiced personality repeating one sentence verbatim is what "
        f"'robotic' means."
    )


@pytest.mark.parametrize("folder", _folders(), ids=lambda f: f.name)
def test_no_pool_contains_a_duplicate_variant(folder):
    """Depth that is not variety buys nothing -- three copies of one line is a
    pool of one wearing a larger number. Cheap to check while widening."""
    for key, variants in _base_pools(folder).items():
        assert len(set(variants)) == len(variants), (
            f"{folder.name}.{key} lists the same variant twice: {variants}"
        )


def test_the_casual_overlay_is_not_gated_on_a_personality_name():
    """THE rule, on the loader. It read `if self._name == "tsundere"`, so a
    second personality shipping casual variants would have had them silently
    ignored. Whether an overlay exists is data; the identity of the personality
    is not a condition."""
    # Comments stripped before scanning. The first version matched the comment
    # in the loader that *documents* the removed check, so the test failed on
    # its own explanation -- the third time that shape has come up in this
    # session, and the reason source sweeps here read code rather than lines.
    raw = (_PERSONALITIES / "__init__.py").read_text(encoding="utf-8")
    src = " ".join(line.split("#", 1)[0] for line in raw.splitlines())
    for folder in _folders():
        assert f'== "{folder.name}"' not in src, (
            f"the loader special-cases the personality '{folder.name}' by name"
        )


def test_the_overlay_key_never_leaks_as_a_response_pool():
    """`_casual_extras` is popped whether or not the flag is on. Left in, it
    would look like a response key called `_casual_extras` to every consumer
    that iterates the pools -- including the floor test above."""
    from assistant.personalities import PersonalityLoader

    for folder in _folders():
        loader = PersonalityLoader(folder.name)
        assert "_casual_extras" not in loader.get_responses(), (
            f"{folder.name} exposes the overlay as a pool"
        )
