"""`_snap_to_ocr` pulls a model's guessed coordinate onto real screen text.

Replays the failing form-fill scenario from a debug log, plus regression cases
for shorter labels so the earlier demos keep working.

**Was a manual script.** It sat in `tests/` with a `def run()` and an
`if __name__ == "__main__"` block, so pytest imported it, collected nothing,
and reported EMPTY -- a file that looks like coverage and asserts nothing. The
2026-08-25 baseline found six of these. The cases below are the originals,
unchanged; only the harness is new.

Run with:  py -3.11 -m pytest tests/test_vision_snap.py -v
"""
import pathlib
import sys
from unittest.mock import patch

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from assistant.automation import vision as computer_agent  # noqa: E402


CASES = []

# ── Case 1: regression scenario from debug.log ─────────────────────────────
# LLM gave (499, 499); OCR blocks were what the screen actually showed.
# Expected: snap to the email input's placeholder text at (682, 570) — that is
# the highest-overlap match (3 words: enter, email — adcress typo doesn't match).
# OR to "Email Address" label at (627, 521) which also has 2 words (email,
# address). Either is correct; both focus the input. (627, 521) is closer to
# the LLM coord, so it ranks first. Either non-(499,499) result is a pass.
CASES.append({
    "name": "form-fill from debug.log — email placeholder/label",
    "search": "Enter your email address",
    "llm_xy": (499, 499),
    "blocks": [
        {"text": "Enter your full name", "x": 665, "y": 339, "confidence": 0.97},
        {"text": "Email Address", "x": 627, "y": 521, "confidence": 1.00},
        {"text": "Enter your email adcress", "x": 682, "y": 570, "confidence": 0.81},
        {"text": "Submit", "x": 800, "y": 800, "confidence": 0.99},
    ],
    "must_not_be": (499, 499),  # any non-LLM coord is a win
    "expect_in": [(627, 521), (682, 570)],
})

# ── Case 2: short label, snap within original radius (regression) ─────────
CASES.append({
    "name": "Save button — close LLM coord, single-word match",
    "search": "Save",
    "llm_xy": (105, 110),
    "blocks": [
        {"text": "Save", "x": 100, "y": 100, "confidence": 0.95},
        {"text": "Cancel", "x": 200, "y": 100, "confidence": 0.95},
    ],
    "expect_in": [(100, 100)],
})

# ── Case 3: multi-word phrase splits across OCR blocks (regression) ───────
# OCR sometimes splits "Bohemian Rhapsody" into separate blocks. With the
# old code, "BOHEMIAN" single-word match within radius would still snap.
CASES.append({
    "name": "Bohemian Rhapsody — split OCR blocks",
    "search": "Bohemian Rhapsody",
    "llm_xy": (300, 400),
    "blocks": [
        {"text": "BOHEMIAN", "x": 290, "y": 395, "confidence": 0.92},
        {"text": "RHAPSODY", "x": 350, "y": 395, "confidence": 0.92},
        {"text": "Other Song", "x": 600, "y": 395, "confidence": 0.92},
    ],
    "expect_in": [(290, 395), (350, 395)],
})

# ── Case 4: no OCR matches at all -> return LLM coord (regression) ─────────
CASES.append({
    "name": "no matching OCR text — fallback to LLM coord",
    "search": "Login",
    "llm_xy": (500, 500),
    "blocks": [
        {"text": "Welcome", "x": 100, "y": 100, "confidence": 0.95},
        {"text": "Settings", "x": 200, "y": 100, "confidence": 0.95},
    ],
    "expect_in": [(500, 500)],
})

# ── Case 5: candidate beyond radius but only 1 match -> no snap ────────────
# Don't snap halfway across the screen on a flimsy single-word match.
CASES.append({
    "name": "lone match too far — no snap",
    "search": "Logout",
    "llm_xy": (100, 100),
    "blocks": [
        {"text": "Logout", "x": 1500, "y": 800, "confidence": 0.95},
    ],
    "expect_in": [(100, 100)],
})

# ── Case 6: 2-word match beyond base radius -> snap (the new behaviour) ────
# The crux fix: when LLM coord is way off but OCR sees a multi-word agreement,
# trust OCR up to the screen-third hard cap.
CASES.append({
    "name": "two-word match at 350px (was beyond old 120px radius)",
    "search": "Email Address",
    "llm_xy": (300, 300),
    "blocks": [
        {"text": "Email Address", "x": 600, "y": 500, "confidence": 1.00},
    ],
    "expect_in": [(600, 500)],
})

# ── Case 7: hard cap — ridiculously far multi-word match still rejected ───
CASES.append({
    "name": "multi-word match beyond hard cap (1/3 of 1920 = 640) — no snap",
    "search": "Email Address",
    "llm_xy": (50, 50),
    "blocks": [
        {"text": "Email Address", "x": 1800, "y": 1000, "confidence": 1.00},
    ],
    "expect_in": [(50, 50)],
})


def _ids():
    return [c["name"] for c in CASES]


@pytest.mark.parametrize("case", CASES, ids=_ids())
def test_snap_lands_on_real_screen_text(case):
    with patch("assistant.io.screen.list_ocr_blocks",
               return_value=case["blocks"]):
        got = computer_agent._snap_to_ocr(*case["llm_xy"], case["search"])

    if "must_not_be" in case:
        assert got != case["must_not_be"], (
            f"{case['name']}: snapped to the model's own guess instead of "
            f"screen text")
    assert got in case["expect_in"], (
        f"{case['name']}: search={case['search']!r} llm={case['llm_xy']} "
        f"-> {got}, expected one of {case['expect_in']}")


def test_the_case_table_is_not_empty():
    """A parametrize over an empty list is a file that passes by not running --
    which is exactly what this file did before it was converted."""
    assert len(CASES) >= 7
    assert any("must_not_be" in c for c in CASES), (
        "no case pins the failure mode this was written for")
