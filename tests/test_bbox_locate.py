"""Verify locate_element_bbox parses Gemini bbox responses correctly and
maps normalized 0-1000 coords to screen pixels. Mocks the Gemini call so
we don't burn API quota on tests."""

from unittest.mock import patch
from assistant import llm


# Cases: (mocked_gemini_response, expected_result_for_1920x1080_screen)
CASES = [
    # ── Happy path: clean JSON → maps to screen pixels ──
    (
        '{"box_2d": [400, 200, 500, 800]}',
        # ymin=400, xmin=200, ymax=500, xmax=800
        # cx_norm=500, cy_norm=450
        # screen_x = 500 * 1920 / 1000 = 960
        # screen_y = 450 * 1080 / 1000 = 486
        (960, 486),
        "centered horizontal band",
    ),
    # ── Top-left corner ──
    (
        '{"box_2d": [0, 0, 100, 100]}',
        # cx_norm=50, cy_norm=50 → 96, 54
        (96, 54),
        "top-left corner",
    ),
    # ── JSON wrapped in extra prose (LLM verbosity) ──
    (
        'Here is the bounding box:\n```json\n{"box_2d": [200, 100, 300, 400]}\n```',
        # cx=250, cy=250 → 480, 270
        (480, 270),
        "JSON wrapped in markdown fences and prose",
    ),
    # ── Element not found → null bbox ──
    (
        '{"box_2d": null}',
        None,
        "Gemini reports element not visible",
    ),
    # ── Missing key ──
    (
        '{"foo": "bar"}',
        None,
        "JSON without box_2d key",
    ),
    # ── Wrong array length ──
    (
        '{"box_2d": [10, 20, 30]}',
        None,
        "bbox has 3 elements not 4",
    ),
    # ── Inverted box (xmin >= xmax) — invalid ──
    (
        '{"box_2d": [100, 500, 200, 300]}',
        None,
        "xmin > xmax — invalid",
    ),
    # ── Out of range coords ──
    (
        '{"box_2d": [100, 100, 100, 1500]}',
        None,
        "xmax > 1000 — out of range",
    ),
    # ── Garbage response ──
    (
        'I cannot find that element on the screen.',
        None,
        "no JSON in response",
    ),
    # ── Empty response ──
    (
        '',
        None,
        "empty response",
    ),
]


def run():
    failures = []
    for resp, expected, note in CASES:
        with patch("assistant.llm._vision_gemini", return_value=resp), \
             patch("pyautogui.size", return_value=(1920, 1080)):
            got = llm.locate_element_bbox("Submit button", "fake_b64")

        ok = got == expected
        status = "OK  " if ok else "FAIL"
        print(f"{status}  expected={expected}  got={got}")
        print(f"      response: {resp[:80]!r}")
        print(f"      note: {note}")
        if not ok:
            failures.append(note)

    print()
    if failures:
        print(f"{len(failures)} FAILED:")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)
    print(f"All {len(CASES)} cases passed.")


if __name__ == "__main__":
    run()
