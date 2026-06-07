"""
screen.py — Screen reading module for the TENKA Voice Assistant.

Provides OCR, window detection, and screen description capabilities
for use by the computer agent and action handlers.

Dependencies: mss, easyocr, pygetwindow, Pillow
"""

import logging
from typing import Optional

logger = logging.getLogger("screen")

# ─── Optional Region Restriction ──────────────────────────────────────────────
# Set to (left, top, width, height) to restrict OCR to a screen region,
# or None for full screen.
SCREEN_REGION: Optional[tuple[int, int, int, int]] = None


# ─── Lazy-loaded globals ─────────────────────────────────────────────────────

_ocr_reader = None  # EasyOCR reader (heavy, loaded once)


def _get_ocr_reader():
    """Lazy-load the EasyOCR reader (downloads model on first run)."""
    global _ocr_reader
    if _ocr_reader is None:
        try:
            import easyocr
            logger.info("[SCREEN] Loading EasyOCR model (first run may download ~100 MB)...")
            _ocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
            logger.info("[SCREEN] EasyOCR model loaded")
        except ImportError:
            logger.error("[SCREEN] easyocr not installed — pip install easyocr")
            raise
        except Exception as e:
            logger.error(f"[SCREEN] Failed to load EasyOCR: {e}")
            raise
    return _ocr_reader


# ─── Screenshot ──────────────────────────────────────────────────────────────


def capture_screenshot(region: Optional[tuple[int, int, int, int]] = None):
    """
    Capture a screenshot of the full screen (or a region).

    Args:
        region: Optional (left, top, width, height) tuple.
                If None, uses SCREEN_REGION or captures full screen.

    Returns:
        A PIL Image of the screenshot.
    """
    try:
        import mss
        from PIL import Image

        region = region or SCREEN_REGION

        with mss.mss() as sct:
            if region:
                monitor = {
                    "left": region[0],
                    "top": region[1],
                    "width": region[2],
                    "height": region[3],
                }
            else:
                monitor = sct.monitors[0]  # Full virtual screen

            screenshot = sct.grab(monitor)
            img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
            logger.info(f"[SCREEN] Screenshot captured: {img.size[0]}x{img.size[1]}")
            return img

    except ImportError:
        logger.error("[SCREEN] mss or Pillow not installed — pip install mss Pillow")
        return None
    except Exception as e:
        logger.error(f"[SCREEN] Screenshot failed: {e}")
        return None


def capture_screenshot_base64(quality: int = 75) -> str | None:
    """
    Capture the full screen and return as a base64-encoded JPEG string.
    Compresses to reduce token usage when sending to vision LLM.
    Returns None on failure.
    """
    try:
        import mss
        import base64
        from PIL import Image
        import io

        with mss.mss() as sct:
            monitor = sct.monitors[0]
            raw = sct.grab(monitor)
            img = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")

        # Resize to max 1600px wide to reduce base64 size
        max_width = 1600
        if img.width > max_width:
            ratio = max_width / img.width
            new_height = int(img.height * ratio)
            img = img.resize((max_width, new_height), Image.LANCZOS)

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        logger.debug(f"[SCREEN] Screenshot base64 size: {len(b64) // 1024}KB")
        return b64

    except Exception as e:
        logger.error(f"[SCREEN] capture_screenshot_base64 failed: {e}")
        return None


# ─── OCR ─────────────────────────────────────────────────────────────────────


def ocr_screen(region: Optional[tuple[int, int, int, int]] = None) -> str:
    """
    Run OCR on the screen (or a region) and return the text.

    Args:
        region: Optional (left, top, width, height). If None, full screen.

    Returns:
        Cleaned text string from the screen, or "" on failure.
    """
    try:
        import numpy as np

        img = capture_screenshot(region)
        if img is None:
            return ""

        reader = _get_ocr_reader()
        img_array = np.array(img)
        results = reader.readtext(img_array, detail=0)

        text = " ".join(results).strip()
        logger.info(f"[SCREEN] OCR result ({len(text)} chars): \"{text[:120]}...\"")
        return text

    except Exception as e:
        logger.error(f"[SCREEN] OCR failed: {e}")
        return ""


# ─── Window Detection ────────────────────────────────────────────────────────


def get_open_windows() -> list[str]:
    """
    Return a list of visible window titles.

    Returns:
        List of window title strings (empty titles filtered out).
    """
    try:
        import pygetwindow as gw

        try:
            windows = gw.getAllTitles()
        except RecursionError:
            logger.warning("[SCREEN] pygetwindow hit recursion limit — returning empty list")
            return []

        # Filter out empty/blank titles
        visible = [w.strip() for w in windows if w.strip()]
        logger.info(f"[SCREEN] Found {len(visible)} windows")
        return visible

    except ImportError:
        logger.error("[SCREEN] pygetwindow not installed — pip install pygetwindow")
        return []
    except Exception as e:
        logger.error(f"[SCREEN] Window detection failed: {e}")
        return []


def get_active_window() -> str:
    """
    Return the title of the currently focused window.

    Returns:
        Window title string, or "" if detection fails.
    """
    try:
        import pygetwindow as gw

        active = gw.getActiveWindow()
        if active and active.title:
            title = active.title.strip()
            logger.info(f"[SCREEN] Active window: \"{title}\"")
            return title
        return ""

    except ImportError:
        logger.error("[SCREEN] pygetwindow not installed")
        return ""
    except Exception as e:
        logger.error(f"[SCREEN] Active window detection failed: {e}")
        return ""


# ─── Text Search ─────────────────────────────────────────────────────────────


def find_text_on_screen(text: str) -> list[tuple[int, int]] | None:
    """
    Find the screen coordinates of a text string using OCR.

    Args:
        text: The text to search for (case-insensitive substring match).

    Returns:
        A list of (x, y) center coordinates where the text was found,
        or None if not found.
    """
    try:
        import numpy as np

        img = capture_screenshot()
        if img is None:
            return None

        reader = _get_ocr_reader()
        img_array = np.array(img)
        results = reader.readtext(img_array, detail=1)

        search_lower = text.lower()
        matches = []
        # For short search terms (<=5 chars), require word-boundary matching
        # to avoid "play" matching "Playlist", "display", etc.
        import re as _re
        _short_term = len(search_lower) <= 5
        if _short_term:
            _word_pattern = _re.compile(r'(?<!\w)' + _re.escape(search_lower) + r'(?!\w)', _re.IGNORECASE)

        for bbox, detected_text, confidence in results:
            detected_lower = detected_text.lower()

            # 1. Standard match — exact or word-boundary for short terms, substring for longer
            if _short_term:
                is_match = bool(_word_pattern.search(detected_text))
            else:
                is_match = search_lower in detected_lower

            # 2. Reverse substring (detected block is inside the search phrase)
            # This handles cases where EasyOCR splits a long phrase like "What's Up Danger"
            # into multiple blocks like "What's" and "Up Danger".
            if not is_match and len(search_lower) >= 6 and len(detected_lower) >= 4:
                if detected_lower in search_lower:
                    is_match = True

            if is_match:
                # Filter 1: High confidence only to avoid false positives
                if confidence < 0.8:  # Lowered slightly to allow for stylized text like in Spotify
                    logger.debug(f"[SCREEN] Skipping '{detected_text}' due to low confidence ({confidence:.2f})")
                    continue
                
                # Filter 2: Discard Python code artifacts often visible in terminal / LLM logs
                if '("' in detected_text or "('" in detected_text:
                    logger.debug(f"[SCREEN] Skipping '{detected_text}' as likely code artifact")
                    continue

                # bbox is [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
                x_center = int((bbox[0][0] + bbox[2][0]) / 2)
                y_center = int((bbox[0][1] + bbox[2][1]) / 2)
                matches.append((x_center, y_center))
                logger.info(
                    f"[SCREEN] Found \"{detected_text}\" at ({x_center}, {y_center}) "
                    f"confidence={confidence:.2f}"
                )

        if matches:
            return matches

        logger.info(f"[SCREEN] Text \"{text}\" not found on screen")
        return None

    except Exception as e:
        logger.error(f"[SCREEN] Text search failed: {e}")
        return None


def list_ocr_blocks(min_confidence: float = 0.6) -> list[dict] | None:
    """
    Run OCR on the full screen and return every detected text block with its
    centre coordinate and confidence. Used by callers that need to score
    candidates against a search phrase (e.g. vision-LLM coord snap) rather
    than substring-matching one word at a time.

    Returns:
        List of dicts {"text": str, "x": int, "y": int, "confidence": float}
        sorted by confidence desc, or None on failure.
    """
    try:
        import numpy as np

        img = capture_screenshot()
        if img is None:
            return None

        reader = _get_ocr_reader()
        img_array = np.array(img)
        results = reader.readtext(img_array, detail=1)

        blocks = []
        for bbox, detected_text, confidence in results:
            if confidence < min_confidence:
                continue
            if not detected_text or not detected_text.strip():
                continue
            x_center = int((bbox[0][0] + bbox[2][0]) / 2)
            y_center = int((bbox[0][1] + bbox[2][1]) / 2)
            blocks.append({
                "text": detected_text,
                "x": x_center,
                "y": y_center,
                "confidence": float(confidence),
            })
        blocks.sort(key=lambda b: b["confidence"], reverse=True)
        return blocks

    except Exception as e:
        logger.error(f"[SCREEN] OCR block listing failed: {e}")
        return None


# ─── LLM-Ready Screen Description ───────────────────────────────────────────


def describe_screen_for_llm() -> str:
    """
    Build a compact screen description for the LLM.

    Combines:
      - List of open window titles
      - Active window title
      - OCR text from the screen (truncated to stay ≤1500 tokens)

    Returns:
        A formatted string suitable for an LLM system/user prompt.
    """
    try:
        parts = []

        # Open windows
        windows = get_open_windows()
        if windows:
            # Limit to 20 window titles to save tokens
            window_list = windows[:20]
            parts.append("OPEN WINDOWS:\n" + "\n".join(f"  - {w}" for w in window_list))
            if len(windows) > 20:
                parts.append(f"  ... and {len(windows) - 20} more")

        # Active window
        active = get_active_window()
        if active:
            parts.append(f"\nACTIVE WINDOW: {active}")

        # OCR text (truncated)
        ocr_text = ocr_screen()
        if ocr_text:
            # Rough token estimate: ~4 chars per token, keep under 1000 tokens
            max_chars = 4000
            if len(ocr_text) > max_chars:
                ocr_text = ocr_text[:max_chars] + "... (truncated)"
            parts.append(f"\nSCREEN TEXT (OCR):\n{ocr_text}")

        if not parts:
            return "Could not read screen content."

        description = "\n".join(parts)
        logger.info(f"[SCREEN] Screen description ready ({len(description)} chars)")
        return description

    except Exception as e:
        logger.error(f"[SCREEN] Screen description failed: {e}")
        return "Error reading screen."
