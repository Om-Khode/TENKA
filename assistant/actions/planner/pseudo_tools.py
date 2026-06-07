"""
pseudo_tools.py — Planner-internal pseudo-tool implementations.

These are tools that the planner executes directly instead of dispatching
through actions.execute(). They handle camera/vision/synthesis/prompting
that lives entirely within the planner's execution loop.

Pseudo-tools:
  - synthesize       : LLM call for mid-plan data processing
  - vision_analyze   : Camera capture + vision LLM
  - prompt_user      : TTS prompt + timed pause for physical action
  - camera_preview   : Live OpenCV preview with overlay, user captures via SPACE
"""

import logging
import re

logger = logging.getLogger("planner")


# --- synthesize ---

async def run_synthesize_step(goal: str, llm_func) -> str:
    """
    Handle the "synthesize" pseudo-tool.
    Calls the LLM to analyze/transform/extract from previous step outputs.
    Uses Cerebras (synthesis task type) — cheap and fast.

    Uses a strict data-only system prompt to prevent personality bleed
    from contaminating structured output that downstream steps depend on.
    """
    system = (
        "You are a data processing assistant. Output ONLY the requested data. "
        "Rules:\n"
        "1. Do NOT add commentary, opinions, personality, or filler text.\n"
        "2. Do NOT start with emotion tags like [sarcastic] or [happy].\n"
        "3. Do NOT add markdown formatting (no **, no headers, no bullets).\n"
        "4. If asked to clean up or format data, output ONLY the cleaned data.\n"
        "5. If asked to convert or transform data, output ONLY the result.\n"
        "6. Be precise and literal. Follow the exact output format requested."
    )

    result = await llm_func(
        goal,
        system_prompt=system,
        task_type="synthesis",
        max_tokens=400,
    )

    if result == "__LLM_UNAVAILABLE__":
        return "Sorry, I couldn't analyze the results — LLM unavailable."

    result = re.sub(
        r'^\[(?:neutral|happy|excited|sad|angry|sarcastic|worried|surprised)\]\s*',
        '', result
    )

    return result


# --- vision_analyze ---

async def run_vision_analyze_step(goal: str, tts_func=None) -> str:
    """
    Handle the "vision_analyze" pseudo-tool.

    Captures a fresh camera image and sends it to the vision LLM with
    a structured prompt. Returns the vision LLM's response directly.

    Unlike camera_look (which goes through actions.py and adds personality),
    this gives the planner raw structured output from the vision model —
    ideal for tasks like reading colors, counting items, extracting text
    from physical objects.
    """
    import asyncio
    from ... import llm as llm_module
    from ... import config as _config

    if not _config.CAMERA_ENABLED:
        return "ERROR: Camera is currently disabled."

    try:
        import cv2
        import base64

        logger.info("[PLANNER] vision_analyze: opening camera...")
        cap = cv2.VideoCapture(_config.CAMERA_INDEX)

        if not cap.isOpened():
            return "ERROR: Couldn't access the camera."

        max_w = getattr(_config, 'CAMERA_MAX_WIDTH', 1280)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, max_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(max_w * 0.75))

        for _ in range(10):
            cap.read()
            await asyncio.sleep(0.1)

        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            return "ERROR: Couldn't capture a frame from the camera."

        logger.info("[PLANNER] vision_analyze: frame captured")

        if tts_func:
            await tts_func("Got it.")

        try:
            from datetime import datetime
            debug_dir = _config.SANDBOX_DIR / "debug_captures"
            debug_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            debug_path = debug_dir / f"vision_{timestamp}.jpg"
            cv2.imwrite(str(debug_path), frame)
            logger.info(f"[PLANNER] vision_analyze: debug saved → {debug_path}")
        except Exception as e:
            logger.debug(f"[PLANNER] debug save failed: {e}")

        _, jpeg_buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        image_b64 = base64.b64encode(jpeg_buf).decode('utf-8')

    except Exception as e:
        logger.error(f"[PLANNER] vision_analyze: camera error: {e}")
        return f"ERROR: Camera capture failed: {e}"

    vision_system = (
        "You are a precise visual analysis assistant. "
        "Follow these rules strictly:\n"
        "1. Answer ONLY what is asked. No commentary, no personality, no filler.\n"
        "2. When asked for structured output (colors, lists, counts, coordinates), "
        "use the EXACT format specified in the question. Do not wrap it in "
        "markdown, headers, or labels unless the question asks for labels.\n"
        "3. If you cannot clearly see something, say 'UNCLEAR' for that item "
        "rather than guessing.\n"
        "4. Do NOT describe the image unless asked to describe it. "
        "Extract the specific data requested.\n"
        "5. Be precise and factual. No hedging, no 'appears to be', no 'it seems'."
    )

    result = (await llm_module.get_vision_response(
        image_base64=image_b64,
        prompt=goal,
        system_prompt=vision_system,
    )).text

    if result == "__LLM_UNAVAILABLE__":
        return "ERROR: Vision LLM unavailable."

    logger.info(f"[PLANNER] vision_analyze result: {result[:150]}")
    return result


# --- prompt_user ---

_PROMPT_USER_PAUSE = 5.0


async def run_prompt_user_step(goal: str, tts_func) -> str:
    """
    Handle the "prompt_user" pseudo-tool.

    Speaks a message via TTS asking the user to perform a physical action
    (rotate an object, hold up a document, etc.), then pauses for a few
    seconds to let them do it. Resumes automatically — no pending state,
    no user speech needed.

    This is fundamentally different from interactive suspension:
    - Interactive suspension: waits for user SPEECH (yes/no, OAuth code, etc.)
    - prompt_user: waits for user PHYSICAL ACTION (timed pause)
    """
    import asyncio

    if tts_func:
        await tts_func(goal)

    logger.info(
        f"[PLANNER] prompt_user: pausing {_PROMPT_USER_PAUSE}s for user action"
    )
    await asyncio.sleep(_PROMPT_USER_PAUSE)

    return f"User was prompted: '{goal}' — paused {_PROMPT_USER_PAUSE}s for action."


# --- camera_preview ---

_OVERLAY_KEYWORDS = {
    "grid_3x3": ["grid_3x3", "3x3", "3 x 3", "grid 3"],
    "grid_4x4": ["grid_4x4", "4x4", "4 x 4", "grid 4"],
    "crosshair": ["crosshair", "cross hair", "center point", "point"],
    "rectangle": ["rectangle", "rect", "document", "card", "page"],
}


def _parse_overlay_type(goal: str) -> str:
    """
    Parse the overlay type from the goal text.
    Returns the overlay name or "none" if no keyword matches.
    """
    goal_lower = goal.lower()
    for overlay_type, keywords in _OVERLAY_KEYWORDS.items():
        for kw in keywords:
            if kw in goal_lower:
                return overlay_type
    if "grid" in goal_lower:
        return "grid_3x3"
    return "none"


def _draw_overlay(display, overlay: str) -> dict:
    """
    Draw the specified overlay on a display frame (in-place).
    The overlay is for visual guidance only — the captured frame is raw.

    Returns a dict with overlay metadata (e.g., grid bounds) so that the
    return value of camera_preview can inform code_executor where to sample.
    """
    import cv2

    h, w = display.shape[:2]
    color = (0, 255, 0)
    thin = 1
    thick = 2
    meta = {}

    if overlay.startswith("grid_"):
        try:
            n = int(overlay.split("_")[1].split("x")[0])
        except (IndexError, ValueError):
            n = 3

        side = min(w, h)
        padding = side // 12
        side = side - 2 * padding
        x0 = (w - side) // 2
        y0 = (h - side) // 2

        meta["grid_x"] = x0
        meta["grid_y"] = y0
        meta["grid_side"] = side
        meta["grid_n"] = n

        cv2.rectangle(display, (x0, y0), (x0 + side, y0 + side), color, thick)

        cell = side // n
        for i in range(1, n):
            x = x0 + cell * i
            cv2.line(display, (x, y0), (x, y0 + side), color, thick)
            y = y0 + cell * i
            cv2.line(display, (x0, y), (x0 + side, y), color, thick)

        cell_num = 1
        for row in range(n):
            for col in range(n):
                lx = x0 + cell * col + 5
                ly = y0 + cell * row + 20
                cv2.putText(display, str(cell_num), (lx, ly),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1, cv2.LINE_AA)
                cell_num += 1

        for row in range(n):
            for col in range(n):
                cx = x0 + cell * col + cell // 2
                cy = y0 + cell * row + cell // 2
                cv2.circle(display, (cx, cy), 3, (0, 200, 255), -1)

    elif overlay == "crosshair":
        cv2.line(display, (w // 2, 0), (w // 2, h), color, thin)
        cv2.line(display, (0, h // 2), (w, h // 2), color, thin)
        cv2.circle(display, (w // 2, h // 2), 15, color, thick)

    elif overlay == "rectangle":
        margin_x = w // 10
        margin_y = h // 10
        cv2.rectangle(display,
                      (margin_x, margin_y),
                      (w - margin_x, h - margin_y),
                      color, thick)
        corner_len = min(w, h) // 15
        corners = [
            (margin_x, margin_y),
            (w - margin_x, margin_y),
            (margin_x, h - margin_y),
            (w - margin_x, h - margin_y),
        ]
        for cx, cy in corners:
            dx = corner_len if cx == margin_x else -corner_len
            cv2.line(display, (cx, cy), (cx + dx, cy), color, thick + 1)
            dy = corner_len if cy == margin_y else -corner_len
            cv2.line(display, (cx, cy), (cx, cy + dy), color, thick + 1)

    return meta


def _camera_preview_blocking(camera_index: int, overlay: str) -> tuple[str | None, dict]:
    """
    Blocking camera preview with overlay. Returns (file_path, metadata).

    Runs in a thread via run_in_executor — OpenCV's highgui (imshow/waitKey)
    requires a thread context.
    """
    import cv2
    from datetime import datetime
    from ... import config as _config

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        logger.error("[CAMERA_PREVIEW] Could not open camera")
        return None, {}

    max_w = getattr(_config, 'CAMERA_MAX_WIDTH', 1280)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, max_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(max_w * 0.75))

    for _ in range(10):
        cap.read()

    captured_path = None
    overlay_meta = {}
    window_name = f"{_config.ASSISTANT_NAME_DISPLAY} Camera - Align and Press SPACE"

    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("[CAMERA_PREVIEW] Failed to read frame")
                break

            display = frame.copy()
            overlay_meta = _draw_overlay(display, overlay)

            h, w = display.shape[:2]
            bar_text = "SPACE = Capture  |  ESC = Cancel"
            if overlay != "none":
                bar_text += f"  |  Overlay: {overlay}"
            cv2.putText(display, bar_text,
                        (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key == 32:  # SPACE
                capture_dir = _config.SANDBOX_DIR / "captures"
                capture_dir.mkdir(parents=True, exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = str(capture_dir / f"preview_{ts}.jpg")
                cv2.imwrite(path, frame)
                captured_path = path
                logger.info(f"[CAMERA_PREVIEW] Captured: {path}")
                break
            elif key == 27:  # ESC
                logger.info("[CAMERA_PREVIEW] Cancelled by user")
                break

    except Exception as e:
        logger.error(f"[CAMERA_PREVIEW] Error in preview loop: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return captured_path, overlay_meta


async def run_camera_preview_step(goal: str, tts_func=None) -> str:
    """
    Handle the "camera_preview" pseudo-tool.

    Opens a live camera preview window with configurable overlay. User aligns
    the target object and presses SPACE to capture. Returns the file path of
    the captured frame.
    """
    import asyncio
    from ... import config as _config

    if not _config.CAMERA_ENABLED:
        return "ERROR: Camera is currently disabled."

    overlay = _parse_overlay_type(goal)
    logger.info(f"[CAMERA_PREVIEW] Overlay: {overlay} | Goal: {goal[:80]}")

    if tts_func:
        overlay_hint = ""
        if overlay.startswith("grid_"):
            overlay_hint = " I'll show a grid overlay to help you align."
        elif overlay == "crosshair":
            overlay_hint = " I'll show a crosshair to help you center it."
        elif overlay == "rectangle":
            overlay_hint = " I'll show a rectangle to help you align the edges."
        await tts_func(
            f"Opening camera preview.{overlay_hint} "
            f"Press SPACE when aligned."
        )

    loop = asyncio.get_running_loop()
    result_path, overlay_meta = await loop.run_in_executor(
        None,
        _camera_preview_blocking,
        _config.CAMERA_INDEX,
        overlay,
    )

    if not result_path:
        return "ERROR: Camera preview cancelled or failed to capture."

    if tts_func:
        await tts_func("Got it.")

    result = f"Image captured: {result_path}"
    if overlay_meta.get("grid_x") is not None:
        gx = overlay_meta["grid_x"]
        gy = overlay_meta["grid_y"]
        gs = overlay_meta["grid_side"]
        gn = overlay_meta["grid_n"]
        result += (
            f" | Grid region: x={gx} y={gy} side={gs} cells={gn}x{gn}"
            f" (crop the image to this square region before sampling cells)"
        )

    return result
