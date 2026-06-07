# assistant/io/overlay/windows.py
"""overlay — Concept A (Pill).

A single Tk Toplevel containing a `tk.Canvas` that draws the entire pill
ourselves: rounded background, accent border, animated glyph, label,
detail, step chip, tier badge. Tk's native widgets can't get this look —
we draw with canvas primitives.

The window stays frameless, always-on-top, click-through (WS_EX_LAYERED |
WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW). Chroma-key transparency fakes
the rounded corners against a solid magenta backdrop.

The pill is pinned top-right of the primary monitor — the design's
`cursor_follows` flag is intentionally ignored in this concept: Pill is
a single fixed surface, no separate cursor badge. (Cursor-follow is
deferred — concept E is the right fit if we want it later.)
"""
from __future__ import annotations

import ctypes
import logging
import math
import tkinter as tk

from . import theme

logger = logging.getLogger("overlay.windows")

# Win32 extended window style flags
_GWL_EXSTYLE = -20
_WS_EX_LAYERED = 0x00080000
_WS_EX_TRANSPARENT = 0x00000020
_WS_EX_TOOLWINDOW = 0x00000080


def _make_click_through(win: tk.Toplevel) -> None:
    try:
        hwnd = ctypes.windll.user32.GetParent(win.winfo_id())
        current = ctypes.windll.user32.GetWindowLongW(hwnd, _GWL_EXSTYLE)
        ctypes.windll.user32.SetWindowLongW(
            hwnd, _GWL_EXSTYLE,
            current | _WS_EX_LAYERED | _WS_EX_TRANSPARENT | _WS_EX_TOOLWINDOW,
        )
    except (AttributeError, OSError) as e:
        logger.warning("[overlay] click-through setup failed: %s", e)


# ─── Rounded-rectangle helper (Canvas can't do real rounded rects) ─────────
def _rounded_rect(canvas: tk.Canvas, x1: int, y1: int, x2: int, y2: int,
                  r: int, **kwargs) -> int:
    """Draw a filled rounded rectangle and return its item id (polygon)."""
    pts = [
        x1 + r, y1,
        x2 - r, y1,
        x2, y1,
        x2, y1 + r,
        x2, y2 - r,
        x2, y2,
        x2 - r, y2,
        x1 + r, y2,
        x1, y2,
        x1, y2 - r,
        x1, y1 + r,
        x1, y1,
    ]
    return canvas.create_polygon(pts, smooth=True, **kwargs)


# ─── Status Pill (Concept A) ───────────────────────────────────────────────
class StatusPill(tk.Toplevel):
    """Single canvas-drawn pill that re-paints on phase change and animates the glyph at ANIM_FPS."""

    def __init__(self, root: tk.Tk) -> None:
        super().__init__(root)
        self.overrideredirect(True)
        self.attributes("-topmost", True)
        self.attributes("-alpha", theme.ALPHA)
        try:
            self.attributes("-transparentcolor", theme.CHROMA_KEY)
        except tk.TclError:
            pass
        self.configure(bg=theme.CHROMA_KEY)  # chroma surround → invisible

        # canvas covers the entire window; bg is chroma key so rounded corners look real
        self.canvas = tk.Canvas(self, bg=theme.CHROMA_KEY, highlightthickness=0, bd=0)
        self.canvas.pack(fill="both", expand=True)

        self._phase = "IDLE"
        self._detail = ""
        self._step: tuple[int, int] | None = None
        self._tier: str | None = None
        self._anim_frame = 0

        self.withdraw()
        self.after(50, lambda: _make_click_through(self))
        self._schedule_anim()

    # ─── public API ─────────────────────────────────────────────────────────
    def update_content(self, phase: str, detail: str,
                       step: tuple[int, int] | None,
                       tier: str | None) -> None:
        self._phase = phase
        self._detail = detail
        self._step = step
        self._tier = tier
        self._anim_frame = 0
        self._repaint()

    def show(self) -> None:
        self.deiconify()

    def hide(self) -> None:
        self.withdraw()

    # ─── layout + paint ─────────────────────────────────────────────────────
    def _repaint(self) -> None:
        self.canvas.delete("all")
        style = theme.style_for(self._phase)
        tier_style = theme.tier_for(self._tier)

        # measure to decide pill width
        label = style.label
        detail = self._detail
        step_text = f"{self._step[0]}/{self._step[1]}" if self._step else ""
        tier_text = tier_style.label if tier_style else ""

        # rough width budget — Tk measures lazily; we approximate then fit
        approx_w = theme.PILL_PAD_X * 2
        approx_w += 18  # glyph slot
        approx_w += 7 * len(label) + 12     # label
        if detail:
            approx_w += 10 + 6 * min(len(detail), 28) + 8  # divider + detail
        if step_text:
            approx_w += 30
        if tier_text:
            approx_w += 38 + len(tier_text) * 5
        w = max(theme.PILL_MIN_W, min(theme.PILL_MAX_W, approx_w))
        h = theme.PILL_HEIGHT
        self.geometry(f"{w}x{h}")
        self._place_top_right(w, h)

        # background pill
        border = theme.mix_hex(style.color, theme.BG_COLOR, theme.BORDER_TINT_PCT)
        _rounded_rect(
            self.canvas, 1, 1, w - 1, h - 1, theme.PILL_CORNER_R,
            fill=theme.BG_COLOR, outline=border, width=1,
        )

        # subtle accent rail on left edge (3px)
        self.canvas.create_rectangle(
            1, h // 2 - 10, 4, h // 2 + 10,
            fill=style.color, outline="",
        )

        # glyph (drawn dynamically per animation frame)
        glyph_cx = theme.PILL_PAD_X + 8
        glyph_cy = h // 2
        self._draw_glyph(style.color, style.glyph, glyph_cx, glyph_cy)

        # label
        x = glyph_cx + 16
        self.canvas.create_text(
            x, glyph_cy - 1, anchor="w", text=label,
            fill=theme.FG_LABEL, font=theme.FONT_LABEL,
        )
        # measure label width
        bbox = self.canvas.bbox("all")
        text_id = self.canvas.create_text(0, -100, text=label, font=theme.FONT_LABEL, anchor="w")
        label_w = self.canvas.bbox(text_id)[2] - self.canvas.bbox(text_id)[0]
        self.canvas.delete(text_id)
        x += label_w + theme.PILL_GAP

        # detail
        if detail:
            self.canvas.create_text(
                x, glyph_cy + 1, anchor="w", text="·",
                fill=theme.FG_DIVIDER, font=theme.FONT_DETAIL,
            )
            x += 8
            self.canvas.create_text(
                x, glyph_cy, anchor="w",
                text=(detail if len(detail) <= 32 else detail[:30] + "…"),
                fill=theme.FG_DETAIL, font=theme.FONT_DETAIL,
            )
            t_id = self.canvas.create_text(0, -100, text=detail[:32], font=theme.FONT_DETAIL, anchor="w")
            detail_w = self.canvas.bbox(t_id)[2] - self.canvas.bbox(t_id)[0]
            self.canvas.delete(t_id)
            x += detail_w + theme.PILL_GAP

        # tier badge (right side)
        right_x = w - theme.PILL_PAD_X
        if tier_style:
            tw = 8 + len(tier_style.label) * 5 + 8
            tx2 = right_x
            tx1 = tx2 - tw
            ty1 = h // 2 - 9
            ty2 = h // 2 + 9
            tbg = theme.mix_hex(tier_style.color, theme.BG_COLOR, 18)
            tborder = theme.mix_hex(tier_style.color, theme.BG_COLOR, 35)
            _rounded_rect(self.canvas, tx1, ty1, tx2, ty2, 6, fill=tbg, outline=tborder, width=1)
            self.canvas.create_text(
                (tx1 + tx2) // 2, h // 2, text=tier_style.label,
                fill=tier_style.color, font=theme.FONT_TIER,
            )
            right_x = tx1 - 6

        # step chip (left of tier badge or right edge)
        if step_text:
            sw = 8 + len(step_text) * 6 + 6
            sx2 = right_x
            sx1 = sx2 - sw
            sy1 = h // 2 - 9
            sy2 = h // 2 + 9
            sbg = theme.mix_hex(style.color, theme.BG_COLOR, 18)
            _rounded_rect(self.canvas, sx1, sy1, sx2, sy2, 6, fill=sbg, outline="")
            self.canvas.create_text(
                (sx1 + sx2) // 2, h // 2, text=step_text,
                fill=style.color, font=theme.FONT_STEP,
            )

    def _place_top_right(self, w: int, h: int) -> None:
        screen_w = self.winfo_screenwidth()
        x = screen_w - w - theme.PILL_MARGIN
        y = theme.PILL_MARGIN
        self.geometry(f"{w}x{h}+{x}+{y}")

    # ─── glyph rendering ────────────────────────────────────────────────────
    def _draw_glyph(self, color: str, kind: str, cx: int, cy: int) -> None:
        size = 14
        r = size // 2
        f = self._anim_frame

        if kind == theme.GLYPH_IDLE:
            self.canvas.create_oval(
                cx - 4, cy - 4, cx + 4, cy + 4,
                fill=color, outline="",
            )
            return

        if kind == theme.GLYPH_SPIN:
            # rotating arc, ~270° sweep
            # one full revolution per ~0.8s → 24 frames at 30fps
            start = (f * (360 / 24)) % 360
            ring_dim = theme.mix_hex(color, theme.BG_COLOR, 22)
            self.canvas.create_oval(
                cx - r, cy - r, cx + r, cy + r,
                outline=ring_dim, width=2,
            )
            self.canvas.create_arc(
                cx - r, cy - r, cx + r, cy + r,
                start=start, extent=120, style="arc",
                outline=color, width=2,
            )
            return

        if kind == theme.GLYPH_PULSE:
            # expanding ring + core dot
            # one cycle per 42 frames at 30fps (1.4s)
            t = (f % 42) / 42  # 0..1
            ring_r = int(3 + t * (r + 2))
            opacity_pct = int(65 * (1 - t))  # fades to 0
            if opacity_pct > 0:
                ring_col = theme.mix_hex(color, theme.BG_COLOR, max(10, opacity_pct))
                self.canvas.create_oval(
                    cx - ring_r, cy - ring_r, cx + ring_r, cy + ring_r,
                    outline=ring_col, width=2,
                )
            # core
            self.canvas.create_oval(
                cx - 3, cy - 3, cx + 3, cy + 3,
                fill=color, outline="",
            )
            return

        if kind == theme.GLYPH_WORK:
            # 3 bouncing dots
            # one cycle per 27 frames (0.9s)
            for i, ox in enumerate((-7, 0, 7)):
                phase = (f + i * 4) % 27
                y_off = -int(math.sin(phase / 27 * math.pi * 2) * 3)
                dot_r = 2
                self.canvas.create_oval(
                    cx + ox - dot_r, cy + y_off - dot_r,
                    cx + ox + dot_r, cy + y_off + dot_r,
                    fill=color, outline="",
                )
            return

        if kind == theme.GLYPH_WAVE:
            # 4 bars scaling vertically
            for i, ox in enumerate((-6, -2, 2, 6)):
                phase = (f + i * 3) % 27
                h_scale = 0.3 + 0.7 * abs(math.sin(phase / 27 * math.pi * 2))
                bar_h = int(10 * h_scale)
                self.canvas.create_rectangle(
                    cx + ox - 1, cy - bar_h // 2,
                    cx + ox + 1, cy + bar_h // 2,
                    fill=color, outline="",
                )
            return

        if kind == theme.GLYPH_CHECK:
            # static check mark
            self.canvas.create_line(
                cx - 5, cy + 1, cx - 1, cy + 4, cx + 5, cy - 3,
                fill=color, width=2, capstyle="round", joinstyle="round",
            )
            return

        if kind == theme.GLYPH_STOP:
            # static X mark — user-cancelled
            self.canvas.create_line(
                cx - 4, cy - 4, cx + 4, cy + 4,
                fill=color, width=2, capstyle="round",
            )
            self.canvas.create_line(
                cx + 4, cy - 4, cx - 4, cy + 4,
                fill=color, width=2, capstyle="round",
            )
            return

    # ─── animation tick ─────────────────────────────────────────────────────
    def _schedule_anim(self) -> None:
        self.after(theme.ANIM_TICK_MS, self._anim_tick)

    def _anim_tick(self) -> None:
        try:
            if self.state() == "normal" and self._phase != "IDLE":
                style = theme.style_for(self._phase)
                # only animated glyphs need a redraw
                if style.glyph in (theme.GLYPH_SPIN, theme.GLYPH_PULSE,
                                   theme.GLYPH_WORK, theme.GLYPH_WAVE):
                    self._anim_frame = (self._anim_frame + 1) % 600
                    self._repaint()
        finally:
            self._schedule_anim()


# ─── Back-compat aliases (kept so existing __main__.py imports still work) ──
# windows.py used to export CursorBadge + CornerToast; concept A collapses
# both into a single StatusPill. The aliases let the rest of the package
# keep working without per-call-site rewrites.
CursorBadge = StatusPill
CornerToast = StatusPill
