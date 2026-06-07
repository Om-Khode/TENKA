# assistant/io/overlay/theme.py
"""overlay theme — Concept A (Pill).

Single source of truth for phase → color/glyph/animation mapping plus
pill geometry, font, timing. Mirrors the design at
docs/superpowers/specs/2026-05-31-cv1-cursor-visibility-design.md §9
(updated 2026-05-31 to match the toast-concepts Concept A).
"""
from __future__ import annotations

from dataclasses import dataclass


# ─── Glyph types (animated by windows.py) ─────────────────────────────────
GLYPH_IDLE  = "idle"   # static small dot
GLYPH_SPIN  = "spin"   # rotating arc (thinking / planning)
GLYPH_PULSE = "pulse"  # expanding ring (listening)
GLYPH_WORK  = "work"   # 3 dots bouncing (reading screen / clicking)
GLYPH_WAVE  = "wave"   # 4 bars scaling (speaking)
GLYPH_CHECK = "check"  # static check mark (done)


@dataclass(frozen=True)
class PhaseStyle:
    color: str       # accent — hex, used for glyph + border + glow
    glyph: str       # one of GLYPH_* — drives animation
    label: str       # human-readable, shown in the pill


# New glyph for STOPPED — an X mark; declared so windows.py can render it.
GLYPH_STOP = "stop"

PHASE_STYLES: dict[str, PhaseStyle] = {
    "IDLE":      PhaseStyle("#6b7280", GLYPH_IDLE,  "Idle"),
    "LISTENING": PhaseStyle("#22d3ee", GLYPH_PULSE, "Listening"),
    "THINKING":  PhaseStyle("#6098ff", GLYPH_SPIN,  "Thinking"),
    "PLANNING":  PhaseStyle("#a78bfa", GLYPH_SPIN,  "Planning"),
    "READING":   PhaseStyle("#c084fc", GLYPH_WORK,  "Reading screen"),
    "VISION":    PhaseStyle("#c084fc", GLYPH_WORK,  "Vision"),
    "CLICKING":  PhaseStyle("#f5b544", GLYPH_WORK,  "Clicking"),
    "TYPING":    PhaseStyle("#f5b544", GLYPH_WORK,  "Typing"),
    "BROWSING":  PhaseStyle("#6098ff", GLYPH_SPIN,  "Browsing"),
    "HEALING":   PhaseStyle("#fb923c", GLYPH_SPIN,  "Healing"),
    "SPEAKING":  PhaseStyle("#34d399", GLYPH_WAVE,  "Speaking"),
    "DONE":      PhaseStyle("#4ade80", GLYPH_CHECK, "Done"),
    "STOPPED":   PhaseStyle("#ef4444", GLYPH_STOP,  "Stopped"),
}

UNKNOWN_PHASE_STYLE = PhaseStyle("#cccccc", GLYPH_SPIN, "Working")


# ─── Tier badges (vision / native / browser) ──────────────────────────────
@dataclass(frozen=True)
class TierStyle:
    color: str
    label: str


TIER_STYLES: dict[str, TierStyle] = {
    "vision":  TierStyle("#c084fc", "VISION"),
    "native":  TierStyle("#f5b544", "NATIVE"),
    "browser": TierStyle("#6098ff", "BROWSER"),
}


# ─── Pill geometry ────────────────────────────────────────────────────────
PILL_HEIGHT = 40
PILL_PAD_X = 13
PILL_GAP = 10           # gap between glyph / label / detail
PILL_CORNER_R = 20      # half of PILL_HEIGHT = full pill
PILL_MIN_W = 120
PILL_MAX_W = 360
PILL_MARGIN = 16        # top/right margin from screen corner


# ─── Color palette ────────────────────────────────────────────────────────
BG_COLOR = "#12141a"           # rgba(18,20,26,.92) — translucent navy
FG_LABEL = "#eef0f3"           # primary label text
FG_DETAIL = "#8a93a2"          # gray detail line
FG_DIVIDER = "#3a3f4b"         # the · dot between label and detail
BORDER_TINT_PCT = 32           # how much of accent color to mix into border
CHROMA_KEY = "#ff00ff"         # transparency chroma key
ALPHA = 0.96


# ─── Fonts (Windows default = Segoe UI) ───────────────────────────────────
FONT_LABEL = ("Segoe UI", 10, "bold")
FONT_DETAIL = ("Segoe UI", 9)
FONT_STEP = ("Segoe UI", 8, "bold")
FONT_TIER = ("Segoe UI", 7, "bold")


# ─── Animation / poll timing ──────────────────────────────────────────────
ANIM_FPS = 30
ANIM_TICK_MS = max(1, int(1000 / ANIM_FPS))
CURSOR_POLL_HZ = 30
CURSOR_REPOSITION_THRESHOLD_PX = 3


# ─── Rotating detail labels ───────────────────────────────────────────────
# Lightweight personality — when a handler has nothing specific to say in
# the detail slot, we rotate through a small hardcoded set so the pill
# never reads as the literal word "step" or "loop 3".

VISION_LOOP_LABELS = (
    "scanning the screen",
    "checking the result",
    "deciding next move",
    "looking for the goal",
    "comparing to plan",
    "double-checking",
)

CODE_GEN_LABELS = (
    "drafting code",
    "rewriting after error",
    "trying a different approach",
    "polishing the call",
    "tightening the fix",
)

# Used when an action object doesn't expose a `.type` we can render
READING_FALLBACK_LABELS = (
    "reading text",
    "finding the element",
    "looking around",
)


def rotating_label(table: tuple[str, ...], idx: int) -> str:
    if not table:
        return ""
    return table[idx % len(table)]


def style_for(phase: str) -> PhaseStyle:
    return PHASE_STYLES.get(phase, UNKNOWN_PHASE_STYLE)


def tier_for(tier: str | None) -> TierStyle | None:
    if not tier:
        return None
    return TIER_STYLES.get(tier)


def mix_hex(fg: str, bg: str, fg_pct: int) -> str:
    """Linearly blend fg into bg by fg_pct (0–100). Approximates CSS color-mix."""
    fg_pct = max(0, min(100, fg_pct))
    fr, fg_, fb = int(fg[1:3], 16), int(fg[3:5], 16), int(fg[5:7], 16)
    br, bg_, bb = int(bg[1:3], 16), int(bg[3:5], 16), int(bg[5:7], 16)
    r = int(fr * fg_pct / 100 + br * (100 - fg_pct) / 100)
    g = int(fg_ * fg_pct / 100 + bg_ * (100 - fg_pct) / 100)
    b = int(fb * fg_pct / 100 + bb * (100 - fg_pct) / 100)
    return f"#{r:02x}{g:02x}{b:02x}"
