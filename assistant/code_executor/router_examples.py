"""router_examples.py — Build the dynamic router-example block from cached templates.

Scans SANDBOX_DIR/scripts/ for saved {slug}.py templates, parses their
v1/v2 headers, infers requires from imports, gates by package availability,
and renders example JSON for the router system prompt.

Cached internally; invalidated by templates._save_template / _delete_template.
"""

import importlib.util
import json as _json
import logging
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path

from .packages import TIER2_ALLOWED_PACKAGES, _IMPORT_TO_PACKAGE, _PACKAGE_IMPORT_NAMES

logger = logging.getLogger("code_executor")


# ─── Data model ────────────────────────────────────────────────────────────

@dataclass
class TemplateInfo:
    slug: str
    goal: str = ""
    params: dict = field(default_factory=dict)
    requires: list[str] = field(default_factory=list)


# ─── Scan ──────────────────────────────────────────────────────────────────

_IMPORT_RE = re.compile(r"^[ \t]*(?:import\s+([\w\.,\t ]+)|from\s+([\w\.]+)\s+import)", re.MULTILINE)


def _parse_header(text: str, slug: str) -> tuple[str, str, dict]:
    """Strip # version: / # GOAL: / # PARAMS: header lines.

    Returns (code, goal, params). Malformed PARAMS yields {} + warning.
    """
    # NOTE: intentionally duplicates templates._read_template_file's header logic.
    # Cannot import from templates — templates imports router_examples for cache
    # invalidation (Task 8). Any header-format change must be mirrored in both.
    goal = ""
    params: dict = {}
    if text.startswith("# version: "):
        text = text[text.index("\n") + 1:]
    if text.startswith("# GOAL: "):
        nl = text.index("\n")
        goal = text[8:nl].strip()
        text = text[nl + 1:]
    if text.startswith("# PARAMS: "):
        nl = text.index("\n")
        raw = text[10:nl].strip()
        try:
            parsed = _json.loads(raw)
            if isinstance(parsed, dict):
                params = parsed
            else:
                logger.warning(f"[CODE] PARAMS header in '{slug}' is not a dict — using {{}}")
        except _json.JSONDecodeError as e:
            logger.warning(f"[CODE] PARAMS header in '{slug}' is malformed JSON ({e}) — using {{}}")
        text = text[nl + 1:]
    return text, goal, params


def _infer_requires(code: str) -> list[str]:
    """Scan import / from-import lines, map to pip names, filter to TIER2 allowlist."""
    found: set[str] = set()
    for m in _IMPORT_RE.finditer(code):
        if m.group(1):  # import X, Y, Z (each may have "as alias")
            for name in m.group(1).split(","):
                top = name.strip().split(".")[0].split()[0]   # drop "as alias"
                if top:
                    found.add(top)
        elif m.group(2):  # from X.Y import Z
            top = m.group(2).split(".")[0].split()[0]
            if top:
                found.add(top)
    pkgs: set[str] = set()
    for import_name in found:
        pip_name = _IMPORT_TO_PACKAGE.get(import_name)
        if pip_name and pip_name in TIER2_ALLOWED_PACKAGES:
            pkgs.add(pip_name)
    return sorted(pkgs)


def scan_templates(scripts_dir: Path) -> list[TemplateInfo]:
    """Scan scripts_dir for *.py templates, parse, return sorted by slug."""
    if not scripts_dir.exists():
        return []
    infos: list[TemplateInfo] = []
    for path in sorted(scripts_dir.glob("*.py")):
        slug = path.stem
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as e:
            logger.warning(f"[CODE] Could not read template '{slug}': {e}")
            continue
        code, goal, params = _parse_header(text, slug)
        requires = _infer_requires(code)
        infos.append(TemplateInfo(slug=slug, goal=goal, params=params, requires=requires))
    return infos


# ─── Availability gate ────────────────────────────────────────────────────

def _is_available(requires: list[str]) -> bool:
    """True if every pip package in `requires` is importable on this host."""
    for pkg in requires:
        import_name = _PACKAGE_IMPORT_NAMES.get(pkg, pkg).replace("-", "_")
        if importlib.util.find_spec(import_name) is None:
            return False
    return True


# ─── Sanitization + render ────────────────────────────────────────────────

_LEADING_ARTICLE_RE = re.compile(r"^(a|an|the|my|to)\s+")
_WS_RE = re.compile(r"\s+")


def _sanitize_goal(goal: str) -> str:
    """Normalize a stored GOAL for inclusion as a // comment in the prompt."""
    g = goal.strip()
    if not g:
        return ""
    # 1. Truncate at first non-URL colon within first 80 chars.
    #    If the colon is FUSED to a word (no space and the previous char is
    #    a letter/digit — e.g. "C:" or "Image:"), also drop that word.
    #    But preserve payload tokens like "50%:" where the previous char is
    #    not alphanumeric.
    head = g[:80]
    idx = head.find(":")
    if idx >= 0 and head[idx + 1:idx + 3] != "//":
        g = g[:idx].rstrip()
        if idx > 0 and head[idx - 1].isalnum():
            # Colon was fused to a word — drop that word too.
            last_space = g.rfind(" ")
            g = g[:last_space] if last_space >= 0 else ""
    # 2. lowercase
    g = g.lower()
    # 3. strip leading article
    g = _LEADING_ARTICLE_RE.sub("", g)
    # 4. collapse whitespace
    g = _WS_RE.sub(" ", g).strip()
    # 5. truncate to 60 chars with ellipsis
    if len(g) > 60:
        g = g[:60] + "…"
    return g


def build_dynamic_examples(scripts_dir: Path) -> str:
    """Return the rendered dynamic-block string. Empty string if nothing to render."""
    lines: list[str] = []
    for info in scan_templates(scripts_dir):
        if not _is_available(info.requires):
            continue
        sanitized = _sanitize_goal(info.goal)
        if sanitized:
            lines.append(f"// {sanitized}")
        json_line = _json.dumps(
            {
                "tier": 2,
                "template_slug": info.slug,
                "requires": info.requires,
                "params": info.params,
            },
            separators=(",", ":"),
        )
        lines.append(json_line)
    return "\n".join(lines)


# ─── Cache + invalidation ─────────────────────────────────────────────────

_cached_examples: str | None = None
_cache_lock = threading.Lock()


def _default_scripts_dir() -> Path:
    """Resolve the production scripts dir lazily (avoids import-time config read)."""
    from .. import config
    return config.SANDBOX_DIR / "scripts"


def get_dynamic_examples() -> str:
    """Return the cached dynamic-block string, building on first call / after invalidate."""
    global _cached_examples
    with _cache_lock:
        if _cached_examples is None:
            _cached_examples = build_dynamic_examples(_default_scripts_dir())
        return _cached_examples


def invalidate() -> None:
    """Clear the cache. Called from templates._save_template / _delete_template."""
    global _cached_examples
    with _cache_lock:
        _cached_examples = None
