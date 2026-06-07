"""templates.py — Template CRUD, parameterization, goal matching, and debug dump."""

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger("code_executor")

from ._utils import _syntax_check

_SCHEMA_VERSION = 2


def _templates_dir() -> Path:
    from .. import config
    d = config.SANDBOX_DIR / "scripts"
    d.mkdir(parents=True, exist_ok=True)
    return d

def _read_template_file(path: Path, slug: str) -> tuple[str, str, dict]:
    """Read and parse a template file, stripping version/goal/params headers.

    Returns (code, stored_goal, stored_params). Empty defaults if a header
    is missing. Malformed # PARAMS: JSON yields {} and logs a warning.
    """
    text = path.read_text(encoding="utf-8")
    stored_goal = ""
    stored_params: dict = {}
    if text.startswith("# version: "):
        first_newline = text.index("\n")
        text = text[first_newline + 1:]
    if text.startswith("# GOAL: "):
        first_newline = text.index("\n")
        stored_goal = text[8:first_newline].strip()
        text = text[first_newline + 1:]
    if text.startswith("# PARAMS: "):
        first_newline = text.index("\n")
        raw = text[10:first_newline].strip()
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                stored_params = parsed
            else:
                logger.warning(f"[CODE] PARAMS header in '{slug}' is not a dict — using {{}}")
        except json.JSONDecodeError as e:
            logger.warning(f"[CODE] PARAMS header in '{slug}' is malformed JSON ({e}) — using {{}}")
        text = text[first_newline + 1:]
    logger.info(f"[CODE] Loaded template '{slug}' from {path.name} (goal: '{stored_goal[:60]}', params: {stored_params})")
    return text, stored_goal, stored_params


def _find_legacy_template(slug: str, tdir: Path) -> Path | None:
    """When a category-prefixed slug has no template, try legacy service-prefixed names.

    e.g. music_play → spotify_play.py, spotify_play_music.py
    """
    from ..core.known_apps import KNOWN_APPS

    cat_to_services: dict[str, list[str]] = {}
    for svc, entry in KNOWN_APPS.items():
        prefix = entry.category.split("_")[0]
        cat_to_services.setdefault(prefix, []).append(svc.replace(" ", "_"))

    parts = slug.split("_", 1)
    if len(parts) < 2:
        return None

    cat_prefix, action = parts
    services = cat_to_services.get(cat_prefix, [])
    if not services:
        return None

    for svc in services:
        exact = tdir / f"{svc}_{action}.py"
        if exact.exists():
            return exact
        for match in sorted(tdir.glob(f"{svc}_{action}_*.py")):
            return match

    return None


def _load_template(slug: str) -> tuple[str | None, str]:
    """Returns (code, stored_goal). Code is None if no template exists.
    Handles versioned (# version: N) and legacy (no version header) formats.
    Falls back to legacy service-prefixed templates when category slug not found."""
    tdir = _templates_dir()
    path = tdir / f"{slug}.py"
    if path.exists():
        code, goal, _params = _read_template_file(path, slug)
        return code, goal

    legacy = _find_legacy_template(slug, tdir)
    if legacy:
        logger.info(f"[CODE] Legacy fallback: '{slug}' → {legacy.name}")
        code, goal, _params = _read_template_file(legacy, slug)
        return code, goal

    return None, ""

def _save_template(slug: str, code: str, goal: str = "", params: dict | None = None) -> None:
    if params:
        code = _parameterize_code(code, params)
    path = _templates_dir() / f"{slug}.py"
    header = f"# version: {_SCHEMA_VERSION}\n"
    if goal:
        header += f"# GOAL: {goal}\n"
    if params:
        header += f"# PARAMS: {json.dumps(params)}\n"
    content = header + code
    path.write_text(content, encoding="utf-8")
    from . import router_examples
    router_examples.invalidate()
    logger.info(f"[CODE] Saved template '{slug}'")


def _parameterize_code(code: str, params: dict) -> str:
    """
    Replace hardcoded param values in generated code with os.environ.get('PARAM_*')
    calls. This makes templates reusable across different inputs.

    The LLM often ignores PARAM_* instructions and hardcodes values from the goal
    text. This function fixes that deterministically after generation.

    Key insight: only replace values that are STANDALONE string literals in the
    code (assignments, comparisons, function args) — NOT values embedded inside
    larger strings like print messages or f-strings.

    Generic — works for any param key/value pair from the router.
    """
    if not params:
        return code

    lines = code.split('\n')

    for key, value in params.items():
        val_str = str(value).strip()
        if not val_str:
            continue

        env_name = f"PARAM_{key.upper()}"

        # Skip if code already uses this PARAM_* via os.environ.get() — already parameterized.
        # Do NOT skip if it appears as a variable assignment (e.g., PARAM_IMAGE_PATH = r"C:\...")
        # — that's a hardcoded value the LLM assigned, which we need to replace.
        if f"os.environ.get('{env_name}'" in code or f'os.environ.get("{env_name}"' in code:
            continue

        env_call = f"os.environ.get('{env_name}', '')"
        val_lower = val_str.lower()
        replaced = False

        # ── Check for PARAM_* variable assignments ────────────────────
        # LLMs sometimes generate: PARAM_IMAGE_PATH = r"C:\Users\..."
        # instead of using os.environ.get(). Replace the hardcoded
        # assignment with the env var lookup.
        _assign_pattern = re.compile(
            rf'^(\s*){re.escape(env_name)}\s*=\s*[rRbBuU]?[\'"].*?[\'"]',
            re.MULTILINE
        )
        _assign_match = _assign_pattern.search(code)
        if _assign_match:
            indent = _assign_match.group(1)
            old_line = _assign_match.group(0)
            new_line = f"{indent}{env_name} = {env_call}"
            code = code.replace(old_line, new_line, 1)
            logger.info(f"[CODE] Parameterized assignment: {env_name} = ... → {env_call}")
            # Update lines since we modified code directly
            lines = code.split('\n')
            continue

        # Build all variants of the value to try matching.
        # LLMs often change slash direction or use raw strings for paths.
        val_variants = [val_str]
        if '/' in val_str or '\\' in val_str:
            val_variants.append(val_str.replace('/', '\\'))
            val_variants.append(val_str.replace('\\', '/'))
        # Deduplicate while preserving order
        val_variants = list(dict.fromkeys(val_variants))

        for i, line in enumerate(lines):
            stripped = line.strip()

            # Skip print/log lines — the value there is part of a message,
            # not a variable we need to parameterize
            if stripped.startswith(('print(', 'print (', 'logger.', 'logging.')):
                continue

            # Skip comments
            if stripped.startswith('#'):
                continue

            # Try to find a standalone quoted value on this line.
            # "Standalone" means the quotes are not nested inside a larger string.
            # We check: the value with its quotes is a full token, not a substring
            # of an f-string or another string literal.
            # Also try path variants (forward/backslash) and r-string prefix.
            # IMPORTANT: try r/R prefix FIRST — if we match without prefix on
            # a raw string like r"C:\...", the 'r' gets left behind as orphan.
            for variant in val_variants:
                for quote in ("'", '"'):
                    for prefix in ("r", "R", ""):
                        literal = f"{prefix}{quote}{variant}{quote}"
                        if literal in line:
                            idx = line.index(literal)
                            before = line[idx - 1] if idx > 0 else ' '
                            if before not in ('"', "'", '\\'):
                                lines[i] = line.replace(literal, env_call, 1)
                                logger.info(f"[CODE] Parameterized: {literal} → {env_call}")
                                replaced = True
                                break
                    if replaced:
                        break
                if replaced:
                    break

            # Also try lowercased version (LLMs often lowercase values)
            if not replaced:
                for quote in ("'", '"'):
                    literal_lower = f"{quote}{val_lower}{quote}"
                    if literal_lower in line and literal_lower != f"{quote}{val_str}{quote}":
                        idx = line.index(literal_lower)
                        before = line[idx - 1] if idx > 0 else ' '
                        if before not in ('"', "'", '\\'):
                            env_call_lower = f"os.environ.get('{env_name}', '').lower()"
                            lines[i] = line.replace(literal_lower, env_call_lower, 1)
                            logger.info(f"[CODE] Parameterized (lowered): {literal_lower} → {env_call_lower}")
                            replaced = True
                            break

            if replaced:
                break

    fixed = '\n'.join(lines)

    # If we inserted os.environ.get() calls, ensure 'import os' is present
    if 'os.environ.get(' in fixed and 'import os' not in fixed:
        # Insert 'import os' after the last existing import line
        fixed_lines = fixed.split('\n')
        last_import_idx = -1
        for idx, ln in enumerate(fixed_lines):
            stripped = ln.strip()
            if stripped.startswith(('import ', 'from ')):
                last_import_idx = idx
        if last_import_idx >= 0:
            fixed_lines.insert(last_import_idx + 1, 'import os')
        else:
            fixed_lines.insert(0, 'import os')
        fixed = '\n'.join(fixed_lines)
        logger.info("[CODE] Injected 'import os' for os.environ.get() calls")

    # Verify syntax — if parameterization broke something, revert
    if _syntax_check(fixed):
        logger.warning("[CODE] Parameterization produced syntax error — keeping original")
        return code

    return fixed

def _delete_template(slug: str) -> None:
    path = _templates_dir() / f"{slug}.py"
    if path.exists():
        path.unlink()
        from . import router_examples
        router_examples.invalidate()
        logger.info(f"[CODE] Deleted broken template '{slug}'")

    # Clean up stale "works" knowledge for this slug.
    # "never" entries are kept — failed approaches remain valuable.
    try:
        from .. import knowledge
        # Detect service from slug prefix using known_apps aliases
        from ..core.known_apps import KNOWN_APPS
        for svc_name, entry in KNOWN_APPS.items():
            prefixes = [svc_name] + entry.aliases
            if any(slug.startswith(p) for p in prefixes):
                deleted = knowledge.delete_slug_works(svc_name, slug)
                if deleted:
                    logger.info(f"[CODE] Cleaned {deleted} 'works' knowledge entries for '{slug}'")
                break
    except Exception as e:
        logger.debug(f"[CODE] Knowledge cleanup skipped: {e}")


# ─── Goal-matching for template reuse ─────────────────────────────────────
# Two-layer check (research-grounded — see CE-DYN spec post-mortem):
#   1. Specificity asymmetry: if the current goal has content nouns the
#      stored goal lacks (via spaCy noun_chunks + PROPN), refuse reuse.
#      Catches the failure where "play some music" (cached, no params) was
#      reused for "play blinding lights" (specific song needed).
#   2. Keyword overlap ≥ 0.50. Raised from 0.20 per GPTCache / LangChain
#      production guidance. 0.20 let any shared verb count as a match.
# spaCy is reused from the same en_core_web_sm model topic_tracker loads;
# we lazy-import here so module import stays zero-cost.

_NLP = None
_NLP_LOAD_FAILED = False

# Tokens that look like nouns but carry no specificity (filler words).
_CHUNK_NOISE = frozenset({
    "i", "you", "we", "they", "it", "me", "us", "them", "he", "she",
    "some", "any", "all", "thing", "things", "stuff", "one", "ones",
})


def _get_nlp_for_matching():
    """Lazy-load spaCy en_core_web_sm. Returns None if unavailable.

    Failure is cached so we don't pay the load cost repeatedly when spaCy
    is missing — matching falls back to keyword-only mode gracefully.
    """
    global _NLP, _NLP_LOAD_FAILED
    if _NLP_LOAD_FAILED:
        return None
    if _NLP is None:
        try:
            import spacy
            _NLP = spacy.load("en_core_web_sm")
        except Exception as e:
            logger.warning(
                f"[CODE] spaCy unavailable for goal-matching specificity ({e}) — "
                "falling back to keyword-only matching"
            )
            _NLP_LOAD_FAILED = True
            return None
    return _NLP


def _content_tokens(nlp, text: str) -> set[str]:
    """Extract content-bearing tokens for specificity comparison.

    Returns a lowercased set. Catches song titles, place names, app names,
    and short imperative-mood phrases (e.g. 'play dare', 'play make you mine')
    that spaCy tags as VERB/PRON rather than NOUN.

    Two-pass:
      1. All tokens that aren't STOP, aren't VERB, aren't PUNCT, aren't in
         the noise filter — captures the "you mine" / "dare" cases where
         spaCy POS-tagging is loose.
      2. PROPN tokens explicitly (catches Title Case proper nouns that may
         not appear in a noun_chunk).
    """
    doc = nlp(text)
    tokens: set[str] = set()
    _SKIP_POS = {"VERB", "AUX", "PUNCT", "SPACE", "PART", "ADP", "DET", "CCONJ", "SCONJ"}
    for tok in doc:
        t = tok.text.lower()
        if (len(t) > 1
                and t not in _CHUNK_NOISE
                and not tok.is_stop
                and tok.pos_ not in _SKIP_POS):
            tokens.add(t)
    # Always include PROPN even if previous pass filtered them (defensive)
    for tok in doc:
        if tok.pos_ == "PROPN" and len(tok.text) > 1:
            tokens.add(tok.text.lower())
    return tokens


def _goal_matches_template(current_goal: str, stored_goal: str) -> bool:
    """Decide if the cached template's stored_goal is similar enough to
    reuse for current_goal.

    Layer 1 — SPECIFICITY ASYMMETRY: if current has content nouns that
    stored lacks (e.g. current="play blinding lights" vs stored="play some
    music"), REJECT. The cached template was generated without the slot
    the new request needs.

    Layer 2 — KEYWORD OVERLAP RATIO ≥ 0.50. Belt-and-suspenders for cases
    spaCy can't catch (or when spaCy is unavailable).
    """
    if not stored_goal:
        return True  # legacy templates without goal comment — allow reuse

    _STOP = frozenset({
        "my", "me", "i", "the", "a", "an", "to", "of", "in", "on",
        "do", "have", "is", "are", "can", "get", "what", "how",
        "please", "just", "from", "for", "and", "or", "all",
    })

    def _keywords(text: str) -> set[str]:
        return {w for w in text.lower().split() if len(w) > 1 and w not in _STOP}

    cur = _keywords(current_goal)
    stored = _keywords(stored_goal)

    if not cur or not stored:
        return True  # can't compare — allow reuse

    # ── Layer 1: specificity asymmetry ────────────────────────────────────
    nlp = _get_nlp_for_matching()
    if nlp is not None:
        cur_content = _content_tokens(nlp, current_goal)
        stored_content = _content_tokens(nlp, stored_goal)
        extra = cur_content - stored_content
        if extra:
            logger.info(
                f"[CODE] Template match: REJECTED (specificity) — "
                f"current has content {sorted(extra)!r} that stored {sorted(stored_content)!r} lacks"
            )
            return False

    # ── Layer 2: keyword overlap ratio ────────────────────────────────────
    overlap = len(cur & stored)
    total = min(len(cur), len(stored))
    ratio = overlap / total if total > 0 else 0

    accepted = ratio >= 0.70
    logger.info(
        f"[CODE] Template match: {'ACCEPTED' if accepted else 'REJECTED'} (overlap) — "
        f"cur={cur}, stored={stored}, overlap={overlap}/{total}={ratio:.2f}, threshold=0.70"
    )
    return accepted


def _dump_code(label: str, code: str, result: str | None = None) -> None:
    """
    Dump full generated/fixed code to SANDBOX_DIR/debug/ for development.
    Each dump overwrites the previous one for the same label.
    Also logs the full code and result at DEBUG level.
    """
    try:
        from .. import config
        debug_dir = config.SANDBOX_DIR / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)

        # Save code
        safe_label = re.sub(r'[^\w\-]', '_', label)
        code_path = debug_dir / f"{safe_label}.py"
        code_path.write_text(code, encoding="utf-8")

        # Save result if provided
        if result is not None:
            result_path = debug_dir / f"{safe_label}_output.txt"
            result_path.write_text(result, encoding="utf-8")

        logger.debug(f"[CODE] Dumped '{label}' to {debug_dir}")
    except Exception as e:
        logger.debug(f"[CODE] Debug dump failed: {e}")

    # Always log full code regardless of dump success
    logger.info(f"[CODE] ── Full code ({label}) ──\n{code}\n── End code ──")
    if result is not None:
        logger.info(f"[CODE] ── Full output ({label}) ──\n{result}\n── End output ──")
