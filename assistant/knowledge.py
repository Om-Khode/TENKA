"""
knowledge.py — Per-service procedural knowledge for TENKA.

Stores lessons learned from code execution successes and failures.
Each service (spotify, gmail, etc.) has its own knowledge file at
SANDBOX_DIR/knowledge/{service}.json.

Two entry types:
  "works"  — patterns confirmed working (saved after successful retry,
             requires user approval).
  "never"  — patterns confirmed broken (saved automatically after
             structural failures, no approval needed).

Knowledge is injected into code gen / fix prompts ONLY when the
relevant service is detected — zero token waste for unrelated tasks.

Storage format: JSON (easy programmatic CRUD, per-entry deletion).
LLM injection format: XML (LLMs parse structured tags better).
User display format: plain English (for approval flow).

Design principles:
  - ZERO service-specific code. Works for any service.
  - Entries are per-service, tagged by slug for surgical deletion.
  - Deduplication by pattern similarity before adding.
  - "never" entries survive template deletion (failed approaches
    are always valuable). Only "works" entries for the deleted
    slug are removed.
  - Structural failures auto-saved. Transient failures skipped.
"""

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("knowledge")

_SCHEMA_VERSION = 1

# ═══════════════════════════════════════════════════════════════════════════════
#  ERROR CATEGORIES — which failures are structural (safe to learn from)
# ═══════════════════════════════════════════════════════════════════════════════

# Structural errors represent genuine "this approach doesn't work with this API"
# lessons. Safe to save automatically — they won't change between runs.
_STRUCTURAL_CATEGORIES = frozenset({
    "field_access",    # wrong dict keys — API returns different structure
    "api_endpoint",    # wrong URL/method — 404, deprecated
    "import",          # wrong module name
    "logic",           # runtime error from wrong API usage
})

# Transient errors might succeed on retry — NOT safe to save as "never".
# Categories: network, timeout, auth, scope, encoding, blocked, unknown, no_output, syntax


def is_structural_error(error_category: str) -> bool:
    """Check if an error category represents a structural (learnable) failure."""
    return error_category in _STRUCTURAL_CATEGORIES


# ═══════════════════════════════════════════════════════════════════════════════
#  FILE I/O
# ═══════════════════════════════════════════════════════════════════════════════

def _knowledge_dir() -> Path:
    from . import config
    d = config.SANDBOX_DIR / "knowledge"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _knowledge_path(service: str) -> Path:
    safe = re.sub(r'[^\w\-]', '_', service.lower())
    return _knowledge_dir() / f"{safe}.json"


def _load_entries(service: str) -> list[dict]:
    """Load all knowledge entries for a service. Returns [] if no file.
    Migrates legacy formats (bare list, {"entries": [...]}) to versioned envelope."""
    path = _knowledge_path(service)
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        migrated = False

        if isinstance(raw, dict) and "version" in raw:
            entries = raw.get("data", [])
        elif isinstance(raw, list):
            entries = raw
            migrated = True
        elif isinstance(raw, dict) and "entries" in raw:
            entries = raw["entries"]
            migrated = True
        else:
            return []

        if not isinstance(entries, list):
            return []

        if migrated:
            _save_entries(service, entries)
            logger.info(f"[KNOWLEDGE] Migrated '{service}' to versioned format")

        return entries
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"[KNOWLEDGE] Failed to load {path}: {e}")
        return []


def _save_entries(service: str, entries: list[dict]) -> None:
    """Save all knowledge entries for a service with schema version envelope."""
    path = _knowledge_path(service)
    try:
        path.write_text(
            json.dumps({"version": _SCHEMA_VERSION, "data": entries}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(f"[KNOWLEDGE] Saved {len(entries)} entries for '{service}'")
    except OSError as e:
        logger.error(f"[KNOWLEDGE] Failed to save {path}: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
#  DEDUPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

def _is_duplicate(existing: list[dict], new_type: str, new_pattern: str) -> bool:
    """
    Check if a similar entry already exists.
    Matches on same type + pattern substring overlap.
    """
    if not new_pattern or not new_pattern.strip():
        return True  # empty pattern — treat as duplicate to skip

    new_clean = new_pattern.strip().lower()

    for entry in existing:
        if entry.get("type") != new_type:
            continue
        existing_clean = entry.get("pattern", "").strip().lower()
        if not existing_clean:
            continue
        # Substring match in either direction — catches duplicates
        # where one is slightly more verbose than the other
        if new_clean in existing_clean or existing_clean in new_clean:
            logger.debug(f"[KNOWLEDGE] Duplicate detected: '{new_clean[:50]}' vs '{existing_clean[:50]}'")
            return True

    return False


# ═══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API — ADD ENTRIES
# ═══════════════════════════════════════════════════════════════════════════════

def add_never_entry(
    service: str,
    slug: str,
    pattern: str,
    reason: str,
    error_category: str,
) -> bool:
    """
    Add a "never" entry — a pattern confirmed broken.
    Saved automatically (no user approval needed).
    Returns True if added, False if duplicate/skipped.
    """
    entries = _load_entries(service)

    if _is_duplicate(entries, "never", pattern):
        logger.info(f"[KNOWLEDGE] Skipping duplicate 'never' for '{service}/{slug}'")
        return False

    entries.append({
        "type": "never",
        "slug": slug,
        "pattern": pattern.strip(),
        "reason": reason.strip(),
        "error_category": error_category,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    _save_entries(service, entries)
    logger.info(f"[KNOWLEDGE] Added 'never' entry for '{service}/{slug}': {reason[:60]}")
    return True


def add_works_entry(
    service: str,
    slug: str,
    pattern: str,
    reason: str,
) -> bool:
    """
    Add a "works" entry — a pattern confirmed working.
    Called AFTER user approves the lesson.
    Returns True if added, False if duplicate/skipped.
    """
    entries = _load_entries(service)

    if _is_duplicate(entries, "works", pattern):
        logger.info(f"[KNOWLEDGE] Skipping duplicate 'works' for '{service}/{slug}'")
        return False

    entries.append({
        "type": "works",
        "slug": slug,
        "pattern": pattern.strip(),
        "reason": reason.strip(),
        "approved": True,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    _save_entries(service, entries)
    logger.info(f"[KNOWLEDGE] Added 'works' entry for '{service}/{slug}': {reason[:60]}")
    return True


# ═══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API — QUERY & DELETE
# ═══════════════════════════════════════════════════════════════════════════════

def get_entries(service: str) -> list[dict]:
    """Get all knowledge entries for a service."""
    return _load_entries(service)


def has_knowledge(service: str) -> bool:
    """Check if any knowledge exists for a service."""
    return len(_load_entries(service)) > 0


def delete_slug_works(service: str, slug: str) -> int:
    """
    Delete only "works" entries for a specific slug.
    Called when a template is deleted (the working pattern may be stale).
    "never" entries are KEPT — failed approaches are still valuable.
    Returns count of deleted entries.
    """
    entries = _load_entries(service)
    before = len(entries)
    entries = [
        e for e in entries
        if not (e.get("type") == "works" and e.get("slug") == slug)
    ]
    after = len(entries)
    deleted = before - after

    if deleted > 0:
        _save_entries(service, entries)
        logger.info(f"[KNOWLEDGE] Deleted {deleted} 'works' entries for '{service}/{slug}'")

    return deleted


def clear_service(service: str) -> bool:
    """Delete ALL knowledge for a service. Returns True if file existed."""
    path = _knowledge_path(service)
    if path.exists():
        path.unlink()
        logger.info(f"[KNOWLEDGE] Cleared all knowledge for '{service}'")
        return True
    return False


# ═══════════════════════════════════════════════════════════════════════════════
#  RENDERING — XML FOR LLM INJECTION
# ═══════════════════════════════════════════════════════════════════════════════

def render_for_llm(service: str) -> str:
    """
    Render knowledge as XML for injection into LLM prompts.
    Returns empty string if no knowledge exists.

    Output format:
      <service_knowledge service="gmail">
      <works slug="email_unread">
      creds = Credentials(token=os.environ.get('GMAIL_ACCESS_TOKEN'))
      service = build('gmail', 'v1', credentials=creds)
      Reason: Google API build() requires credentials= param
      </works>
      <never slug="email_unread" error="field_access">
      build('gmail', 'v1', auth=None)
      Reason: auth= is not a valid parameter for build()
      </never>
      </service_knowledge>
    """
    entries = _load_entries(service)
    if not entries:
        return ""

    works_entries = [e for e in entries if e.get("type") == "works"]
    never_entries = [e for e in entries if e.get("type") == "never"]

    if not works_entries and not never_entries:
        return ""

    parts = [f'<service_knowledge service="{service}">']

    for e in works_entries:
        slug = e.get("slug", "unknown")
        pattern = e.get("pattern", "")
        reason = e.get("reason", "")
        parts.append(f'<works slug="{slug}">')
        parts.append(pattern)
        if reason:
            parts.append(f"Reason: {reason}")
        parts.append("</works>")

    for e in never_entries:
        slug = e.get("slug", "unknown")
        err_cat = e.get("error_category", "unknown")
        pattern = e.get("pattern", "")
        reason = e.get("reason", "")
        parts.append(f'<never slug="{slug}" error="{err_cat}">')
        parts.append(pattern)
        if reason:
            parts.append(f"Reason: {reason}")
        parts.append("</never>")

    parts.append("</service_knowledge>")
    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
#  RENDERING — PLAIN ENGLISH FOR USER APPROVAL
# ═══════════════════════════════════════════════════════════════════════════════

def render_for_user(works_pattern: str, works_reason: str,
                    never_pattern: str | None = None,
                    never_reason: str | None = None) -> str:
    """
    Render a proposed knowledge entry in plain English for user approval.
    Used in the TTS/chat approval flow.
    """
    parts = ["I learned something from fixing that task."]

    parts.append(f"What works: {works_reason}")
    if never_pattern and never_reason:
        parts.append(f"What to avoid: {never_reason}")

    parts.append("Should I save this for next time?")
    return " ".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
#  LESSON EXTRACTION — PROMPTS FOR LLM
# ═══════════════════════════════════════════════════════════════════════════════

EXTRACT_FAILURE_PROMPT = """\
Analyze this failed Python code and its error.

Failed code:
```
{code_snippet}
```

Error: {error}
Error category: {category}
{discovery_context}
Respond ONLY with a JSON object. No markdown, no explanation.
The "pattern" field must be the EXACT line(s) of code that caused the failure — copy verbatim from the code above.
The "reason" field must state the actual error and what the correct code should be.
If DISCOVERY DATA is provided, use it as ground truth for correct field names/structure.

{{"pattern": "exact failing code line(s)", "reason": "what is wrong and what the correct code should be"}}
"""

EXTRACT_SUCCESS_PROMPT = """\
A Python script was broken, then fixed. Here is the DIFF showing what changed:

{diff}

The original error was: {error}
{discovery_context}
The diff shows lines removed from broken code (-) and lines added in fixed code (+).
Identify the ONE change that actually fixed the error. Ignore cosmetic changes (error handling, limit params, variable renames) — focus on the change that addressed the root cause.

Respond ONLY with a JSON object. No markdown, no explanation.
The "pattern" field must be the EXACT line(s) from the FIXED code (+ lines) that solved the problem — copy verbatim.
The "reason" field must state what was wrong and why this specific change fixes it.
If DISCOVERY DATA is provided, reference the actual field names/structure it revealed.

{{"pattern": "exact working code line(s) from the fix", "reason": "what was wrong and why this change fixes it"}}
"""

CHECK_NOVELTY_PROMPT = """\
Existing knowledge for this service:
{existing_knowledge}

Proposed new lesson:
Pattern: {new_pattern}
Reason: {new_reason}

Is this lesson ALREADY covered by the existing knowledge above?
Respond with ONLY "YES" or "NO".
"""


def build_failure_extraction_prompt(code_snippet: str, error: str,
                                     error_category: str,
                                     discovery_data: str = "") -> str:
    """Build the prompt to extract a lesson from a failed attempt."""
    disc_ctx = ""
    if discovery_data:
        disc_ctx = f"\nDISCOVERY DATA (actual API response structure — this is ground truth):\n{discovery_data[:1000]}\n"
    return EXTRACT_FAILURE_PROMPT.format(
        code_snippet=code_snippet[:500],
        error=error[:200],
        category=error_category,
        discovery_context=disc_ctx,
    )


def build_success_extraction_prompt(diff: str, error: str,
                                     discovery_data: str = "") -> str:
    """Build the prompt to extract a lesson from a successful fix."""
    disc_ctx = ""
    if discovery_data:
        disc_ctx = f"\nDISCOVERY DATA (actual API response structure — this is ground truth):\n{discovery_data[:1000]}\n"
    return EXTRACT_SUCCESS_PROMPT.format(
        diff=diff[:800],
        error=error[:200],
        discovery_context=disc_ctx,
    )


def build_novelty_check_prompt(service: str, new_pattern: str,
                                new_reason: str) -> str:
    """Build a prompt to check if a lesson is already known."""
    existing = render_for_llm(service)
    if not existing:
        return ""  # no existing knowledge — always novel
    return CHECK_NOVELTY_PROMPT.format(
        existing_knowledge=existing,
        new_pattern=new_pattern,
        new_reason=new_reason,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  UTILITY — EXTRACT RELEVANT CODE SNIPPET
# ═══════════════════════════════════════════════════════════════════════════════

def extract_relevant_snippet(code: str, error: str, max_lines: int = 15) -> str:
    """
    Extract the most relevant portion of code for lesson extraction.
    Focuses on lines around the failure point rather than the full script.
    """
    lines = code.splitlines()

    # Try to find the failing line from error traceback
    line_match = re.search(r'line (\d+)', error)
    if line_match:
        target_line = int(line_match.group(1))
        start = max(0, target_line - 5)
        end = min(len(lines), target_line + 5)
        return "\n".join(lines[start:end])

    # If no line number, return auth setup + API call sections
    # (most common failure area)
    relevant = []
    for line in lines:
        stripped = line.strip()
        if any(kw in stripped for kw in (
            "import ", "from ", "os.environ", "ACCESS_TOKEN",
            "Credentials(", "build(", "Spotify(", "requests.",
            ".get(", ".post(", ".execute(", "print(",
        )):
            relevant.append(line)

    if relevant:
        return "\n".join(relevant[:max_lines])

    # Fallback — first and last few lines
    if len(lines) > max_lines:
        return "\n".join(lines[:7] + ["..."] + lines[-5:])
    return code


def compute_code_diff(broken: str, fixed: str, max_lines: int = 20) -> str:
    """
    Compute a simple line diff between broken and fixed code.
    Returns only the lines that changed, with context.
    Deterministic — no LLM needed.
    """
    broken_lines = broken.strip().splitlines()
    fixed_lines = fixed.strip().splitlines()

    changes = []
    # Find removed lines (in broken but not in fixed)
    fixed_set = set(ln.strip() for ln in fixed_lines)
    for i, ln in enumerate(broken_lines):
        if ln.strip() and ln.strip() not in fixed_set:
            changes.append(f"- (broken line {i+1}): {ln.strip()}")

    # Find added lines (in fixed but not in broken)
    broken_set = set(ln.strip() for ln in broken_lines)
    for i, ln in enumerate(fixed_lines):
        if ln.strip() and ln.strip() not in broken_set:
            changes.append(f"+ (fixed  line {i+1}): {ln.strip()}")

    if not changes:
        return ""

    return "\n".join(changes[:max_lines])
