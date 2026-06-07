"""TODO Classifier — pure regex, zero LLM cost.

Decomposes a TODO text like "Type 'John' in First Name" into structured
fields the action-signature matcher uses without an LLM call. Each TODO
falls into one of four kinds:

  type   — keyboard input into a field (verbs: type, enter, input, fill)
  select — pick a value from a dropdown/list (verbs: select, choose, pick)
  click  — single click action (verbs: click, press, tap, hit)
  other  — anything that doesn't pattern-match; falls through to the LLM
           updater as a last resort.

Quote handling: matched-pair stripping for ', ", `, ‘, ’, “, ”. Internal
quotes (e.g. apostrophes inside names) are preserved.
Suffix scrubbing: trailing "dropdown"/"menu"/"list" stripped from select
fields; trailing "button"/"link"/"tab"/"icon" stripped from click targets.
"""

import re as _re

_TODO_TYPE_VERB_RE = _re.compile(r"^\s*(type|enter|input|fill)\s+(.+)$", _re.IGNORECASE)
_TODO_SELECT_VERB_RE = _re.compile(r"^\s*(select|choose|pick)\s+(.+)$", _re.IGNORECASE)
_TODO_CLICK_VERB_RE = _re.compile(r"^\s*(click|press|tap|hit)\s+(.+)$", _re.IGNORECASE)

_TODO_TYPE_CONNECTOR_RE = _re.compile(r"\s+(into|in)\s+(.+)$", _re.IGNORECASE)
_TODO_SELECT_CONNECTOR_RE = _re.compile(r"\s+(from|in)\s+(.+)$", _re.IGNORECASE)

_TODO_DROPDOWN_SUFFIX_RE = _re.compile(
    r"\s+(dropdown|drop[-\s]down|menu|list|combobox|combo\s*box|select|picker)\s*\.?\s*$",
    _re.IGNORECASE,
)
_TODO_CLICK_SUFFIX_RE = _re.compile(
    r"\s+(button|link|tab|icon|menu\s*item|toggle|checkbox|radio|option)\s*\.?\s*$",
    _re.IGNORECASE,
)
_TODO_LEADING_THE_RE = _re.compile(r"^the\s+", _re.IGNORECASE)


def _strip_matched_quotes(s: str) -> str:
    """Strip ONE pair of matched surrounding quotes. Internal quotes preserved.

    Handles straight single, double, backtick, and smart curly variants.
    No-op when the leading/trailing characters don't form a recognized pair.
    """
    if not isinstance(s, str):
        return ""
    s = s.strip()
    if len(s) < 2:
        return s
    pairs = [
        ("'", "'"), ('"', '"'), ("`", "`"),
        ("‘", "’"),  # ‘ ’
        ("“", "”"),  # “ ”
    ]
    for opener, closer in pairs:
        if s.startswith(opener) and s.endswith(closer):
            return s[len(opener):-len(closer)].strip()
    return s


def _classify_todo(text: str) -> dict:
    """
    Classify a TODO text into structured fields for action-signature matching.

    Returns a dict with keys:
      kind   — "type" | "select" | "click" | "other"
      target — for "click": the element being clicked (without "button" etc.)
      field  — for "type"/"select": the field/dropdown label
      value  — for "type"/"select": the literal value to type or option to pick

    `target`/`field`/`value` are empty strings when not applicable to the kind.
    Always returns a fully-populated dict (no None values) so callers can
    safely `.get()` without defensive None handling.
    """
    default = {"kind": "other", "target": "", "field": "", "value": ""}
    if not isinstance(text, str):
        return default
    s = text.strip()
    if not s:
        return default
    # Strip a single trailing period (LLMs often punctuate).
    if s.endswith("."):
        s = s[:-1].rstrip()

    # --- Type-style ---
    m = _TODO_TYPE_VERB_RE.match(s)
    if m:
        rest = m.group(2)
        conn = _TODO_TYPE_CONNECTOR_RE.search(rest)
        if conn:
            value_raw = rest[: conn.start()].strip()
            field_raw = conn.group(2).strip()
            return {
                "kind": "type",
                "target": "",
                "field": _strip_matched_quotes(field_raw),
                "value": _strip_matched_quotes(value_raw),
            }

    # --- Select-style ---
    m = _TODO_SELECT_VERB_RE.match(s)
    if m:
        rest = m.group(2)
        conn = _TODO_SELECT_CONNECTOR_RE.search(rest)
        if conn:
            value_raw = rest[: conn.start()].strip()
            field_raw = conn.group(2).strip()
            field_raw = _TODO_DROPDOWN_SUFFIX_RE.sub("", field_raw).strip()
            field_raw = _TODO_LEADING_THE_RE.sub("", field_raw).strip()
            return {
                "kind": "select",
                "target": "",
                "field": _strip_matched_quotes(field_raw),
                "value": _strip_matched_quotes(value_raw),
            }

    # --- Click-style ---
    m = _TODO_CLICK_VERB_RE.match(s)
    if m:
        rest = m.group(2).strip()
        rest = _TODO_LEADING_THE_RE.sub("", rest)
        rest = _TODO_CLICK_SUFFIX_RE.sub("", rest).strip()
        target_clean = _strip_matched_quotes(rest)
        if target_clean:
            return {
                "kind": "click",
                "target": target_clean,
                "field": "",
                "value": "",
            }

    return default


def _make_todo_dict(todo_id: int, text: str) -> dict:
    """
    Factory for a TODO dict with standard fields populated.
    Centralizes the schema so set_initial_todos and add_todo can't drift apart.
    """
    classified = _classify_todo(text)
    return {
        "id": todo_id,
        "task": text,
        "done": False,
        # Classifier output
        "kind": classified["kind"],
        "target": classified["target"],
        "field": classified["field"],
        "value": classified["value"],
        # Visual-confirm state
        "pending_visual_confirm": False,
        "confirm_strikes": 0,
        # Fix A: True when TODO was marked done because vision confirm gave up.
        "confirm_abandoned": False,
        # Engagement timestamps
        "batch_marked_done": -1,
        "batch_deferred": -1,
    }
