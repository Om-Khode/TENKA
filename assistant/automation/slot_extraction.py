"""slot_extraction.py — regex-first slot extraction for manifest-based dispatch.

A phrase may contain {slot_name} placeholders. This module turns the phrase
into a regex and matches it against the utterance. If a required slot is
empty, the dispatcher BAILS rather than silently substituting (per
feedback_respect_user_pinned_values.md).

For v1 there is no LLM fallback — bail with a clear reason and let the
classifier route the utterance through computer_task. LLM slot extraction is a
possible v1.x feature.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class SlotExtractionResult:
    ok: bool
    slots: dict[str, str]
    reason: str = ""


_SLOT_RE = re.compile(r"\{(\w+)\}")


def extract_slots(
    *, utterance: str, phrase: str, slot_names: list[str],
) -> SlotExtractionResult:
    """Match utterance against phrase; extract {slot} captures.

    If `slot_names` contains a name not present as a placeholder in `phrase`,
    the regex won't capture it and the function returns ok=False with the
    slot-empty reason. Caller is responsible for keeping slot_names and
    phrase placeholders in sync.
    """
    pattern = re.escape(phrase)
    found_slots = _SLOT_RE.findall(phrase)
    if len(found_slots) != len(set(found_slots)):
        return SlotExtractionResult(
            ok=False, slots={},
            reason=f"duplicate slot name in phrase '{phrase}'",
        )
    for slot in found_slots:
        escaped = re.escape("{" + slot + "}")
        pattern = pattern.replace(escaped, rf"(?P<{slot}>.+?)")
    pattern = "^" + pattern + "$"

    m = re.match(pattern, utterance.strip(), re.IGNORECASE)
    if m is None:
        return SlotExtractionResult(
            ok=False, slots={},
            reason=f"utterance '{utterance}' does not match phrase '{phrase}'",
        )

    slots = {}
    for slot in slot_names:
        value = (m.groupdict().get(slot) or "").strip()
        if not value:
            return SlotExtractionResult(
                ok=False, slots={},
                reason=f"slot '{slot}' empty in '{utterance}'",
            )
        slots[slot] = value

    return SlotExtractionResult(ok=True, slots=slots)
