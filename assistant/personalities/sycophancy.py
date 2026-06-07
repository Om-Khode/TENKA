"""Sycophancy opener-strip filter.

Removes known sycophantic opening phrases from LLM responses.
Only applied for warm_honest personality. Microsecond cost.
"""

import re

_EMOTION_TAG_RE = re.compile(r"^(\[[\w]+\]\s*)")

_SYCOPHANTIC_OPENERS = re.compile(
    r"^(?:"
    r"great question[!.]?\s*"
    r"|that'?s (?:a )?(?:really |very )?(?:good|great|excellent|thoughtful|interesting) (?:question|point|idea|observation)[!.]?\s*"
    r"|absolutely[!.]?\s*"
    r"|you'?re absolutely right[!.]?\s*"
    r"|what a (?:fantastic|brilliant|great|wonderful) idea[!.]?\s*"
    r"|i love that(?:\s+idea)?[!.]?\s*"
    r"|that'?s (?:brilliant|amazing|fantastic)[!.]?\s*"
    r")",
    re.IGNORECASE,
)


def strip_sycophantic_opener(text: str) -> str:
    if not text:
        return text

    tag_prefix = ""
    body = text

    tag_match = _EMOTION_TAG_RE.match(body)
    if tag_match:
        tag_prefix = tag_match.group(1)
        body = body[tag_match.end():]

    stripped = _SYCOPHANTIC_OPENERS.sub("", body, count=1)

    if stripped != body and stripped and stripped[0].islower():
        stripped = stripped[0].upper() + stripped[1:]

    return tag_prefix + stripped
