"""Sentence boundary detection for streaming TTS.

Accumulates LLM token chunks and emits complete sentences using pysbd.
Lazy-initializes the segmenter on first use — zero startup cost if
streaming is never invoked.
"""

from __future__ import annotations


class SentenceBuffer:
    """Accumulates token chunks, emits complete sentences."""

    def __init__(self, min_length: int = 20):
        self._buffer: str = ""
        self._segmenter = None
        self._min_length = min_length

    def _ensure_segmenter(self):
        if self._segmenter is None:
            import pysbd
            self._segmenter = pysbd.Segmenter(language="en", clean=False)

    def add(self, token: str) -> list[str]:
        """Append a token chunk; return any complete sentences.

        The last segment is always held back — it may be incomplete
        until more tokens arrive.  Short segments (< min_length) are
        accumulated with the next sentence to avoid choppy TTS.
        """
        self._ensure_segmenter()
        self._buffer += token

        segments = self._segmenter.segment(self._buffer)
        if len(segments) < 2:
            return []

        complete = segments[:-1]
        self._buffer = segments[-1]

        merged: list[str] = []
        carry = ""
        for seg in complete:
            seg = seg.strip()
            candidate = (carry + " " + seg).strip() if carry else seg
            if len(candidate) < self._min_length:
                carry = candidate
            else:
                merged.append(candidate)
                carry = ""

        if carry:
            self._buffer = carry + " " + self._buffer

        return merged

    def flush(self) -> str | None:
        """Return whatever text remains in the buffer, or None if empty."""
        text = self._buffer.strip()
        self._buffer = ""
        return text if text else None
