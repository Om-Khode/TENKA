"""telemetry.py — Interaction event tracking facade.

Thin delegation layer over storage.repos.telemetry.TelemetryRepo.
Provides TurnTracker (per-turn context object) and ContextVar-based
LLM accumulation for zero-touch metadata collection.
"""

import json
import logging
import re
import time
from collections import Counter
from contextvars import ContextVar, Token
from datetime import datetime
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .storage.repos.telemetry import TelemetryRepo

logger = logging.getLogger("telemetry")

_repo: Optional["TelemetryRepo"] = None


def init_telemetry_db() -> None:
    global _repo
    from .storage.db import get_db
    from .storage.repos.telemetry import TelemetryRepo

    db = get_db()
    if db is None:
        raise RuntimeError(
            "telemetry.init_telemetry_db() called before storage.db.init_db(). "
            "Call init_db() first."
        )
    _repo = TelemetryRepo(db)
    logger.info("[TELEMETRY] Initialized")


def _get_repo() -> "TelemetryRepo":
    if _repo is None:
        init_telemetry_db()
    assert _repo is not None
    return _repo


# ─── ContextVar ─────────────────────────────────────────────────────────────

_current_tracker: ContextVar["TurnTracker | None"] = ContextVar(
    "current_turn_tracker", default=None
)


def set_current_tracker(tracker: "TurnTracker") -> Token:
    return _current_tracker.set(tracker)


def get_current_tracker() -> "TurnTracker | None":
    return _current_tracker.get()


def reset_current_tracker(token: Token) -> None:
    _current_tracker.reset(token)


# ─── Failure marking ────────────────────────────────────────────────────────


def mark_action_failure(error_class: str, reason: str = "") -> None:
    """Mark the current turn as a failed action.

    Handlers that detect a non-recoverable failure (planner gave up,
    vision agent exhausted retries, browser verification failed) should
    call this so the telemetry outcome reflects reality rather than
    defaulting to "success" because the handler returned a graceful string.

    No-op when no tracker is set in the current context (e.g. background
    tasks). First failure wins — subsequent calls in the same turn keep
    the original error_class.
    """
    tracker = _current_tracker.get()
    if tracker is None:
        return
    if tracker.action_outcome == "failure":
        return
    tracker.action_outcome = "failure"
    # Cap error_class to keep DB rows compact and avoid leaking huge payloads.
    cls = (error_class or "UnknownError")[:200]
    tracker.error_class = cls
    if reason:
        logger.debug(f"[TELEMETRY] action_failure: {cls}: {reason[:200]}")


# ─── TurnTracker ────────────────────────────────────────────────────────────


class TurnTracker:
    """Per-turn context object that accumulates telemetry and writes on save()."""

    def __init__(self, session_id: str, input_modality: str, transcript: str) -> None:
        self.session_id = session_id
        self.input_modality = input_modality
        self.transcript = transcript
        self._start = time.monotonic()
        # Wall-clock anchor for cross-turn timing checks (e.g. the 30s
        # correction window). Captured at construction = right after STT
        # handed us the transcript, BEFORE the intent classifier runs.
        # Using datetime.now() at check time would unfairly penalise the
        # user when the classifier stalls on an LLM timeout. See F11b.
        self.utterance_wall_time: datetime = datetime.now()

        self.intent_detected: str | None = None
        self.intent_source: str | None = None
        self.action_dispatched: str | None = None
        self.action_outcome: str = "skipped"
        self.error_class: str | None = None

        self.latency_stt_ms: int | None = None
        self.latency_intent_ms: int | None = None
        self.latency_action_ms: int | None = None
        self.latency_tts_ms: int | None = None
        self.latency_total_ms: int | None = None

        self.llm_calls_count: int = 0
        self.llm_tokens_in: int = 0
        self.llm_tokens_out: int = 0
        self.fallback_chain_depth: int = 0
        self.vision_calls_count: int = 0
        self.llm_providers_used: Counter = Counter()
        self.llm_models_used: Counter = Counter()

    def record_llm_result(self, result) -> None:
        self.llm_calls_count += 1
        self.llm_tokens_in += result.tokens_in or 0
        self.llm_tokens_out += result.tokens_out or 0
        self.fallback_chain_depth = max(
            self.fallback_chain_depth, result.fallback_depth
        )
        # Provider/model are best-effort: streaming and some fallback paths
        # may pass "none" or empty. Skip those to avoid noise.
        provider = getattr(result, "provider", None)
        model = getattr(result, "model", None)
        if provider and provider != "none":
            self.llm_providers_used[provider] += 1
        if model and model != "none":
            self.llm_models_used[model] += 1

    def record_vision_call(self, result) -> None:
        self.vision_calls_count += 1
        self.record_llm_result(result)

    def save(self) -> None:
        self.latency_total_ms = int((time.monotonic() - self._start) * 1000)
        # JSON-encode counters; NULL when no calls happened so SQL queries can
        # filter on `IS NOT NULL` cleanly.
        providers_json = (
            json.dumps(dict(self.llm_providers_used))
            if self.llm_providers_used else None
        )
        models_json = (
            json.dumps(dict(self.llm_models_used))
            if self.llm_models_used else None
        )
        try:
            _get_repo().create(
                session_id=self.session_id,
                timestamp=datetime.now().isoformat(),
                input_modality=self.input_modality,
                transcript=self.transcript,
                intent_detected=self.intent_detected,
                intent_source=self.intent_source,
                action_dispatched=self.action_dispatched,
                action_outcome=self.action_outcome,
                error_class=self.error_class,
                latency_total_ms=self.latency_total_ms,
                latency_stt_ms=self.latency_stt_ms,
                latency_intent_ms=self.latency_intent_ms,
                latency_action_ms=self.latency_action_ms,
                latency_tts_ms=self.latency_tts_ms,
                llm_calls_count=self.llm_calls_count,
                llm_tokens_in=self.llm_tokens_in,
                llm_tokens_out=self.llm_tokens_out,
                fallback_chain_depth=self.fallback_chain_depth,
                vision_calls_count=self.vision_calls_count,
                llm_providers_used=providers_json,
                llm_models_used=models_json,
            )
        except Exception as e:
            logger.warning(f"[TELEMETRY] Failed to save event: {e}")


# ─── Correction Detection ───────────────────────────────────────────────────

# Explicit correction phrases at the start of a transcript. Generic across
# domains — no app names, no task-specific vocabulary. Case-insensitive.
# Each word/multi-word phrase here is anchored at the start so mid-sentence
# usage doesn't trigger ("I wrong-clicked the wrong button" doesn't match
# because "I" doesn't open the alternation).
_CORRECTION_PHRASE_RE = re.compile(
    r"^\s*("
    # Reversal / hesitation openers
    r"no|nope|wait|actually|i meant|i mean|stop|cancel|"
    r"never mind|nevermind|let me|hold on|"
    # Wrong-result phrases — F11a: 'wrong button' surfaced live as a
    # natural correction that wasn't caught.
    r"wrong|that's wrong|that is wrong|that was wrong|"
    # Negation / rejection
    r"not that|not it|that's not it|that isn't it|"
    # Selection-redirect phrases — common follow-up to ambiguous UI clicks.
    r"the other one|other one|something else|"
    # Undo intent
    r"undo|undo that"
    r")\b",
    re.IGNORECASE,
)


def _is_explicit_correction(transcript: str | None) -> bool:
    """True if transcript opens with a correction phrase."""
    if not transcript:
        return False
    return bool(_CORRECTION_PHRASE_RE.match(transcript))


def check_correction(tracker: TurnTracker) -> None:
    """Decide whether the current turn corrects the previous one.

    A correction requires BOTH:
      1. The previous turn happened within 30 seconds.
      2. There is an explicit signal that the user is retrying — either the
         previous turn outright failed/was skipped, OR the current transcript
         opens with a correction phrase (e.g. "no", "wait", "actually").

    `same_intent_repeated` is recorded independently: it fires whenever the
    same intent is detected within 30s, regardless of correction status.
    """
    try:
        repo = _get_repo()
        prev = repo.get_last_event(tracker.session_id)
        if not prev:
            return
        prev_time = datetime.fromisoformat(prev["timestamp"])
        # Measure from when the user finished speaking (tracker construction
        # = post-STT, pre-classifier), NOT from check time. Otherwise an
        # LLM stall during intent classification can blow the 30s window
        # even when the user spoke promptly. See F11b.
        elapsed_ms = (tracker.utterance_wall_time - prev_time).total_seconds() * 1000
        if elapsed_ms >= 30_000:
            return

        prev_outcome = (prev.get("action_outcome") or "").lower()
        prev_failed = prev_outcome in ("failure", "skipped")
        explicit = _is_explicit_correction(tracker.transcript)
        is_correction = prev_failed or explicit
        same_intent = tracker.intent_detected == prev["intent_detected"]

        # Always record same_intent_repeated when applicable; only flag
        # user_corrected when we have an actual correction signal.
        if is_correction or same_intent:
            repo.mark_correction(
                prev["id"],
                user_corrected=is_correction,
                same_intent=same_intent,
            )

        # ─── manifest-based correction feedback ──────────────────────────────────
        # When the corrected turn was an manifest_dispatch, feed the signal back
        # to the manifest dispatcher so it bumps the primary selector's
        # failure counter (demote-after-3 swap).
        if is_correction and prev.get("intent_detected") == "manifest_dispatch":
            try:
                from .automation import manifest_runtime
                disp = manifest_runtime.get_dispatcher()
                if disp is not None:
                    disp.record_last_dispatch_correction()
            except Exception as e:
                logger.debug(
                    f"[TELEMETRY] manifest-based correction feedback failed (non-critical): {e}"
                )
    except Exception as e:
        logger.debug(f"[TELEMETRY] Correction check failed (non-critical): {e}")
