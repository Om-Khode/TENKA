"""storage/repos/telemetry.py — Interaction event persistence."""

import logging
from datetime import datetime, timedelta

from ..db import Database

logger = logging.getLogger("telemetry")


class TelemetryRepo:
    """CRUD for interaction_events table."""

    def __init__(self, db: Database) -> None:
        self._db = db

    def create(
        self,
        *,
        session_id: str,
        timestamp: str,
        input_modality: str,
        transcript: str | None,
        intent_detected: str | None,
        intent_source: str | None,
        action_dispatched: str | None,
        action_outcome: str | None,
        error_class: str | None,
        latency_total_ms: int | None,
        latency_stt_ms: int | None,
        latency_intent_ms: int | None,
        latency_action_ms: int | None,
        latency_tts_ms: int | None,
        llm_calls_count: int,
        llm_tokens_in: int,
        llm_tokens_out: int,
        fallback_chain_depth: int,
        vision_calls_count: int,
        llm_providers_used: str | None = None,
        llm_models_used: str | None = None,
    ) -> int:
        cursor = self._db.execute(
            "INSERT INTO interaction_events ("
            "  session_id, timestamp, input_modality, transcript,"
            "  intent_detected, intent_source,"
            "  action_dispatched, action_outcome, error_class,"
            "  latency_total_ms, latency_stt_ms, latency_intent_ms,"
            "  latency_action_ms, latency_tts_ms,"
            "  llm_calls_count, llm_tokens_in, llm_tokens_out,"
            "  fallback_chain_depth, vision_calls_count,"
            "  llm_providers_used, llm_models_used"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                session_id, timestamp, input_modality, transcript,
                intent_detected, intent_source,
                action_dispatched, action_outcome, error_class,
                latency_total_ms, latency_stt_ms, latency_intent_ms,
                latency_action_ms, latency_tts_ms,
                llm_calls_count, llm_tokens_in, llm_tokens_out,
                fallback_chain_depth, vision_calls_count,
                llm_providers_used, llm_models_used,
            ),
        )
        self._db.commit()
        return cursor.lastrowid

    def get_last_event(self, session_id: str) -> dict | None:
        row = self._db.fetchone(
            "SELECT * FROM interaction_events "
            "WHERE session_id = ? ORDER BY id DESC LIMIT 1",
            (session_id,),
        )
        return dict(row) if row else None

    def mark_correction(
        self, event_id: int, *, user_corrected: bool, same_intent: bool
    ) -> None:
        self._db.execute(
            "UPDATE interaction_events "
            "SET user_corrected_within_30s = ?, same_intent_repeated = ? "
            "WHERE id = ?",
            (int(user_corrected), int(same_intent), event_id),
        )
        self._db.commit()

    def cleanup(self, retention_days: int = 90) -> int:
        cutoff = (datetime.now() - timedelta(days=retention_days)).isoformat()
        cursor = self._db.execute(
            "DELETE FROM interaction_events WHERE timestamp < ?",
            (cutoff,),
        )
        self._db.commit()
        deleted = cursor.rowcount
        if deleted:
            logger.info(f"[TELEMETRY] Cleanup: removed {deleted} event(s) older than {retention_days}d")
        return deleted
