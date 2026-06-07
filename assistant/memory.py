"""
memory.py — Conversation memory facade.

Thin delegation layer over storage.repos.memory.MemoryRepo.
All persistence (SQLite, FAISS, ID-maps) lives in the repo.
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("memory")

_repo: Optional["MemoryRepo"] = None


def init_memory() -> None:
    """Initialize the memory repo. Requires init_db() to have been called first."""
    global _repo
    from .storage.db import get_db
    from .storage.repos.memory import MemoryRepo
    from . import config

    db = get_db()
    if db is None:
        raise RuntimeError(
            "memory.init_memory() called before storage.db.init_db(). "
            "Call init_db() first."
        )

    data_dir: Path = config.SANDBOX_DIR / "memory"
    data_dir.mkdir(parents=True, exist_ok=True)

    _repo = MemoryRepo(db, data_dir)
    _repo.init_vector_store()
    logger.info(f"[MEMORY] Initialized (data_dir={data_dir})")


def _get_repo() -> "MemoryRepo":
    if _repo is None:
        init_memory()
    assert _repo is not None
    return _repo


def save_turn(user_input: str, intent: str, response: str, session_id: str) -> int:
    return _get_repo().save_turn(user_input, intent, response, session_id)


def get_recent(n: int = 10, session_id: str = "") -> list[dict]:
    return _get_repo().get_recent(n, session_id=session_id)


def build_recent_context(
    limit: int = 25, header: str = "RECENT CONVERSATION HISTORY:",
    session_id: str = "",
) -> str:
    return _get_repo().build_recent_context(limit, header, session_id=session_id)


def search_conversations(query: str, limit: int = 5) -> list[dict]:
    return _get_repo().search_conversations(query, limit)


def hybrid_search_conversations(query: str, limit: int = 5) -> list[dict]:
    return _get_repo().hybrid_search_conversations(query, limit)


def hybrid_search_facts(query: str, limit: int = 10) -> list[dict]:
    return _get_repo().hybrid_search_facts(query, limit)


def search_recording_sessions(query: str, limit: int = 3) -> list[dict]:
    return _get_repo().search_recording_sessions(query, limit)


def summarize_session(session_id: str) -> str:
    return _get_repo().summarize_session(session_id)


def save_fact(key: str, value: str, source: str = "user") -> None:
    _get_repo().save_fact(key, value, source)


def search_facts(key: str) -> list[dict]:
    return _get_repo().search_facts(key)


def save_typed_fact(
    key: str, value: str, source: str, memory_type: str, expires_at: str | None = None
) -> None:
    _get_repo().save_typed_fact(key, value, source, memory_type, expires_at)


def get_active_facts(query: str | None = None) -> list[dict]:
    return _get_repo().get_active_facts(query)


def cleanup_expired() -> int:
    return _get_repo().cleanup_expired()


def delete_fact(key_pattern: str) -> int:
    return _get_repo().delete_fact(key_pattern)


def save_chunk(session_id: str, chunk_index: int, transcript: str) -> None:
    _get_repo().save_chunk(session_id, chunk_index, transcript)


def get_session_transcript(session_id: str) -> list[dict]:
    return _get_repo().get_session_transcript(session_id)


def list_sessions(limit: int = 10) -> list[dict]:
    return _get_repo().list_sessions(limit)


def warm_embed_model():
    return _get_repo()._get_embed_model()
