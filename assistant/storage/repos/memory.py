"""
storage/repos/memory.py — Conversation memory persistence.

Owns SQLite tables (conversations, facts, recording_sessions),
FAISS vector indices, and ID-map files. The top-level memory.py
facade delegates here.
"""

import json
import logging
import os
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import faiss
    from sentence_transformers import SentenceTransformer
    HAS_VECTOR_DEPS = True
except ImportError:
    HAS_VECTOR_DEPS = False

from ..db import Database

logger = logging.getLogger("memory")

warnings.filterwarnings("ignore", message=".*position_ids.*")

_EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_EMBED_DIM = 384
_ID_MAP_VERSION = 1
_MIN_SIMILARITY = 0.25


class MemoryRepo:
    """Conversation, fact, and recording persistence with optional FAISS search."""

    def __init__(self, db: Database, data_dir: Path) -> None:
        self._db = db
        self._data_dir = data_dir

        self._index_path = data_dir / "vector.index"
        self._id_map_path = data_dir / "vector_ids.json"
        self._rs_index_path = data_dir / "vector_rs.index"
        self._rs_id_map_path = data_dir / "vector_rs_ids.json"

        self._embed_model: Optional["SentenceTransformer"] = None
        self._faiss_index: Optional["faiss.IndexFlatIP"] = None
        self._id_map: list[int] = []
        self._rs_faiss_index: Optional["faiss.IndexFlatIP"] = None
        self._rs_id_map: list[int] = []
        self._facts_index_path = data_dir / "vector_facts.index"
        self._facts_id_map_path = data_dir / "vector_facts_ids.json"
        self._facts_faiss_index: Optional["faiss.IndexFlatIP"] = None
        self._facts_id_map: list[int] = []

    # ─── Embedding ──────────────────────────────────────────────────────

    def _get_embed_model(self) -> Optional["SentenceTransformer"]:
        if not HAS_VECTOR_DEPS:
            return None
        if self._embed_model is None:
            try:
                logger.info(f"[MEMORY] Loading embedding model: {_EMBED_MODEL_NAME}")
                self._embed_model = SentenceTransformer(_EMBED_MODEL_NAME)
            except Exception as e:
                logger.warning(f"[MEMORY] Failed to load embedding model: {e}")
                return None
        return self._embed_model

    def _embed(self, texts: list[str]) -> Optional[np.ndarray]:
        if not texts:
            return None
        model = self._get_embed_model()
        embeddings = None
        if model:
            try:
                embeddings = model.encode(texts)
            except Exception as e:
                logger.warning(f"[MEMORY] Local embedding failed: {e}")
        if embeddings is None:
            api_key = os.getenv("JINA_API_KEY", "")
            if api_key:
                try:
                    import requests
                    logger.info("[MEMORY] Falling back to Jina API for embeddings")
                    url = "https://api.jina.ai/v1/embeddings"
                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {api_key}",
                    }
                    data = {"model": "jina-embeddings-v2-base-en", "input": texts}
                    response = requests.post(url, headers=headers, json=data, timeout=10)
                    response.raise_for_status()
                    res_data = response.json()
                    embeddings = np.array(
                        [item["embedding"] for item in res_data["data"]],
                        dtype=np.float32,
                    )
                except Exception as e:
                    logger.warning(f"[MEMORY] Jina API embedding fallback failed: {e}")
        if embeddings is not None:
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)
            norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norm[norm == 0] = 1.0
            embeddings = embeddings / norm
            return embeddings.astype(np.float32)
        return None

    # ─── Vector Store Init ──────────────────────────────────────────────

    def _load_id_map(self, path: Path) -> list[int]:
        """Load an ID-map file, handling both legacy (bare list) and versioned formats."""
        if not path.exists():
            return []
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            self._save_id_map(path, data)
            return data
        if isinstance(data, dict) and data.get("version") == _ID_MAP_VERSION:
            return data["ids"]
        logger.warning(f"[MEMORY] Unrecognised id-map format at {path}, rebuilding")
        return []

    def _save_id_map(self, path: Path, ids: list[int]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"version": _ID_MAP_VERSION, "ids": ids}, f)

    # ─── Reciprocal Rank Fusion ───────────────────────────────────────────

    @staticmethod
    def _rrf_fuse(
        *ranked_lists: list[tuple[int, float]],
        k: int = 60,
        limit: int = 10,
    ) -> list[tuple[int, float]]:
        scores: dict[int, float] = {}
        for ranked in ranked_lists:
            for rank, (item_id, _score) in enumerate(ranked):
                scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank + 1)
        fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return fused[:limit]

    # ─── FTS5 Search ──────────────────────────────────────────────────────

    @staticmethod
    def _sanitize_fts_query(query: str) -> str:
        tokens = query.split()
        sanitized = []
        for t in tokens:
            t = t.replace('"', "")
            if t:
                sanitized.append(f'"{t}"')
        return " ".join(sanitized)

    def _search_facts_fts(self, query: str, limit: int = 10) -> list[tuple[int, float]]:
        fts_query = self._sanitize_fts_query(query)
        if not fts_query:
            return []
        try:
            rows = self._db.fetchall(
                "SELECT rowid, rank FROM facts_fts WHERE facts_fts MATCH ? "
                "ORDER BY rank LIMIT ?",
                (fts_query, limit),
            )
            return [(row["rowid"], -float(row["rank"])) for row in rows]
        except Exception as e:
            logger.warning(f"[MEMORY] FTS5 facts search failed: {e}")
            return []

    def _search_conversations_fts(self, query: str, limit: int = 10) -> list[tuple[int, float]]:
        fts_query = self._sanitize_fts_query(query)
        if not fts_query:
            return []
        try:
            rows = self._db.fetchall(
                "SELECT rowid, rank FROM conversations_fts "
                "WHERE conversations_fts MATCH ? ORDER BY rank LIMIT ?",
                (fts_query, limit),
            )
            return [(row["rowid"], -float(row["rank"])) for row in rows]
        except Exception as e:
            logger.warning(f"[MEMORY] FTS5 conversations search failed: {e}")
            return []

    def init_vector_store(self) -> None:
        if not HAS_VECTOR_DEPS:
            logger.warning(
                "[MEMORY] Vector search dependencies (faiss, sentence-transformers) "
                "not found. Vector search disabled."
            )
            return
        try:
            # Conversations index
            if self._index_path.exists():
                self._faiss_index = faiss.read_index(str(self._index_path))
                self._id_map = self._load_id_map(self._id_map_path)
                logger.info(f"[MEMORY] Loaded conversation index with {len(self._id_map)} entries.")
            else:
                self._faiss_index = faiss.IndexFlatIP(_EMBED_DIM)
                self._id_map = []
                logger.info("[MEMORY] Created fresh conversation index.")

            # Sync new conversations
            placeholder = ",".join(["?"] * len(self._id_map)) if self._id_map else "-1"
            new_rows = self._db.fetchall(
                f"SELECT id, user_input, response FROM conversations "
                f"WHERE id NOT IN ({placeholder}) ORDER BY id ASC",
                tuple(self._id_map),
            )
            if new_rows:
                logger.info(f"[MEMORY] Indexing {len(new_rows)} new conversation turns.")
                texts = [f"{row['user_input']} {row['response']}" for row in new_rows]
                embeddings = self._embed(texts)
                if embeddings is not None:
                    self._faiss_index.add(embeddings)
                    self._id_map.extend([row["id"] for row in new_rows])
                    faiss.write_index(self._faiss_index, str(self._index_path))
                    self._save_id_map(self._id_map_path, self._id_map)

            # Recording sessions index
            if self._rs_index_path.exists():
                self._rs_faiss_index = faiss.read_index(str(self._rs_index_path))
                self._rs_id_map = self._load_id_map(self._rs_id_map_path)
                logger.info(f"[MEMORY] Loaded recording index with {len(self._rs_id_map)} entries.")
            else:
                self._rs_faiss_index = faiss.IndexFlatIP(_EMBED_DIM)
                self._rs_id_map = []
                logger.info("[MEMORY] Created fresh recording index.")

            # Sync new recordings
            placeholder_rs = ",".join(["?"] * len(self._rs_id_map)) if self._rs_id_map else "-1"
            new_rs_rows = self._db.fetchall(
                f"SELECT id, transcript FROM recording_sessions "
                f"WHERE id NOT IN ({placeholder_rs}) ORDER BY id ASC",
                tuple(self._rs_id_map),
            )
            if new_rs_rows:
                logger.info(f"[MEMORY] Indexing {len(new_rs_rows)} new recording chunks.")
                rs_texts = [row["transcript"] for row in new_rs_rows]
                rs_embeddings = self._embed(rs_texts)
                if rs_embeddings is not None:
                    self._rs_faiss_index.add(rs_embeddings)
                    self._rs_id_map.extend([row["id"] for row in new_rs_rows])
                    faiss.write_index(self._rs_faiss_index, str(self._rs_index_path))
                    self._save_id_map(self._rs_id_map_path, self._rs_id_map)

            # Facts index
            if self._facts_index_path.exists():
                self._facts_faiss_index = faiss.read_index(str(self._facts_index_path))
                self._facts_id_map = self._load_id_map(self._facts_id_map_path)
                logger.info(f"[MEMORY] Loaded facts index with {len(self._facts_id_map)} entries.")
            else:
                self._facts_faiss_index = faiss.IndexFlatIP(_EMBED_DIM)
                self._facts_id_map = []
                logger.info("[MEMORY] Created fresh facts index.")

            # Sync new facts
            ph_facts = ",".join(["?"] * len(self._facts_id_map)) if self._facts_id_map else "-1"
            new_facts = self._db.fetchall(
                f"SELECT id, key, value FROM facts "
                f"WHERE id NOT IN ({ph_facts}) ORDER BY id ASC",
                tuple(self._facts_id_map),
            )
            if new_facts:
                logger.info(f"[MEMORY] Indexing {len(new_facts)} new facts.")
                texts = [f"{row['key']} {row['value']}" for row in new_facts]
                embeddings = self._embed(texts)
                if embeddings is not None:
                    self._facts_faiss_index.add(embeddings)
                    self._facts_id_map.extend([row["id"] for row in new_facts])
                    faiss.write_index(self._facts_faiss_index, str(self._facts_index_path))
                    self._save_id_map(self._facts_id_map_path, self._facts_id_map)

        except Exception as e:
            logger.warning(f"[MEMORY] Failed to initialize vector store: {e}")
            self._faiss_index = None
            self._rs_faiss_index = None
            self._facts_faiss_index = None

    # ─── Incremental Indexing ───────────────────────────────────────────

    def _index_new_turn(self, row_id: int, user_input: str, response: str) -> None:
        if self._faiss_index is None:
            return
        try:
            embedding = self._embed([f"{user_input} {response}"])
            if embedding is not None:
                self._faiss_index.add(embedding)
                self._id_map.append(row_id)
                faiss.write_index(self._faiss_index, str(self._index_path))
                self._save_id_map(self._id_map_path, self._id_map)
        except Exception as e:
            logger.warning(f"[MEMORY] Failed to index new turn: {e}")

    def _index_new_chunk(self, row_id: int, transcript: str) -> None:
        if self._rs_faiss_index is None:
            return
        try:
            embedding = self._embed([transcript])
            if embedding is not None:
                self._rs_faiss_index.add(embedding)
                self._rs_id_map.append(row_id)
                faiss.write_index(self._rs_faiss_index, str(self._rs_index_path))
                self._save_id_map(self._rs_id_map_path, self._rs_id_map)
        except Exception as e:
            logger.warning(f"[MEMORY] Failed to index new recording chunk: {e}")

    def _index_new_fact(self, fact_id: int, key: str, value: str) -> None:
        if self._facts_faiss_index is None:
            return
        try:
            embedding = self._embed([f"{key} {value}"])
            if embedding is not None:
                self._facts_faiss_index.add(embedding)
                self._facts_id_map.append(fact_id)
                faiss.write_index(self._facts_faiss_index, str(self._facts_index_path))
                self._save_id_map(self._facts_id_map_path, self._facts_id_map)
        except Exception as e:
            logger.warning(f"[MEMORY] Failed to index new fact: {e}")

    def _search_facts_semantic(self, query: str, limit: int = 10) -> list[tuple[int, float]]:
        if self._facts_faiss_index is None or not self._facts_id_map:
            return []
        try:
            query_embedding = self._embed([query])
            if query_embedding is None:
                return []
            search_limit = min(limit, self._facts_faiss_index.ntotal)
            if search_limit == 0:
                return []
            distances, indices = self._facts_faiss_index.search(query_embedding, search_limit)
            results = []
            for i, idx in enumerate(indices[0]):
                if idx == -1 or idx >= len(self._facts_id_map):
                    continue
                score = float(distances[0][i])
                if score < _MIN_SIMILARITY:
                    continue
                results.append((self._facts_id_map[idx], score))
            results.sort(key=lambda x: x[1], reverse=True)
            return results
        except Exception as e:
            logger.warning(f"[MEMORY] Semantic facts search failed: {e}")
            return []

    # ─── Conversation Storage ───────────────────────────────────────────

    def save_turn(self, user_input: str, intent: str, response: str, session_id: str) -> int:
        cur = self._db.execute(
            "INSERT INTO conversations (timestamp, user_input, intent, response, session_id) "
            "VALUES (?, ?, ?, ?, ?)",
            (datetime.now().isoformat(), user_input, intent, response, session_id),
        )
        row_id = cur.lastrowid
        self._db.commit()
        logger.debug(f"[MEMORY] Saved turn: intent={intent}, session={session_id}")
        self._index_new_turn(row_id, user_input, response)
        return row_id

    def get_recent(self, n: int = 10, session_id: str = "") -> list[dict]:
        if session_id:
            rows = self._db.fetchall(
                "SELECT * FROM conversations WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                (session_id, n),
            )
        else:
            rows = self._db.fetchall(
                "SELECT * FROM conversations ORDER BY id DESC LIMIT ?", (n,)
            )
        return [dict(row) for row in reversed(rows)]

    def build_recent_context(
        self, limit: int = 25, header: str = "RECENT CONVERSATION HISTORY:",
        session_id: str = "",
    ) -> str:
        try:
            turns = self.get_recent(limit, session_id=session_id)
            if not turns:
                return ""
            lines: list[str] = []
            if header:
                lines.append(header)
            for t in turns:
                lines.append(f"User: {t['user_input']}")
                lines.append(f"Assistant: {t['response']}")
            return "\n".join(lines)
        except Exception:
            return ""

    def summarize_session(self, session_id: str) -> str:
        rows = self._db.fetchall(
            "SELECT * FROM conversations WHERE session_id = ? ORDER BY id ASC",
            (session_id,),
        )
        if not rows:
            return f"No conversations found for session '{session_id}'."
        lines = []
        for row in rows:
            r = dict(row)
            lines.append(
                f"[{r['timestamp']}] User: {r['user_input']}\n"
                f"  → ({r['intent']}) {r['response']}"
            )
        return "\n\n".join(lines)

    # ─── Semantic Search ────────────────────────────────────────────────

    def search_conversations(self, query: str, limit: int = 5) -> list[dict]:
        results = []
        semantic_success = False
        if self._faiss_index is not None and self._id_map:
            try:
                query_embedding = self._embed([query])
                if query_embedding is not None:
                    search_limit = min(limit * 2, 20)
                    distances, indices = self._faiss_index.search(query_embedding, search_limit)
                    indices = indices[0]
                    distances = distances[0]
                    for i, idx in enumerate(indices):
                        if idx == -1 or idx >= len(self._id_map):
                            continue
                        row_id = self._id_map[idx]
                        row = self._db.fetchone(
                            "SELECT * FROM conversations WHERE id = ?", (row_id,)
                        )
                        if row:
                            res = dict(row)
                            res["similarity_score"] = float(distances[i])
                            results.append(res)
                    results.sort(key=lambda x: x["similarity_score"], reverse=True)
                    results = results[:limit]
                    semantic_success = True
                    logger.debug(
                        f"[MEMORY] Semantic search for '{query}' returned {len(results)} results."
                    )
            except Exception as e:
                logger.warning(f"[MEMORY] Semantic search failed, falling back to SQL: {e}")
        if not semantic_success:
            logger.debug(f"[MEMORY] Falling back to SQL LIKE search for '{query}'.")
            pattern = f"%{query}%"
            rows = self._db.fetchall(
                "SELECT * FROM conversations "
                "WHERE user_input LIKE ? OR response LIKE ? "
                "ORDER BY id DESC LIMIT ?",
                (pattern, pattern, limit),
            )
            results = [dict(row) for row in rows]
            for r in results:
                r["similarity_score"] = 0.0
        return results

    def search_recording_sessions(self, query: str, limit: int = 3) -> list[dict]:
        results = []
        semantic_success = False
        if self._rs_faiss_index is not None and self._rs_id_map:
            try:
                query_embedding = self._embed([query])
                if query_embedding is not None:
                    search_limit = min(limit * 2, 20)
                    distances, indices = self._rs_faiss_index.search(
                        query_embedding, search_limit
                    )
                    indices = indices[0]
                    distances = distances[0]
                    for i, idx in enumerate(indices):
                        if idx == -1 or idx >= len(self._rs_id_map):
                            continue
                        row_id = self._rs_id_map[idx]
                        row = self._db.fetchone(
                            "SELECT * FROM recording_sessions WHERE id = ?", (row_id,)
                        )
                        if row:
                            res = dict(row)
                            res["similarity_score"] = float(distances[i])
                            results.append(res)
                    results.sort(key=lambda x: x["similarity_score"], reverse=True)
                    results = results[:limit]
                    semantic_success = True
                    logger.debug(
                        f"[MEMORY] Semantic recording search for '{query}' "
                        f"returned {len(results)} results."
                    )
            except Exception as e:
                logger.warning(f"[MEMORY] Semantic recording search failed: {e}")
        if not semantic_success:
            logger.debug(f"[MEMORY] Falling back to SQL search for recording '{query}'.")
            pattern = f"%{query}%"
            rows = self._db.fetchall(
                "SELECT * FROM recording_sessions WHERE transcript LIKE ? "
                "ORDER BY id DESC LIMIT ?",
                (pattern, limit),
            )
            results = [dict(row) for row in rows]
            for r in results:
                r["similarity_score"] = 0.0
        return results

    # ─── Fact Storage ───────────────────────────────────────────────────

    _EXPIRY_DAYS = {
        "fact": 30,
        "how_to": 14,
        "blocker": 14,
    }
    _VALID_TYPES = frozenset({"preference", "identity", "fact", "how_to", "blocker"})

    def save_typed_fact(
        self,
        key: str,
        value: str,
        source: str,
        memory_type: str,
        expires_at: str | None = None,
    ) -> None:
        if memory_type not in self._VALID_TYPES:
            logger.warning(f"[MEMORY] Invalid memory_type '{memory_type}', coercing to 'fact'")
            memory_type = "fact"

        if expires_at is None:
            expiry_days = self._EXPIRY_DAYS.get(memory_type)
            if expiry_days is not None:
                expires_at = (datetime.now() + timedelta(days=expiry_days)).isoformat()

        cur = self._db.execute(
            "INSERT INTO facts (timestamp, key, value, source, memory_type, expires_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (datetime.now().isoformat(), key, value, source, memory_type, expires_at),
        )
        self._db.commit()
        logger.debug(f"[MEMORY] Saved typed fact: {key}={value} (type={memory_type}, expires={expires_at})")
        self._index_new_fact(cur.lastrowid, key, value)

    def save_fact(self, key: str, value: str, source: str = "user") -> None:
        self.save_typed_fact(key, value, source, memory_type="fact")

    def get_active_facts(self, query: str | None = None) -> list[dict]:
        now = datetime.now().isoformat()
        if query:
            pattern = f"%{query}%"
            rows = self._db.fetchall(
                "SELECT * FROM facts "
                "WHERE (expires_at IS NULL OR expires_at > ?) AND key LIKE ? "
                "ORDER BY id DESC",
                (now, pattern),
            )
        else:
            rows = self._db.fetchall(
                "SELECT * FROM facts "
                "WHERE expires_at IS NULL OR expires_at > ? "
                "ORDER BY id DESC",
                (now,),
            )
        return [dict(row) for row in rows]

    def cleanup_expired(self) -> int:
        now = datetime.now().isoformat()
        cursor = self._db.execute(
            "DELETE FROM facts WHERE expires_at IS NOT NULL AND expires_at < ?",
            (now,),
        )
        self._db.commit()
        count = cursor.rowcount
        if count > 0:
            logger.info(f"[MEMORY] Cleaned up {count} expired fact(s)")
        return count

    def delete_fact(self, key_pattern: str) -> int:
        pattern = f"%{key_pattern}%"
        cursor = self._db.execute(
            "DELETE FROM facts WHERE key LIKE ?",
            (pattern,),
        )
        self._db.commit()
        count = cursor.rowcount
        if count > 0:
            logger.info(f"[MEMORY] Deleted {count} fact(s) matching '{key_pattern}'")
        return count

    def search_facts(self, key: str) -> list[dict]:
        pattern = f"%{key}%"
        rows = self._db.fetchall(
            "SELECT * FROM facts WHERE key LIKE ? ORDER BY id DESC", (pattern,)
        )
        return [dict(row) for row in rows]

    # ─── Hybrid Search ─────────────────────────────────────────────────────

    def hybrid_search_facts(self, query: str, limit: int = 10) -> list[dict]:
        if not query or not query.strip():
            return []

        fetch_limit = limit * 2
        semantic_results = self._search_facts_semantic(query, limit=fetch_limit)
        fts_results = self._search_facts_fts(query, limit=fetch_limit)

        sem_top = f"{semantic_results[0][1]:.3f}" if semantic_results else "-"
        fts_top = f"{fts_results[0][1]:.3f}" if fts_results else "-"
        logger.info(
            f"[MEMORY] Hybrid facts: query={query!r}, "
            f"semantic={len(semantic_results)} (top={sem_top}), "
            f"fts={len(fts_results)} (top={fts_top})"
        )

        if not semantic_results and not fts_results:
            logger.info(f"[MEMORY] Hybrid facts: no results, falling back to LIKE")
            pattern = f"%{query}%"
            rows = self._db.fetchall(
                "SELECT * FROM facts WHERE (key LIKE ? OR value LIKE ?) "
                "ORDER BY id DESC LIMIT ?",
                (pattern, pattern, limit),
            )
            results = [dict(row) for row in rows]
            for r in results:
                r["rrf_score"] = 0.0
            return results

        fused = self._rrf_fuse(semantic_results, fts_results, limit=limit * 2)
        if not fused:
            return []

        fused_ids = [item_id for item_id, _score in fused]
        score_map = {item_id: score for item_id, score in fused}

        placeholders = ",".join(["?"] * len(fused_ids))
        now = datetime.now().isoformat()
        rows = self._db.fetchall(
            f"SELECT * FROM facts WHERE id IN ({placeholders}) "
            f"AND (expires_at IS NULL OR expires_at > ?)",
            (*fused_ids, now),
        )

        results = []
        for row in rows:
            r = dict(row)
            r["rrf_score"] = score_map.get(r["id"], 0.0)
            results.append(r)

        results.sort(key=lambda x: x["rrf_score"], reverse=True)
        final = results[:limit]
        if final:
            top = final[0]
            logger.info(f"[MEMORY] Hybrid facts: returning {len(final)}, top={top['key']}={top['value'][:40]}")
        return final

    def hybrid_search_conversations(self, query: str, limit: int = 5) -> list[dict]:
        if not query or not query.strip():
            return []

        fetch_limit = limit * 2
        semantic_results: list[tuple[int, float]] = []
        if self._faiss_index is not None and self._id_map:
            try:
                query_embedding = self._embed([query])
                if query_embedding is not None:
                    search_limit = min(fetch_limit, self._faiss_index.ntotal)
                    if search_limit > 0:
                        distances, indices = self._faiss_index.search(
                            query_embedding, search_limit
                        )
                        for i, idx in enumerate(indices[0]):
                            if idx == -1 or idx >= len(self._id_map):
                                continue
                            score = float(distances[0][i])
                            if score < _MIN_SIMILARITY:
                                continue
                            semantic_results.append(
                                (self._id_map[idx], score)
                            )
                        semantic_results.sort(key=lambda x: x[1], reverse=True)
            except Exception as e:
                logger.warning(f"[MEMORY] Semantic conversation search failed: {e}")

        fts_results = self._search_conversations_fts(query, limit=fetch_limit)

        sem_top = f"{semantic_results[0][1]:.3f}" if semantic_results else "-"
        fts_top = f"{fts_results[0][1]:.3f}" if fts_results else "-"
        logger.info(
            f"[MEMORY] Hybrid convos: query={query!r}, "
            f"semantic={len(semantic_results)} (top={sem_top}), "
            f"fts={len(fts_results)} (top={fts_top})"
        )

        if not semantic_results and not fts_results:
            logger.info(f"[MEMORY] Hybrid convos: no results, falling back to LIKE")
            pattern = f"%{query}%"
            rows = self._db.fetchall(
                "SELECT * FROM conversations "
                "WHERE user_input LIKE ? OR response LIKE ? "
                "ORDER BY id DESC LIMIT ?",
                (pattern, pattern, limit),
            )
            results = [dict(row) for row in rows]
            for r in results:
                r["rrf_score"] = 0.0
            return results

        fused = self._rrf_fuse(semantic_results, fts_results, limit=limit * 2)
        if not fused:
            return []

        fused_ids = [item_id for item_id, _score in fused]
        score_map = {item_id: score for item_id, score in fused}

        placeholders = ",".join(["?"] * len(fused_ids))
        rows = self._db.fetchall(
            f"SELECT * FROM conversations WHERE id IN ({placeholders})",
            tuple(fused_ids),
        )

        results = []
        for row in rows:
            r = dict(row)
            r["rrf_score"] = score_map.get(r["id"], 0.0)
            results.append(r)

        results.sort(key=lambda x: x["rrf_score"], reverse=True)
        final = results[:limit]
        logger.info(f"[MEMORY] Hybrid convos: returning {len(final)}")
        return final

    # ─── Recording Storage ──────────────────────────────────────────────

    def save_chunk(self, session_id: str, chunk_index: int, transcript: str) -> None:
        cur = self._db.execute(
            "INSERT INTO recording_sessions (session_id, chunk_index, timestamp, transcript) "
            "VALUES (?, ?, ?, ?)",
            (session_id, chunk_index, datetime.now().isoformat(), transcript),
        )
        row_id = cur.lastrowid
        self._db.commit()
        logger.debug(f"[MEMORY] Saved recording chunk {chunk_index} for session {session_id}")
        self._index_new_chunk(row_id, transcript)

    def get_session_transcript(self, session_id: str) -> list[dict]:
        rows = self._db.fetchall(
            "SELECT * FROM recording_sessions WHERE session_id = ? ORDER BY chunk_index ASC",
            (session_id,),
        )
        return [dict(row) for row in rows]

    def list_sessions(self, limit: int = 10) -> list[dict]:
        rows = self._db.fetchall(
            """
            SELECT
                session_id,
                COUNT(*)        AS chunk_count,
                MIN(timestamp)  AS started_at,
                MAX(timestamp)  AS ended_at
            FROM recording_sessions
            GROUP BY session_id
            ORDER BY started_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [dict(row) for row in rows]
