"""Knowledge Graph repository — entities, facts, relationships.

Sits alongside the flat `facts` table. hybrid-retrieval retrieval contract untouched.
See docs/superpowers/specs/2026-05-31-kg-1-knowledge-graph-design.md.
"""

# ─── Imports ───────────────────────────────────────────────────────────────
import json
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from assistant.storage.db import Database

logger = logging.getLogger("storage.repos.knowledge_graph")


# ─── Helpers ───────────────────────────────────────────────────────────────
def _canonicalize(name: str) -> str:
    """Lowercase + whitespace-strip + collapse-internal-whitespace."""
    return " ".join(name.lower().strip().split())


def _now_iso() -> str:
    return datetime.now().isoformat()


# ─── Repo ──────────────────────────────────────────────────────────────────
class KnowledgeGraphRepo:
    """CRUD + dedup + 1-hop traversal over kg_* tables."""

    _COSINE_THRESHOLD = 0.85
    _COSINE_SUBSET_THRESHOLD = 0.75

    def __init__(self, db: "Database", embed_model_loader: Callable):
        """`db` is the singleton Database. `embed_model_loader` is a
        zero-arg callable that returns a sentence-transformers model
        (lazy — only called when cosine dedup is needed)."""
        self._db = db
        self._load_embed = embed_model_loader
        self._embed_model = None  # lazy

    # ─── Entity ops ────────────────────────────────────────────────────────
    def upsert_entity(
        self, entity_type: str, name: str, source: str,
        properties: dict | None = None, confidence: float = 1.0,
        source_turn_id: str | None = None,
    ) -> tuple[int, bool]:
        """Returns (entity_id, was_created).

        1. Exact (type, canonical_name) match → return existing id.
        2. Cosine fallback (added in Task 3).
        3. Else INSERT.

        source_turn_id is recorded only on the INSERT path — when an
        existing row is matched (exact or cosine), provenance stays with
        the FIRST turn that introduced the entity.
        """
        canon = _canonicalize(name)
        existing = self._db.fetchone(
            "SELECT id FROM kg_entities WHERE type = ? AND canonical_name = ?",
            (entity_type, canon),
        )
        if existing:
            return existing["id"], False

        merged_id = self._try_cosine_merge(entity_type, canon, name)
        if merged_id is not None:
            return merged_id, False

        return self._insert_entity(
            entity_type, canon, name, source, properties or {}, confidence,
            source_turn_id=source_turn_id,
        ), True

    def _insert_entity(
        self, entity_type: str, canon: str, display_name: str,
        source: str, properties: dict, confidence: float,
        source_turn_id: str | None = None,
    ) -> int:
        now = _now_iso()
        cur = self._db.execute(
            "INSERT INTO kg_entities (type, canonical_name, display_name, "
            "properties_json, source, confidence, created_at, updated_at, "
            "source_turn_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (entity_type, canon, display_name, json.dumps(properties),
             source, confidence, now, now, source_turn_id),
        )
        self._db.commit()
        return cur.lastrowid

    def _try_cosine_merge(self, entity_type: str, canon: str, original: str) -> int | None:
        """Embed `canon`; cosine-match against same-type entities.

        Two merge paths:
        1. Strong cosine (>= _COSINE_THRESHOLD).
        2. Token-subset (one name's tokens fully contained in the other's)
           AND cosine >= _COSINE_SUBSET_THRESHOLD. Catches "Aanya" -> "Aanya Sharma"
           where the embedding similarity sits in the 0.75-0.85 band.
           Same-token-set comparison is the all-or-nothing extreme of this rule.
        """
        candidates = self._db.fetchall(
            "SELECT id, canonical_name, properties_json FROM kg_entities WHERE type = ?",
            (entity_type,),
        )
        if not candidates:
            return None

        try:
            model = self._get_embed_model()
            target_vec = model.encode(canon, normalize_embeddings=True)
            cand_names = [c["canonical_name"] for c in candidates]
            cand_vecs = model.encode(cand_names, normalize_embeddings=True)
            import numpy as np
            sims = np.asarray(cand_vecs) @ np.asarray(target_vec)
            best_idx = int(sims.argmax())
            best_sim = float(sims[best_idx])
        except Exception as e:
            logger.debug(f"[KG] cosine merge failed (non-critical): {e}")
            return None

        if best_sim >= self._COSINE_THRESHOLD:
            return self._apply_alias_merge(candidates[best_idx], original)

        if best_sim >= self._COSINE_SUBSET_THRESHOLD:
            target_tokens = set(canon.split())
            cand_tokens = set(candidates[best_idx]["canonical_name"].split())
            if target_tokens and cand_tokens and (
                target_tokens <= cand_tokens or cand_tokens <= target_tokens
            ):
                return self._apply_alias_merge(candidates[best_idx], original)

        return None

    def _apply_alias_merge(self, match_row, original: str) -> int:
        try:
            props = json.loads(match_row["properties_json"]) if match_row["properties_json"] else {}
        except json.JSONDecodeError:
            props = {}
        aliases = props.setdefault("aliases", [])
        if original not in aliases:
            aliases.append(original)
        self._db.execute(
            "UPDATE kg_entities SET properties_json = ?, updated_at = ? WHERE id = ?",
            (json.dumps(props), _now_iso(), match_row["id"]),
        )
        self._db.commit()
        return match_row["id"]

    def _get_embed_model(self):
        if self._embed_model is None:
            self._embed_model = self._load_embed()
        return self._embed_model

    def get_entity(self, entity_id: int) -> dict | None:
        row = self._db.fetchone(
            "SELECT * FROM kg_entities WHERE id = ?", (entity_id,)
        )
        return dict(row) if row else None

    # ─── Fact ops ──────────────────────────────────────────────────────────
    def add_fact(
        self, subject_id: int, predicate: str, object: str,
        source: str, confidence: float = 1.0,
        expires_at: str | None = None,
        event_at: str | None = None,
        source_turn_id: str | None = None,
    ) -> int:
        """UPSERT with knowledge-graph item B fact-invalidation semantics.

        - Same (subject, predicate, object) repeated → UPSERT max(confidence),
          refresh verified_at, clear invalid_at (restoration), refresh
          expires_at if provided.
        - Same (subject, predicate) with a different object: mark all
          currently-valid rows for that (subject, predicate) as invalid_at=now,
          then UPSERT/INSERT the new triple.

        UNIQUE(subject_id, predicate, object) means the restoration path is
        the only way to re-validate an originally-invalidated triple — we
        cannot INSERT a second row with the same triple.
        """
        now = _now_iso()

        # Step 1: invalidate any currently-valid rows for (subject, predicate)
        # whose object differs from the incoming one. Restoration of the same
        # triple is handled in step 2 (it will UPDATE that row, including
        # clearing invalid_at).
        self._db.execute(
            "UPDATE kg_facts SET invalid_at = ? "
            "WHERE subject_id = ? AND predicate = ? AND object != ? "
            "AND invalid_at IS NULL",
            (now, subject_id, predicate, object),
        )

        # Step 2: UPSERT the (subject, predicate, object) row itself.
        existing = self._db.fetchone(
            "SELECT id, confidence FROM kg_facts "
            "WHERE subject_id = ? AND predicate = ? AND object = ?",
            (subject_id, predicate, object),
        )
        if existing:
            new_conf = max(float(existing["confidence"]), float(confidence))
            # Build UPDATE dynamically: only overwrite expires_at / event_at
            # when a fresh value is provided. None means "leave alone."
            sets = ["confidence = ?", "verified_at = ?", "invalid_at = NULL"]
            params: list = [new_conf, now]
            if expires_at is not None:
                sets.append("expires_at = ?")
                params.append(expires_at)
            if event_at is not None:
                sets.append("event_at = ?")
                params.append(event_at)
            params.append(existing["id"])
            self._db.execute(
                f"UPDATE kg_facts SET {', '.join(sets)} WHERE id = ?",
                tuple(params),
            )
            self._db.commit()
            return existing["id"]

        cur = self._db.execute(
            "INSERT INTO kg_facts (subject_id, predicate, object, "
            "confidence, source, verified_at, expires_at, event_at, "
            "created_at, source_turn_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (subject_id, predicate, object, confidence, source,
             None, expires_at, event_at, now, source_turn_id),
        )
        self._db.commit()
        return cur.lastrowid

    def get_facts_for_entity(self, entity_id: int, limit: int = 20) -> list[dict]:
        rows = self._db.fetchall(
            "SELECT * FROM kg_facts WHERE subject_id = ? "
            "AND invalid_at IS NULL "
            "ORDER BY confidence DESC, id DESC LIMIT ?",
            (entity_id, limit),
        )
        return [dict(r) for r in rows]

    def cleanup_expired_facts(self) -> int:
        now = _now_iso()
        cur = self._db.execute(
            "DELETE FROM kg_facts WHERE expires_at IS NOT NULL AND expires_at < ?",
            (now,),
        )
        self._db.commit()
        count = cur.rowcount
        if count > 0:
            logger.info(f"[KG] Cleaned up {count} expired KG fact(s)")
        return count

    # ─── Relationship ops ──────────────────────────────────────────────────
    def add_relationship(
        self, from_id: int, to_id: int, rel_type: str,
        source: str, confidence: float = 1.0,
        properties: dict | None = None,
        source_turn_id: str | None = None,
    ) -> int:
        existing = self._db.fetchone(
            "SELECT id FROM kg_relationships "
            "WHERE from_id = ? AND to_id = ? AND type = ?",
            (from_id, to_id, rel_type),
        )
        if existing:
            return existing["id"]
        cur = self._db.execute(
            "INSERT INTO kg_relationships (from_id, to_id, type, "
            "properties_json, confidence, source, created_at, source_turn_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (from_id, to_id, rel_type, json.dumps(properties or {}),
             confidence, source, _now_iso(), source_turn_id),
        )
        self._db.commit()
        return cur.lastrowid

    def get_neighbors(
        self, entity_id: int, depth: int = 1, limit: int = 10,
    ) -> list[dict]:
        """BFS up to `depth` hops. Returns
        [{entity, rel_type, direction, distance}, ...] in BFS order."""
        visited = {entity_id}
        frontier = [entity_id]
        results: list[dict] = []
        for d in range(1, depth + 1):
            if not frontier:
                break
            placeholders = ",".join("?" * len(frontier))
            edges = self._db.fetchall(
                f"SELECT from_id, to_id, type FROM kg_relationships "
                f"WHERE from_id IN ({placeholders}) OR to_id IN ({placeholders})",
                tuple(frontier) + tuple(frontier),
            )
            next_frontier: list[int] = []
            for e in edges:
                if e["from_id"] in frontier and e["to_id"] not in visited:
                    other, direction = e["to_id"], "out"
                elif e["to_id"] in frontier and e["from_id"] not in visited:
                    other, direction = e["from_id"], "in"
                else:
                    continue
                entity = self.get_entity(other)
                if entity is None:
                    continue
                results.append({
                    "entity": entity, "rel_type": e["type"],
                    "direction": direction, "distance": d,
                })
                visited.add(other)
                next_frontier.append(other)
                if len(results) >= limit:
                    return results
            frontier = next_frontier
        return results

    # ─── Compound query ────────────────────────────────────────────────────
    def find_entities_by_name(
        self, name: str, entity_type: str | None = None,
    ) -> list[dict]:
        """Exact match on canonical_name OR any alias entry in properties_json."""
        canon = _canonicalize(name)
        params: tuple = (canon,)
        sql = "SELECT * FROM kg_entities WHERE canonical_name = ?"
        if entity_type is not None:
            sql += " AND type = ?"
            params = (canon, entity_type)
        rows = self._db.fetchall(sql, params)
        results = [dict(r) for r in rows]
        if results:
            return results

        # Alias scan — LIKE on JSON. Cheap because aliases are short and rare.
        like = f'%"{name}"%'
        params2: tuple = (like,)
        sql2 = "SELECT * FROM kg_entities WHERE properties_json LIKE ?"
        if entity_type is not None:
            sql2 += " AND type = ?"
            params2 = (like, entity_type)
        rows2 = self._db.fetchall(sql2, params2)
        return [dict(r) for r in rows2]

    def expand_entity_context(
        self, entity_id: int, fact_limit: int = 5, neighbor_limit: int = 5,
        commitment_limit: int = 3,
    ) -> dict:
        """Returns {entity, facts, neighbors, commitments}. Sole read-time
        method called from build_kg_context. Commitments are open ones
        only — fulfilled commitments rarely shape the next answer."""
        entity = self.get_entity(entity_id)
        if entity is None:
            return {"entity": None, "facts": [], "neighbors": [],
                    "commitments": []}
        facts = self.get_facts_for_entity(entity_id, limit=fact_limit)
        neighbors = self.get_neighbors(entity_id, depth=1, limit=neighbor_limit)
        commitments = self.list_open_commitments(
            owner_id=entity_id, limit=commitment_limit,
        )
        return {
            "entity": entity, "facts": facts, "neighbors": neighbors,
            "commitments": commitments,
        }

    # ─── Commitment ops (E) ───────────────────────────────────────────
    def add_commitment(
        self, owner_id: int, promise_text: str, *,
        source: str, when_due: str | None = None,
        source_turn_id: str | None = None, reminder_id: int | None = None,
    ) -> int:
        """Insert a new open commitment. Returns the row id."""
        cur = self._db.execute(
            "INSERT INTO kg_commitments "
            "(owner_id, promise_text, when_due, created_at, source, "
            "source_turn_id, reminder_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (owner_id, promise_text, when_due, _now_iso(), source,
             source_turn_id, reminder_id),
        )
        self._db.commit()
        return cur.lastrowid

    def list_open_commitments(
        self, owner_id: int | None = None, limit: int = 20,
    ) -> list[dict]:
        """Return open commitments (fulfilled_at IS NULL), newest first.
        Filter by owner if given."""
        if owner_id is None:
            rows = self._db.fetchall(
                "SELECT * FROM kg_commitments WHERE fulfilled_at IS NULL "
                "ORDER BY id DESC LIMIT ?",
                (limit,),
            )
        else:
            rows = self._db.fetchall(
                "SELECT * FROM kg_commitments "
                "WHERE owner_id = ? AND fulfilled_at IS NULL "
                "ORDER BY id DESC LIMIT ?",
                (owner_id, limit),
            )
        return [dict(r) for r in rows]

    def mark_commitment_fulfilled(self, commitment_id: int) -> bool:
        """Stamp fulfilled_at on a commitment. Returns False if not found
        or already fulfilled."""
        cur = self._db.execute(
            "UPDATE kg_commitments SET fulfilled_at = ? "
            "WHERE id = ? AND fulfilled_at IS NULL",
            (_now_iso(), commitment_id),
        )
        self._db.commit()
        return cur.rowcount > 0

    def find_commitments_by_text(self, query: str, limit: int = 5) -> list[dict]:
        """LIKE-match on promise_text for cancellation / lookup. Cheap because
        the table is small (one row per turn-with-promise, not per turn)."""
        like = f"%{query.strip().lower()}%"
        rows = self._db.fetchall(
            "SELECT * FROM kg_commitments "
            "WHERE LOWER(promise_text) LIKE ? "
            "ORDER BY (fulfilled_at IS NULL) DESC, id DESC LIMIT ?",
            (like, limit),
        )
        return [dict(r) for r in rows]
