"""Knowledge graph domain facade.

Wraps storage/repos/knowledge_graph with TENKA-level concerns: extraction
pre-filter, ingest-turn orchestration, query-time context block builder.
"""

# ─── Imports ───────────────────────────────────────────────────────────────
import logging
import os
import re
from datetime import datetime
from typing import Literal

logger = logging.getLogger("assistant.knowledge_graph")

# ─── Module state ──────────────────────────────────────────────────────────
_repo = None


# ─── Temporal grounding helper (item A) ───────────────────────────────
_MONTHS = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)


# Bare relative phrases the LLM sometimes silently resolves to absolute
# dates (often with the wrong year). Used by _strip_unfounded_event_at.
_RELATIVE_PHRASE_RE = re.compile(
    r"\b(last|this|next|past|previous|coming|upcoming)\s+"
    r"(week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"january|february|march|april|may|june|july|august|september|october|november|december)\b"
    r"|"
    r"\b(yesterday|today|tomorrow|tonight|recently|earlier|later|just\s+now|a\s+while\s+ago)\b",
    re.IGNORECASE,
)

# An explicit calendar anchor: a 4-digit year or a numeric date pattern.
# Month names alone are NOT explicit because "last March" / "next March" are
# both relative — only a year or absolute date disambiguates.
_EXPLICIT_DATE_RE = re.compile(
    r"\b(19|20)\d{2}\b"
    r"|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
)


def _strip_unfounded_event_at(text: str, event_at: str | None) -> str | None:
    """Conservative defense: keep event_at ONLY when (a) the value parses as
    an ISO date and (b) the source text contains an explicit calendar anchor
    (4-digit year or numeric date). Otherwise drop.

    Strict because Flash-Lite hallucinates years aggressively: 'in December'
    becomes 2023-12, 'joined Voyager' becomes 2024-05-16, schema placeholders
    like 'YYYY-MM-DD' echo verbatim. Accepts the trade-off of dropping
    legitimate but unanchored values ('I moved on Jan 15' with no year) in
    exchange for never persisting a fabricated year.
    """
    if not event_at:
        return None
    s = event_at.strip()
    parseable = False
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            datetime.strptime(s, fmt)
            parseable = True
            break
        except ValueError:
            continue
    if not parseable:
        return None
    if _EXPLICIT_DATE_RE.search(text):
        return event_at
    return None


def _relative_date(iso_str, now: datetime) -> str | None:
    """Render an ISO date string as a human-readable relative phrase.

    The LLM never computes datetimes (CLAUDE.md gotcha) — Python does it
    here and passes a literal string into the [KNOWLEDGE] block.

    Accepts year-only ("2024"), year-month ("2026-03"), or full
    ("2026-06-03"). Returns None for empty/unparseable input so callers can
    cleanly skip the suffix.
    """
    if not iso_str:
        return None
    s = str(iso_str).strip()
    parsed = None
    fmt_used = None
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            parsed = datetime.strptime(s, fmt)
            fmt_used = fmt
            break
        except ValueError:
            continue
    if parsed is None or fmt_used is None:
        return None

    if fmt_used == "%Y":
        return f"in {parsed.year}"

    if fmt_used == "%Y-%m":
        if parsed.year == now.year:
            return f"in {_MONTHS[parsed.month - 1]}"
        return f"in {_MONTHS[parsed.month - 1]} {parsed.year}"

    delta_days = (parsed.date() - now.date()).days
    if delta_days == 0:
        return "today"
    if delta_days == -1:
        return "yesterday"
    if delta_days == 1:
        return "tomorrow"
    if -6 <= delta_days <= -2:
        return f"{abs(delta_days)} days ago"
    if 2 <= delta_days <= 6:
        return f"in {delta_days} days"
    if -13 <= delta_days <= -7:
        return "last week"
    if 7 <= delta_days <= 13:
        return "next week"
    if -30 <= delta_days <= -14:
        return f"{abs(delta_days) // 7} weeks ago"
    if 14 <= delta_days <= 30:
        return f"in {delta_days // 7} weeks"
    if parsed.year == now.year:
        return f"in {_MONTHS[parsed.month - 1]}"
    return f"in {parsed.year}"


def init_kg() -> None:
    """Initialize the KG repo singleton. Safe to call multiple times."""
    global _repo
    if _repo is not None:
        return
    from assistant.storage.db import get_db
    from assistant.storage.repos.knowledge_graph import KnowledgeGraphRepo

    db = get_db()
    if db is None:
        logger.warning("[KG] init_kg called before storage.init_db; skipping")
        return

    def _load_embed_model():
        # Reuse the existing FAISS embed model. Wrapped in a closure so the
        # repo gets a zero-arg loader without storing the model itself.
        from assistant import memory as _mem
        _mem.warm_embed_model()
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("all-MiniLM-L6-v2")

    _repo = KnowledgeGraphRepo(db, embed_model_loader=_load_embed_model)


def _get_repo():
    if _repo is None:
        init_kg()
    return _repo


# ─── Pre-filter ────────────────────────────────────────────────────────────
_PERSONAL_SIGNALS = (
    "my ", "i am", "i'm", "i live", "i work", "i have", "i like", "i love",
    "i hate", "i study", "i go to", "call me", "name is", "age is",
    "i'm from", "i am from", "my name", "my age", "my phone", "my email",
    "my job", "my city", "my country",
)

_RESPONSE_CUES = (
    "you mentioned", "your ", "they ", "you said", "earlier you",
)

_CAPITALIZED_NOUN = re.compile(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b")
_EMOJI_ONLY = re.compile(r"^[\W_]+$", re.UNICODE)


def _has_entity_signal(text: str) -> bool:
    """Cheap pre-filter to decide whether to call the LLM at all.

    Returns True if the text plausibly contains an entity, fact, or
    relationship worth extracting. Designed to be permissive — the LLM
    itself is the final filter. The goal is only to skip obvious
    pleasantries ("ok thanks", "hmm", emoji-only replies) to save tokens.
    """
    if not text or not text.strip():
        return False
    stripped = text.strip()
    if len(stripped.split()) <= 2:
        return False
    if _EMOJI_ONLY.match(stripped):
        return False
    lowered = stripped.lower()
    if any(s in lowered for s in _PERSONAL_SIGNALS):
        return True
    if any(c in lowered for c in _RESPONSE_CUES):
        return True
    if _CAPITALIZED_NOUN.search(stripped):
        return True
    return False


# ─── Ingest facade ─────────────────────────────────────────────────────────
from assistant.llm.contracts import ask_for_entity_extraction, ask_for_kg_followup

# Only assistant replies from these intents are ingested into the KG.
# Others (web_search, code_executor, set_reminder, store_memory, planner, ...)
# produce ephemeral or action-confirmation output that pollutes the graph with
# weather data, error codes, storage labels, news fragments, etc. The user's
# own message (source="user_msg") is ALWAYS ingested regardless of intent.
_CONVERSATIONAL_REPLY_INTENTS = frozenset({
    "small_talk",
    "unknown",
    "memory_query",
})


def _ingest_enabled() -> bool:
    return os.environ.get("KG_INGEST_ENABLED", "true").lower() not in {"false", "0", "no"}


async def ingest_turn(
    text: str, source: Literal["user_msg", "tenka_resp"],
    reply_intent: str | None = None,
    source_turn_id: str | None = None,
) -> None:
    """Pre-filter → extract → resolve → store. Never raises.

    `reply_intent` is the intent label of the turn TENKA just resolved (e.g.
    "small_talk", "web_search", "set_reminder"). For source="tenka_resp" we
    skip ingest entirely unless the intent is in the conversational set —
    action-handler replies are ephemeral and pollute the KG. Ignored for
    source="user_msg" (the user's own words always count).

    `source_turn_id` (H) is an opaque provenance string written into
    the source_turn_id column on every kg_entity / kg_fact / kg_relationship
    inserted by this call. Callers pick the format (TENKA main loop uses
    f"{session_id}:{conv_row_id}"). NULL is fine — legacy rows already
    have NULL there.
    """
    if not _ingest_enabled():
        return
    if source == "tenka_resp" and reply_intent is not None \
            and reply_intent not in _CONVERSATIONAL_REPLY_INTENTS:
        logger.debug(f"[KG] skipping tenka_resp ingest for intent={reply_intent!r}")
        return
    if not _has_entity_signal(text):
        return
    # knowledge-graph Session 3 Path A + Session 2 Issue 1: fetch the active tracker
    # once and use it for BOTH deterministic pre-resolution (Path A) AND
    # the context-hint string (Issue 1). Belt-and-braces — Path A handles
    # the LLM-doesn't-use-hint failure mode the right way (zero-LLM,
    # deterministic substitution), and the hint stays as a fallback for
    # cases pre-resolution can't reach (e.g. possessive without pronoun).
    _tracker = None
    if source == "user_msg":
        try:
            from . import topic_tracker as _tt
            _tracker = _tt.get_active()
        except Exception as e:
            logger.debug(f"[KG] topic-tracker fetch failed (non-critical): {e}")

    resolved_text = text
    if _tracker is not None:
        try:
            resolved_text = _tracker.resolve_query(text)
        except Exception as e:
            logger.debug(f"[KG] topic-resolve failed (non-critical): {e}")

    context_hint: str | None = None
    if _tracker is not None:
        try:
            context_hint = _tracker.get_topic_hint()
        except Exception as e:
            logger.debug(f"[KG] topic-hint fetch failed (non-critical): {e}")

    try:
        payload = await ask_for_entity_extraction(
            resolved_text, source, context_hint=context_hint,
        )
    except Exception as e:
        logger.debug(f"[KG] extraction failed (non-critical): {e}")
        return
    # original_text=text (raw, pre-resolution) on purpose: the date-grounding
    # check in _strip_unfounded_event_at must verify against what the user
    # literally typed, not the pronoun-substituted form.
    try:
        _persist_extraction(
            payload, source, original_text=text,
            source_turn_id=source_turn_id,
        )
    except Exception as e:
        logger.warning(f"[KG] persist failed (non-critical): {e}")


# ─── Persistence ───────────────────────────────────────────────────────────
_VALID_ENTITY_TYPES = {"person", "project", "tool", "place", "concept", "event"}
_VALID_REL_TYPES = {"manages", "uses", "part_of", "related_to", "parent_of", "knows"}

# knowledge-graph E: the owner of a user-side promise. "user" is the canonical
# self-reference seen in family-fact lifting ("user has_uncle Damien").
_USER_OWNER_NAME = "user"

# Defense-in-depth against LLM leaking pronouns as entities/relationship
# endpoints (live-test bug: "she uses Figma" → junk concept entity "she").
# Compared against the canonicalized name (lowercased, whitespace-stripped).
_PRONOUNS = frozenset({
    "i", "me", "my", "mine", "myself",
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself",
    "she", "her", "hers", "herself",
    "it", "its", "itself",
    "we", "us", "our", "ours", "ourselves",
    "they", "them", "their", "theirs", "themselves",
    "this", "that", "these", "those",
})


def _is_pronoun(name: str) -> bool:
    return _canonicalize_for_filter(name) in _PRONOUNS


def _canonicalize_for_filter(name: str) -> str:
    return " ".join(str(name).lower().strip().split())

# Defense-in-depth against template-echo noise: facts the assistant asserts
# about itself (replying with our own templated key:value confirmations) get
# down-weighted so user_msg evidence wins on dedup/cleanup. Generic --
# user_msg is the source-of-truth modality; tenka_resp is reinforcement.
_TENKA_RESP_CONFIDENCE_FACTOR = 0.7


def _adjust_confidence(raw, source: str, default: float = 1.0) -> float:
    try:
        conf = float(raw if raw is not None else default)
    except (TypeError, ValueError):
        conf = default
    if source == "tenka_resp":
        conf *= _TENKA_RESP_CONFIDENCE_FACTOR
    return conf


def _persist_extraction(
    payload: dict, source: str, original_text: str = "",
    source_turn_id: str | None = None,
) -> None:
    repo = _get_repo()
    if repo is None:
        return

    # Resolve entity names → ids (upsert pass)
    name_to_id: dict[str, int] = {}
    for ent in payload.get("entities", []):
        if not isinstance(ent, dict):
            continue
        etype = str(ent.get("type", "")).lower().strip()
        ename = str(ent.get("name", "")).strip()
        if etype not in _VALID_ENTITY_TYPES or not ename:
            continue
        if _is_pronoun(ename):
            logger.debug(f"[KG] dropping pronoun entity {ename!r}")
            continue
        conf = _adjust_confidence(ent.get("confidence"), source)
        eid, _ = repo.upsert_entity(
            entity_type=etype, name=ename, source=source, confidence=conf,
            source_turn_id=source_turn_id,
        )
        name_to_id[ename.lower()] = eid

    # Facts — need a subject; auto-upsert subject as "concept" if missing
    for fact in payload.get("facts", []):
        if not isinstance(fact, dict):
            continue
        subj = str(fact.get("subject", "")).strip()
        pred = str(fact.get("predicate", "")).strip().lower().replace(" ", "_")
        obj = str(fact.get("object", "")).strip()
        if not (subj and pred and obj):
            continue
        if _is_pronoun(subj):
            logger.debug(f"[KG] dropping fact with pronoun subject {subj!r}")
            continue
        # knowledge-graph Session 4 livetest defense: drop nonsense self-referent
        # facts where the LLM emitted (X, predicate, X). Flash-Lite
        # occasionally mis-labels the object as the subject (e.g.
        # "Aanya works at Razorpay" → (Razorpay, works_at, Razorpay)).
        # The signal is corrupt either way — better to drop than store.
        if _canonicalize_for_filter(subj) == _canonicalize_for_filter(obj):
            logger.debug(
                f"[KG] dropping self-referent fact ({subj!r}, {pred!r}, {obj!r})"
            )
            continue
        sid = name_to_id.get(subj.lower())
        if sid is None:
            sid, _ = repo.upsert_entity(
                entity_type="concept", name=subj, source=source,
                confidence=_adjust_confidence(0.5, source, default=0.5),
                source_turn_id=source_turn_id,
            )
            name_to_id[subj.lower()] = sid
        conf = _adjust_confidence(fact.get("confidence"), source)
        event_at = str(fact.get("event_at") or "").strip() or None
        event_at = _strip_unfounded_event_at(original_text, event_at)
        repo.add_fact(
            sid, pred, obj, source=source, confidence=conf,
            event_at=event_at, source_turn_id=source_turn_id,
        )

    # Relationships
    for rel in payload.get("relationships", []):
        if not isinstance(rel, dict):
            continue
        rtype = str(rel.get("type", "")).lower().strip()
        if rtype not in _VALID_REL_TYPES:
            continue
        a = str(rel.get("from", "")).strip()
        b = str(rel.get("to", "")).strip()
        if not (a and b):
            continue
        if _is_pronoun(a) or _is_pronoun(b):
            logger.debug(f"[KG] dropping relationship with pronoun endpoint: {a!r} -> {b!r}")
            continue
        fid = name_to_id.get(a.lower()) or repo.upsert_entity(
            entity_type="concept", name=a, source=source,
            confidence=_adjust_confidence(0.5, source, default=0.5),
            source_turn_id=source_turn_id,
        )[0]
        tid = name_to_id.get(b.lower()) or repo.upsert_entity(
            entity_type="concept", name=b, source=source,
            confidence=_adjust_confidence(0.5, source, default=0.5),
            source_turn_id=source_turn_id,
        )[0]
        conf = _adjust_confidence(rel.get("confidence"), source)
        repo.add_relationship(
            fid, tid, rtype, source=source, confidence=conf,
            source_turn_id=source_turn_id,
        )

    # Commitments — knowledge-graph E. Defensive about extractor output: skip rows
    # without promise text, and resolve owner to an entity id (auto-upsert
    # as 'person' so the FK stays valid). 'user' upserts as a single
    # canonical self-entity that other knowledge-graph work already touches.
    for com in payload.get("commitments", []):
        if not isinstance(com, dict):
            continue
        promise = str(com.get("promise", "")).strip()
        if not promise:
            continue
        owner_name = str(com.get("owner", "") or _USER_OWNER_NAME).strip()
        if not owner_name:
            owner_name = _USER_OWNER_NAME
        if _is_pronoun(owner_name):
            owner_name = _USER_OWNER_NAME
        owner_id = name_to_id.get(owner_name.lower())
        if owner_id is None:
            owner_id, _ = repo.upsert_entity(
                entity_type="person", name=owner_name, source=source,
                confidence=_adjust_confidence(0.5, source, default=0.5),
                source_turn_id=source_turn_id,
            )
            name_to_id[owner_name.lower()] = owner_id
        when_due_raw = str(com.get("when_due") or "").strip() or None
        # Reuse the same date-grounding defense as facts — strips
        # LLM-hallucinated dates not present in the user's text.
        when_due = _strip_unfounded_event_at(original_text, when_due_raw)
        repo.add_commitment(
            owner_id, promise, source=source,
            when_due=when_due, source_turn_id=source_turn_id,
        )


# ─── Query-time context builder ────────────────────────────────────────────
def _query_injection_enabled() -> bool:
    return os.environ.get("KG_QUERY_INJECTION_ENABLED", "true").lower() not in {"false", "0", "no"}


_PRONOUN_HINTS = re.compile(r"\b(my|your|they|them|their)\s+(\w+)", re.I)


def build_kg_context(
    user_message: str, max_blocks: int = 5, char_budget: int = 600,
) -> str | None:
    """Resolve entity mentions in user_message → expand 1-hop →
    format compact block. Returns None if nothing resolves OR if
    query injection is disabled by env var."""
    if not _query_injection_enabled():
        return None
    if not user_message or not user_message.strip():
        return None

    repo = _get_repo()
    if repo is None:
        return None

    mentions = _extract_mentions(user_message)
    if not mentions:
        return None

    resolved_ids: list[int] = []
    seen: set[int] = set()
    for m in mentions:
        rows = repo.find_entities_by_name(m)
        for r in rows:
            if r["id"] not in seen:
                seen.add(r["id"])
                resolved_ids.append(r["id"])
        if len(resolved_ids) >= max_blocks:
            break

    if not resolved_ids:
        return None

    lines: list[str] = ["[KNOWLEDGE]"]
    for eid in resolved_ids[:max_blocks]:
        ctx = repo.expand_entity_context(eid, fact_limit=3, neighbor_limit=3)
        line = _format_entity_block(ctx)
        if line:
            lines.append(line)
    block = "\n".join(lines)
    if len(block) > char_budget:
        block = block[: char_budget - 3].rstrip() + "..."
    return block


def _extract_mentions(text: str) -> list[str]:
    """Capitalized nouns + pronoun-target words. Order preserved, deduped."""
    seen: set[str] = set()
    out: list[str] = []
    for m in _CAPITALIZED_NOUN.finditer(text):
        s = m.group(0).strip()
        if s.lower() not in seen:
            seen.add(s.lower())
            out.append(s)
    for m in _PRONOUN_HINTS.finditer(text):
        s = m.group(2).strip()
        if len(s) >= 3 and s.lower() not in seen:
            seen.add(s.lower())
            out.append(s)
    return out


def _format_entity_block(ctx: dict, now: datetime | None = None) -> str:
    ent = ctx.get("entity")
    if ent is None:
        return ""
    if now is None:
        now = datetime.now()
    facts = ctx.get("facts", [])
    neighbors = ctx.get("neighbors", [])
    commitments = ctx.get("commitments", [])
    parts: list[str] = []
    for f in facts:
        line = f"{f['predicate'].replace('_', ' ')} {f['object']}"
        when = _relative_date(f.get("event_at"), now)
        if when:
            line = f"{line} ({when})"
        parts.append(line)
    for n in neighbors:
        parts.append(
            f"{n['rel_type'].replace('_', ' ')} {n['entity']['display_name']}"
        )
    for c in commitments:
        line = f"open promise: {c['promise_text']}"
        when = _relative_date(c.get("when_due"), now)
        if when:
            line = f"{line} (due {when})"
        elif c.get("when_due"):
            line = f"{line} (due {c['when_due']})"
        parts.append(line)
    if not parts:
        return f"{ent['display_name']} ({ent['type']})."
    return f"{ent['display_name']} ({ent['type']}): {'; '.join(parts)}."


# ─── knowledge-graph D — Multi-hop COT expansion ──────────────────────────────────────
async def expand_multi_hop(
    question: str, seed_ids: list[int], *,
    max_iter: int = 3, fact_limit: int = 3, neighbor_limit: int = 3,
    char_budget: int = 600,
) -> dict:
    """D — Cognee-pattern multi-hop COT expansion.

    Starts from `seed_ids`, expands 1-hop, asks the LLM whether the gathered
    context is sufficient for `question`. If not, follows the LLM-named entity
    one more hop. Loops up to `max_iter` times (default 3) so LLM cost is
    bounded. The "deep" companion to the sync `build_kg_context` 1-hop path —
    callers reach for this only when the cheap path returns too little.

    Returns:
        {
          "context_block":   "[KNOWLEDGE]\\n..." or "",
          "visited_ids":     sorted list[int] of every entity touched,
          "iterations":      int (1..max_iter),
          "stopped_reason":  one of {no_seeds, no_repo, sufficient,
                             max_iter, no_follow_up, unresolvable},
        }

    Never raises. LLM verdict failure short-circuits as 'sufficient'.
    """
    EMPTY = {"context_block": "", "visited_ids": [], "iterations": 0,
             "stopped_reason": "no_seeds"}
    if not seed_ids:
        return EMPTY

    repo = _get_repo()
    if repo is None:
        return {**EMPTY, "stopped_reason": "no_repo"}

    visited: set[int] = set()
    blocks: list[str] = []
    frontier: list[int] = list(seed_ids)
    iterations = 0
    stopped = "max_iter"

    for i in range(max_iter):
        iterations = i + 1
        new_ids = [eid for eid in frontier if eid not in visited]
        if not new_ids:
            stopped = "no_follow_up"
            break
        for eid in new_ids:
            ctx = repo.expand_entity_context(
                eid, fact_limit=fact_limit, neighbor_limit=neighbor_limit,
            )
            block = _format_entity_block(ctx)
            if block:
                blocks.append(block)
            visited.add(eid)

        context_str = "[KNOWLEDGE]\n" + "\n".join(blocks) if blocks else ""
        verdict = await ask_for_kg_followup(question, context_str)
        if verdict.get("sufficient", True):
            stopped = "sufficient"
            break

        follow_up_name = verdict.get("follow_up")
        if not follow_up_name:
            stopped = "no_follow_up"
            break
        rows = repo.find_entities_by_name(follow_up_name)
        next_frontier = [r["id"] for r in rows if r["id"] not in visited]
        if not next_frontier:
            stopped = "unresolvable"
            break
        frontier = next_frontier

    block_out = ""
    if blocks:
        block_out = "[KNOWLEDGE]\n" + "\n".join(blocks)
        if len(block_out) > char_budget:
            block_out = block_out[: char_budget - 3].rstrip() + "..."

    return {
        "context_block": block_out,
        "visited_ids": sorted(visited),
        "iterations": iterations,
        "stopped_reason": stopped,
    }


# ─── Public helpers for memory_query fallback ──────────────────────────────
def search_entities(query: str) -> list[dict]:
    repo = _get_repo()
    if repo is None:
        return []
    return repo.find_entities_by_name(query)


def get_entity_with_context(entity_id: int) -> dict:
    repo = _get_repo()
    if repo is None:
        return {}
    return repo.expand_entity_context(entity_id)


# ─── knowledge-graph E — Commitment facade ────────────────────────────────────────────
def list_open_commitments_for_user(limit: int = 10) -> list[dict]:
    """Return open commitments where the owner is the canonical 'user'
    self-entity. Empty list when no such entity exists yet."""
    repo = _get_repo()
    if repo is None:
        return []
    rows = repo.find_entities_by_name(_USER_OWNER_NAME, entity_type="person")
    if not rows:
        return []
    return repo.list_open_commitments(owner_id=rows[0]["id"], limit=limit)


def list_open_commitments_for_entity(
    entity_id: int, limit: int = 10,
) -> list[dict]:
    repo = _get_repo()
    if repo is None:
        return []
    return repo.list_open_commitments(owner_id=entity_id, limit=limit)


# ─── knowledge-graph H — Provenance query helper ──────────────────────────────────────
def why_do_you_think_that(
    *, fact_id: int | None = None, entity_id: int | None = None,
    commitment_id: int | None = None,
) -> dict | None:
    """Zep-style read path — given a KG row, return the originating
    conversation row (the turn that produced it).

    Pass exactly one of fact_id / entity_id / commitment_id. Returns a
    dict with the row's source_turn_id and the matched conversations
    row (or None when the KG row lacks provenance / the conv row was
    purged). Never raises.
    """
    repo = _get_repo()
    if repo is None:
        return None
    db = repo._db
    if fact_id is not None:
        row = db.fetchone(
            "SELECT source_turn_id FROM kg_facts WHERE id = ?", (fact_id,),
        )
    elif entity_id is not None:
        row = db.fetchone(
            "SELECT source_turn_id FROM kg_entities WHERE id = ?", (entity_id,),
        )
    elif commitment_id is not None:
        row = db.fetchone(
            "SELECT source_turn_id FROM kg_commitments WHERE id = ?",
            (commitment_id,),
        )
    else:
        return None
    if row is None or row["source_turn_id"] is None:
        return None
    tid = str(row["source_turn_id"])
    # source_turn_id format: "{session_id}:{conv_row_id}". Be permissive
    # about other shapes — caller may have set a free-form id.
    conv_row = None
    if ":" in tid:
        try:
            conv_id = int(tid.rsplit(":", 1)[1])
        except ValueError:
            conv_id = None
        if conv_id is not None:
            conv_row = db.fetchone(
                "SELECT * FROM conversations WHERE id = ?", (conv_id,),
            )
    return {
        "source_turn_id": tid,
        "conversation": dict(conv_row) if conv_row else None,
    }
