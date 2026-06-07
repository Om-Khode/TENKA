"""
topic_tracker.py — Context-aware topic tracking .

Extracts named entities and noun phrases from each utterance using spaCy,
maintains a 3-item recency stack, and resolves pronoun references to the
most recent topic. Zero LLM calls.
"""

import logging
import re

logger = logging.getLogger("topic_tracker")

_PRONOUNS = re.compile(
    r'\b(it|that|this|they|them|there|here|he|she|his|her|its|their)\b',
    re.IGNORECASE,
)

# knowledge-graph livetest bug #1: 'this/that' need extra care because they double as
# demonstrative determiners ('this week', 'that bet'). Substituting them
# wholesale corrupted real turns: "what did I commit to this week" became
# "what did I commit to Friday week". Match only at this set; resolve_query
# uses a peek-ahead to skip determiner usages.
_DEMONSTRATIVES = frozenset({"this", "that"})
_FOLLOWED_BY_WORD_RE = re.compile(r"^\s+[A-Za-z]")

_LOCATION_PRONOUNS = re.compile(r'\b(there|here)\b', re.IGNORECASE)

_ENTITY_LABELS = {"PERSON", "ORG", "GPE", "EVENT", "WORK_OF_ART", "LOC",
                  "NORP", "FAC", "PRODUCT", "LAW"}

_MAX_STACK = 3
_DECAY_TURNS = 5


# ─── Module-level singleton accessor (Session 2 Issue 1) ─────────────
# main.py registers the live tracker here at boot so other domain modules
# (e.g. knowledge_graph) can read topic state without circular imports or
# duplicating the tracker instance.
_active_tracker = None


def set_active(tracker) -> None:
    global _active_tracker
    _active_tracker = tracker


def get_active():
    return _active_tracker


class TopicTracker:

    def __init__(self):
        self._stack: list[tuple[str, str, int]] = []
        self._nlp = None

    # ─── Internal ────────────────────────────────────────────────────────────

    def _ensure_nlp(self):
        if self._nlp is None:
            import spacy
            self._nlp = spacy.load("en_core_web_sm")
        return self._nlp

    # ─── Public API ──────────────────────────────────────────────────────────

    def push_turn(self, text: str, turn_number: int) -> None:
        self._stack = [
            (e, l, t) for e, l, t in self._stack
            if turn_number - t <= _DECAY_TURNS
        ]

        nlp = self._ensure_nlp()
        doc = nlp(text)

        # knowledge-graph Session 4 livetest fix: separate named entities from generic
        # noun chunks so we can rank them. Insertions happen at position 0,
        # so the LAST inserted ends up at the TOP — push noun_chunks FIRST,
        # then named entities, so proper nouns win on resolve_query.
        # Without this, turn 1 "My best friend is Aanya" left "My best
        # friend" at the top of the stack; turn 2 "She works at..." then
        # resolved "she" → "my best friend" and the LLM emitted nonsense
        # facts like (backend engineer, works_at, Razorpay).
        named_ents: list[tuple[str, str]] = []
        for ent in doc.ents:
            if ent.label_ in _ENTITY_LABELS:
                named_ents.append((ent.text, ent.label_))

        # knowledge-graph livetest bug #2: split noun_chunks by syntactic role.
        # Copula predicates ("Y" in "X is Y") are the definition pattern —
        # they introduce the referent the next pronoun likely points to —
        # and should outrank the generic subject chunk ("My best friend").
        # Without this, lowercase names ("My best friend is sarvesh") never
        # get prioritized because spaCy doesn't NER-tag lowercase tokens.
        attr_chunks: list[tuple[str, str]] = []
        other_chunks: list[tuple[str, str]] = []
        for chunk in doc.noun_chunks:
            chunk_text = chunk.text.strip()
            if (chunk_text.lower() not in {"i", "you", "we", "me", "it", "they",
                                           "he", "she", "them", "this", "that",
                                           "what", "who", "which", "where",
                                           "when", "how", "there", "here"}
                    and len(chunk_text) > 1
                    and not (len(chunk) == 1 and chunk.root.dep_ == "ROOT")
                    and not any(
                        chunk_text.lower() in e[0].lower()
                        or e[0].lower() in chunk_text.lower()
                        for e in named_ents
                    )):
                if chunk.root.dep_ == "attr":
                    attr_chunks.append((chunk_text, "NOUN_CHUNK_ATTR"))
                else:
                    other_chunks.append((chunk_text, "NOUN_CHUNK"))

        # Token-level rescue for lowercase copula predicates that spaCy
        # doesn't put in noun_chunks. "My best friend is sarvesh" → spaCy
        # tags 'sarvesh' as ADJ/acomp because lowercase looks like an
        # adjective. Pick it up by token-walking instead: any AUX-be
        # complement that is NOT already covered by a chunk or named ent
        # is treated as a definition-pattern referent. Ranks alongside
        # attr_chunks.
        chunk_token_lc: set[str] = set()
        for chunk in doc.noun_chunks:
            for tok in chunk:
                chunk_token_lc.add(tok.text.lower())
        for tok in doc:
            if (tok.dep_ in {"attr", "acomp"}
                    and tok.head.pos_ == "AUX"
                    and tok.head.lemma_ == "be"
                    and len(tok.text) > 1
                    and tok.text.lower() not in chunk_token_lc
                    and not any(
                        tok.text.lower() in e[0].lower()
                        or e[0].lower() in tok.text.lower()
                        for e in named_ents
                    )):
                attr_chunks.append((tok.text, "COPULA_PRED"))

        # Insertions all go at index 0, so the LAST inserted ends up at the
        # TOP. Order: other → attr → named. Top priority is named > attr >
        # other.
        for entity_text, label in other_chunks + attr_chunks + named_ents:
            normalized = entity_text.lower()
            self._stack = [
                (e, l, t) for e, l, t in self._stack
                if e.lower() != normalized
            ]
            self._stack.insert(0, (entity_text, label, turn_number))

        self._stack = self._stack[:_MAX_STACK]

    def resolve_query(self, text: str) -> str:
        if not self._stack:
            return text

        # knowledge-graph C-Q2: bail only when there's nothing to resolve. The previous
        # version also bailed when the sentence contained any non-pronoun
        # entity ("Figma" suppressed resolving "she"), which broke realistic
        # follow-ups like "she likes <unrelated tool>".
        if not _PRONOUNS.search(text):
            return text

        top_entity = self._stack[0][0]
        # knowledge-graph livetest bug #1: walk matches and skip 'this/that' when
        # they're acting as determiners ('this week', 'that bet'). Only
        # the first eligible match is substituted, preserving the original
        # count=1 contract.
        for m in _PRONOUNS.finditer(text):
            word = m.group(0).lower()
            if word in _DEMONSTRATIVES:
                tail = text[m.end():m.end() + 30]
                if _FOLLOWED_BY_WORD_RE.match(tail):
                    continue
            resolved = text[:m.start()] + top_entity + text[m.end():]
            logger.debug(
                f"[TOPIC] Resolved '{text}' → '{resolved}' (topic: {top_entity})"
            )
            return resolved
        return text

    def get_topic_hint(self) -> str | None:
        if not self._stack:
            return None
        return f"Active topic: {self._stack[0][0]}"
