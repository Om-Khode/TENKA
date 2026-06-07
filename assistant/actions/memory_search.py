"""Memory query and fact storage handlers."""

import json
import logging
import re

from .registry import tool_registry

logger = logging.getLogger("actions")

# Simple "X is Y" pattern — covers most "remember that" statements
_IS_PATTERN = re.compile(
    r"^(?:my\s+)?(.+?)\s+(?:is|are|was|were)\s+(.+)$", re.I
)

# knowledge-graph E — flag commitment-shaped recall queries so memory_query pulls
# open promises into the synthesis context. Word-bounded to avoid false
# positives like "promising results" or "committee".
_COMMITMENT_QUERY_RE = re.compile(
    r"\b(promise[sd]?|commit(?:ted|ment|ments)?|owe[ds]?|"
    r"agreed\s+to|said\s+(?:i|i'?ll)\s+would|"
    r"supposed\s+to|going\s+to\s+(?:do|send|finish))\b",
    re.IGNORECASE,
)


def _is_commitment_query(query: str) -> bool:
    return bool(_COMMITMENT_QUERY_RE.search(query or ""))


@tool_registry.decorator("store_memory")
async def handle_store_memory(params: dict, llm_response: str, bridge=None) -> str:
    """Store a user-stated fact into memory."""
    from .. import memory
    from ..llm.contracts import ask_for_intent, ask_for_memory_type

    content = params.get("content", "").strip()
    if not content:
        return "What should I remember? You didn't say anything."

    # Fast path: try regex extraction for "X is Y" patterns
    m = _IS_PATTERN.match(content)
    if m:
        key = re.sub(r"\s+", "_", m.group(1).strip().lower())
        value = m.group(2).strip()
    else:
        # Cheap LLM call to extract key/value
        extract_prompt = (
            f'The user said: "remember that {content}"\n'
            f"Extract a short key and value. Return ONLY this JSON:\n"
            f'{{"key": "short_snake_case_key", "value": "the fact to remember"}}\n'
            f"Examples:\n"
            f'  "I like biryani" → {{"key": "favorite_food", "value": "biryani"}}\n'
            f'  "I\'m allergic to peanuts" → {{"key": "allergy", "value": "peanuts"}}\n'
            f"Return ONLY JSON."
        )
        raw = await ask_for_intent(
            extract_prompt,
            max_tokens=60,
            system_prompt="You are a JSON extraction utility. Return only the requested JSON.",
        )
        try:
            data = json.loads(raw.strip().strip("```json").strip("```").strip())
            key = data.get("key", "").strip()
            value = data.get("value", content).strip()
        except (json.JSONDecodeError, AttributeError):
            key = re.sub(r"\s+", "_", content[:40].lower())
            value = content

    if not key:
        key = re.sub(r"\s+", "_", content[:40].lower())

    memory_type = await ask_for_memory_type(key, value)
    memory.save_typed_fact(key=key, value=value, source="user", memory_type=memory_type)
    logger.info(f"Stored fact: {key} = {value} (type={memory_type})")
    return f"Got it, I'll remember that. {key.replace('_', ' ').title()}: {value}."


@tool_registry.decorator("forget_memory")
async def handle_forget_memory(params: dict, llm_response: str, bridge=None) -> str:
    """Delete a stored fact from memory."""
    from .. import memory

    content = params.get("content", "").strip()
    if not content:
        return "What should I forget? You didn't say anything."

    key_pattern = re.sub(r"\s+", "_", content.lower())
    count = memory.delete_fact(key_pattern)

    if count > 0:
        return "Forgotten."

    count_raw = memory.delete_fact(content)
    if count_raw > 0:
        return "Forgotten."

    return "I don't have anything about that."


@tool_registry.decorator("memory_query")
async def handle_memory_query(params: dict, llm_response: str, bridge=None) -> str:
    """Search past conversations and facts to answer a recall question."""
    from .. import memory
    from ..llm.contracts import ask_for_synthesis

    query = params.get("query", "")
    if not query:
        return "What am I supposed to remember? Give me a hint, dummy."

    user_question = llm_response if llm_response and llm_response.strip() else query

    conv_results = memory.hybrid_search_conversations(query, limit=5)
    fact_results = memory.hybrid_search_facts(query, limit=10)
    recording_results = memory.search_recording_sessions(query, limit=3)

    logger.info(
        f"[MEMORY_QUERY] query={query!r}, facts={len(fact_results)}, "
        f"convos={len(conv_results)}, recordings={len(recording_results)}"
    )

    # knowledge-graph fallback: when hybrid-retrieval (facts + convos + recordings) has nothing,
    # try the knowledge graph.
    kg_results: list[dict] = []
    if not fact_results and not conv_results and not recording_results:
        try:
            from .. import knowledge_graph
            kg_results = knowledge_graph.search_entities(query)
        except Exception as e:
            logger.debug(f"[MEMORY_QUERY] KG fallback failed: {e}")

    # knowledge-graph E — commitment lookup: when the query talks about promises,
    # surface open commitments from the KG before synthesis runs. The
    # generic memory_query path then composes them with whatever facts /
    # convos turned up. Owner resolution: if the query names an entity in
    # the KG, scope to that person; otherwise scope to 'user'.
    commitment_results: list[dict] = []
    if _is_commitment_query(query):
        try:
            from .. import knowledge_graph
            ent_matches = knowledge_graph.search_entities(query) if not kg_results else kg_results
            owner_id = None
            for r in ent_matches:
                if r.get("type") == "person":
                    owner_id = r["id"]
                    break
            if owner_id is not None:
                commitment_results = knowledge_graph.list_open_commitments_for_entity(owner_id)
            else:
                commitment_results = knowledge_graph.list_open_commitments_for_user()
        except Exception as e:
            logger.debug(f"[MEMORY_QUERY] commitment lookup failed: {e}")

    # knowledge-graph D — deep multi-hop fallback. When the cheap 1-hop returned KG
    # entity matches but no facts / conversations turned up to ground a
    # synthesis, escalate to expand_multi_hop for COT-style traversal.
    # Caps at 3 hops so cost stays bounded.
    multi_hop_block: str | None = None
    if kg_results and not fact_results and not conv_results and not recording_results:
        try:
            from .. import knowledge_graph
            seed_ids = [r["id"] for r in kg_results[:3]]
            mh = await knowledge_graph.expand_multi_hop(user_question, seed_ids)
            if mh.get("context_block"):
                multi_hop_block = mh["context_block"]
                logger.info(
                    f"[MEMORY_QUERY] multi-hop expansion: "
                    f"iters={mh.get('iterations')}, "
                    f"stopped={mh.get('stopped_reason')}"
                )
        except Exception as e:
            logger.debug(f"[MEMORY_QUERY] multi-hop expansion failed: {e}")

    if (
        not conv_results
        and not fact_results
        and not recording_results
        and not kg_results
        and not commitment_results
    ):
        return "I don't have any memory of that. We may not have discussed it before."

    context_parts = []

    if fact_results:
        context_parts.append("KNOWN FACTS:")
        for r in fact_results:
            context_parts.append(f"  {r['key']}: {r['value']}")

    if conv_results:
        context_parts.append("PAST CONVERSATIONS:")
        for r in conv_results:
            context_parts.append(
                f"  [{r['timestamp']}] User: {r['user_input']} → {r['response']}"
            )

    if recording_results:
        context_parts.append("RECORDING SESSIONS:")
        for r in recording_results:
            context_parts.append(
                f"  [{r['timestamp']}] (session: {r['session_id']}, chunk {r['chunk_index']}): {r['transcript']}"
            )

    if kg_results:
        from .. import knowledge_graph
        context_parts.append("Knowledge graph:")
        for ent in kg_results[:3]:
            ctx = knowledge_graph.get_entity_with_context(ent["id"])
            context_parts.append(_format_entity_for_synth(ctx))

    if multi_hop_block:
        context_parts.append("Knowledge graph (deep):")
        context_parts.append(multi_hop_block)

    if commitment_results:
        context_parts.append("OPEN PROMISES:")
        for c in commitment_results:
            line = f"  - {c['promise_text']}"
            if c.get("when_due"):
                line = f"{line} (due {c['when_due']})"
            context_parts.append(line)

    context = "\n".join(context_parts)

    prompt = (
        f"The user asked: \"{user_question}\"\n\n"
        f"Here is what I found in my memory:\n{context}\n\n"
        f"Answer the user's question using ONLY the memory above. "
        f"Be concise and conversational. "
        f"Match your current personality tone, not the tone of past assistant responses."
    )

    response = await ask_for_synthesis(prompt)
    if response == "__LLM_UNAVAILABLE__":
        return f"Here's what I found: {context[:500]}"

    return response


def _format_entity_for_synth(ctx: dict) -> str:
    """Format a single KG entity-with-context block for the synthesis prompt."""
    ent = ctx.get("entity") or {}
    facts = ctx.get("facts", [])
    neighbors = ctx.get("neighbors", [])
    lines = [f"- {ent.get('display_name', '')} ({ent.get('type', '')}):"]
    for f in facts:
        lines.append(f"    {f['predicate'].replace('_', ' ')} {f['object']}")
    for n in neighbors:
        lines.append(
            f"    {n['rel_type'].replace('_', ' ')} {n['entity']['display_name']}"
        )
    return "\n".join(lines)
