"""
llm/ — LLM dispatch package for TENKA.

Re-exports the public API from router.py so that existing
``from . import llm`` / ``llm.get_llm_response(...)`` callers
keep working unchanged.
"""

# Bind submodules as attributes so external callers (and pytest's
# monkeypatch dotted-string lookups) can resolve `assistant.llm.contracts`
# without first triggering an explicit `import assistant.llm.contracts`.
# The `from .contracts import name` lines below do NOT reliably set this
# binding — knowledge-graph livetest carry-overs surfaced the gap in Session 4.
from . import contracts as contracts  # noqa: F401  (re-export)
from . import router as router  # noqa: F401
from . import providers as providers  # noqa: F401
from . import prompts as prompts  # noqa: F401

from .router import (
    # Task routing
    TASK_MODEL_MAP,
    PROVIDERS,
    # Result types
    LLMResult,
    StreamingLLMResult,
    # Text API
    get_llm_response,
    get_llm_response_stream,
    chat,
    # Vision API
    get_vision_response,
    _vision_yes_no_sync,
    # Utilities
    classify_emotion,
    extract_facts,
    locate_element_bbox,
    parse_emotion_tag,
)

from .providers import provider_registry
from .providers.base import Provider, ProviderResult
from .providers.groq import GROQ_KEYS

from .contracts import (
    ask_for_intent,
    ask_for_synthesis,
    ask_for_plan,
    ask_for_code_gen,
    ask_for_small_talk,
    ask_for_personality_reflection,
    ask_for_agent_verify,
    ask_for_default,
    stream_for_synthesis,
    stream_for_small_talk,
    ask_for_memory_type,
    ask_for_entity_extraction,
)

from .prompts import (
    build_personality_prompt,
    build_intent_prompt,
    get_system_prompt,
)
