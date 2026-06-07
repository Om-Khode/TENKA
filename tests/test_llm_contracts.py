"""
test_llm_contracts.py — Unit tests for llm/contracts.py.

Verifies each contract dispatches with the correct task_type and defaults.
Does NOT call real LLMs — patches get_llm_response.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

from assistant.llm.contracts import (
    ask_for_intent,
    ask_for_synthesis,
    ask_for_plan,
    ask_for_code_gen,
    ask_for_small_talk,
    ask_for_personality_reflection,
    ask_for_agent_verify,
    ask_for_default,
)

# ─── Stubs ──────────────────────────────────────────────────────────────────

_captured_calls: list[dict] = []


async def _fake_get_llm_response(
    prompt, system_prompt=None, json_mode=False,
    max_tokens=256, task_type="default", temperature=None,
):
    _captured_calls.append({
        "prompt": prompt,
        "system_prompt": system_prompt,
        "json_mode": json_mode,
        "max_tokens": max_tokens,
        "task_type": task_type,
        "temperature": temperature,
    })
    return SimpleNamespace(text=f"__FAKE_{task_type}__")


_patch = patch("assistant.llm.contracts.get_llm_response", _fake_get_llm_response)


def _run(coro):
    return asyncio.run(coro)


def setup_function():
    _captured_calls.clear()
    _patch.start()


def teardown_function():
    _patch.stop()


# ─── Task-type dispatch ────────────────────────────────────────────────────

def test_ask_for_intent_dispatches_correct_task_type():
    result = _run(ask_for_intent("classify this"))
    assert result == "__FAKE_intent__"
    assert _captured_calls[-1]["task_type"] == "intent"


def test_ask_for_synthesis_dispatches_correct_task_type():
    result = _run(ask_for_synthesis("summarize this"))
    assert result == "__FAKE_synthesis__"
    assert _captured_calls[-1]["task_type"] == "synthesis"


def test_ask_for_plan_dispatches_correct_task_type():
    result = _run(ask_for_plan("plan steps"))
    assert result == "__FAKE_agent_plan__"
    assert _captured_calls[-1]["task_type"] == "agent_plan"


def test_ask_for_code_gen_dispatches_correct_task_type():
    result = _run(ask_for_code_gen("write code"))
    assert result == "__FAKE_code_gen__"
    assert _captured_calls[-1]["task_type"] == "code_gen"


def test_ask_for_small_talk_dispatches_correct_task_type():
    result = _run(ask_for_small_talk("hello"))
    assert result == "__FAKE_small_talk__"
    assert _captured_calls[-1]["task_type"] == "small_talk"


def test_ask_for_personality_reflection_dispatches_correct_task_type():
    result = _run(ask_for_personality_reflection("reflect"))
    assert result == "__FAKE_personality_reflection__"
    assert _captured_calls[-1]["task_type"] == "personality_reflection"


def test_ask_for_agent_verify_dispatches_correct_task_type():
    result = _run(ask_for_agent_verify("verify"))
    assert result == "__FAKE_agent_verify__"
    assert _captured_calls[-1]["task_type"] == "agent_verify"


def test_ask_for_default_dispatches_correct_task_type():
    result = _run(ask_for_default("generic"))
    assert result == "__FAKE_default__"
    assert _captured_calls[-1]["task_type"] == "default"


# ─── Parameter forwarding ──────────────────────────────────────────────────

def test_max_tokens_forwarded():
    _run(ask_for_synthesis("test", max_tokens=400))
    assert _captured_calls[-1]["max_tokens"] == 400


def test_temperature_forwarded():
    _run(ask_for_intent("test", temperature=0))
    assert _captured_calls[-1]["temperature"] == 0


def test_system_prompt_forwarded():
    _run(ask_for_intent("test", system_prompt="You are a parser."))
    assert _captured_calls[-1]["system_prompt"] == "You are a parser."


def test_json_mode_forwarded():
    _run(ask_for_plan("test", json_mode=True))
    assert _captured_calls[-1]["json_mode"] is True


def test_default_max_tokens_is_256():
    _run(ask_for_synthesis("test"))
    assert _captured_calls[-1]["max_tokens"] == 256


def test_default_system_prompt_is_none():
    _run(ask_for_synthesis("test"))
    assert _captured_calls[-1]["system_prompt"] is None


def test_default_temperature_is_none():
    _run(ask_for_synthesis("test"))
    assert _captured_calls[-1]["temperature"] is None


def test_default_json_mode_is_false():
    _run(ask_for_synthesis("test"))
    assert _captured_calls[-1]["json_mode"] is False


# ─── All 8 contracts exist ─────────────────────────────────────────────────

def test_all_contracts_are_callable():
    contracts = [
        ask_for_intent, ask_for_synthesis, ask_for_plan,
        ask_for_code_gen, ask_for_small_talk,
        ask_for_personality_reflection, ask_for_agent_verify,
        ask_for_default,
    ]
    for c in contracts:
        assert callable(c), f"{c.__name__} is not callable"
    assert len(contracts) == 8


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
