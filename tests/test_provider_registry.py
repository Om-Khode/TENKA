"""
tests/test_provider_registry.py — Provider registry and protocol tests.
"""

import pytest
from assistant.llm.providers.base import Provider, ProviderResult
from assistant.llm.providers import provider_registry


@pytest.fixture(autouse=True)
def _snapshot_registry():
    snapshot = provider_registry.list_all()
    yield
    provider_registry.reset()
    for k, v in snapshot.items():
        provider_registry.register(k, v)


def test_all_providers_registered():
    expected = {"gemini", "groq", "cerebras", "ollama"}
    assert set(provider_registry.keys()) == expected


def test_providers_implement_protocol():
    for name, provider in provider_registry.list_all().items():
        assert isinstance(provider, Provider)
        assert provider.name == name


def test_provider_result_fields():
    r = ProviderResult("hello", 10, 5)
    assert r.text == "hello" and r.tokens_in == 10 and r.tokens_out == 5


def test_provider_result_defaults():
    r = ProviderResult("hello")
    assert r.tokens_in is None and r.tokens_out is None


def test_gemini_has_vision():
    assert callable(getattr(provider_registry.require("gemini"), "vision", None))


def test_gemini_has_stream():
    assert hasattr(provider_registry.require("gemini"), "stream")


def test_groq_has_vision():
    assert callable(getattr(provider_registry.require("groq"), "vision", None))


def test_groq_has_stream():
    assert hasattr(provider_registry.require("groq"), "stream")


def test_groq_has_vision_yes_no_sync():
    assert callable(getattr(provider_registry.require("groq"), "vision_yes_no_sync", None))


def test_cerebras_has_stream():
    assert hasattr(provider_registry.require("cerebras"), "stream")


def test_ollama_has_stream():
    assert hasattr(provider_registry.require("ollama"), "stream")


def test_cerebras_no_vision():
    assert not hasattr(provider_registry.require("cerebras"), "vision")


def test_ollama_no_vision():
    assert not hasattr(provider_registry.require("ollama"), "vision")


def test_re_export_from_llm_package():
    """Verify that GROQ_KEYS, provider_registry, etc. are accessible from llm/."""
    from assistant.llm import provider_registry as pr, Provider as P, ProviderResult as PR, GROQ_KEYS as gk
    assert pr is provider_registry
    assert P is Provider
    assert PR is ProviderResult
    assert isinstance(gk, list)
