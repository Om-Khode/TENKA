# assistant/io/api/__init__.py
"""TENKA Studio daemon.

Layering: io/api may import assistant.core, assistant.config and third-party
packages only. Never storage, actions, automation, llm or main — every piece of
assistant data arrives through the injected StudioRuntime protocols.
"""
