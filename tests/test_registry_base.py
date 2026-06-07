"""Tests for assistant.core.registry.RegistryBase."""

import threading

import pytest

from assistant.core.registry import RegistryBase


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture()
def reg() -> RegistryBase:
    """Fresh RegistryBase for each test."""
    return RegistryBase(name="test")


# ─── Basic registration ────────────────────────────────────────────────────────

def test_register_and_get(reg):
    reg.register("alpha", 42)
    assert reg.get("alpha") == 42


def test_get_missing_returns_none(reg):
    assert reg.get("nonexistent") is None


def test_require_returns_value(reg):
    reg.register("beta", "hello")
    assert reg.require("beta") == "hello"


def test_require_missing_raises(reg):
    with pytest.raises(KeyError, match="test registry has no entry 'missing'"):
        reg.require("missing")


def test_duplicate_raises(reg):
    reg.register("dup", 1)
    with pytest.raises(ValueError, match="already registered"):
        reg.register("dup", 2)


# ─── Introspection ────────────────────────────────────────────────────────────

def test_has(reg):
    reg.register("x", object())
    assert reg.has("x") is True
    assert reg.has("y") is False


def test_keys(reg):
    reg.register("a", 1)
    reg.register("b", 2)
    assert sorted(reg.keys()) == ["a", "b"]


def test_list_all_returns_snapshot(reg):
    reg.register("k", 99)
    snapshot = reg.list_all()
    assert snapshot["k"] == 99

    # Mutating the snapshot must not affect the registry
    snapshot["injected"] = -1
    assert reg.get("injected") is None


# ─── Reset ────────────────────────────────────────────────────────────────────

def test_reset_clears_all(reg):
    reg.register("p", "v")
    reg.reset()
    assert reg.get("p") is None
    assert reg.keys() == []


# ─── Decorator ────────────────────────────────────────────────────────────────

def test_decorator_registers_callable(reg):
    @reg.decorator("my_fn")
    def my_fn():
        return "result"

    assert reg.get("my_fn") is my_fn
    assert my_fn() == "result"


def test_decorator_duplicate_raises(reg):
    @reg.decorator("once")
    def first():
        pass

    with pytest.raises(ValueError, match="already registered"):
        @reg.decorator("once")
        def second():
            pass


# ─── Return value ─────────────────────────────────────────────────────────────

def test_register_returns_object(reg):
    sentinel = object()
    returned = reg.register("obj", sentinel)
    assert returned is sentinel


# ─── Thread safety ────────────────────────────────────────────────────────────

def test_thread_safety():
    """4 threads each register 100 unique keys; expect 400 total, no errors."""
    registry: RegistryBase[int] = RegistryBase(name="thread-test")
    errors: list[Exception] = []

    def worker(thread_id: int) -> None:
        try:
            for i in range(100):
                registry.register(f"t{thread_id}_k{i}", thread_id * 100 + i)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(tid,)) for tid in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"Thread errors: {errors}"
    assert len(registry.keys()) == 400
