"""Shared test setup. Puts the project root on sys.path so `from assistant import X` works."""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


_GUARDED_MODULES = [
    "assistant.io.screen",
    "assistant.llm",
    "assistant.intent",
    "assistant.automation.browser.automation",
    "assistant.automation.browser.cdp",
    "assistant.automation.browser.dom_orchestrator",
    "assistant.automation.native",
    "assistant.automation.router",
    "assistant.automation.vision",
    "assistant.automation.vision.agent",
    "assistant.automation.vision.verifier",
    "assistant.automation.vision.todo_classifier",
    "assistant.automation.vision._parsing",
    "playwright.async_api",
    "pyautogui",
]

_SENTINEL = object()


@pytest.fixture(autouse=True)
def _restore_sys_modules():
    """Prevent test-installed sys.modules stubs from leaking across test files."""
    saved = {k: sys.modules.get(k, _SENTINEL) for k in _GUARDED_MODULES}
    yield
    for k, original in saved.items():
        if original is _SENTINEL:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = original


class FakeTerminator:
    """Minimal seam mirroring the real terminator API surface manifest-based uses.

    Elements can be registered under their automation_id (lookup-by-id, the
    classic path) AND/OR under their accessible name (lookup-by-name, the
    F8 path for apps that don't expose automation_id). The find_element
    contract: prefer automation_id when non-empty; otherwise fall back to
    name. At least one must be non-empty.
    """

    def __init__(self):
        self.last_call = None
        self.elements: dict[str, dict] = {}        # by automation_id
        self.elements_by_name: dict[str, dict] = {}  # by accessible name
        self.fail_on: set[str] = set()              # keys (id or name) to force-miss

    def send_key(self, key: str) -> None:
        self.last_call = ("send_key", key)

    def find_element(
        self, *, automation_id: str = "", name: str = "",
        control_type: str = "", window: str = "",
    ):
        if automation_id:
            if automation_id in self.fail_on or automation_id not in self.elements:
                raise LookupError(f"NoSuchElement: automation_id={automation_id!r}")
            return self.elements[automation_id]
        if name:
            if name in self.fail_on or name not in self.elements_by_name:
                raise LookupError(f"NoSuchElement: name={name!r}")
            return self.elements_by_name[name]
        raise LookupError("NoSuchElement: both automation_id and name are empty")

    def click(self, element) -> None:
        self.last_call = ("click", element.get("automation_id") or element.get("name") or "?")


@pytest.fixture
def fake_terminator():
    """Reusable FakeTerminator instance for manifest-based primitive/dispatcher tests."""
    return FakeTerminator()
