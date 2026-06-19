"""
test_known_issues_fixes.py — Regression tests for the batched Known-Issues fixes.

Covers:
  KI-1: get_text steps stripped from search goals (not just type/write).
  KI-6: deterministic window-param pinning to the real focused window.
  KI-7: SMTC media verification gated on playback intent, not app names.
  KI-8: code_executor synthesis prompts force output values into the reply.

KI-1/KI-8 live inside large async handlers, so they are verified by source
inspection (matching the repo's existing deterministic-fix tests). KI-6's pin
loop was extracted into `_pin_step_windows` and is unit-tested directly; KI-7's
gate and verifier short-circuit are tested behaviourally with monkeypatching.
"""

import asyncio
import os
import pytest

_ROOT = os.path.dirname(os.path.dirname(__file__))
_ROUTER = os.path.join(_ROOT, "assistant", "automation", "router.py")
_ORCH = os.path.join(_ROOT, "assistant", "code_executor", "orchestrator.py")


def _read(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ─── KI-7: SMTC gated on playback intent ────────────────────────────
class TestKI7MusicPlaybackGate:
    @pytest.fixture
    def is_playback(self):
        from assistant.automation.vision.verifier import _is_music_playback_goal
        return _is_music_playback_goal

    @pytest.mark.parametrize("goal", [
        "play lo-fi on spotify",
        "pause the music",
        "skip this song",
        "play despacito",
        "shuffle my playlist",
        "resume playback",
        "next track",
    ])
    def test_playback_goals_match(self, is_playback, goal):
        assert is_playback(goal) is True, f"expected playback intent for: {goal!r}"

    @pytest.mark.parametrize("goal", [
        "close spotify app",
        "open spotify",
        "minimize spotify",
        "switch to spotify",
        "quit the music app",
        "is spotify running",
    ])
    def test_app_management_goals_do_not_match(self, is_playback, goal):
        assert is_playback(goal) is False, f"app-management goal must NOT trigger SMTC: {goal!r}"

    def test_display_is_not_play(self, is_playback):
        # word-boundary regex must not match "play" inside "display"
        assert is_playback("open display settings") is False

    def test_smtc_only_called_under_playback_gate(self):
        """The _get_now_playing() SMTC query must sit behind the playback gate."""
        src = _read(os.path.join(_ROOT, "assistant", "automation", "vision", "verifier.py"))
        # the only call to _get_now_playing() must be inside an _is_music_playback_goal block
        assert "_get_now_playing()" in src
        assert "if _is_music_playback_goal(goal):" in src
        # the old app-name-based trigger must be gone
        assert "+ _music_apps" not in src, "app-name trigger for SMTC must be removed"


# ─── KI-1: get_text stripped from search goals ──────────────────────
class TestKI1SearchGetTextStrip:
    def test_search_in_type_words(self):
        src = _read(_ROUTER)
        # locate the _TYPE_WORDS set definition and assert "search" is a member
        assert '"search"' in src
        idx = src.index("_TYPE_WORDS = {")
        line_end = src.index("}", idx)
        type_words_literal = src[idx:line_end]
        assert "search" in type_words_literal, "_TYPE_WORDS must include 'search' for KI-1"


# ─── KI-6: deterministic window pinning ─────────────────────────────
class TestKI6WindowPinning:
    def test_pin_loop_present(self):
        src = _read(_ROUTER)
        assert "Pinned" in src and "running_window" in src, "KI-6 window-pin log must exist"
        # the override must force the real window onto interaction steps
        assert 'params["window"] = running_window' in src, (
            "KI-6 must deterministically overwrite hallucinated window params"
        )

    def test_pin_targets_interaction_actions(self):
        src = _read(_ROUTER)
        # the pin must apply to click/type/get_text steps
        assert '("click", "type", "get_text")' in src


# ─── KI-6: deterministic window pinning — behavioural ───────────────
class TestKI6PinHelper:
    @pytest.fixture
    def pin(self):
        from assistant.automation.router import _pin_step_windows
        return _pin_step_windows

    def test_rewrites_hallucinated_window(self, pin):
        steps = [{"action": "click", "params": {
            "selector": "name:X", "window": "Spotify - Web Player: Music for everyone"}}]
        n = pin(steps, "Spotify Premium")
        assert n == 1
        assert steps[0]["params"]["window"] == "Spotify Premium"

    def test_leaves_correct_window_untouched(self, pin):
        steps = [{"action": "type", "params": {"text": "hi", "window": "Spotify Premium"}}]
        assert pin(steps, "Spotify Premium") == 0
        assert steps[0]["params"]["window"] == "Spotify Premium"

    def test_ignores_non_interaction_actions(self, pin):
        # focus/open/close use 'name', not 'window' — must not be rewritten
        steps = [
            {"action": "focus", "params": {"name": "Spotify Premium"}},
            {"action": "close", "params": {"name": "Wrong Window Title"}},
        ]
        assert pin(steps, "Spotify Premium") == 0
        assert steps[1]["params"]["name"] == "Wrong Window Title"

    def test_step_without_window_param_is_safe(self, pin):
        steps = [{"action": "get_text", "params": {"selector": "name:Y"}}]
        assert pin(steps, "Spotify Premium") == 0
        assert "window" not in steps[0]["params"]

    def test_no_running_window_is_noop(self, pin):
        steps = [{"action": "click", "params": {"window": "Whatever"}}]
        assert pin(steps, None) == 0
        assert steps[0]["params"]["window"] == "Whatever"


# ─── KI-7: verifier SMTC short-circuit — behavioural ────────────────
class TestKI7VerifierShortCircuit:
    """Drives the real _verify_goal: the SMTC query (_get_now_playing) must be
    skipped for app-management goals and used for genuine playback goals."""

    def _wire(self, monkeypatch, now_playing):
        import assistant.io.screen as screen_mod
        import assistant.llm.contracts as contracts_mod
        from assistant.automation.vision import verifier

        calls = {"smtc": 0}

        def fake_now_playing():
            calls["smtc"] += 1
            return now_playing

        async def fake_vision_verify(*a, **k):
            return '{"achieved": true, "result": "vision-path", "remaining": ""}'

        monkeypatch.setattr(screen_mod, "get_active_window", lambda: "Spotify Premium")
        monkeypatch.setattr(screen_mod, "capture_screenshot_base64", lambda: None)
        monkeypatch.setattr(screen_mod, "describe_screen_for_llm", lambda: "Spotify visible")
        monkeypatch.setattr(contracts_mod, "ask_for_agent_verify", fake_vision_verify)
        monkeypatch.setattr(verifier, "_get_now_playing", fake_now_playing)
        return verifier, calls

    async def _dummy_llm(self, *a, **k):
        return ""

    def test_close_goal_skips_smtc(self, monkeypatch):
        verifier, calls = self._wire(
            monkeypatch, {"is_playing": True, "title": "Other Song", "artist": "X"})
        res = asyncio.run(verifier._verify_goal("close spotify app", self._dummy_llm))
        assert calls["smtc"] == 0, "SMTC must NOT be queried for an app-management goal"
        # it fell through to vision verification, not the SMTC 'wrong song' path
        assert res["result"] == "vision-path"
        assert "wrong" not in res.get("remaining", "").lower()

    def test_playback_goal_uses_smtc(self, monkeypatch):
        verifier, calls = self._wire(
            monkeypatch, {"is_playing": True, "title": "Blinding Lights", "artist": "The Weeknd"})
        res = asyncio.run(verifier._verify_goal("play blinding lights in spotify", self._dummy_llm))
        assert calls["smtc"] == 1, "SMTC must be queried for a genuine playback goal"
        assert res["achieved"] is True
        assert "blinding lights" in res["result"].lower()


# ─── KI-8: synthesis includes output values ─────────────────────────
class TestKI8SynthesisValues:
    def test_both_success_prompts_demand_values(self):
        src = _read(_ORCH)
        marker = "State the key output values"
        count = src.count(marker)
        assert count >= 2, (
            f"both code_executor success-synthesis prompts must demand output values "
            f"(found {count} occurrences of {marker!r})"
        )

    def test_prompts_warn_user_cannot_see_output(self):
        src = _read(_ORCH)
        assert "cannot see the raw output" in src


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
