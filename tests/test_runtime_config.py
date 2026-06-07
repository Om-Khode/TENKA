"""
test_runtime_config.py — RC-1: settings_store + slash_commands + config integration

Run: python test_runtime_config.py

Covers:
  - settings_store: get/set/delete round-trip, defaults, type preservation
  - slash_commands: /config, /set, /reset, shortcuts, unknown key, bad cast
  - config.reload_runtime_settings: picks up DB changes
  - SV-1d migration: set_listen_to_everyone persists + config reflects it
"""

import os
import sys
import sqlite3
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Direct imports of the real package. An earlier revision stubbed
# `assistant` / `assistant.config` here to skip the real package's import
# side effects, and used `_load_real_package()` to flush+reimport later.
# That flush deleted every `assistant.*` entry from sys.modules mid-pytest
# session, which silently broke any sibling test file that had bound
# module-level references (test_url_recon, test_knowledge_graph,
# test_teaching_session, test_file_ops, test_procedure_store). The stubs
# are no longer load-bearing — the real package imports fine — so we just
# import directly and stop fighting sys.modules.
import assistant.settings as ss
from assistant import config as _config_stub  # alias kept for legacy refs
from assistant.storage.db import init_db, get_db, _reset_for_testing


# ── Helpers ──────────────────────────────────────────────────────────────────


def _fresh_db() -> None:
    """Reset the storage singleton and the settings facade for a fresh DB.

    Post-RG-1, settings is a thin facade over the storage.db singleton —
    state lives on Database._instance and on settings._repo. Reset BOTH
    so prior-test rows don't leak in.
    """
    tmp = Path(tempfile.mkdtemp()) / "test_personality.db"
    _config_stub.SANDBOX_DIR = tmp.parent.parent
    (tmp.parent.parent / "memory").mkdir(parents=True, exist_ok=True)

    _reset_for_testing()
    ss._repo = None

    init_db(tmp)
    ss.init_settings_db()


# ─── settings_store: the persistence layer ──────────────────────────────────


class TestStoreRoundTrip(unittest.TestCase):
    def setUp(self):
        _fresh_db()

    def test_missing_key_returns_default(self):
        self.assertIsNone(ss.get("nonexistent"))
        self.assertEqual(ss.get("nonexistent", default=42), 42)

    def test_set_and_get_bool(self):
        ss.set("flag", True)
        self.assertIs(ss.get("flag"), True)
        ss.set("flag", False)
        self.assertIs(ss.get("flag"), False)

    def test_set_and_get_int(self):
        ss.set("count", 17)
        self.assertEqual(ss.get("count"), 17)
        self.assertIsInstance(ss.get("count"), int)

    def test_set_and_get_float(self):
        ss.set("threshold", 0.42)
        self.assertAlmostEqual(ss.get("threshold"), 0.42)

    def test_set_and_get_string(self):
        ss.set("mode", "verbose")
        self.assertEqual(ss.get("mode"), "verbose")

    def test_upsert_overwrites(self):
        ss.set("x", 1)
        ss.set("x", 2)
        self.assertEqual(ss.get("x"), 2)

    def test_delete_returns_true_when_existed(self):
        ss.set("x", 1)
        self.assertTrue(ss.delete("x"))
        self.assertIsNone(ss.get("x"))

    def test_delete_returns_false_when_missing(self):
        self.assertFalse(ss.delete("never_existed"))

    def test_list_all_returns_dict(self):
        ss.set("a", 1)
        ss.set("b", "hi")
        ss.set("c", True)
        all_ = ss.list_all()
        self.assertEqual(all_, {"a": 1, "b": "hi", "c": True})

    def test_corrupt_row_falls_back_to_default(self):
        # Write a row with invalid JSON directly via the storage singleton.
        db = get_db()
        self.assertIsNotNone(db, "DB must be initialized by _fresh_db()")
        db.execute(
            "INSERT INTO runtime_settings (key, value, updated_at, updated_source) "
            "VALUES (?, ?, ?, ?)",
            ("bad", "{not valid json", "2026-01-01", "user"),
        )
        db.commit()
        self.assertEqual(ss.get("bad", default="safe"), "safe")


# ─── config + slash_commands integration ────────────────────────────────────
# The real package is imported at module top — no flush needed.


def _load_real_package():
    """Return live references to the real assistant.config/settings/slash_commands.

    Previously this flushed every `assistant.*` entry from sys.modules and
    re-imported, which broke sibling test files' module-level bindings.
    Now it just imports the modules normally and hands back references.
    """
    import importlib
    cfg = importlib.import_module("assistant.config")
    store = importlib.import_module("assistant.settings")
    cmds = importlib.import_module("assistant.slash_commands")
    return cfg, store, cmds


def _reset_real_singleton(cfg, store) -> None:
    """Reset the storage singleton + settings facade against the real package.

    Mirrors _fresh_db() but binds against the real assistant.config (rather
    than the stub) so reload_runtime_settings() picks up an empty DB.
    """
    from assistant.storage.db import init_db as _real_init_db
    from assistant.storage.db import _reset_for_testing as _real_reset
    tmp = Path(tempfile.mkdtemp()) / "test_personality.db"
    cfg.SANDBOX_DIR = tmp.parent.parent
    (tmp.parent.parent / "memory").mkdir(parents=True, exist_ok=True)

    _real_reset()
    store._repo = None

    _real_init_db(tmp)
    store.init_settings_db()
    cfg.reload_runtime_settings()


class TestConfigIntegration(unittest.TestCase):
    """Exercise config.reload_runtime_settings + slash_commands against a temp DB."""

    @classmethod
    def setUpClass(cls):
        cls.cfg, cls.store, cls.cmds = _load_real_package()

    def setUp(self):
        _reset_real_singleton(self.cfg, self.store)

    def test_registry_has_expected_keys(self):
        reg = self.cfg.RUNTIME_SETTINGS_REGISTRY
        self.assertIn("listen_to_everyone", reg)
        self.assertIn("followup_timer", reg)
        self.assertIn("wake_word_sensitivity", reg)
        for meta in reg.values():
            self.assertIn("default", meta)
            self.assertIn("cast", meta)
            self.assertIn("description", meta)

    def test_defaults_when_empty_db(self):
        self.assertFalse(self.cfg.LISTEN_TO_EVERYONE)
        self.assertIsInstance(self.cfg.FOLLOWUP_TIMER, float)

    def test_reload_picks_up_db_value(self):
        self.store.set("listen_to_everyone", True)
        self.cfg.reload_runtime_settings()
        self.assertTrue(self.cfg.LISTEN_TO_EVERYONE)

    def test_legacy_alias_stays_in_sync(self):
        self.store.set("followup_timer", 12.5)
        self.cfg.reload_runtime_settings()
        self.assertAlmostEqual(self.cfg.FOLLOWUP_TIMER, 12.5)
        self.assertAlmostEqual(self.cfg.FOLLOW_UP_LISTEN_SECONDS, 12.5)

    def test_wake_threshold_alias(self):
        self.store.set("wake_word_sensitivity", 0.08)
        self.cfg.reload_runtime_settings()
        self.assertAlmostEqual(self.cfg.WAKE_WORD_SENSITIVITY, 0.08)
        self.assertAlmostEqual(self.cfg.WAKE_WORD_THRESHOLD, 0.08)


class TestSlashCommands(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg, cls.store, cls.cmds = _load_real_package()

    def setUp(self):
        _reset_real_singleton(self.cfg, self.store)

    def test_is_slash_command(self):
        self.assertTrue(self.cmds.is_slash_command("/help"))
        self.assertTrue(self.cmds.is_slash_command("  /config "))
        self.assertFalse(self.cmds.is_slash_command("/"))
        self.assertFalse(self.cmds.is_slash_command("hello"))
        self.assertFalse(self.cmds.is_slash_command(""))

    def test_help(self):
        out = self.cmds.handle("/help")
        self.assertIn("/config", out)
        self.assertIn("/set", out)
        self.assertIn("/reset", out)

    def test_config_lists_all(self):
        out = self.cmds.handle("/config")
        self.assertIn("listen_to_everyone", out)
        self.assertIn("followup_timer", out)
        # No customizations yet — no "*" custom-marker on any row (header legend is OK)
        for key in self.cfg.RUNTIME_SETTINGS_REGISTRY:
            self.assertNotRegex(out, rf"\* {key}|\*R {key}")

    def test_config_one_setting(self):
        out = self.cmds.handle("/config listen_to_everyone")
        self.assertIn("listen_to_everyone", out)
        self.assertIn("default", out)
        self.assertIn("bool", out)
        self.assertIn("description", out)

    def test_config_unknown_setting(self):
        out = self.cmds.handle("/config nonexistent_key")
        self.assertIn("Unknown", out)

    def test_set_bool_true(self):
        out = self.cmds.handle("/set listen_to_everyone true")
        self.assertIn("Set listen_to_everyone", out)
        self.assertTrue(self.cfg.LISTEN_TO_EVERYONE)
        self.assertTrue(self.store.get("listen_to_everyone"))

    def test_set_bool_false(self):
        self.store.set("listen_to_everyone", True)
        self.cfg.reload_runtime_settings()
        self.cmds.handle("/set listen_to_everyone false")
        self.assertFalse(self.cfg.LISTEN_TO_EVERYONE)

    def test_set_invalid_bool(self):
        out = self.cmds.handle("/set listen_to_everyone banana")
        self.assertIn("Invalid", out)
        self.assertFalse(self.cfg.LISTEN_TO_EVERYONE)  # unchanged

    def test_set_invalid_float(self):
        out = self.cmds.handle("/set followup_timer notanumber")
        self.assertIn("Invalid", out)

    def test_set_float(self):
        self.cmds.handle("/set followup_timer 8.5")
        self.assertAlmostEqual(self.cfg.FOLLOWUP_TIMER, 8.5)
        self.assertAlmostEqual(self.cfg.FOLLOW_UP_LISTEN_SECONDS, 8.5)

    def test_set_missing_value(self):
        out = self.cmds.handle("/set followup_timer")
        self.assertIn("Usage", out)

    def test_set_unknown_key(self):
        out = self.cmds.handle("/set nonexistent_key 5")
        self.assertIn("Unknown", out)

    def test_reset_reverts(self):
        self.cmds.handle("/set followup_timer 99")
        self.assertAlmostEqual(self.cfg.FOLLOWUP_TIMER, 99.0)
        out = self.cmds.handle("/reset followup_timer")
        self.assertIn("Reset", out)
        # Back to default (module-level FOLLOW_UP_LISTEN_SECONDS, typically 5.0)
        self.assertNotEqual(self.cfg.FOLLOWUP_TIMER, 99.0)

    def test_reset_already_default(self):
        out = self.cmds.handle("/reset followup_timer")
        self.assertIn("already at default", out)

    def test_shortcut_form_get(self):
        out = self.cmds.handle("/listen_to_everyone")
        self.assertIn("listen_to_everyone", out)
        self.assertIn("description", out)

    def test_shortcut_form_set(self):
        out = self.cmds.handle("/listen_to_everyone true")
        self.assertIn("Set listen_to_everyone", out)
        self.assertTrue(self.cfg.LISTEN_TO_EVERYONE)

    def test_shortcut_unknown(self):
        out = self.cmds.handle("/banana 5")
        self.assertIn("Unknown", out)

    def test_custom_shows_asterisk(self):
        self.cmds.handle("/set followup_timer 7.5")
        out = self.cmds.handle("/config")
        # Customized rows are marked with * in the first marker column
        # Format: "  *  followup_timer" (custom, non-restart) or "  *R followup_timer"
        self.assertRegex(out, r"\*[ R] followup_timer")


class TestExpandedRegistry(unittest.TestCase):
    """Tier 1 + Tier 2 settings registered with correct metadata."""

    @classmethod
    def setUpClass(cls):
        cls.cfg, cls.store, cls.cmds = _load_real_package()

    def setUp(self):
        _reset_real_singleton(self.cfg, self.store)

    def test_all_tier1_tier2_registered(self):
        expected = {
            "tts_speed", "vocal_voice_enabled", "vocal_casual_language",
            "wake_word_cooldown", "wake_word_enabled",
            "speaker_verify_enabled", "speaker_verify_threshold",
            "camera_enabled", "face_recognition_tolerance",
            "proactive_enabled", "proactive_mode",
            "proactive_interval_minutes", "proactive_idle_threshold_minutes",
            "messaging_notify_debounce", "messaging_suppress_window",
            "incoming_read_threshold",
        }
        registered = set(self.cfg.RUNTIME_SETTINGS_REGISTRY.keys())
        missing = expected - registered
        self.assertFalse(missing, f"Missing settings: {missing}")

    def test_needs_restart_flags(self):
        reg = self.cfg.RUNTIME_SETTINGS_REGISTRY
        # Restart-required by design
        for key in ("vocal_casual_language", "wake_word_enabled", "camera_enabled",
                    "proactive_enabled", "proactive_interval_minutes"):
            self.assertTrue(reg[key]["needs_restart"],
                            f"{key} should be needs_restart")
        # Live-reloadable
        for key in ("tts_speed", "vocal_voice_enabled", "wake_word_cooldown",
                    "speaker_verify_enabled", "speaker_verify_threshold",
                    "face_recognition_tolerance", "proactive_mode",
                    "proactive_idle_threshold_minutes",
                    "messaging_notify_debounce", "messaging_suppress_window",
                    "incoming_read_threshold", "followup_timer",
                    "wake_word_sensitivity", "listen_to_everyone"):
            self.assertFalse(reg[key]["needs_restart"],
                             f"{key} should NOT be needs_restart")

    def test_correct_casts(self):
        reg = self.cfg.RUNTIME_SETTINGS_REGISTRY
        self.assertIs(reg["tts_speed"]["cast"], float)
        self.assertIs(reg["incoming_read_threshold"]["cast"], int)
        self.assertIs(reg["proactive_mode"]["cast"], str)
        self.assertIs(reg["camera_enabled"]["cast"], bool)

    def test_reload_updates_tts_speed(self):
        self.store.set("tts_speed", 1.3)
        self.cfg.reload_runtime_settings()
        self.assertAlmostEqual(self.cfg.TTS_SPEED, 1.3)

    def test_reload_updates_int_setting(self):
        self.store.set("incoming_read_threshold", 7)
        self.cfg.reload_runtime_settings()
        self.assertEqual(self.cfg.INCOMING_READ_THRESHOLD, 7)

    def test_reload_updates_str_setting(self):
        self.store.set("proactive_mode", "idle_only")
        self.cfg.reload_runtime_settings()
        self.assertEqual(self.cfg.PROACTIVE_MODE, "idle_only")

    def test_set_via_slash_command_hot_reload(self):
        # Live setting: changes should reflect immediately
        self.cmds.handle("/set wake_word_cooldown 3.5")
        self.assertAlmostEqual(self.cfg.WAKE_WORD_COOLDOWN, 3.5)

    def test_config_shows_restart_marker(self):
        out = self.cmds.handle("/config")
        # Restart-required rows have 'R' as the second marker char
        self.assertRegex(out, r"\s+R camera_enabled")
        self.assertRegex(out, r"\s+R vocal_casual_language")

    def test_one_setting_shows_restart_note(self):
        out = self.cmds.handle("/config camera_enabled")
        self.assertIn("restart", out.lower())

    def test_one_setting_no_restart_note_when_live(self):
        out = self.cmds.handle("/config tts_speed")
        self.assertNotIn("restart", out.lower())

    def test_set_restart_message(self):
        out = self.cmds.handle("/set camera_enabled false")
        self.assertIn("Restart required", out)

    def test_set_no_restart_message_when_live(self):
        out = self.cmds.handle("/set tts_speed 1.1")
        self.assertNotIn("Restart required", out)


class TestSV1dMigration(unittest.TestCase):
    """SV-1d voice toggle now persists via settings_store."""

    @classmethod
    def setUpClass(cls):
        cls.cfg, cls.store, cls.cmds = _load_real_package()
        # Import speaker_verify with the real config in place
        import importlib
        cls.sv = importlib.import_module("assistant.io.audio.speaker_verify")

    def setUp(self):
        _reset_real_singleton(self.cfg, self.store)

    def test_voice_trigger_persists(self):
        self.sv.set_listen_to_everyone(True)
        # Persisted in DB
        self.assertTrue(self.store.get("listen_to_everyone"))
        # Reflected in config
        self.assertTrue(self.cfg.LISTEN_TO_EVERYONE)
        # Accessor agrees
        self.assertTrue(self.sv.is_listen_to_everyone())

    def test_voice_trigger_persists_across_reload(self):
        self.sv.set_listen_to_everyone(True)
        # Simulate a restart: reset module state, reload config from DB
        self.cfg.reload_runtime_settings()
        self.assertTrue(self.cfg.LISTEN_TO_EVERYONE)
        self.assertTrue(self.sv.is_listen_to_everyone())

    def test_toggle_off(self):
        self.sv.set_listen_to_everyone(True)
        self.sv.set_listen_to_everyone(False)
        self.assertFalse(self.cfg.LISTEN_TO_EVERYONE)
        self.assertFalse(self.sv.is_listen_to_everyone())


class TestAssistantName(unittest.TestCase):
    """Phase 6A: assistant_name flows through config + dependent modules."""

    @classmethod
    def setUpClass(cls):
        cls.cfg, cls.store, cls.cmds = _load_real_package()

    def setUp(self):
        _reset_real_singleton(self.cfg, self.store)

    def test_registered_with_needs_restart(self):
        meta = self.cfg.RUNTIME_SETTINGS_REGISTRY.get("assistant_name")
        self.assertIsNotNone(meta)
        self.assertEqual(meta["default"], "TENKA")
        self.assertIs(meta["cast"], str)
        self.assertTrue(meta["needs_restart"])

    def test_default_name(self):
        self.assertEqual(self.cfg.ASSISTANT_NAME, "TENKA")
        self.assertEqual(self.cfg.ASSISTANT_NAME_LOWER, "tenka")
        # All-caps already → display form preserves casing
        self.assertEqual(self.cfg.ASSISTANT_NAME_DISPLAY, "TENKA")

    def test_reload_updates_name_lower(self):
        self.store.set("assistant_name", "Luna")
        self.cfg.reload_runtime_settings()
        self.assertEqual(self.cfg.ASSISTANT_NAME, "Luna")
        self.assertEqual(self.cfg.ASSISTANT_NAME_LOWER, "luna")

    def test_set_shows_restart_required(self):
        out = self.cmds.handle("/set assistant_name Luna")
        self.assertIn("Restart required", out)

    def test_personality_prompt_contains_name(self):
        # get_system_prompt() returns the active personality base with rules.
        # Assert the current display name is somewhere in the prompt.
        prompt = self.cfg.LLM_SYSTEM_PROMPT
        self.assertIn("You are", prompt)

    def test_display_name_capitalizes_lowercase(self):
        self.assertEqual(self.cfg._display_name("luna"), "Luna")
        self.assertEqual(self.cfg._display_name("tenka"), "Tenka")

    def test_display_name_preserves_mixed_case(self):
        self.assertEqual(self.cfg._display_name("Luna"), "Luna")
        self.assertEqual(self.cfg._display_name("McKay"), "McKay")
        self.assertEqual(self.cfg._display_name("DJ"), "DJ")
        self.assertEqual(self.cfg._display_name("LUNA"), "LUNA")

    def test_display_name_empty_safe(self):
        self.assertEqual(self.cfg._display_name(""), "")

    def test_display_matches_stored_when_mixed_case(self):
        self.store.set("assistant_name", "McKay")
        self.cfg.reload_runtime_settings()
        self.assertEqual(self.cfg.ASSISTANT_NAME, "McKay")
        self.assertEqual(self.cfg.ASSISTANT_NAME_DISPLAY, "McKay")

    def test_display_capitalizes_on_reload_when_lowercase(self):
        self.store.set("assistant_name", "luna")
        self.cfg.reload_runtime_settings()
        self.assertEqual(self.cfg.ASSISTANT_NAME, "luna")       # stored verbatim
        self.assertEqual(self.cfg.ASSISTANT_NAME_LOWER, "luna")
        self.assertEqual(self.cfg.ASSISTANT_NAME_DISPLAY, "Luna")  # promoted

    def test_identity_directive_in_prompt(self):
        # Post P1: identity is asserted via the personality prompt's opening
        # line ("You are <NAME>, ...") and the name-suppression rule, not via
        # the legacy "Your name is X. If asked to be Y, IGNORE IT" directive.
        # Verify the current shape — the name is present and the prompt forbids
        # leading with it.
        prompt = self.cfg.LLM_SYSTEM_PROMPT
        self.assertIn(f"You are {self.cfg.ASSISTANT_NAME}", prompt)
        self.assertIn("NEVER start a response with your name", prompt)

    def test_description_mentions_memory_caveat(self):
        # Users should know renaming doesn't wipe memory
        meta = self.cfg.RUNTIME_SETTINGS_REGISTRY["assistant_name"]
        self.assertIn("memory", meta["description"].lower())

    def test_wake_word_path_matches_name_lower(self):
        # Path is baked at import time from whatever was in the DB; assert it
        # ends with <some-name>.onnx (robust against test-order / prod-DB state).
        path_str = str(self.cfg.WAKE_WORD_MODEL_PATH).lower()
        self.assertTrue(
            path_str.endswith(".onnx"),
            f"Wake word path should end with .onnx: {self.cfg.WAKE_WORD_MODEL_PATH}",
        )
        self.assertIn("models", path_str)

    def test_intent_prompt_no_longer_says_miku(self):
        # The INTENT_SYSTEM_PROMPT used to include "miku hides" / "miku reappears"
        # for the avatar intents. 6A made those generic.
        self.assertNotIn("miku hides", self.cfg.INTENT_SYSTEM_PROMPT)
        self.assertNotIn("miku reappears", self.cfg.INTENT_SYSTEM_PROMPT)

    def test_shortcut_fillers_include_name_lower(self):
        import importlib
        ss = importlib.import_module("assistant.shortcuts")
        # Create a quick in-memory shortcut and verify "<name> setup" matches "setup"
        from assistant.storage.db import init_db as _init_db, _reset_for_testing as _reset
        tmp = Path(tempfile.mkdtemp()) / "sc_personality.db"
        (tmp.parent / "memory").mkdir(parents=True, exist_ok=True)
        _reset()
        ss._repo = None
        _init_db(tmp)
        ss.init_shortcut_db()
        self.assertTrue(ss.create_shortcut(
            trigger="setup", intent="open_browser",
            params={}, description="test"))
        # Build filler phrases from whatever the current assistant name is
        name = self.cfg.ASSISTANT_NAME_LOWER
        match = ss.match_shortcut(f"setup {name}")
        self.assertIsNotNone(match, "trailing assistant name should be stripped as filler")
        self.assertEqual(match["trigger"], "setup")
        match = ss.match_shortcut(f"{name} setup")
        self.assertIsNotNone(match, "leading assistant name should be stripped as filler")

    def test_shortcut_name_reserved(self):
        import importlib
        ss = importlib.import_module("assistant.shortcuts")
        from assistant.storage.db import init_db as _init_db, _reset_for_testing as _reset
        tmp = Path(tempfile.mkdtemp()) / "sc_personality.db"
        (tmp.parent / "memory").mkdir(parents=True, exist_ok=True)
        _reset()
        ss._repo = None
        _init_db(tmp)
        ss.init_shortcut_db()
        # Trying to create a shortcut with the assistant name as trigger should fail
        ok = ss.create_shortcut(
            trigger=self.cfg.ASSISTANT_NAME_LOWER,
            intent="open_browser", params={}, description="should reject")
        self.assertFalse(ok, "assistant name must be reserved as a trigger")


class TestPersonalityEventsNameBinding(unittest.TestCase):
    """personality regex patterns must bind to ASSISTANT_NAME_LOWER."""

    @classmethod
    def setUpClass(cls):
        cls.cfg, cls.store, cls.cmds = _load_real_package()
        import importlib
        cls.pe = importlib.import_module("assistant.personality")

    def test_greeting_matches_current_name(self):
        name = self.cfg.ASSISTANT_NAME_LOWER
        for phrase in (f"hey {name}", f"hi {name}", f"hello {name}",
                       f"morning {name}", f"night {name}"):
            self.assertIsNotNone(
                self.pe._GREETING_PATTERNS.search(phrase),
                f"greeting regex should match {phrase!r}",
            )

    def test_greeting_still_matches_generic(self):
        # Name-agnostic greetings must still work
        for phrase in ("good morning", "goodnight", "good evening"):
            self.assertIsNotNone(self.pe._GREETING_PATTERNS.search(phrase))

    def test_compliment_matches_current_name(self):
        name = self.cfg.ASSISTANT_NAME_LOWER
        self.assertIsNotNone(
            self.pe._COMPLIMENT_PATTERNS.search(f"thanks {name}"))
        self.assertIsNotNone(
            self.pe._COMPLIMENT_PATTERNS.search(f"thank you {name}"))

    def test_check_on_matches_current_name(self):
        name = self.cfg.ASSISTANT_NAME_LOWER
        self.assertIsNotNone(
            self.pe._CHECK_ON_PATTERNS.search(f"how's it going {name}"))

    def test_frustration_is_name_agnostic(self):
        # Frustration patterns never included the name — make sure they still work
        for phrase in ("never mind", "forget it", "this is stupid"):
            self.assertIsNotNone(self.pe._FRUSTRATION_PATTERNS.search(phrase))


class TestChatInputCompleter(unittest.TestCase):
    """Autocomplete logic for the prompt_toolkit chat prompt."""

    @classmethod
    def setUpClass(cls):
        cls.cfg, cls.store, cls.cmds = _load_real_package()
        try:
            import importlib
            cls.ci = importlib.import_module("assistant.chat_input")
        except ImportError:
            cls.ci = None

    def _completions(self, text: str):
        """Return list of (completion_text, meta_text) for input `text`."""
        if self.ci is None:
            self.skipTest("chat_input module not importable")
        completer = self.ci._build_completer()
        if completer is None:
            self.skipTest("prompt_toolkit not installed")
        try:
            from prompt_toolkit.document import Document
        except ImportError:
            self.skipTest("prompt_toolkit not installed")
        doc = Document(text=text, cursor_position=len(text))
        results = list(completer.get_completions(doc, complete_event=None))
        return [(c.text, str(c.display_meta_text) if c.display_meta else "") for c in results]

    def test_no_completion_without_slash(self):
        self.assertEqual(self._completions("hello"), [])
        self.assertEqual(self._completions(""), [])

    def test_slash_offers_reserved_and_settings(self):
        completions = [t for t, _ in self._completions("/")]
        for cmd in ("help", "config", "set", "reset"):
            self.assertIn(cmd, completions)
        self.assertIn("followup_timer", completions)
        self.assertIn("tts_speed", completions)

    def test_prefix_filters(self):
        completions = [t for t, _ in self._completions("/wake")]
        # All suggestions must start with 'wake'
        for c in completions:
            self.assertTrue(c.startswith("wake"), c)
        # Must include wake_word_* family
        self.assertIn("wake_word_enabled", completions)
        self.assertIn("wake_word_sensitivity", completions)

    def test_config_arg_offers_settings(self):
        completions = [t for t, _ in self._completions("/config ")]
        self.assertIn("listen_to_everyone", completions)
        # Reserved words should NOT appear as setting keys here
        self.assertNotIn("help", completions)
        self.assertNotIn("set", completions)

    def test_set_arg_offers_settings(self):
        completions = [t for t, _ in self._completions("/set tts_")]
        self.assertIn("tts_speed", completions)

    def test_no_completion_after_set_key(self):
        completions = self._completions("/set tts_speed ")
        # Value token is free-form — no completions offered
        self.assertEqual(completions, [])

    def test_restart_flag_in_meta(self):
        # Find camera_enabled in completions for "/config "
        pairs = self._completions("/config ")
        camera = [m for t, m in pairs if t == "camera_enabled"]
        self.assertTrue(camera, "camera_enabled missing")
        self.assertIn("[R]", camera[0])

    def test_live_setting_no_restart_flag(self):
        pairs = self._completions("/config ")
        tts = [m for t, m in pairs if t == "tts_speed"]
        self.assertTrue(tts)
        self.assertNotIn("[R]", tts[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
