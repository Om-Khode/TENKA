"""Tests for P1: Swappable Personality Bases."""

import sqlite3
import tempfile
import unittest
from pathlib import Path


class TestDBMigrationV9(unittest.TestCase):
    """Test schema v8 -> v9 migration preserves existing trait rows."""

    def test_migration_tags_existing_rows_as_tsundere(self):
        """Pre-P1 trait rows get personality_id='tsundere' after migration."""
        from assistant.storage.db import Database

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            # Step 1: build a real v8 DB by running the migration chain with
            # _LATEST_VERSION temporarily clamped to 8. This is more faithful
            # than hand-crafting subset tables and survives future migrations.
            original_latest = Database._LATEST_VERSION
            Database._LATEST_VERSION = 8
            try:
                v8 = Database(db_path)
                try:
                    traits = [
                        ("trust", 0.35, 0.10, 0.85, "2026-01-01"),
                        ("warmth", 0.30, 0.10, 0.85, "2026-01-01"),
                        ("sass", 0.75, 0.30, 0.95, "2026-01-01"),
                        ("openness", 0.40, 0.10, 0.85, "2026-01-01"),
                        ("patience", 0.40, 0.15, 0.80, "2026-01-01"),
                        ("playfulness", 0.65, 0.25, 0.90, "2026-01-01"),
                    ]
                    v8._conn.executemany(
                        "INSERT INTO personality_state "
                        "(trait, value, floor_val, ceiling_val, updated_at) "
                        "VALUES (?, ?, ?, ?, ?)",
                        traits,
                    )
                    v8._conn.commit()
                finally:
                    v8.close()
            finally:
                Database._LATEST_VERSION = original_latest

            # Step 2: reopen — runs v9 onward. v9 is the migration under test.
            db = Database(db_path)
            try:
                rows = db.fetchall(
                    "SELECT personality_id, trait FROM personality_state"
                )
                personality_ids = {r["personality_id"] for r in rows}
                self.assertEqual(personality_ids, {"tsundere"})
                self.assertEqual(len(rows), 6)
            finally:
                db.close()

    def test_personality_log_has_personality_id_column(self):
        """personality_log gains personality_id column with default 'tsundere'."""
        from assistant.storage.db import Database

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = Database(db_path)

            cols = db.fetchall("PRAGMA table_info(personality_log)")
            col_names = [c["name"] for c in cols]
            self.assertIn("personality_id", col_names)
            db.close()

    def test_composite_primary_key(self):
        """personality_state PK is now (personality_id, trait)."""
        from assistant.storage.db import Database

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = Database(db_path)

            # Insert two rows with same trait but different personality_id
            db.execute(
                "INSERT INTO personality_state "
                "(personality_id, trait, value, floor_val, ceiling_val, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                ("tsundere", "trust", 0.35, 0.10, 0.85, "2026-01-01"),
            )
            db.execute(
                "INSERT INTO personality_state "
                "(personality_id, trait, value, floor_val, ceiling_val, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                ("warm_honest", "trust", 0.55, 0.10, 0.95, "2026-01-01"),
            )
            db.commit()

            rows = db.fetchall(
                "SELECT personality_id, trait FROM personality_state "
                "WHERE trait = 'trust'"
            )
            self.assertEqual(len(rows), 2)

            # Duplicate (personality_id, trait) should fail
            with self.assertRaises(Exception):
                db.execute(
                    "INSERT INTO personality_state "
                    "(personality_id, trait, value, floor_val, ceiling_val, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    ("tsundere", "trust", 0.99, 0.10, 0.95, "2026-01-01"),
                )
            db.close()


class TestPersonalityLoader(unittest.TestCase):
    """Test PersonalityLoader loads/switches personalities correctly."""

    def test_load_each_builtin(self):
        """Each built-in personality loads prompt, traits, and config."""
        from assistant.personalities import PersonalityLoader

        for name in PersonalityLoader.BUILTIN:
            loader = PersonalityLoader(name)
            self.assertTrue(len(loader.get_prompt_base()) > 50)
            defaults = loader.get_trait_defaults()
            self.assertEqual(set(defaults.keys()),
                             {"trust", "warmth", "sass", "openness", "patience", "playfulness"})
            self.assertIn(loader.get_emotion_mode(), ("full", "neutral"))

    def test_warm_honest_has_modifiers(self):
        """warm_honest loads 6 traits x 3 tiers of modifiers."""
        from assistant.personalities import PersonalityLoader

        loader = PersonalityLoader("warm_honest")
        mods = loader.get_modifiers()
        self.assertEqual(set(mods.keys()),
                         {"trust", "warmth", "sass", "openness", "patience", "playfulness"})
        for trait_mods in mods.values():
            self.assertEqual(set(trait_mods.keys()), {"low", "mid", "high"})

    def test_minimal_has_no_modifiers(self):
        """minimal personality returns empty modifiers dict."""
        from assistant.personalities import PersonalityLoader

        loader = PersonalityLoader("minimal")
        self.assertEqual(loader.get_modifiers(), {})

    def test_invalid_personality_raises(self):
        """Unknown personality name raises ValueError."""
        from assistant.personalities import PersonalityLoader

        with self.assertRaises(ValueError):
            PersonalityLoader("nonexistent")

    def test_prompt_template_substitution(self):
        """Prompt text has {ASSISTANT_NAME} replaced with config value."""
        from assistant.personalities import PersonalityLoader
        from assistant import config

        loader = PersonalityLoader("warm_honest")
        prompt = loader.get_prompt_base()
        self.assertNotIn("{ASSISTANT_NAME}", prompt)
        self.assertIn(config.ASSISTANT_NAME_DISPLAY, prompt)

    def test_tsundere_emotion_mode_full(self):
        """tsundere has emotion_mode 'full'."""
        from assistant.personalities import PersonalityLoader

        loader = PersonalityLoader("tsundere")
        self.assertEqual(loader.get_emotion_mode(), "full")

    def test_warm_honest_emotion_mode_neutral(self):
        """warm_honest has emotion_mode 'neutral'."""
        from assistant.personalities import PersonalityLoader

        loader = PersonalityLoader("warm_honest")
        self.assertEqual(loader.get_emotion_mode(), "neutral")

    def test_reflection_hints_present(self):
        """All personalities have identity, drift_check, character_anchor."""
        from assistant.personalities import PersonalityLoader

        for name in PersonalityLoader.BUILTIN:
            loader = PersonalityLoader(name)
            hints = loader.get_reflection_hints()
            self.assertIn("identity", hints)
            self.assertIn("drift_check", hints)
            self.assertIn("character_anchor", hints)

    def test_responses_loaded(self):
        """Each personality has response pools with at least 'error' key."""
        from assistant.personalities import PersonalityLoader

        for name in PersonalityLoader.BUILTIN:
            loader = PersonalityLoader(name)
            responses = loader.get_responses()
            self.assertIn("error", responses)
            self.assertIsInstance(responses["error"], list)
            self.assertTrue(len(responses["error"]) >= 1)


_WH_DEFAULTS = {
    "trust": {"initial": 0.55, "floor": 0.10, "ceiling": 0.95},
    "warmth": {"initial": 0.70, "floor": 0.10, "ceiling": 0.90},
    "sass": {"initial": 0.30, "floor": 0.30, "ceiling": 0.95},
    "openness": {"initial": 0.50, "floor": 0.05, "ceiling": 0.85},
    "patience": {"initial": 0.70, "floor": 0.20, "ceiling": 0.90},
    "playfulness": {"initial": 0.55, "floor": 0.20, "ceiling": 0.95},
}


class TestPersonalitySwitching(unittest.TestCase):
    """Test per-personality trait isolation in the DB."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._db_path = Path(self._tmpdir.name) / "test.db"

        from assistant.storage.db import Database
        self._db = Database(self._db_path)

        from assistant.storage.repos.personality import PersonalityRepo
        self._repo = PersonalityRepo(self._db, personality_id="tsundere")

    def tearDown(self):
        self._db.close()
        self._tmpdir.cleanup()

    def test_traits_isolated_per_personality(self):
        """Different personality_ids have independent trait values."""
        from assistant.storage.repos.personality import PersonalityRepo

        ts_traits = self._repo.get_current_traits()
        self.assertIn("trust", ts_traits)
        self.assertAlmostEqual(ts_traits["trust"], 0.30)

        wh_repo = PersonalityRepo(
            self._db, personality_id="warm_honest",
            trait_defaults=_WH_DEFAULTS,
        )
        wh_traits = wh_repo.get_current_traits()
        self.assertAlmostEqual(wh_traits["trust"], 0.55)
        self.assertAlmostEqual(wh_traits["warmth"], 0.70)

    def test_update_traits_scoped(self):
        """update_traits only affects the repo's personality_id."""
        from assistant.storage.repos.personality import PersonalityRepo

        self._repo.update_traits({"trust": 0.02}, "test", trigger="event")
        ts_traits = self._repo.get_current_traits()

        wh_repo = PersonalityRepo(
            self._db, personality_id="warm_honest",
            trait_defaults=_WH_DEFAULTS,
        )
        wh_traits = wh_repo.get_current_traits()

        self.assertNotAlmostEqual(ts_traits["trust"], wh_traits["trust"])

    def test_constructor_seeds_with_personality_defaults(self):
        """Constructor uses trait_defaults, not hardcoded tsundere baselines."""
        from assistant.storage.repos.personality import PersonalityRepo, TRAIT_DEFAULTS

        wh_repo = PersonalityRepo(
            self._db, personality_id="warm_honest",
            trait_defaults=_WH_DEFAULTS,
        )
        wh_traits = wh_repo.get_current_traits()

        self.assertAlmostEqual(wh_traits["warmth"], 0.70)
        self.assertNotAlmostEqual(wh_traits["warmth"], TRAIT_DEFAULTS["warmth"]["initial"])

        self.assertAlmostEqual(wh_traits["sass"], 0.30)
        self.assertNotAlmostEqual(wh_traits["sass"], TRAIT_DEFAULTS["sass"]["initial"])

    def test_no_trait_defaults_falls_back_to_module_defaults(self):
        """Without trait_defaults kwarg, seeds with TRAIT_DEFAULTS."""
        from assistant.storage.repos.personality import PersonalityRepo, TRAIT_DEFAULTS

        repo = PersonalityRepo(self._db, personality_id="fallback_test")
        traits = repo.get_current_traits()
        self.assertAlmostEqual(traits["trust"], TRAIT_DEFAULTS["trust"]["initial"])

    def test_seed_does_not_overwrite_evolved_traits(self):
        """Re-creating repo doesn't reset traits that have evolved."""
        from assistant.storage.repos.personality import PersonalityRepo

        wh_repo = PersonalityRepo(
            self._db, personality_id="warm_honest",
            trait_defaults=_WH_DEFAULTS,
        )
        wh_repo.update_traits({"trust": 0.02}, "bonding", trigger="event")
        evolved_trust = wh_repo.get_current_traits()["trust"]
        self.assertNotAlmostEqual(evolved_trust, 0.55)

        wh_repo2 = PersonalityRepo(
            self._db, personality_id="warm_honest",
            trait_defaults=_WH_DEFAULTS,
        )
        self.assertAlmostEqual(wh_repo2.get_current_traits()["trust"], evolved_trust)


class TestSycophancyFilter(unittest.TestCase):
    """Test opener-strip regex catches sycophantic phrases."""

    def test_strips_great_question(self):
        from assistant.personalities.sycophancy import strip_sycophantic_opener
        self.assertEqual(
            strip_sycophantic_opener("Great question! Here's the answer."),
            "Here's the answer.",
        )

    def test_strips_thats_brilliant(self):
        from assistant.personalities.sycophancy import strip_sycophantic_opener
        self.assertEqual(
            strip_sycophantic_opener("That's brilliant! Let me explain."),
            "Let me explain.",
        )

    def test_strips_absolutely(self):
        from assistant.personalities.sycophancy import strip_sycophantic_opener
        self.assertEqual(
            strip_sycophantic_opener("Absolutely! I can do that."),
            "I can do that.",
        )

    def test_no_false_positive_legitimate(self):
        from assistant.personalities.sycophancy import strip_sycophantic_opener
        self.assertEqual(
            strip_sycophantic_opener("Here's what I found."),
            "Here's what I found.",
        )

    def test_no_false_positive_mid_sentence(self):
        from assistant.personalities.sycophancy import strip_sycophantic_opener
        self.assertEqual(
            strip_sycophantic_opener("That's a great restaurant downtown."),
            "That's a great restaurant downtown.",
        )

    def test_strips_after_emotion_tag(self):
        from assistant.personalities.sycophancy import strip_sycophantic_opener
        self.assertEqual(
            strip_sycophantic_opener("[happy] Great question! Here's the answer."),
            "[happy] Here's the answer.",
        )

    def test_capitalizes_remaining(self):
        from assistant.personalities.sycophancy import strip_sycophantic_opener
        result = strip_sycophantic_opener("That's a really good point. let me think.")
        self.assertTrue(result[0].isupper() or result[0] == "[")


class TestPromptBuilding(unittest.TestCase):
    """Test personality-aware prompt construction."""

    def test_tsundere_has_emotion_tag_rule(self):
        """tsundere personality includes emotion tag instruction."""
        from assistant.llm.prompts import _build_personality_rules
        rules = _build_personality_rules("full")
        self.assertIn("emotion tag", rules.lower())

    def test_warm_honest_no_emotion_tag_rule(self):
        """warm_honest personality does NOT include emotion tag instruction."""
        from assistant.llm.prompts import _build_personality_rules
        rules = _build_personality_rules("neutral")
        self.assertNotIn("emotion tag", rules.lower())

    def test_minimal_no_modifiers_injected(self):
        """minimal personality injects no trait modifiers."""
        from assistant.personalities import PersonalityLoader
        loader = PersonalityLoader("minimal")
        self.assertEqual(loader.get_modifiers(), {})

    def test_get_system_prompt_returns_string(self):
        """get_system_prompt() returns a non-empty string."""
        from assistant.llm.prompts import get_system_prompt
        prompt = get_system_prompt()
        self.assertIsInstance(prompt, str)
        self.assertTrue(len(prompt) > 50)


class TestReflection(unittest.TestCase):
    """Test personality-aware reflection prompts."""

    def test_warm_honest_drift_check(self):
        """warm_honest reflection includes sycophancy drift check."""
        from assistant.personalities import PersonalityLoader

        loader = PersonalityLoader("warm_honest")
        hints = loader.get_reflection_hints()
        self.assertIn("flatter", hints["drift_check"].lower())

    def test_tsundere_drift_check(self):
        """tsundere reflection includes character-break check."""
        from assistant.personalities import PersonalityLoader

        loader = PersonalityLoader("tsundere")
        hints = loader.get_reflection_hints()
        self.assertIn("break character", hints["drift_check"].lower())

    def test_no_hardcoded_brand_in_reflection_prompt(self):
        """Reflection prompt must not hardcode a specific persona brand."""
        from assistant.reflection import _build_reflection_prompt

        prompt = _build_reflection_prompt("{}", "no interactions", "[]")
        self.assertNotIn("Miku", prompt)
        self.assertNotIn("tsundere anime", prompt)


class TestPersonalitySettingSwitching(unittest.TestCase):
    """Test /set personality switching end-to-end."""

    def test_personality_in_runtime_registry(self):
        """personality setting is registered in RUNTIME_SETTINGS_REGISTRY."""
        from assistant import config
        self.assertIn("personality", config.RUNTIME_SETTINGS_REGISTRY)

    def test_personality_setting_default(self):
        """Default personality setting is warm_honest."""
        from assistant import config
        meta = config.RUNTIME_SETTINGS_REGISTRY["personality"]
        self.assertEqual(meta["default"], "warm_honest")


class TestWellbeingSafeguard(unittest.TestCase):
    """Test wellbeing check-in constants."""

    def test_wellbeing_checkin_text(self):
        """Wellbeing check-in variants exist and are under 120 chars."""
        from assistant.main import _WELLBEING_CHECKINS
        self.assertTrue(len(_WELLBEING_CHECKINS) >= 3)
        for text in _WELLBEING_CHECKINS:
            self.assertLess(len(text), 120)


class TestEmotionModeGating(unittest.TestCase):
    """Test emotion mode config from personality loader."""

    def test_neutral_mode_for_warm_honest(self):
        """warm_honest has emotion_mode neutral."""
        from assistant.personalities import PersonalityLoader
        loader = PersonalityLoader("warm_honest")
        self.assertEqual(loader.get_emotion_mode(), "neutral")

    def test_full_mode_for_tsundere(self):
        """tsundere has emotion_mode full."""
        from assistant.personalities import PersonalityLoader
        loader = PersonalityLoader("tsundere")
        self.assertEqual(loader.get_emotion_mode(), "full")


class TestStartupMigration(unittest.TestCase):
    """Test existing-user vs new-user personality detection."""

    def test_existing_user_gets_tsundere(self):
        """Pre-P1 user with evolved traits defaults to tsundere."""
        from assistant.storage.db import Database

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            # Build a real v8 DB via the migration chain (clamped to v8) and
            # insert a pre-P1 trait row, then reopen at LATEST so v9 migrates.
            original_latest = Database._LATEST_VERSION
            Database._LATEST_VERSION = 8
            try:
                v8 = Database(db_path)
                try:
                    v8._conn.execute(
                        "INSERT INTO personality_state "
                        "(trait, value, floor_val, ceiling_val, updated_at) "
                        "VALUES ('trust', 0.35, 0.10, 0.85, '2026-01-01')"
                    )
                    v8._conn.commit()
                finally:
                    v8.close()
            finally:
                Database._LATEST_VERSION = original_latest

            db = Database(db_path)
            try:
                row = db.fetchone(
                    "SELECT COUNT(*) AS cnt FROM personality_state WHERE personality_id = 'tsundere'"
                )
                self.assertGreater(row["cnt"], 0)
            finally:
                db.close()


class TestSwitchPersistence(unittest.TestCase):
    """Test that switch_personality persists the choice to the metadata table."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._db_path = Path(self._tmpdir.name) / "test.db"

        import assistant.storage.db as db_mod
        self._old_instance = db_mod._instance
        db_mod._instance = None

        from assistant.storage.db import init_db
        self._db = init_db(self._db_path)

        from assistant.storage.repos.personality import PersonalityRepo
        import assistant.personality as p_mod
        self._old_repo = p_mod._repo
        p_mod._repo = PersonalityRepo(self._db, personality_id="tsundere")

    def tearDown(self):
        import assistant.storage.db as db_mod
        import assistant.personality as p_mod
        p_mod._repo = self._old_repo
        self._db.close()
        db_mod._instance = self._old_instance
        self._tmpdir.cleanup()

    def test_switch_updates_metadata(self):
        from assistant.personality import switch_personality

        self._db.execute(
            "INSERT INTO metadata (key, value, updated_at) "
            "VALUES ('active_personality', 'tsundere', '2026-01-01')"
        )
        self._db.commit()

        result = switch_personality("warm_honest")
        self.assertIn("warm-honest", result)

        row = self._db.fetchone(
            "SELECT value FROM metadata WHERE key = 'active_personality'"
        )
        self.assertEqual(row["value"], "warm_honest")

    def test_switch_resets_conversation_counter(self):
        """Switching personality resets conversation counter to prevent cross-contamination."""
        from assistant.personality import switch_personality, increment_conversation_count

        self._db.execute(
            "INSERT INTO metadata (key, value, updated_at) "
            "VALUES ('active_personality', 'tsundere', '2026-01-01')"
        )
        self._db.commit()

        for _ in range(15):
            increment_conversation_count()

        switch_personality("warm_honest")

        row = self._db.fetchone(
            "SELECT value FROM metadata WHERE key = 'conversation_count'"
        )
        self.assertEqual(row["value"], "0")


class TestTraitSeedingMigration(unittest.TestCase):
    """Test one-time migration fixes wrongly-seeded personalities."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._db_path = Path(self._tmpdir.name) / "test.db"

        from assistant.storage.db import Database
        self._db = Database(self._db_path)

    def tearDown(self):
        self._db.close()
        self._tmpdir.cleanup()

    def test_migrates_wrongly_seeded_warm_honest(self):
        """warm_honest seeded with TRAIT_DEFAULTS gets corrected."""
        from assistant.storage.repos.personality import TRAIT_DEFAULTS

        now = "2026-01-01"
        for trait, vals in TRAIT_DEFAULTS.items():
            self._db.execute(
                "INSERT INTO personality_state "
                "(personality_id, trait, value, floor_val, ceiling_val, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                ("warm_honest", trait, vals["initial"], vals["floor"], vals["ceiling"], now),
            )
        self._db.commit()

        from assistant.personality import _migrate_wrongly_seeded_traits
        _migrate_wrongly_seeded_traits(self._db)

        row = self._db.fetchone(
            "SELECT value FROM personality_state "
            "WHERE personality_id = 'warm_honest' AND trait = 'warmth'"
        )
        self.assertAlmostEqual(row["value"], 0.70)

        row = self._db.fetchone(
            "SELECT value FROM personality_state "
            "WHERE personality_id = 'warm_honest' AND trait = 'trust'"
        )
        self.assertAlmostEqual(row["value"], 0.55)

    def test_does_not_touch_tsundere(self):
        """tsundere rows are left alone even if they match TRAIT_DEFAULTS."""
        from assistant.storage.repos.personality import PersonalityRepo, TRAIT_DEFAULTS
        PersonalityRepo(self._db, personality_id="tsundere")

        from assistant.personality import _migrate_wrongly_seeded_traits
        _migrate_wrongly_seeded_traits(self._db)

        row = self._db.fetchone(
            "SELECT value FROM personality_state "
            "WHERE personality_id = 'tsundere' AND trait = 'warmth'"
        )
        self.assertAlmostEqual(row["value"], TRAIT_DEFAULTS["warmth"]["initial"])

    def test_migration_runs_only_once(self):
        """Second call is a no-op (flag prevents re-run)."""
        from assistant.personality import _migrate_wrongly_seeded_traits
        _migrate_wrongly_seeded_traits(self._db)

        flag = self._db.fetchone(
            "SELECT value FROM metadata WHERE key = 'trait_seeding_v2'"
        )
        self.assertEqual(flag["value"], "done")

        _migrate_wrongly_seeded_traits(self._db)

    def test_does_not_touch_evolved_traits(self):
        """If warm_honest traits have evolved away from TRAIT_DEFAULTS, skip."""
        from assistant.storage.repos.personality import TRAIT_DEFAULTS

        now = "2026-01-01"
        for trait, vals in TRAIT_DEFAULTS.items():
            value = vals["initial"] + 0.05
            self._db.execute(
                "INSERT INTO personality_state "
                "(personality_id, trait, value, floor_val, ceiling_val, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                ("warm_honest", trait, value, vals["floor"], vals["ceiling"], now),
            )
        self._db.commit()

        from assistant.personality import _migrate_wrongly_seeded_traits
        _migrate_wrongly_seeded_traits(self._db)

        row = self._db.fetchone(
            "SELECT value FROM personality_state "
            "WHERE personality_id = 'warm_honest' AND trait = 'warmth'"
        )
        self.assertAlmostEqual(row["value"], TRAIT_DEFAULTS["warmth"]["initial"] + 0.05)


class TestWarmHonestTuning(unittest.TestCase):
    """Test warm_honest tier tuning migration."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._db_path = Path(self._tmpdir.name) / "test.db"
        from assistant.storage.db import Database
        self._db = Database(self._db_path)

    def tearDown(self):
        self._db.close()
        self._tmpdir.cleanup()

    def test_bumps_warmth_to_high_tier(self):
        """warmth 0.65 → 0.70, sass 0.35 → 0.30."""
        now = "2026-01-01"
        for trait, val in [("warmth", 0.65), ("sass", 0.35)]:
            self._db.execute(
                "INSERT INTO personality_state "
                "(personality_id, trait, value, floor_val, ceiling_val, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                ("warm_honest", trait, val, 0.10, 0.95, now),
            )
        self._db.commit()

        from assistant.personality import _tune_warm_honest_tiers
        _tune_warm_honest_tiers(self._db)

        warmth = self._db.fetchone(
            "SELECT value FROM personality_state "
            "WHERE personality_id = 'warm_honest' AND trait = 'warmth'"
        )
        self.assertAlmostEqual(warmth["value"], 0.70)

        sass = self._db.fetchone(
            "SELECT value FROM personality_state "
            "WHERE personality_id = 'warm_honest' AND trait = 'sass'"
        )
        self.assertAlmostEqual(sass["value"], 0.30)

    def test_skips_if_already_tuned(self):
        """Doesn't overwrite if warmth is already at 0.70."""
        now = "2026-01-01"
        self._db.execute(
            "INSERT INTO personality_state "
            "(personality_id, trait, value, floor_val, ceiling_val, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("warm_honest", "warmth", 0.70, 0.10, 0.90, now),
        )
        self._db.commit()

        from assistant.personality import _tune_warm_honest_tiers
        _tune_warm_honest_tiers(self._db)

        flag = self._db.fetchone(
            "SELECT value FROM metadata WHERE key = 'wh_tuning_v1'"
        )
        self.assertEqual(flag["value"], "done")


class TestSwitchFlag(unittest.TestCase):
    """Test personality switch flag for context bleeding prevention."""

    def test_switch_flag_set_on_switch(self):
        from assistant.personalities import (
            set_active_personality, consume_switch_flag,
        )
        set_active_personality("tsundere")
        self.assertTrue(consume_switch_flag())

    def test_switch_flag_consumed_once(self):
        from assistant.personalities import (
            set_active_personality, consume_switch_flag,
        )
        set_active_personality("minimal")
        self.assertTrue(consume_switch_flag())
        self.assertFalse(consume_switch_flag())


class TestFeatureFlags(unittest.TestCase):
    """Test personality feature flags are data-driven."""

    def test_warm_honest_has_sycophancy_filter(self):
        from assistant.personalities import PersonalityLoader
        loader = PersonalityLoader("warm_honest")
        flags = loader.get_feature_flags()
        self.assertTrue(flags["sycophancy_filter"])
        self.assertTrue(flags["wellbeing_checkin"])

    def test_tsundere_no_sycophancy_filter(self):
        from assistant.personalities import PersonalityLoader
        loader = PersonalityLoader("tsundere")
        flags = loader.get_feature_flags()
        self.assertFalse(flags["sycophancy_filter"])
        self.assertFalse(flags["wellbeing_checkin"])

    def test_minimal_no_sycophancy_filter(self):
        from assistant.personalities import PersonalityLoader
        loader = PersonalityLoader("minimal")
        flags = loader.get_feature_flags()
        self.assertFalse(flags["sycophancy_filter"])
        self.assertFalse(flags["wellbeing_checkin"])


class TestResponseAutoKwargs(unittest.TestCase):
    """Test personality_say auto-injects assistant_name_lower."""

    def test_assistant_name_lower_injected(self):
        from unittest.mock import patch
        from assistant.actions.responses import personality_say

        mock_responses = {"face_need_name": [
            "Say '{assistant_name_lower} this is' and your name."
        ]}

        # get_active_loader is imported lazily inside personality_say, so the
        # name doesn't exist on assistant.actions.responses — patch the source.
        with patch("assistant.personalities.get_active_loader") as mock_loader:
            mock_loader.return_value.get_responses.return_value = mock_responses
            result = personality_say("face_need_name")

        from assistant import config
        self.assertIn(config.ASSISTANT_NAME_LOWER, result)
        self.assertNotIn("{assistant_name_lower}", result)


if __name__ == "__main__":
    unittest.main()
