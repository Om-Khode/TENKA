"""
test_schema_versioning.py — Tests for P4+P9 schema versioning.

Covers: faces.py, knowledge.py, credentials.py, code_executor/templates.py.
All on-disk artifacts must use versioned envelopes.

Run: python -m pytest tests/test_schema_versioning.py -v
"""

import sys
import os
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# --- faces.py schema versioning ---

class TestFacesVersioning:

    def _make_faces_dir(self, tmp_path):
        faces_dir = tmp_path / "faces"
        faces_dir.mkdir()
        return faces_dir

    def test_save_writes_versioned_envelope(self, tmp_path):
        faces_dir = self._make_faces_dir(tmp_path)
        with patch("assistant.faces.get_faces_dir", return_value=faces_dir):
            from assistant.faces import save_encodings
            save_encodings([{"name": "Alice", "encodings": [[1.0, 2.0]]}])

        raw = json.loads((faces_dir / "encodings.json").read_text())
        assert "version" in raw
        assert raw["version"] == 1
        assert isinstance(raw["data"], list)
        assert raw["data"][0]["name"] == "Alice"

    def test_load_reads_versioned_format(self, tmp_path):
        faces_dir = self._make_faces_dir(tmp_path)
        enc_path = faces_dir / "encodings.json"
        enc_path.write_text(json.dumps({
            "version": 1,
            "data": [{"name": "Bob", "encodings": [[3.0]], "added": "2026-01-01", "updated": "2026-01-01"}]
        }))

        with patch("assistant.faces.get_faces_dir", return_value=faces_dir):
            from assistant.faces import load_encodings
            entries = load_encodings()

        assert len(entries) == 1
        assert entries[0]["name"] == "Bob"

    def test_load_migrates_bare_list(self, tmp_path):
        """Legacy bare list format should auto-migrate to versioned."""
        faces_dir = self._make_faces_dir(tmp_path)
        enc_path = faces_dir / "encodings.json"
        enc_path.write_text(json.dumps([
            {"name": "Charlie", "encodings": [[1.0]], "added": "2026-01-01", "updated": "2026-01-01"}
        ]))

        with patch("assistant.faces.get_faces_dir", return_value=faces_dir):
            from assistant.faces import load_encodings
            entries = load_encodings()

        assert len(entries) == 1
        assert entries[0]["name"] == "Charlie"

        # File should now be versioned
        raw = json.loads(enc_path.read_text())
        assert raw["version"] == 1
        assert isinstance(raw["data"], list)

    def test_load_migrates_old_single_encoding(self, tmp_path):
        """Legacy {encoding: []} format should migrate to {encodings: [[]]}."""
        faces_dir = self._make_faces_dir(tmp_path)
        enc_path = faces_dir / "encodings.json"
        enc_path.write_text(json.dumps([
            {"name": "Dan", "encoding": [1.0, 2.0], "added": "2026-01-01"}
        ]))

        with patch("assistant.faces.get_faces_dir", return_value=faces_dir):
            from assistant.faces import load_encodings
            entries = load_encodings()

        assert entries[0]["encodings"] == [[1.0, 2.0]]
        assert "encoding" not in entries[0]


# --- knowledge.py schema versioning ---

class TestKnowledgeVersioning:

    def test_save_writes_versioned_envelope(self, tmp_path):
        with patch("assistant.knowledge._knowledge_dir", return_value=tmp_path):
            from assistant.knowledge import _save_entries
            _save_entries("testservice", [{"type": "works", "pattern": "x"}])

        raw = json.loads((tmp_path / "testservice.json").read_text())
        assert raw["version"] == 1
        assert isinstance(raw["data"], list)
        assert raw["data"][0]["type"] == "works"

    def test_load_reads_versioned_format(self, tmp_path):
        path = tmp_path / "testservice.json"
        path.write_text(json.dumps({
            "version": 1,
            "data": [{"type": "never", "slug": "s", "pattern": "p", "reason": "r"}]
        }))

        with patch("assistant.knowledge._knowledge_dir", return_value=tmp_path):
            from assistant.knowledge import _load_entries
            entries = _load_entries("testservice")

        assert len(entries) == 1
        assert entries[0]["type"] == "never"

    def test_load_migrates_bare_list(self, tmp_path):
        """Legacy bare list format should auto-migrate."""
        path = tmp_path / "testservice.json"
        path.write_text(json.dumps([{"type": "works", "pattern": "x"}]))

        with patch("assistant.knowledge._knowledge_dir", return_value=tmp_path):
            from assistant.knowledge import _load_entries
            entries = _load_entries("testservice")

        assert len(entries) == 1

        # File should now be versioned
        raw = json.loads(path.read_text())
        assert raw["version"] == 1

    def test_load_migrates_legacy_dict_format(self, tmp_path):
        """Legacy {"service": ..., "entries": [...]} format should migrate."""
        path = tmp_path / "testservice.json"
        path.write_text(json.dumps({
            "service": "testservice",
            "entries": [{"type": "never", "pattern": "y"}]
        }))

        with patch("assistant.knowledge._knowledge_dir", return_value=tmp_path):
            from assistant.knowledge import _load_entries
            entries = _load_entries("testservice")

        assert len(entries) == 1
        assert entries[0]["pattern"] == "y"

        raw = json.loads(path.read_text())
        assert raw["version"] == 1


# --- credentials.py schema versioning ---

class TestCredentialsVersioning:

    def test_save_raw_writes_versioned(self, tmp_path):
        from assistant.credentials import _save_raw
        path = tmp_path / "test.json"
        _save_raw(path, {"client_id": "encrypted_value"})

        raw = json.loads(path.read_text())
        assert raw["version"] == 1
        assert raw["data"]["client_id"] == "encrypted_value"

    def test_load_raw_reads_versioned(self, tmp_path):
        from assistant.credentials import _load_raw
        path = tmp_path / "test.json"
        path.write_text(json.dumps({"version": 1, "data": {"key": "val"}}))

        data = _load_raw(path)
        assert data == {"key": "val"}

    def test_load_raw_reads_legacy_bare_dict(self, tmp_path):
        """Legacy bare dict (no version) should be loaded correctly."""
        from assistant.credentials import _load_raw
        path = tmp_path / "test.json"
        path.write_text(json.dumps({"client_id": "old_encrypted"}))

        data = _load_raw(path)
        assert data == {"client_id": "old_encrypted"}

    def test_load_raw_nonexistent_returns_empty(self, tmp_path):
        from assistant.credentials import _load_raw
        path = tmp_path / "nonexistent.json"
        assert _load_raw(path) == {}

    def test_set_credential_migrates_on_write(self, tmp_path):
        """Writing a credential to a legacy file should upgrade to versioned."""
        svc_path = tmp_path / "svc.json"
        svc_path.write_text(json.dumps({"old_key": "old_val"}))

        with patch("assistant.credentials._service_path", return_value=svc_path), \
             patch("assistant.credentials._encrypt", return_value="enc_new"):
            from assistant.credentials import set_credential
            set_credential("svc", "new_key", "plaintext")

        raw = json.loads(svc_path.read_text())
        assert raw["version"] == 1
        assert raw["data"]["old_key"] == "old_val"
        assert raw["data"]["new_key"] == "enc_new"


# --- code_executor/templates.py schema versioning ---

class TestTemplateVersioning:

    def test_save_writes_version_header(self, tmp_path):
        with patch("assistant.code_executor.templates._templates_dir", return_value=tmp_path):
            from assistant.code_executor.templates import _save_template
            _save_template("test_slug", "print('hello')", goal="say hello")

        content = (tmp_path / "test_slug.py").read_text(encoding="utf-8")
        assert content.startswith("# version: 1\n")
        assert "# GOAL: say hello" in content
        assert "print('hello')" in content

    def test_save_without_goal(self, tmp_path):
        with patch("assistant.code_executor.templates._templates_dir", return_value=tmp_path):
            from assistant.code_executor.templates import _save_template
            _save_template("no_goal", "x = 1")

        content = (tmp_path / "no_goal.py").read_text(encoding="utf-8")
        assert content.startswith("# version: 1\n")
        assert "GOAL" not in content

    def test_load_reads_versioned_template(self, tmp_path):
        path = tmp_path / "versioned.py"
        path.write_text("# version: 1\n# GOAL: test goal\nprint('hi')", encoding="utf-8")

        with patch("assistant.code_executor.templates._templates_dir", return_value=tmp_path):
            from assistant.code_executor.templates import _load_template
            code, goal = _load_template("versioned")

        assert code == "print('hi')"
        assert goal == "test goal"

    def test_load_reads_legacy_template(self, tmp_path):
        """Legacy templates without version header should still load."""
        path = tmp_path / "legacy.py"
        path.write_text("# GOAL: old goal\nprint('legacy')", encoding="utf-8")

        with patch("assistant.code_executor.templates._templates_dir", return_value=tmp_path):
            from assistant.code_executor.templates import _load_template
            code, goal = _load_template("legacy")

        assert code == "print('legacy')"
        assert goal == "old goal"

    def test_load_reads_bare_legacy_template(self, tmp_path):
        """Very old templates with no headers at all should still load."""
        path = tmp_path / "bare.py"
        path.write_text("x = 42", encoding="utf-8")

        with patch("assistant.code_executor.templates._templates_dir", return_value=tmp_path):
            from assistant.code_executor.templates import _load_template
            code, goal = _load_template("bare")

        assert code == "x = 42"
        assert goal == ""

    def test_roundtrip_save_load(self, tmp_path):
        with patch("assistant.code_executor.templates._templates_dir", return_value=tmp_path):
            from assistant.code_executor.templates import _save_template, _load_template
            _save_template("rt", "result = 2 + 2", goal="add numbers")
            code, goal = _load_template("rt")

        assert code == "result = 2 + 2"
        assert goal == "add numbers"

    def test_legacy_fallback_finds_service_template(self, tmp_path):
        """music_play should find spotify_play_music.py via category→service fallback."""
        (tmp_path / "spotify_play_music.py").write_text(
            "# version: 1\n# GOAL: play music\nsp.start_playback()", encoding="utf-8"
        )

        with patch("assistant.code_executor.templates._templates_dir", return_value=tmp_path):
            from assistant.code_executor.templates import _load_template
            code, goal = _load_template("music_play")

        assert code is not None
        assert "start_playback" in code
        assert goal == "play music"

    def test_legacy_fallback_exact_action_match(self, tmp_path):
        """music_pause should find spotify_pause.py (exact action match)."""
        (tmp_path / "spotify_pause.py").write_text(
            "# version: 1\n# GOAL: pause music\nsp.pause_playback()", encoding="utf-8"
        )

        with patch("assistant.code_executor.templates._templates_dir", return_value=tmp_path):
            from assistant.code_executor.templates import _load_template
            code, goal = _load_template("music_pause")

        assert code is not None
        assert "pause_playback" in code

    def test_legacy_fallback_prefers_exact_over_glob(self, tmp_path):
        """When both spotify_skip.py and spotify_skip_track.py exist, exact wins."""
        (tmp_path / "spotify_skip.py").write_text("exact", encoding="utf-8")
        (tmp_path / "spotify_skip_track.py").write_text("glob", encoding="utf-8")

        with patch("assistant.code_executor.templates._templates_dir", return_value=tmp_path):
            from assistant.code_executor.templates import _load_template
            code, _ = _load_template("music_skip")

        assert code == "exact"

    def test_legacy_fallback_no_match_returns_none(self, tmp_path):
        """No legacy template → returns (None, '')."""
        with patch("assistant.code_executor.templates._templates_dir", return_value=tmp_path):
            from assistant.code_executor.templates import _load_template
            code, goal = _load_template("music_nonexistent")

        assert code is None
        assert goal == ""

    def test_direct_slug_preferred_over_legacy(self, tmp_path):
        """Direct music_play.py is used even when spotify_play.py also exists."""
        (tmp_path / "music_play.py").write_text("direct", encoding="utf-8")
        (tmp_path / "spotify_play.py").write_text("legacy", encoding="utf-8")

        with patch("assistant.code_executor.templates._templates_dir", return_value=tmp_path):
            from assistant.code_executor.templates import _load_template
            code, _ = _load_template("music_play")

        assert code == "direct"


# --- Source-level checks: every module has _SCHEMA_VERSION ---

class TestSchemaVersionConstants:

    def test_faces_has_schema_version(self):
        from assistant.faces import _SCHEMA_VERSION
        assert isinstance(_SCHEMA_VERSION, int)
        assert _SCHEMA_VERSION >= 1

    def test_knowledge_has_schema_version(self):
        from assistant.knowledge import _SCHEMA_VERSION
        assert isinstance(_SCHEMA_VERSION, int)
        assert _SCHEMA_VERSION >= 1

    def test_credentials_has_schema_version(self):
        from assistant.credentials import _SCHEMA_VERSION
        assert isinstance(_SCHEMA_VERSION, int)
        assert _SCHEMA_VERSION >= 1

    def test_templates_has_schema_version(self):
        from assistant.code_executor.templates import _SCHEMA_VERSION
        assert isinstance(_SCHEMA_VERSION, int)
        assert _SCHEMA_VERSION >= 1
