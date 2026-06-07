"""Tests for the PR4 setup wizard (`scripts/setup.py`).

The wizard is stdlib-only and runs before deps are installed, so the
tests must also avoid pulling in any third-party imports from inside
setup.py at import time. They do.

We mock `subprocess.run`, `input`, and use a tmp_path .env / marker so
the real working tree is untouched.
"""
import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest


# ─── Load scripts/setup.py as a module ──────────────────────────────
_SETUP_PATH = Path(__file__).resolve().parent.parent / "scripts" / "setup.py"


@pytest.fixture(scope="module")
def setup_mod():
    spec = importlib.util.spec_from_file_location("tenka_setup_wizard", _SETUP_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ─── Helpers ────────────────────────────────────────────────────────
def _redirect_paths(setup_mod, tmp_path, monkeypatch):
    """Point the wizard's module-level paths at tmp_path so writes are sandboxed."""
    monkeypatch.setattr(setup_mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(setup_mod, "ENV_PATH", tmp_path / ".env")
    monkeypatch.setattr(setup_mod, "ENV_EXAMPLE_PATH", tmp_path / ".env.example")
    monkeypatch.setattr(setup_mod, "MARKER_PATH", tmp_path / ".tenka_setup.json")
    monkeypatch.setattr(setup_mod, "REQUIREMENTS_PATH", tmp_path / "requirements.txt")


# ─── 1. Python version check ────────────────────────────────────────
def test_python_version_passes_on_311(setup_mod, monkeypatch):
    monkeypatch.setattr(setup_mod.sys, "version_info",
                        SimpleNamespace(major=3, minor=11, micro=9))
    args = SimpleNamespace(force=False, no_launch=True)
    marker: dict = {}
    setup_mod.step_python_version(marker, args)
    assert marker["steps"]["python_version"]["ok"] is True


def test_python_version_fails_below_min(setup_mod, monkeypatch):
    monkeypatch.setattr(setup_mod.sys, "version_info",
                        SimpleNamespace(major=3, minor=10, micro=0))
    args = SimpleNamespace(force=False, no_launch=True)
    with pytest.raises(SystemExit) as exc:
        setup_mod.step_python_version({}, args)
    assert exc.value.code == 2


def test_python_version_warns_above_max_but_continues(setup_mod, monkeypatch, capsys):
    monkeypatch.setattr(setup_mod.sys, "version_info",
                        SimpleNamespace(major=3, minor=13, micro=0))
    args = SimpleNamespace(force=False, no_launch=True)
    marker: dict = {}
    setup_mod.step_python_version(marker, args)
    out = capsys.readouterr().out
    assert "newer than the tested version" in out
    assert marker["steps"]["python_version"]["ok"] is True


# ─── 2. .env merge: preserves existing, writes new, no clobber ──────
def test_env_merge_preserves_existing_value_when_no_update(setup_mod):
    existing = "GROQ_API_KEY=oldkey\nGEMINI_API_KEY=stayme\n"
    out = setup_mod.merge_env_text(existing, {})
    assert "GROQ_API_KEY=oldkey" in out
    assert "GEMINI_API_KEY=stayme" in out


def test_env_merge_replaces_value_in_place(setup_mod):
    existing = "# my header\nGROQ_API_KEY=oldkey\n# trailing\n"
    out = setup_mod.merge_env_text(existing, {"GROQ_API_KEY": "NEW"})
    lines = out.splitlines()
    # The comment lines and order must be preserved
    assert lines[0] == "# my header"
    assert "GROQ_API_KEY=NEW" in lines
    assert "GROQ_API_KEY=oldkey" not in out
    assert "# trailing" in out


def test_env_merge_appends_missing_keys_under_wizard_header(setup_mod):
    existing = "GROQ_API_KEY=oldkey\n"
    out = setup_mod.merge_env_text(existing, {"NEWKEY": "v1"})
    assert "GROQ_API_KEY=oldkey" in out  # untouched
    assert "Added by setup wizard" in out
    assert "NEWKEY=v1" in out


def test_env_merge_uses_template_when_existing_is_empty(setup_mod):
    template = "# header\nGEMINI_API_KEY=\nGROQ_API_KEY=\n"
    out = setup_mod.merge_env_text("", {"GEMINI_API_KEY": "abc"}, template)
    # The placeholder line is replaced in-place, not appended
    assert "GEMINI_API_KEY=abc" in out
    assert out.count("GEMINI_API_KEY=") == 1
    assert "Added by setup wizard" not in out


def test_env_merge_round_trips_through_parse(setup_mod):
    existing = "GROQ_API_KEY=k1\nGEMINI_API_KEY=k2\n"
    parsed = setup_mod.parse_env(existing)
    assert parsed == {"GROQ_API_KEY": "k1", "GEMINI_API_KEY": "k2"}


# ─── 3. Atomic write ────────────────────────────────────────────────
def test_atomic_write_creates_file(setup_mod, tmp_path):
    target = tmp_path / "out.txt"
    setup_mod._atomic_write(target, "hello\n")
    assert target.read_text(encoding="utf-8") == "hello\n"


def test_atomic_write_leaves_no_tmp_artifacts(setup_mod, tmp_path):
    target = tmp_path / "out.txt"
    setup_mod._atomic_write(target, "hello\n")
    leftovers = [p.name for p in tmp_path.iterdir() if p.name.startswith(".out.txt.")]
    assert leftovers == [], f"temp file leaked: {leftovers}"


def test_atomic_write_does_not_corrupt_existing_on_failure(setup_mod, tmp_path):
    target = tmp_path / "out.txt"
    target.write_text("ORIGINAL", encoding="utf-8")
    with patch("os.replace", side_effect=OSError("boom")):
        with pytest.raises(OSError):
            setup_mod._atomic_write(target, "NEW")
    assert target.read_text(encoding="utf-8") == "ORIGINAL"


# ─── 4. Marker is schema-versioned + idempotent ─────────────────────
def test_marker_round_trip_is_versioned(setup_mod, tmp_path, monkeypatch):
    _redirect_paths(setup_mod, tmp_path, monkeypatch)
    setup_mod.save_marker({"steps": {"x": {"ok": True}}})
    data = json.loads((tmp_path / ".tenka_setup.json").read_text(encoding="utf-8"))
    assert data["version"] == setup_mod.MARKER_SCHEMA_VERSION
    assert "updated" in data
    assert data["steps"]["x"]["ok"] is True


def test_load_marker_rejects_wrong_schema_version(setup_mod, tmp_path, monkeypatch):
    _redirect_paths(setup_mod, tmp_path, monkeypatch)
    (tmp_path / ".tenka_setup.json").write_text(
        json.dumps({"version": 999, "steps": {"x": {"ok": True}}}), encoding="utf-8",
    )
    assert setup_mod.load_marker() == {}


def test_load_marker_returns_empty_on_corrupt_json(setup_mod, tmp_path, monkeypatch):
    _redirect_paths(setup_mod, tmp_path, monkeypatch)
    (tmp_path / ".tenka_setup.json").write_text("not json", encoding="utf-8")
    assert setup_mod.load_marker() == {}


def test_pip_install_skipped_when_marker_matches_requirements_hash(
    setup_mod, tmp_path, monkeypatch
):
    _redirect_paths(setup_mod, tmp_path, monkeypatch)
    (tmp_path / "requirements.txt").write_text("playwright\n", encoding="utf-8")
    req_sha = setup_mod._file_sha(tmp_path / "requirements.txt")
    marker = {"steps": {"pip_install": {"requirements_sha": req_sha, "ok": True}}}
    args = SimpleNamespace(force=False, no_launch=True)
    calls = []
    monkeypatch.setattr(setup_mod.subprocess, "run",
                        lambda *a, **kw: calls.append(a) or SimpleNamespace(returncode=0))
    setup_mod.step_pip_install(marker, args)
    assert calls == [], "pip install ran despite up-to-date marker"


def test_pip_install_runs_when_requirements_hash_changes(
    setup_mod, tmp_path, monkeypatch
):
    _redirect_paths(setup_mod, tmp_path, monkeypatch)
    (tmp_path / "requirements.txt").write_text("playwright\n", encoding="utf-8")
    marker = {"steps": {"pip_install": {"requirements_sha": "STALE", "ok": True}}}
    args = SimpleNamespace(force=False, no_launch=True)
    calls = []
    monkeypatch.setattr(setup_mod.subprocess, "run",
                        lambda *a, **kw: calls.append(a) or SimpleNamespace(returncode=0))
    setup_mod.step_pip_install(marker, args)
    assert len(calls) == 1
    assert marker["steps"]["pip_install"]["ok"] is True


def test_force_flag_reruns_completed_pip_install(
    setup_mod, tmp_path, monkeypatch
):
    _redirect_paths(setup_mod, tmp_path, monkeypatch)
    (tmp_path / "requirements.txt").write_text("playwright\n", encoding="utf-8")
    req_sha = setup_mod._file_sha(tmp_path / "requirements.txt")
    marker = {"steps": {"pip_install": {"requirements_sha": req_sha, "ok": True}}}
    args = SimpleNamespace(force=True, no_launch=True)
    calls = []
    monkeypatch.setattr(setup_mod.subprocess, "run",
                        lambda *a, **kw: calls.append(a) or SimpleNamespace(returncode=0))
    setup_mod.step_pip_install(marker, args)
    assert len(calls) == 1, "--force did not re-run pip install"


# ─── 5. THE rule: PROVIDERS is pure data, no app-specific branches ──
def test_providers_is_data_driven(setup_mod):
    assert isinstance(setup_mod.PROVIDERS, list) and len(setup_mod.PROVIDERS) >= 2
    required_fields = {"key", "name", "url", "tier", "blurb"}
    for p in setup_mod.PROVIDERS:
        assert required_fields.issubset(p.keys()), f"missing fields in {p}"
        assert p["tier"] in ("primary", "fallback", "optional")
        # Conventional env var name: uppercase + underscores. We allow any
        # such name (Groq has `_API_KEY`; HF uses `_TOKEN`; others may differ).
        assert re.fullmatch(r"[A-Z][A-Z0-9_]*", p["key"]), f"bad env key: {p['key']}"


def test_gemini_is_listed_as_primary(setup_mod):
    """Regression: the old .env.example was missing GEMINI_API_KEY entirely
    even though Gemini is the primary LLM. PR4 fixes this."""
    gemini = [p for p in setup_mod.PROVIDERS if p["key"] == "GEMINI_API_KEY"]
    assert len(gemini) == 1
    assert gemini[0]["tier"] == "primary"


# ─── 6. Region / timezone auto-detect is generic (no hardcoded IN) ──
def test_autodetect_region_returns_two_letter_or_empty(setup_mod):
    result = setup_mod.autodetect_region()
    assert result == "" or (len(result) == 2 and result.isupper())


def test_autodetect_region_no_hardcoded_country(setup_mod):
    """THE rule: no app/locale-specific defaults. If detection fails, return ''."""
    src = Path(setup_mod.__file__).read_text(encoding="utf-8")
    assert 'return "IN"' not in src
    assert 'return "US"' not in src


def test_autodetect_timezone_returns_iana_or_empty(setup_mod):
    result = setup_mod.autodetect_timezone()
    # Either empty (Windows fallback) or an IANA name with a slash
    assert result == "" or "/" in result


# ─── 7. TZ alias normalisation ──────────────────────────────────────
def test_canonicalize_tz_renames_calcutta_to_kolkata(setup_mod):
    assert setup_mod._canonicalize_tz("Asia/Calcutta") == "Asia/Kolkata"


def test_canonicalize_tz_passthrough_for_modern_names(setup_mod):
    assert setup_mod._canonicalize_tz("Asia/Kolkata") == "Asia/Kolkata"
    assert setup_mod._canonicalize_tz("America/New_York") == "America/New_York"
    assert setup_mod._canonicalize_tz("Europe/Berlin") == "Europe/Berlin"


def test_canonicalize_tz_passthrough_for_unknown_names(setup_mod):
    # Unknown / made-up names round-trip unchanged — we never invent a rename.
    assert setup_mod._canonicalize_tz("Mars/Olympus_Mons") == "Mars/Olympus_Mons"
    assert setup_mod._canonicalize_tz("") == ""


def test_canonicalize_tz_covers_known_renames(setup_mod):
    # The aliases TENKA users are most likely to see returned by Windows
    # tzlocal mappings. Failure here means the table drifted out of date.
    expected = {
        "Asia/Calcutta": "Asia/Kolkata",
        "Asia/Saigon": "Asia/Ho_Chi_Minh",
        "Europe/Kiev": "Europe/Kyiv",
    }
    for old, new in expected.items():
        assert setup_mod._canonicalize_tz(old) == new, f"{old} should map to {new}"
