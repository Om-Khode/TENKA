"""Tests for CE-DYN: dynamic router examples from saved templates."""

# Imports used by later CE-DYN test clusters (Tasks 3+):
import importlib
import json
import threading
from pathlib import Path

import pytest


# ─── Cluster 0: packages.py refactor ───────────────────────────────────────

def test_packages_module_exports_constants():
    """packages.py exposes the three constants previously defined in routing.py."""
    from assistant.code_executor import packages

    assert isinstance(packages.TIER2_ALLOWED_PACKAGES, frozenset)
    assert "requests" in packages.TIER2_ALLOWED_PACKAGES
    assert "spotipy" in packages.TIER2_ALLOWED_PACKAGES
    assert "opencv-python" in packages.TIER2_ALLOWED_PACKAGES

    assert isinstance(packages._PACKAGE_IMPORT_NAMES, dict)
    assert packages._PACKAGE_IMPORT_NAMES["opencv-python"] == "cv2"
    assert packages._PACKAGE_IMPORT_NAMES["beautifulsoup4"] == "bs4"

    assert isinstance(packages._IMPORT_TO_PACKAGE, dict)
    assert packages._IMPORT_TO_PACKAGE["cv2"] == "opencv-python"
    assert packages._IMPORT_TO_PACKAGE["bs4"] == "beautifulsoup4"


def test_routing_re_exports_same_constants_for_backwards_compat():
    """routing.py still exposes the constants (now imported from packages.py)."""
    from assistant.code_executor import routing, packages

    assert routing.TIER2_ALLOWED_PACKAGES is packages.TIER2_ALLOWED_PACKAGES
    assert routing._PACKAGE_IMPORT_NAMES is packages._PACKAGE_IMPORT_NAMES
    assert routing._IMPORT_TO_PACKAGE is packages._IMPORT_TO_PACKAGE


# ─── Cluster 1: header parsing (v1 + v2 + malformed) ───────────────────────

@pytest.fixture
def _isolated_sandbox(tmp_path, monkeypatch):
    """Point assistant.config.SANDBOX_DIR at a fresh tmp dir.

    Returns the scripts/ subdir (created on demand by templates._templates_dir).
    """
    from assistant import config
    monkeypatch.setattr(config, "SANDBOX_DIR", tmp_path)
    scripts = tmp_path / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    return scripts


@pytest.fixture(autouse=True)
def _fresh_router_examples_cache():
    """Clear the router_examples cache around every test in this module so the
    module-level _cached_examples never leaks state between tests."""
    from assistant.code_executor import router_examples
    router_examples.invalidate()
    yield
    router_examples.invalidate()


def test_save_template_writes_v2_header_with_params(_isolated_sandbox):
    """_save_template writes # version: 2, # GOAL:, and # PARAMS: when params present."""
    from assistant.code_executor.templates import _save_template

    _save_template(
        "music_play_song",
        "import os\nprint('hi')\n",
        goal="play blinding lights",
        params={"song_name": "Blinding Lights"},
    )

    path = _isolated_sandbox / "music_play_song.py"
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# version: 2\n")
    assert "# GOAL: play blinding lights\n" in text
    assert '# PARAMS: {"song_name": "Blinding Lights"}\n' in text


def test_save_template_omits_params_header_when_empty(_isolated_sandbox):
    """_save_template does NOT write # PARAMS: when params is empty/None."""
    from assistant.code_executor.templates import _save_template

    _save_template(
        "weather_current",
        "import os\nprint('hi')\n",
        goal="what's the weather",
        params=None,
    )

    text = (_isolated_sandbox / "weather_current.py").read_text(encoding="utf-8")
    assert "# PARAMS:" not in text


def test_save_template_omits_params_header_when_empty_dict(_isolated_sandbox):
    """_save_template does NOT write # PARAMS: when params is an empty dict."""
    from assistant.code_executor.templates import _save_template

    _save_template(
        "weather_current",
        "import os\nprint('hi')\n",
        goal="what's the weather",
        params={},
    )

    text = (_isolated_sandbox / "weather_current.py").read_text(encoding="utf-8")
    assert "# PARAMS:" not in text


def test_read_template_file_parses_v2_with_params(_isolated_sandbox):
    """_read_template_file returns (code, goal, params) for a v2 template."""
    from assistant.code_executor.templates import _read_template_file, _save_template

    _save_template(
        "music_play_song",
        "import os\nprint('hi')\n",
        goal="play blinding lights",
        params={"song_name": "Blinding Lights"},
    )

    path = _isolated_sandbox / "music_play_song.py"
    code, goal, params = _read_template_file(path, "music_play_song")
    assert "import os" in code
    assert goal == "play blinding lights"
    assert params == {"song_name": "Blinding Lights"}


def test_read_template_file_v1_returns_empty_params(_isolated_sandbox):
    """v1 templates (no # PARAMS: line) load with params={}."""
    from assistant.code_executor.templates import _read_template_file

    path = _isolated_sandbox / "legacy.py"
    path.write_text(
        "# version: 1\n# GOAL: legacy goal\nprint('hi')\n",
        encoding="utf-8",
    )

    code, goal, params = _read_template_file(path, "legacy")
    assert goal == "legacy goal"
    assert params == {}
    assert "print('hi')" in code
    assert "# version:" not in code
    assert "# GOAL:" not in code


def test_read_template_file_no_headers(_isolated_sandbox):
    """Bare template (no version/goal/params headers) loads cleanly."""
    from assistant.code_executor.templates import _read_template_file

    path = _isolated_sandbox / "bare.py"
    path.write_text("print('hi')\n", encoding="utf-8")

    code, goal, params = _read_template_file(path, "bare")
    assert "print('hi')" in code
    assert goal == ""
    assert params == {}


def test_read_template_file_malformed_params_falls_back(_isolated_sandbox, caplog):
    """Malformed # PARAMS: JSON returns {} and logs a warning."""
    import logging
    from assistant.code_executor.templates import _read_template_file

    path = _isolated_sandbox / "broken.py"
    path.write_text(
        "# version: 2\n# GOAL: x\n# PARAMS: {not valid json}\nprint('hi')\n",
        encoding="utf-8",
    )

    with caplog.at_level(logging.WARNING, logger="code_executor"):
        code, goal, params = _read_template_file(path, "broken")
    assert params == {}
    assert any("PARAMS" in r.getMessage() for r in caplog.records)


# ─── Cluster 2: scan + requires inference ──────────────────────────────────

def test_scan_returns_template_info_for_v2(_isolated_sandbox):
    """scan_templates parses v2 templates into TemplateInfo dataclasses."""
    from assistant.code_executor.router_examples import scan_templates

    (_isolated_sandbox / "music_play_song.py").write_text(
        '# version: 2\n'
        '# GOAL: play blinding lights\n'
        '# PARAMS: {"song_name": "Blinding Lights"}\n'
        'import os, spotipy\nprint(1)\n',
        encoding="utf-8",
    )

    infos = scan_templates(_isolated_sandbox)
    assert len(infos) == 1
    info = infos[0]
    assert info.slug == "music_play_song"
    assert info.goal == "play blinding lights"
    assert info.params == {"song_name": "Blinding Lights"}
    assert info.requires == ["spotipy"]  # 'os' is stdlib, dropped


def test_scan_orders_alphabetically_by_slug(_isolated_sandbox):
    """scan_templates returns infos sorted by slug for deterministic prompts."""
    from assistant.code_executor.router_examples import scan_templates

    for slug in ("zebra", "apple", "mango"):
        (_isolated_sandbox / f"{slug}.py").write_text(
            "# version: 2\n# GOAL: x\nimport os\nprint(1)\n",
            encoding="utf-8",
        )

    slugs = [i.slug for i in scan_templates(_isolated_sandbox)]
    assert slugs == ["apple", "mango", "zebra"]


def test_scan_infers_requires_from_imports(_isolated_sandbox):
    """import X / from X import Y are mapped to TIER2_ALLOWED_PACKAGES pip names."""
    from assistant.code_executor.router_examples import scan_templates

    (_isolated_sandbox / "t.py").write_text(
        "# version: 2\n# GOAL: x\n"
        "import requests\n"
        "import cv2\n"
        "from googleapiclient.discovery import build\n"
        "from bs4 import BeautifulSoup\n"
        "print(1)\n",
        encoding="utf-8",
    )

    info = scan_templates(_isolated_sandbox)[0]
    assert set(info.requires) == {
        "requests", "opencv-python", "google-api-python-client", "beautifulsoup4",
    }


def test_scan_drops_stdlib_imports(_isolated_sandbox):
    """Pure-stdlib templates yield requires=[]."""
    from assistant.code_executor.router_examples import scan_templates

    (_isolated_sandbox / "stdlib_only.py").write_text(
        "# version: 2\n# GOAL: x\n"
        "import os, sys, json, re\nprint(1)\n",
        encoding="utf-8",
    )

    info = scan_templates(_isolated_sandbox)[0]
    assert info.requires == []


def test_scan_drops_unknown_packages(_isolated_sandbox):
    """Imports not in TIER2_ALLOWED_PACKAGES are dropped silently."""
    from assistant.code_executor.router_examples import scan_templates

    (_isolated_sandbox / "t.py").write_text(
        "# version: 2\n# GOAL: x\n"
        "import nonexistent_pkg\nimport requests\nprint(1)\n",
        encoding="utf-8",
    )

    info = scan_templates(_isolated_sandbox)[0]
    assert info.requires == ["requests"]


def test_scan_handles_missing_dir(tmp_path):
    """scan_templates returns [] if scripts dir does not exist."""
    from assistant.code_executor.router_examples import scan_templates

    assert scan_templates(tmp_path / "does_not_exist") == []


def test_scan_handles_aliased_imports(_isolated_sandbox):
    """import X as Y / from X import Y as Z should still resolve to the pip package."""
    from assistant.code_executor.router_examples import scan_templates

    (_isolated_sandbox / "t.py").write_text(
        "# version: 2\n# GOAL: x\n"
        "import numpy as np\n"
        "import cv2 as cv\n"
        "import requests\n"
        "print(1)\n",
        encoding="utf-8",
    )

    info = scan_templates(_isolated_sandbox)[0]
    assert set(info.requires) == {"numpy", "opencv-python", "requests"}


# ─── Cluster 3: availability gate ──────────────────────────────────────────

def test_is_available_passes_when_all_installed(monkeypatch):
    """_is_available returns True when find_spec succeeds for every pkg."""
    import importlib.util as iu
    from assistant.code_executor import router_examples

    monkeypatch.setattr(iu, "find_spec", lambda name: object())
    assert router_examples._is_available(["requests", "spotipy"]) is True


def test_is_available_fails_when_any_missing(monkeypatch):
    """_is_available returns False if any single pkg fails the spec lookup."""
    import importlib.util as iu
    from assistant.code_executor import router_examples

    def _fake(name):
        return None if name == "spotipy" else object()
    monkeypatch.setattr(iu, "find_spec", _fake)
    assert router_examples._is_available(["requests", "spotipy"]) is False


def test_is_available_empty_requires_always_passes():
    """Pure-stdlib templates (requires=[]) always pass the gate."""
    from assistant.code_executor import router_examples
    assert router_examples._is_available([]) is True


def test_is_available_uses_import_name_mapping(monkeypatch):
    """Gate looks up the IMPORT name, not the pip name (e.g. opencv-python → cv2)."""
    import importlib.util as iu
    from assistant.code_executor import router_examples

    seen: list[str] = []

    def _fake(name):
        seen.append(name)
        return object()

    monkeypatch.setattr(iu, "find_spec", _fake)
    router_examples._is_available(["opencv-python", "google-api-python-client"])
    assert "cv2" in seen
    assert "googleapiclient" in seen


def test_is_available_fallback_normalises_hyphen_to_underscore(monkeypatch):
    """Unmapped pip names with hyphens are normalised to underscores for find_spec.

    Defensive: every hyphenated package currently in TIER2_ALLOWED_PACKAGES has a
    mapping entry, so this fallback path is dormant. But if a future contributor
    adds a hyphenated package without updating _PACKAGE_IMPORT_NAMES, this test
    seals the contract that the gate normalises hyphens, not asks importlib to
    look up a name with a hyphen (which would always fail).
    """
    import importlib.util as iu
    from assistant.code_executor import router_examples

    seen: list[str] = []

    def _fake(name):
        seen.append(name)
        return object()

    monkeypatch.setattr(iu, "find_spec", _fake)
    # Patch the _PACKAGE_IMPORT_NAMES NAME inside router_examples (not the source
    # module — router_examples imported the dict by name, so that binding is the
    # one in effect at call time).
    monkeypatch.setattr(router_examples, "_PACKAGE_IMPORT_NAMES", {})

    assert router_examples._is_available(["some-new-pkg"]) is True
    assert seen == ["some_new_pkg"]


# ─── Cluster 4: sanitization + render ──────────────────────────────────────

def test_sanitize_truncates_at_path_colon():
    """A colon that's NOT followed by // (i.e. a path) truncates the goal."""
    from assistant.code_executor.router_examples import _sanitize_goal
    assert _sanitize_goal("Capture the image at C:\\foo\\bar and process") \
        == "capture the image at"


def test_sanitize_colon_fused_to_word_returns_empty():
    """When the colon is fused directly to a word (no space before it AND
    the previous char is a letter/digit), and there's no payload before
    the fused word, sanitize to empty.

    Covers two shapes: 'C:\\foo' (drive letter) and 'Image: details'
    (label prefix). Both have the colon attached to a letter token with
    no preceding payload words.
    """
    from assistant.code_executor.router_examples import _sanitize_goal
    assert _sanitize_goal("C:\\foo\\bar") == ""
    assert _sanitize_goal("Image: details") == ""


def test_sanitize_preserves_payload_when_colon_not_fused_to_letter():
    """When the colon follows a non-alphanumeric char (like '%' or ' '),
    the partial-word drop does NOT fire — payload tokens are preserved."""
    from assistant.code_executor.router_examples import _sanitize_goal
    # '50%:' — colon after '%' (not alphanumeric) → keep '50%'
    assert _sanitize_goal("Set volume to 50%: done") == "set volume to 50%"
    # 'key :' — colon after space → truncate cleanly, no extra drop
    assert _sanitize_goal("key : value") == "key"


def test_sanitize_preserves_url_colons():
    """A colon followed by // (URL) does NOT truncate."""
    from assistant.code_executor.router_examples import _sanitize_goal
    assert _sanitize_goal("Open https://example.com/foo") \
        == "open https://example.com/foo"


def test_sanitize_strips_leading_articles():
    """Leading a/an/the/my/to are stripped (after lowercasing)."""
    from assistant.code_executor.router_examples import _sanitize_goal
    assert _sanitize_goal("The current weather") == "current weather"
    assert _sanitize_goal("My unread emails") == "unread emails"
    assert _sanitize_goal("To do list") == "do list"


def test_sanitize_collapses_whitespace_and_lowercases():
    from assistant.code_executor.router_examples import _sanitize_goal
    assert _sanitize_goal("  Play   BLINDING   lights  ") == "play blinding lights"


def test_sanitize_truncates_at_60_chars_with_ellipsis():
    from assistant.code_executor.router_examples import _sanitize_goal
    long = "play " + ("x" * 100)
    out = _sanitize_goal(long)
    assert len(out) == 61  # 60 chars + ellipsis char
    assert out.endswith("…")


def test_sanitize_empty_returns_empty():
    from assistant.code_executor.router_examples import _sanitize_goal
    assert _sanitize_goal("") == ""
    assert _sanitize_goal("   ") == ""


def test_build_dynamic_examples_renders_gated_templates(_isolated_sandbox, monkeypatch):
    """Available templates produce // goal\n{json} lines, sorted by slug."""
    import importlib.util as iu
    from assistant.code_executor import router_examples

    (_isolated_sandbox / "music_play_song.py").write_text(
        '# version: 2\n# GOAL: play blinding lights\n'
        '# PARAMS: {"song_name": "Blinding Lights"}\n'
        'import os, spotipy\nprint(1)\n', encoding="utf-8")
    (_isolated_sandbox / "weather_current.py").write_text(
        '# version: 2\n# GOAL: what\'s the weather\n'
        'import requests\nprint(1)\n', encoding="utf-8")

    monkeypatch.setattr(iu, "find_spec", lambda name: object())

    out = router_examples.build_dynamic_examples(_isolated_sandbox)
    lines = out.strip().split("\n")
    assert lines == [
        "// play blinding lights",
        '{"tier":2,"template_slug":"music_play_song","requires":["spotipy"],"params":{"song_name":"Blinding Lights"}}',
        "// what's the weather",
        '{"tier":2,"template_slug":"weather_current","requires":["requests"],"params":{}}',
    ]


def test_build_dynamic_examples_omits_unavailable(_isolated_sandbox, monkeypatch):
    """Templates whose deps are missing do NOT appear in the rendered block."""
    import importlib.util as iu
    from assistant.code_executor import router_examples

    (_isolated_sandbox / "music_play_song.py").write_text(
        '# version: 2\n# GOAL: play x\nimport spotipy\nprint(1)\n', encoding="utf-8")
    (_isolated_sandbox / "weather_current.py").write_text(
        '# version: 2\n# GOAL: weather\nimport requests\nprint(1)\n', encoding="utf-8")

    def _fake(name):
        return None if name == "spotipy" else object()

    monkeypatch.setattr(iu, "find_spec", _fake)
    out = router_examples.build_dynamic_examples(_isolated_sandbox)
    assert "music_play_song" not in out
    assert "weather_current" in out


def test_build_dynamic_examples_omits_comment_when_goal_empty(_isolated_sandbox, monkeypatch):
    """Templates without a # GOAL: header render JSON only, no // comment."""
    import importlib.util as iu
    from assistant.code_executor import router_examples

    (_isolated_sandbox / "legacy.py").write_text(
        "import requests\nprint(1)\n", encoding="utf-8")

    monkeypatch.setattr(iu, "find_spec", lambda name: object())
    out = router_examples.build_dynamic_examples(_isolated_sandbox)
    assert "//" not in out
    assert "legacy" in out


def test_build_dynamic_examples_empty_dir_returns_empty_string(tmp_path):
    """No templates → empty string (not whitespace, not '[]', empty)."""
    from assistant.code_executor import router_examples
    assert router_examples.build_dynamic_examples(tmp_path) == ""


def test_build_dynamic_examples_alphabetical_slug_order(_isolated_sandbox, monkeypatch):
    """Output is sorted by slug — critical for Gemini's implicit prefix cache."""
    import importlib.util as iu
    from assistant.code_executor import router_examples

    for slug in ("weather_current", "audio_test", "music_play"):
        (_isolated_sandbox / f"{slug}.py").write_text(
            f"# version: 2\n# GOAL: {slug}\nimport requests\nprint(1)\n",
            encoding="utf-8",
        )
    monkeypatch.setattr(iu, "find_spec", lambda name: object())

    out = router_examples.build_dynamic_examples(_isolated_sandbox)
    slug_positions = {s: out.find(f'"template_slug":"{s}"') for s in ("audio_test", "music_play", "weather_current")}
    assert slug_positions["audio_test"] < slug_positions["music_play"] < slug_positions["weather_current"]


# ─── Cluster 5: cache + invalidation ───────────────────────────────────────


def test_get_dynamic_examples_caches_first_call(_isolated_sandbox, monkeypatch):
    """Second call to get_dynamic_examples returns the same cached string."""
    import importlib.util as iu
    from assistant import config
    from assistant.code_executor import router_examples

    monkeypatch.setattr(iu, "find_spec", lambda name: object())
    monkeypatch.setattr(config, "SANDBOX_DIR", _isolated_sandbox.parent)

    (_isolated_sandbox / "weather_current.py").write_text(
        '# version: 2\n# GOAL: weather\nimport requests\nprint(1)\n', encoding="utf-8")

    first = router_examples.get_dynamic_examples()
    assert "weather_current" in first

    # Add a second template — should NOT appear until invalidate
    (_isolated_sandbox / "music_play.py").write_text(
        '# version: 2\n# GOAL: play\nimport spotipy\nprint(1)\n', encoding="utf-8")

    second = router_examples.get_dynamic_examples()
    assert second is first  # same cached string object
    assert "music_play" not in second


def test_invalidate_forces_rebuild_on_next_call(_isolated_sandbox, monkeypatch):
    """After invalidate, the next get_dynamic_examples picks up new templates."""
    import importlib.util as iu
    from assistant import config
    from assistant.code_executor import router_examples

    monkeypatch.setattr(iu, "find_spec", lambda name: object())
    monkeypatch.setattr(config, "SANDBOX_DIR", _isolated_sandbox.parent)

    (_isolated_sandbox / "weather_current.py").write_text(
        '# version: 2\n# GOAL: weather\nimport requests\nprint(1)\n', encoding="utf-8")
    router_examples.get_dynamic_examples()  # prime cache

    (_isolated_sandbox / "music_play.py").write_text(
        '# version: 2\n# GOAL: play\nimport spotipy\nprint(1)\n', encoding="utf-8")
    router_examples.invalidate()

    rebuilt = router_examples.get_dynamic_examples()
    assert "music_play" in rebuilt


def test_concurrent_get_and_invalidate_thread_safe(_isolated_sandbox, monkeypatch):
    """No exception under concurrent get + invalidate calls."""
    import importlib.util as iu
    from assistant import config
    from assistant.code_executor import router_examples

    monkeypatch.setattr(iu, "find_spec", lambda name: object())
    monkeypatch.setattr(config, "SANDBOX_DIR", _isolated_sandbox.parent)
    (_isolated_sandbox / "weather_current.py").write_text(
        '# version: 2\n# GOAL: x\nimport requests\nprint(1)\n', encoding="utf-8")

    errors: list[BaseException] = []
    results: list[str] = []
    results_lock = threading.Lock()

    def _worker():
        try:
            for _ in range(50):
                v = router_examples.get_dynamic_examples()
                with results_lock:
                    results.append(v)
                router_examples.invalidate()
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    ts = [threading.Thread(target=_worker) for _ in range(4)]
    for t in ts: t.start()
    for t in ts: t.join()
    assert errors == []
    assert all(isinstance(v, str) for v in results)
    assert all("weather_current" in v or v == "" for v in results)


def test_save_template_invalidates_router_examples_cache(_isolated_sandbox, monkeypatch):
    """_save_template clears the router_examples cache so new templates appear."""
    import importlib.util as iu
    from assistant import config
    from assistant.code_executor import router_examples
    from assistant.code_executor.templates import _save_template

    monkeypatch.setattr(iu, "find_spec", lambda name: object())
    monkeypatch.setattr(config, "SANDBOX_DIR", _isolated_sandbox.parent)

    (_isolated_sandbox / "weather_current.py").write_text(
        '# version: 2\n# GOAL: weather\nimport requests\nprint(1)\n', encoding="utf-8")
    router_examples.get_dynamic_examples()  # prime cache

    _save_template(
        "music_play",
        "import spotipy\nprint(1)\n",
        goal="play music",
        params=None,
    )

    rebuilt = router_examples.get_dynamic_examples()
    assert "music_play" in rebuilt
    assert "weather_current" in rebuilt


def test_delete_template_invalidates_router_examples_cache(_isolated_sandbox, monkeypatch):
    """_delete_template clears the cache so the removed template disappears."""
    import importlib.util as iu
    from assistant import config
    from assistant.code_executor import router_examples
    from assistant.code_executor.templates import _delete_template

    monkeypatch.setattr(iu, "find_spec", lambda name: object())
    monkeypatch.setattr(config, "SANDBOX_DIR", _isolated_sandbox.parent)

    (_isolated_sandbox / "weather_current.py").write_text(
        '# version: 2\n# GOAL: weather\nimport requests\nprint(1)\n', encoding="utf-8")
    router_examples.get_dynamic_examples()

    _delete_template("weather_current")

    rebuilt = router_examples.get_dynamic_examples()
    assert "weather_current" not in rebuilt


# ─── Cluster 6: prompt assembly ────────────────────────────────────────────

def test_static_base_examples_always_present(tmp_path, monkeypatch):
    """Empty scripts dir → prompt still contains the 3 static base examples."""
    from assistant import config
    from assistant.code_executor.prompts import get_router_system_prompt

    monkeypatch.setattr(config, "SANDBOX_DIR", tmp_path)

    prompt = get_router_system_prompt()
    assert '"tier":1,"template_slug":null' in prompt
    assert '"template_slug":"weather_current"' in prompt
    assert '"template_slug":"md_to_docx"' in prompt


def test_dynamic_examples_append_after_base(_isolated_sandbox, monkeypatch):
    """With cached templates, dynamic block follows the static base."""
    import importlib.util as iu
    from assistant import config
    from assistant.code_executor.prompts import get_router_system_prompt

    monkeypatch.setattr(iu, "find_spec", lambda name: object())
    monkeypatch.setattr(config, "SANDBOX_DIR", _isolated_sandbox.parent)

    (_isolated_sandbox / "music_play_song.py").write_text(
        '# version: 2\n# GOAL: play blinding lights\n'
        '# PARAMS: {"song_name": "Blinding Lights"}\n'
        'import spotipy\nprint(1)\n', encoding="utf-8")

    prompt = get_router_system_prompt()
    base_idx = prompt.index('"template_slug":"md_to_docx"')
    dynamic_idx = prompt.index('"template_slug":"music_play_song"')
    assert base_idx < dynamic_idx


def test_no_legacy_router_examples_string_in_prompt(tmp_path, monkeypatch):
    """Sentinel: assert the old hardcoded slugs are NOT present from the base set.

    music_play / music_play_song / email_unread / messaging_* / volume_set /
    image_grid_colors were in the deleted ROUTER_EXAMPLES. They must only
    appear in the prompt now if they exist as cached templates on disk.
    """
    from assistant import config
    from assistant.code_executor.prompts import get_router_system_prompt

    monkeypatch.setattr(config, "SANDBOX_DIR", tmp_path)  # empty dir

    prompt = get_router_system_prompt()
    for legacy_slug in (
        "music_play_song", "music_play", "email_unread", "email_draft",
        "messaging_read", "messaging_send", "volume_set", "image_grid_colors",
    ):
        assert f'"template_slug":"{legacy_slug}"' not in prompt, \
            f"legacy slug '{legacy_slug}' leaked into base prompt"


def test_service_registry_no_longer_exports_router_examples():
    """ROUTER_EXAMPLES must be deleted from service_registry."""
    from assistant import service_registry
    assert not hasattr(service_registry, "ROUTER_EXAMPLES"), \
        "ROUTER_EXAMPLES should have been deleted as part of CE-DYN"


def test_routing_uses_getter_not_constant():
    """Router call path resolves through get_router_system_prompt, not a stale constant."""
    from assistant.code_executor import prompts

    assert not hasattr(prompts, "_ROUTER_SYSTEM_PROMPT"), \
        "_ROUTER_SYSTEM_PROMPT constant should be removed (use get_router_system_prompt instead)"
    assert callable(getattr(prompts, "get_router_system_prompt", None))


# ─── Cluster 7: goal-match specificity + threshold (post-livetest fix) ─────
#
# Bug surfaced during CE-DYN live-test 2026-05-30: "play some music" cached a
# generic spotipy template with no song-name param. "play blinding lights"
# then matched the cached template (overlap ratio 0.33 ≥ 0.20 threshold),
# played generic music instead of the requested song. Two-layer fix in
# _goal_matches_template: spaCy-based specificity asymmetry check (primary)
# + raise overlap threshold 0.20 → 0.50 (secondary).

def test_goal_match_rejects_specific_for_generic_stored():
    """The actual live-test failure: cached generic 'play some music' MUST NOT
    be reused for a specific song request 'play blinding lights'."""
    from assistant.code_executor.templates import _goal_matches_template
    assert _goal_matches_template("play blinding lights", "play some music") is False


def test_goal_match_accepts_exact_repeat():
    """The same goal twice should always reuse the cached template."""
    from assistant.code_executor.templates import _goal_matches_template
    assert _goal_matches_template("play blinding lights", "play blinding lights") is True


def test_goal_match_accepts_generic_to_generic_paraphrase():
    """Two equally-generic goals should still match (no specificity asymmetry,
    high overlap ratio)."""
    from assistant.code_executor.templates import _goal_matches_template
    assert _goal_matches_template("play music", "play some music") is True


def test_goal_match_rejects_different_specific_songs():
    """Two specific-but-different songs should NOT match — even though
    the cached template might be parameterized, the specific-content tokens
    differ. Force regen so the new song lands in the cached params."""
    from assistant.code_executor.templates import _goal_matches_template
    assert _goal_matches_template("play counting stars", "play blinding lights") is False


def test_goal_match_rejects_low_overlap_unrelated_goals():
    """Sanity: goals that share only the verb (no specificity asymmetry
    because both are generic) should fail the 0.70 overlap floor.

    Threshold raised from 0.50 → 0.70 after livetest #3 exposed that
    "play dare" vs "play make you mine" hit exactly 0.50 (1/2 ratio, just
    the verb "play" overlapping) and got falsely ACCEPTED. 0.70 forces
    real content-word overlap, not just the verb."""
    from assistant.code_executor.templates import _goal_matches_template
    # 1/2 = 0.50 — well below the 0.70 floor → REJECT
    assert _goal_matches_template("open tab", "open calendar") is False
    # 1/3 = 0.33 → REJECT
    assert _goal_matches_template("open tab now", "open calendar today morning") is False


def test_goal_match_rejects_play_dare_vs_play_make_you_mine_via_overlap():
    """Concrete livetest #3 failure: 'play make you mine' MUST NOT be
    accepted against 'play dare'. With the 0.70 threshold, the 1/2 overlap
    ratio (just 'play' shared) is correctly rejected — independent of
    whatever the spaCy specificity check decides about those tokens."""
    from assistant.code_executor.templates import _goal_matches_template
    assert _goal_matches_template("play make you mine", "play dare") is False
    # Same in the other direction
    assert _goal_matches_template("play dare", "play make you mine") is False


def test_goal_match_allows_specificity_subset():
    """If stored has the same content nouns as current (or more), it's not
    a specificity gap. E.g. stored='play blinding lights' / current='blinding
    lights' — current is a subset; reuse is fine."""
    from assistant.code_executor.templates import _goal_matches_template
    # current_content - stored_content == {} → specificity check passes
    # overlap: cur={"blinding", "lights"}, stored={"play", "blinding", "lights"} → 2/2 = 1.0 → ACCEPT
    assert _goal_matches_template("blinding lights", "play blinding lights") is True


def test_goal_match_empty_stored_always_accepts():
    """Legacy templates with no GOAL header (empty stored_goal) — preserved
    behavior, always reuse."""
    from assistant.code_executor.templates import _goal_matches_template
    assert _goal_matches_template("any goal at all", "") is True


def test_goal_match_falls_back_when_spacy_unavailable(monkeypatch):
    """If spaCy fails to load, matcher falls back to keyword-overlap-only
    (no specificity check). Test by faking the load failure."""
    from assistant.code_executor import templates
    monkeypatch.setattr(templates, "_NLP", None)
    monkeypatch.setattr(templates, "_NLP_LOAD_FAILED", True)

    # Without specificity check, this would have to fall back to overlap-only.
    # cur={"play", "blinding", "lights"}, stored={"play", "some", "music"} → 1/3 = 0.33 → REJECT (0.50 floor)
    assert _goal_match_via_templates("play blinding lights", "play some music") is False

    # An exact match still works on overlap alone.
    assert _goal_match_via_templates("play music", "play music") is True


def _goal_match_via_templates(a, b):
    """Helper — avoids re-importing inside the monkeypatch test."""
    from assistant.code_executor.templates import _goal_matches_template
    return _goal_matches_template(a, b)


# ─── Cluster 8: slug disambiguation on goal mismatch (livetest fix #2) ─────
#
# Bug: orchestrator appended `_{first_word_of_goal}` when goal-match failed.
# For verb-led goals ("play X", "open Y", "send Z") the first word is the
# verb, producing slugs like spotify_play_music_play, _play_play, _play_play_play
# on each successive specific request. Fix: use sorted param-key signature
# instead — stable across requests with the same param shape.

def test_disambiguate_no_params_keeps_slug():
    """Generic request (no params) overwrites the cached generic template."""
    from assistant.code_executor.orchestrator import _disambiguate_slug_on_mismatch
    assert _disambiguate_slug_on_mismatch("music_play", None) == "music_play"
    assert _disambiguate_slug_on_mismatch("music_play", {}) == "music_play"


def test_disambiguate_with_params_appends_sorted_param_keys():
    """Parameterized request appends the sorted param-key signature."""
    from assistant.code_executor.orchestrator import _disambiguate_slug_on_mismatch
    assert _disambiguate_slug_on_mismatch("music_play", {"song_name": "x"}) == "music_play__song_name"
    # Sorted order is deterministic — same params in different insertion
    # order still produce the same slug.
    assert (
        _disambiguate_slug_on_mismatch("email_send", {"to": "a", "subject": "b"})
        == _disambiguate_slug_on_mismatch("email_send", {"subject": "b", "to": "a"})
        == "email_send__subject_to"
    )


def test_disambiguate_idempotent_for_same_param_shape():
    """Successive specific requests with the same param shape share one slug —
    no accumulation. This is the fix for the _play_play_play bug."""
    from assistant.code_executor.orchestrator import _disambiguate_slug_on_mismatch
    s1 = _disambiguate_slug_on_mismatch("spotify_play_music", {"music_to_play": "blinding lights"})
    s2 = _disambiguate_slug_on_mismatch("spotify_play_music", {"music_to_play": "counting stars"})
    s3 = _disambiguate_slug_on_mismatch("spotify_play_music", {"music_to_play": "make you mine"})
    assert s1 == s2 == s3 == "spotify_play_music__music_to_play"


def test_disambiguate_old_behavior_did_not_use_first_word():
    """Negative test: the previous implementation suffixed with goal's first
    word ('play', 'open', 'send'), producing _play, _play_play accumulation.
    The new implementation never references the goal text."""
    from assistant.code_executor.orchestrator import _disambiguate_slug_on_mismatch
    # No matter what goal text, the result depends only on slug + params.
    assert _disambiguate_slug_on_mismatch("x", None) == "x"
    assert _disambiguate_slug_on_mismatch("x", {}) == "x"
    assert _disambiguate_slug_on_mismatch("x", {"k": "v"}) == "x__k"


# ─── Cluster 9: launcher skips when app already running (livetest fix #3) ──
#
# Live-test bug: user said "play some music" while Spotify was already
# playing. Cached script's active-device check still emitted APP_NOT_READY,
# orchestrator said "Opening Spotify, one moment..." (FALSE — already open)
# and polled for 20s before falling back to the "use ANY device" hint.
# Fix: native.is_app_running() probes pygetwindow; orchestrator skips TTS +
# open_app + poll when the window is already open.

def test_is_app_running_returns_true_when_window_matches(monkeypatch):
    """is_app_running matches case-insensitive substring against window titles."""
    from assistant.automation import native

    class _FakeWin:
        def __init__(self, title): self.title = title

    fake_windows = [_FakeWin("Notepad - file.txt"), _FakeWin("Spotify Premium - John Doe")]
    fake_gw = type("FakeGW", (), {"getAllWindows": staticmethod(lambda: fake_windows)})
    monkeypatch.setitem(__import__("sys").modules, "pygetwindow", fake_gw)

    assert native.is_app_running("spotify") is True
    assert native.is_app_running("SPOTIFY") is True  # case-insensitive
    assert native.is_app_running("notepad") is True


def test_is_app_running_returns_false_when_no_match(monkeypatch):
    """is_app_running returns False when no process AND no window matches."""
    from assistant.automation import native
    import sys

    # Mock psutil to return no matching processes (otherwise real psutil
    # sees whatever is actually running on the test machine).
    class _FakeProc:
        def __init__(self, name): self.info = {"name": name}
    class _FakePsutil:
        @staticmethod
        def process_iter(attrs):
            return [_FakeProc("python.exe"), _FakeProc("explorer.exe")]
    monkeypatch.setitem(sys.modules, "psutil", _FakePsutil)

    class _FakeWin:
        def __init__(self, title): self.title = title

    fake_windows = [_FakeWin("Notepad - file.txt"), _FakeWin("Chrome - Github")]
    fake_gw = type("FakeGW", (), {"getAllWindows": staticmethod(lambda: fake_windows)})
    monkeypatch.setitem(sys.modules, "pygetwindow", fake_gw)

    assert native.is_app_running("spotify") is False
    assert native.is_app_running("discord") is False


def test_is_app_running_ignores_blank_titles(monkeypatch):
    """Windows with empty/whitespace titles do not count as a match —
    avoids false positives on system/hidden windows."""
    from assistant.automation import native

    class _FakeWin:
        def __init__(self, title): self.title = title

    fake_windows = [_FakeWin(""), _FakeWin("   "), _FakeWin("spotify")]
    fake_gw = type("FakeGW", (), {"getAllWindows": staticmethod(lambda: fake_windows)})
    monkeypatch.setitem(__import__("sys").modules, "pygetwindow", fake_gw)

    assert native.is_app_running("spotify") is True
    # An empty-titled fake window shouldn't trigger a substring match on ""
    assert native.is_app_running("") is False


def test_is_app_running_returns_false_when_pygetwindow_unavailable(monkeypatch):
    """If both psutil and pygetwindow are unavailable, return False
    gracefully so the launcher falls back to the open_app + poll flow."""
    from assistant.automation import native
    import sys

    # Make both backends raise ModuleNotFoundError on import
    monkeypatch.setitem(sys.modules, "psutil", None)
    monkeypatch.setitem(sys.modules, "pygetwindow", None)

    assert native.is_app_running("spotify") is False


# ─── Cluster 10: disambiguation idempotence (livetest fix #4) ──────────────
#
# Bug: when router learned a disambiguated slug from the dynamic catalog
# (e.g. `spotify_play_music__music_title`) and emitted it for a different
# specific request with the same param shape, _disambiguate_slug_on_mismatch
# appended `__music_title` AGAIN, producing slugs like
# `spotify_play_music__music_title__music_title__music_title`.

def test_disambiguate_idempotent_when_slug_already_has_suffix():
    """If the input slug ALREADY ends with the param-key suffix (router
    picked the disambiguated slug from the catalog), don't append again."""
    from assistant.code_executor.orchestrator import _disambiguate_slug_on_mismatch
    # Input slug already ends with the suffix this param shape would produce
    assert (
        _disambiguate_slug_on_mismatch("spotify_play_music__music_title", {"music_title": "x"})
        == "spotify_play_music__music_title"
    )
    # Multi-param suffix: sorted keys joined with _
    assert (
        _disambiguate_slug_on_mismatch("email_send__body_subject_to", {"to": "a", "subject": "b", "body": "c"})
        == "email_send__body_subject_to"
    )


def test_disambiguate_different_param_shape_still_appends():
    """If the input slug ends with a DIFFERENT param suffix, append the new
    one — different param shapes deserve separate cached templates."""
    from assistant.code_executor.orchestrator import _disambiguate_slug_on_mismatch
    # Slug has __music_title suffix, but new params are {"volume_level"} — append
    assert (
        _disambiguate_slug_on_mismatch("spotify_play_music__music_title", {"volume_level": "50"})
        == "spotify_play_music__music_title__volume_level"
    )


def test_disambiguate_no_double_suffix_after_3_consecutive_mismatches():
    """The bug as observed in live-test: 'play DARE' → 'play make you mine'
    → 'play counting stars' produced ..__music_title__music_title__music_title.
    With the fix, slug stays stable across all three."""
    from assistant.code_executor.orchestrator import _disambiguate_slug_on_mismatch
    base = "spotify_play_music"
    params = {"music_title": "..."}  # same param shape for each request

    # First mismatch: bare slug → disambiguated
    s1 = _disambiguate_slug_on_mismatch(base, params)
    assert s1 == "spotify_play_music__music_title"

    # Second mismatch: router picks the disambiguated slug from catalog, same param shape
    s2 = _disambiguate_slug_on_mismatch(s1, params)
    assert s2 == s1, f"slug grew on 2nd mismatch: {s2!r}"

    # Third mismatch: same again
    s3 = _disambiguate_slug_on_mismatch(s2, params)
    assert s3 == s1, f"slug grew on 3rd mismatch: {s3!r}"


# ─── Cluster 11: prompt teaches available-vs-active pattern (livetest fix #5) ─
#
# Bug: code-gen prompt taught LLM to bail with APP_NOT_READY whenever no
# device is "active" — caused Spotify scripts to give up immediately when
# Spotify was open but no playback session was active. Fix: rewrite the
# prompt section to teach "if available list non-empty, transfer/activate
# one before bailing". Generic across Spotify, music apps, messaging,
# browser, smart-home — any SDK with the available/active pattern.

def test_prompt_teaches_available_vs_active_distinction():
    """The tier-2 code-gen prompt must explicitly teach that available !=
    active, and tell the LLM to activate an available item before bailing."""
    from assistant.code_executor import prompts
    p = prompts._CODE_GEN_SYSTEM_PROMPT_TIER2.lower()
    # Must mention both concepts
    assert "available" in p
    assert "active" in p
    # Must give concrete guidance to transfer/activate/focus before bailing
    assert any(verb in p for verb in ("transfer_playback", "transfer", "activate"))
    # Must condition APP_NOT_READY on the available listing being empty
    # (not just "no active device" as before)
    assert "available" in p and "empty" in p


def test_prompt_no_longer_unconditionally_emits_app_not_ready_on_no_active():
    """Negative test: the old wording 'no active device/instance is found:
    print APP_NOT_READY' unconditionally is the bug. The new wording must
    GATE the APP_NOT_READY emission on the AVAILABLE listing being empty,
    not just no-active."""
    from assistant.code_executor import prompts
    p = prompts._CODE_GEN_SYSTEM_PROMPT_TIER2
    # The exact phrase from before must be gone — it told the LLM to bail
    # whenever no ACTIVE device was found, ignoring the available list.
    assert "no active device/instance is found:\n  print(\"APP_NOT_READY" not in p
    assert "no active device\\instance is found:\n  print(\"APP_NOT_READY" not in p


# ─── Cluster 12: router teaches entity extraction (livetest fix #6) ────────
#
# Bug C from livetest #3: router emitted params={} for "play dare", so the
# LLM hardcoded "dare" as the search query in the generated template. Every
# subsequent "play X" matched the cached template and played "dare". Fix:
# teach the router to extract specific named values as params.

def test_router_prompt_teaches_param_extraction():
    """The router head must explicitly instruct on param extraction with
    concrete examples — without this the LLM emits params={} and the
    generated code hardcodes the first-seen value."""
    from assistant.code_executor.prompts import get_router_system_prompt
    p = get_router_system_prompt().lower()
    # Must teach the concept by name
    assert "param extraction" in p or "extract it as a param" in p
    # Must show at least one example with a specific value being extracted
    # to a snake_case key (the format the parameterizer expects)
    assert "song" in p or "contact" in p or "city" in p
    # Must mention the consequence the LLM needs to avoid
    assert ("hardcode" in p or "hardcoded" in p
            or "replay" in p or "without this" in p)


def test_router_prompt_keeps_generic_no_param_examples():
    """Counter-balance: generic verbs without specific entities should still
    map to params={}. Without these examples the LLM might over-extract."""
    from assistant.code_executor.prompts import get_router_system_prompt
    p = get_router_system_prompt().lower()
    # At least one example showing 'generic' → params={}
    assert "params={}" in p
    assert "no specific" in p or "generic" in p


# ─── Cluster 13: is_app_running uses psutil for tray-minimized apps (#7) ───
#
# Bug E from livetest #3: Spotify was minimized to system tray, pygetwindow
# returned no matching windows, is_app_running returned False, launcher
# fired full TTS+open+poll. Fix: psutil.process_iter first (catches
# tray-only processes), pygetwindow fallback.

def test_is_app_running_finds_tray_only_process(monkeypatch):
    """is_app_running returns True when the process exists even if no window
    is visible (tray-minimized case from livetest)."""
    from assistant.automation import native
    import sys

    # Fake psutil: pretend Spotify.exe is running
    class _FakeProc:
        def __init__(self, name): self.info = {"name": name}

    class _FakePsutil:
        @staticmethod
        def process_iter(attrs):
            return [_FakeProc("explorer.exe"), _FakeProc("Spotify.exe"), _FakeProc("python.exe")]

    monkeypatch.setitem(sys.modules, "psutil", _FakePsutil)
    # Make pygetwindow return NO matching windows — simulates tray-only state
    class _FakeWin:
        def __init__(self, title): self.title = title
    fake_gw = type("FakeGW", (), {"getAllWindows": staticmethod(lambda: [_FakeWin("Notepad")])})
    monkeypatch.setitem(sys.modules, "pygetwindow", fake_gw)

    assert native.is_app_running("spotify") is True


def test_is_app_running_falls_back_to_pygetwindow_when_psutil_fails(monkeypatch):
    """If psutil is unavailable / errors, fall back to pygetwindow (preserves
    the old behavior for environments without psutil)."""
    from assistant.automation import native
    import sys

    class _BadPsutil:
        @staticmethod
        def process_iter(attrs):
            raise OSError("simulated psutil failure")

    monkeypatch.setitem(sys.modules, "psutil", _BadPsutil)

    class _FakeWin:
        def __init__(self, title): self.title = title
    fake_gw = type("FakeGW", (), {"getAllWindows": staticmethod(lambda: [_FakeWin("Spotify Free")])})
    monkeypatch.setitem(sys.modules, "pygetwindow", fake_gw)

    assert native.is_app_running("spotify") is True


def test_is_app_running_returns_false_when_both_backends_fail(monkeypatch):
    """If both psutil and pygetwindow are unavailable, fall back to False —
    safer than crashing. Launcher will fire the full open_app flow."""
    from assistant.automation import native
    import sys

    monkeypatch.setitem(sys.modules, "psutil", None)
    monkeypatch.setitem(sys.modules, "pygetwindow", None)

    assert native.is_app_running("spotify") is False


# ─── Cluster 14: knowledge-injection toggle (CE-DYN post-mortem 2026-05-30) ──
#
# Decision: turn OFF dynamic knowledge injection into code-gen prompts for
# v1.0. Stale "never" entries accumulated from situational failures become
# permanent dogma and mislead future generations. WRITES still happen so v1.1
# has historical data to analyze; only READS (prompt injection) are gated.

def test_config_inject_knowledge_defaults_off():
    """Default for v1.0 is OFF — must be explicitly enabled via env var."""
    from assistant import config
    assert hasattr(config, "CODE_EXECUTOR_INJECT_KNOWLEDGE"), \
        "config.CODE_EXECUTOR_INJECT_KNOWLEDGE must exist"
    # Default loaded at import time. If user has the env var set we can't
    # assert False, but we CAN assert the var EXISTS and is a bool.
    assert isinstance(config.CODE_EXECUTOR_INJECT_KNOWLEDGE, bool)


def test_config_inject_knowledge_respects_env_var(monkeypatch):
    """Setting the env var to 'true' enables injection (for v1.1 / dev opt-in)."""
    import importlib
    monkeypatch.setenv("CODE_EXECUTOR_INJECT_KNOWLEDGE", "true")
    from assistant import config
    importlib.reload(config)
    assert config.CODE_EXECUTOR_INJECT_KNOWLEDGE is True
    # Reset for other tests — set back to false then reload
    monkeypatch.setenv("CODE_EXECUTOR_INJECT_KNOWLEDGE", "false")
    importlib.reload(config)
    assert config.CODE_EXECUTOR_INJECT_KNOWLEDGE is False


def test_knowledge_writes_still_happen_when_injection_off():
    """Sanity: the knowledge module's write APIs are independent of the
    injection toggle. Disabling injection MUST NOT disable collection —
    v1.1 needs the data."""
    from assistant import knowledge
    # The write functions still exist and are callable (we don't actually
    # write here — just confirm the API is intact regardless of the toggle).
    assert callable(getattr(knowledge, "add_never_entry", None))
    assert callable(getattr(knowledge, "render_for_llm", None))


def test_knowledge_render_for_llm_unaffected_by_toggle(_isolated_sandbox):
    """The knowledge module itself doesn't read the toggle — it just renders
    whatever is on disk. The orchestrator is what gates the injection.
    Confirmed by inspection — if render_for_llm started checking the flag,
    that would break the v1.1 transition (we want to flip the flag and have
    injection start working immediately, no other code changes)."""
    import inspect
    from assistant import knowledge
    src = inspect.getsource(knowledge.render_for_llm)
    assert "CODE_EXECUTOR_INJECT_KNOWLEDGE" not in src, \
        "knowledge.render_for_llm should not check the injection toggle " \
        "— that gate belongs in the orchestrator caller."
