"""Tests for _apply_replace_blocks — XML fix/replace patching in code_executor."""

import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

for mod_name in (
    "assistant.io.audio.tts", "assistant.io.audio.stt",
    "assistant.io.audio.speaker_verify", "assistant.io.unity_bridge",
    "assistant.io.audio.wake_word",
):
    if mod_name not in sys.modules:
        sys.modules[mod_name] = types.ModuleType(mod_name)

from assistant.code_executor.retry import _apply_replace_blocks


SAMPLE_CODE = """\
import os
import spotipy

sp = spotipy.Spotify(auth=os.environ.get('TOKEN'))

try:
    devices = sp.devices()
    active = None
    for d in devices.get('devices', []):
        if d.get('is_active'):
            active = d['id']
            break
    if active:
        sp.start_playback(device_id=active)
        print("Playing.")
    else:
        print("No device.")
except Exception as e:
    print(f"Error: {e}")
"""


class TestDirectMatch:

    def test_single_line_replacement(self):
        llm = '<fix><old>    devices = sp.devices()</old><new>    devices_resp = sp.devices()</new></fix>'
        result = _apply_replace_blocks(SAMPLE_CODE, llm)
        assert result is not None
        assert "devices_resp = sp.devices()" in result
        assert "    devices_resp" in result

    def test_multi_line_replacement_preserves_indent(self):
        llm = (
            '<fix><old>    devices = sp.devices()\n'
            '    active = None</old>'
            '<new>    devices_resp = sp.devices()\n'
            '    devices = devices_resp.get("devices", [])\n'
            '    active = None</new></fix>'
        )
        result = _apply_replace_blocks(SAMPLE_CODE, llm)
        assert result is not None
        assert '    devices_resp = sp.devices()' in result
        assert '    devices = devices_resp.get("devices", [])' in result
        assert '    active = None' in result


class TestReindentedMatch:

    def test_llm_drops_indent_single_line(self):
        """LLM omits leading spaces in <old> — should still match via dedented logic."""
        llm = '<fix><old>devices = sp.devices()</old><new>devices_resp = sp.devices()</new></fix>'
        result = _apply_replace_blocks(SAMPLE_CODE, llm)
        assert result is not None
        assert "    devices_resp = sp.devices()" in result

    def test_llm_drops_indent_multi_line(self):
        """LLM omits leading spaces — multi-line replacement gets re-indented."""
        llm = (
            '<fix><old>devices = sp.devices()\n'
            'active = None</old>'
            '<new>devices_resp = sp.devices()\n'
            'devices = devices_resp.get("devices", [])\n'
            'active = None</new></fix>'
        )
        result = _apply_replace_blocks(SAMPLE_CODE, llm)
        assert result is not None
        lines = result.splitlines()
        idx = next(i for i, l in enumerate(lines) if 'devices_resp' in l)
        assert lines[idx].startswith('    ')
        assert lines[idx + 1].startswith('    ')
        assert lines[idx + 2].startswith('    ')

    def test_nested_indent_preserved(self):
        """Replacement block with deeper nesting gets correct relative indentation."""
        llm = (
            '<fix><old>active = None\n'
            'for d in devices.get(\'devices\', []):\n'
            '    if d.get(\'is_active\'):\n'
            '        active = d[\'id\']\n'
            '        break</old>'
            '<new>active = None\n'
            'if devices:\n'
            '    active = devices[0][\'id\']</new></fix>'
        )
        result = _apply_replace_blocks(SAMPLE_CODE, llm)
        assert result is not None
        lines = result.splitlines()
        idx = next(i for i, l in enumerate(lines) if 'if devices:' in l)
        assert lines[idx].startswith('    ')
        assert lines[idx + 1].startswith('        ')


class TestNormalizedFallback:

    def test_single_line_extra_spaces(self):
        """Single-line with extra internal whitespace still matches via normalization."""
        code = "    result  =  compute( x )\n"
        llm = '<fix><old>result  =  compute( x )</old><new>result = compute(x)</new></fix>'
        result = _apply_replace_blocks(code, llm)
        assert result is not None
        assert "result = compute(x)" in result


class TestEdgeCases:

    def test_empty_old_skipped(self):
        llm = '<fix><old>  </old><new>something</new></fix>'
        result = _apply_replace_blocks(SAMPLE_CODE, llm)
        assert result is None

    def test_no_match_returns_none_for_all_blocks(self):
        llm = '<fix><old>this_does_not_exist()</old><new>replaced()</new></fix>'
        result = _apply_replace_blocks(SAMPLE_CODE, llm)
        assert result is None

    def test_empty_new_deletes_old(self):
        llm = '<fix><old>    active = None</old><new></new></fix>'
        result = _apply_replace_blocks(SAMPLE_CODE, llm)
        assert result is not None
        assert "active = None" not in result

    def test_sibling_format(self):
        """Handles malformed <old>/<new> siblings (not nested in <fix>)."""
        llm = '<old>    devices = sp.devices()</old><new>    d = sp.devices()</new>'
        result = _apply_replace_blocks(SAMPLE_CODE, llm)
        assert result is not None
        assert "    d = sp.devices()" in result

    def test_multiple_blocks_applied(self):
        llm = (
            '<fix><old>    devices = sp.devices()</old>'
            '<new>    d = sp.devices()</new></fix>'
            '<fix><old>        print("Playing.")</old>'
            '<new>        print("Music started.")</new></fix>'
        )
        result = _apply_replace_blocks(SAMPLE_CODE, llm)
        assert result is not None
        assert "d = sp.devices()" in result
        assert "Music started." in result


class TestIndentationBugRegression:
    """Regression tests for the SyntaxError: unexpected indent bug."""

    def test_stripped_new_text_doesnt_lose_indent(self):
        """The old .strip() bug: multi-line new_text lost first-line indent."""
        code = "    x = get_data()\n    process(x)\n"
        llm = (
            '<fix><old>    x = get_data()</old>'
            '<new>    resp = get_data()\n'
            '    x = resp["items"]</new></fix>'
        )
        result = _apply_replace_blocks(code, llm)
        assert result is not None
        lines = result.splitlines()
        assert lines[0] == '    resp = get_data()'
        assert lines[1] == '    x = resp["items"]'

    def test_dedented_replacement_reindents_correctly(self):
        """LLM sends unindented code for a block that's inside a try."""
        code = (
            "try:\n"
            "    data = fetch()\n"
            "    value = data['key']\n"
            "except:\n"
            "    pass\n"
        )
        llm = (
            '<fix><old>data = fetch()\n'
            'value = data[\'key\']</old>'
            '<new>response = fetch()\n'
            'data = response.get("items", [])\n'
            'value = data[0] if data else None</new></fix>'
        )
        result = _apply_replace_blocks(code, llm)
        assert result is not None
        lines = result.splitlines()
        assert lines[1] == '    response = fetch()'
        assert lines[2] == '    data = response.get("items", [])'
        assert lines[3] == '    value = data[0] if data else None'
