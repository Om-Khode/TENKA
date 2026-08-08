"""Files, commands and system — with confinement proven, not asserted."""
import pytest

from assistant.actions import studio_runtime_system as srs
from assistant.actions.studio_runtime_system import (
    LiveCommandRuntime, LiveFileRuntime, resolve_within,
)


@pytest.fixture()
def root(tmp_path):
    (tmp_path / "notes.md").write_text("hello", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "deep.txt").write_text("deep", encoding="utf-8")
    return tmp_path


def test_a_plain_relative_path_resolves(root):
    assert resolve_within(root, "notes.md") == (root / "notes.md").resolve()


def test_a_nested_path_resolves(root):
    assert resolve_within(root, "sub/deep.txt") == (root / "sub" / "deep.txt").resolve()


@pytest.mark.parametrize("attack", [
    "../outside.txt",
    "../../outside.txt",
    "sub/../../outside.txt",
    "..\\outside.txt",
    "sub\\..\\..\\outside.txt",
    "/etc/passwd",
    "C:\\Windows\\System32\\config\\SAM",
    "\\\\server\\share\\file.txt",
    "sub/./../../outside.txt",
])
def test_traversal_is_refused(root, attack):
    with pytest.raises(ValueError):
        resolve_within(root, attack)


def test_a_symlink_pointing_outside_is_refused(root, tmp_path):
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    link = root / "escape.txt"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation needs developer mode or admin on Windows")
    with pytest.raises(ValueError):
        resolve_within(root, "escape.txt")


def test_an_empty_path_is_refused(root):
    for junk in ("", "   ", ".", "..", None):
        with pytest.raises((ValueError, TypeError)):
            resolve_within(root, junk)


# ─── Extra attacks the brief's list doesn't cover ──────────────────────────
# Windows-specific forms that look like ordinary relative paths and would
# resolve-then-check straight through: an alternate-data-stream suffix, a
# drive-relative form, an extended-length \\?\ prefix, and a reserved device
# name that Windows redirects regardless of the containing directory.
@pytest.mark.parametrize("attack", [
    "notes.md:secret",              # NTFS alternate data stream
    "D:\\outside.txt",               # absolute path on another drive
    "\\\\?\\C:\\Windows\\System32",  # extended-length / device-namespace prefix
])
def test_windows_specific_traversal_forms_are_refused(root, attack):
    with pytest.raises(ValueError):
        resolve_within(root, attack)


@pytest.mark.parametrize("device", ["CON", "con", "NUL", "COM1", "LPT1"])
def test_reserved_device_names_are_refused(root, device):
    """CON, NUL, COM1... are redirected to hardware devices by Windows even
    when addressed through a real directory path — opening one for a
    'file read' can block or misbehave. Confinement's job is the whole
    files domain staying inside the sandbox, not just staying under root."""
    with pytest.raises(ValueError):
        resolve_within(root, device)


@pytest.mark.asyncio
async def test_listing_an_unknown_root_raises():
    with pytest.raises(KeyError):
        await LiveFileRuntime().listing("not-a-root")


@pytest.mark.asyncio
async def test_roots_are_the_three_the_page_claims():
    assert sorted(await LiveFileRuntime().roots()) == ["desktop", "documents", "downloads"]


@pytest.mark.asyncio
@pytest.mark.parametrize("attack", ["desktop/../../etc", "desktop/..", "../desktop",
                                    "desktop\\..\\..\\windows"])
async def test_a_nested_path_cannot_climb_out_of_its_root(attack):
    with pytest.raises((ValueError, KeyError)):
        await LiveFileRuntime().listing(attack)


def test_classification_is_extension_driven_and_brand_free():
    from pathlib import Path as P
    from assistant.actions.studio_runtime_system import _classify
    assert _classify(P("a/b.py")) == ("code", "python")
    assert _classify(P("a/b.md")) == ("text", "")
    assert _classify(P("a/b.png")) == ("image", "")
    assert _classify(P("a/b.dat")) == ("binary", "")


@pytest.mark.asyncio
async def test_command_catalogue_declares_a_grant_per_entry():
    for command in await LiveCommandRuntime().catalogue():
        assert command.required_grant, f"{command.command_id} declares no grant"


@pytest.mark.asyncio
async def test_running_an_unknown_command_fails_without_raising():
    outcome = await LiveCommandRuntime().run("no_such_command")
    assert outcome.ok is False


def test_the_catalogue_names_no_application():
    """THE rule: commands are OS capabilities, not applications."""
    import pathlib
    source = pathlib.Path("assistant/actions/studio_runtime_system.py").read_text(encoding="utf-8")
    for brand in ("chrome", "firefox", "vscode", "spotify", "notepad", "edge"):
        assert brand not in source.lower(), f"catalogue names '{brand}'"


# ─── Backup state accessor (added to orchestrator — see report) ───────────
# Pure settings reads through a real temp SQLite DB, same style as
# tests/test_studio_runtime_live.py. Never touches a provider, never
# uploads/downloads anything.
class TestBackupStateAccessor:
    @pytest.fixture()
    def backup_db(self, tmp_path):
        from assistant.storage.db import init_db, _reset_for_testing
        _reset_for_testing()
        init_db(tmp_path / "test.db")
        yield
        _reset_for_testing()

    def test_get_state_defaults_when_nothing_configured(self, backup_db):
        from assistant.io.backup import orchestrator
        state = orchestrator.get_state()
        assert state["enabled"] is False
        assert state["provider"] == "google_drive"
        assert state["last_backup_at"] == ""
        assert state["last_result"] == ""
        assert state["size_bytes"] == 0

    def test_get_state_reflects_stored_settings(self, backup_db):
        from assistant.storage.db import get_db
        from assistant.storage.repos.settings import SettingsRepo
        from assistant.io.backup import orchestrator

        settings = SettingsRepo(get_db())
        settings.set("backup_enabled", True, source="test")
        settings.set("backup_provider", "google_drive", source="test")
        settings.set("backup_last_backup_at", "2026-08-07T04:00:00Z", source="test")
        settings.set("backup_last_backup_status", "success", source="test")
        settings.set("backup_last_backup_size_bytes", 18432112, source="test")

        state = orchestrator.get_state()
        assert state == {
            "enabled": True,
            "provider": "google_drive",
            "last_backup_at": "2026-08-07T04:00:00Z",
            "last_result": "success",
            "size_bytes": 18432112,
        }


# ─── Review fix 1: a bare root is not a valid delete/rename target ────────
# is_protected_path() guards Windows/Program Files/drive roots, never these
# three user-data roots -- nothing downstream catches delete("desktop") on
# its own. The guard must raise before any real filesystem call, which is
# exactly why these are safe to run for real: _resolve() (inside the
# to_thread call) raises ValueError before _delete_sync/_rename_sync ever run.
@pytest.mark.asyncio
@pytest.mark.parametrize("root_name", ["desktop", "documents", "downloads"])
async def test_deleting_a_root_itself_is_refused(root_name):
    with pytest.raises(ValueError):
        await LiveFileRuntime().delete(root_name)


@pytest.mark.asyncio
@pytest.mark.parametrize("root_name", ["desktop", "documents", "downloads"])
async def test_renaming_a_root_itself_is_refused(root_name):
    with pytest.raises(ValueError):
        await LiveFileRuntime().rename(root_name, "x")


@pytest.mark.asyncio
@pytest.mark.parametrize("root_name", ["desktop", "documents", "downloads"])
async def test_deleting_a_path_that_traverses_back_to_root_is_refused(root_name):
    with pytest.raises(ValueError):
        await LiveFileRuntime().delete(f"{root_name}/sub/..")


@pytest.mark.asyncio
@pytest.mark.parametrize("root_name", ["desktop", "documents", "downloads"])
async def test_renaming_a_path_that_traverses_back_to_root_is_refused(root_name):
    with pytest.raises(ValueError):
        await LiveFileRuntime().rename(f"{root_name}/sub/..", "x")


@pytest.mark.asyncio
async def test_listing_a_bare_root_is_still_allowed():
    """listing() is the one caller that legitimately means the root itself --
    it's how the client shows the top level of "desktop" at all."""
    entries = await LiveFileRuntime().listing("desktop")
    assert isinstance(entries, list)


# ─── Review fix 2: read() is bounded for every content kind ───────────────
# No test in the original suite called read() at all. _MAX_PREVIEW_BYTES is
# monkeypatched small so "oversized" doesn't require gigabyte fixtures --
# the mechanism under test is "does the cap apply", not "how big is default".
@pytest.mark.parametrize("suffix,write", [
    (".txt", lambda p: p.write_text("x" * 1000, encoding="utf-8")),
    (".py", lambda p: p.write_text("y" * 1000, encoding="utf-8")),
])
def test_read_caps_text_and_code_previews(tmp_path, monkeypatch, suffix, write):
    monkeypatch.setattr(srs, "_MAX_PREVIEW_BYTES", 100)
    target = tmp_path / f"big{suffix}"
    write(target)
    content = LiveFileRuntime._read_sync(f"desktop/big{suffix}", target)
    assert content.truncated is True
    assert len(content.text) == 100


def test_read_caps_an_image_preview_without_reading_the_bytes_at_all(tmp_path, monkeypatch):
    monkeypatch.setattr(srs, "_MAX_PREVIEW_BYTES", 100)
    target = tmp_path / "big.png"
    target.write_bytes(b"\x89PNG" + b"0" * 1000)
    content = LiveFileRuntime._read_sync("desktop/big.png", target)
    assert content.truncated is True
    assert content.text == ""  # size check short-circuits before read_bytes()


def test_read_of_a_formerly_unbounded_readable_extension_is_now_capped(tmp_path, monkeypatch):
    """.env used to fall through _classify() to "binary", which delegated to
    file_manager.read_file()'s unbounded path.read_text(). It is classified
    as "text" now, so it takes the same capped path.open().read(cap) as any
    other text file -- this is the exact gap the review flagged."""
    monkeypatch.setattr(srs, "_MAX_PREVIEW_BYTES", 100)
    target = tmp_path / "secrets.env"
    target.write_text("z" * 1000, encoding="utf-8")
    content = LiveFileRuntime._read_sync("desktop/secrets.env", target)
    assert content.content_kind == "text"
    assert content.truncated is True
    assert len(content.text) == 100


def test_read_of_a_binary_extension_never_reads_the_whole_file(tmp_path):
    """.dat is neither a READABLE_EXTENSIONS suffix nor a rich document --
    file_manager.read_file() returns a short "can't read" message without
    calling read_text() at all, so this path is bounded independent of file
    size (no monkeypatched cap needed to prove it)."""
    target = tmp_path / "blob.dat"
    target.write_bytes(b"\x00" * 2000)
    content = LiveFileRuntime._read_sync("desktop/blob.dat", target)
    assert content.content_kind == "binary"
    assert len(content.text) < 200


# ─── Review fix 3: lock_workstation reports what Win32 actually returned ──
# ctypes.windll is replaced wholesale with a fake before any call -- the real
# Win32 API is never reached, so this does not execute the command the
# task's dispatch prohibits. Only lock_workstation is exercised: it is the
# one branch with a documented, checkable return value (LockWorkStation is a
# BOOL; keybd_event is VOID and has nothing to check -- see the source
# comment). volume_up/down and screenshot are never invoked by these tests.
class _FakeUser32:
    def __init__(self, lock_result: int) -> None:
        self._lock_result = lock_result
        self.lock_calls = 0

    def LockWorkStation(self):
        self.lock_calls += 1
        return self._lock_result


class _FakeWindll:
    def __init__(self, user32) -> None:
        self.user32 = user32


def test_lock_workstation_reports_failure_when_win32_returns_zero(monkeypatch):
    import ctypes
    fake_user32 = _FakeUser32(lock_result=0)
    monkeypatch.setattr(ctypes, "windll", _FakeWindll(fake_user32))

    outcome = LiveCommandRuntime._run_sync("lock_workstation")

    assert outcome.ok is False
    assert fake_user32.lock_calls == 1


def test_lock_workstation_reports_success_when_win32_returns_nonzero(monkeypatch):
    import ctypes
    fake_user32 = _FakeUser32(lock_result=1)
    monkeypatch.setattr(ctypes, "windll", _FakeWindll(fake_user32))

    outcome = LiveCommandRuntime._run_sync("lock_workstation")

    assert outcome.ok is True
    assert fake_user32.lock_calls == 1
