"""Starting the Studio daemon must not tear down TENKA's own logging.

uvicorn applies its default logging config through `logging.config.dictConfig`,
which calls `logging.shutdown()` on every handler in the process. That closed
main.py's `debug.log` FileHandler; `FileHandler.close()` nulls `stream`, and
`emit()` deliberately refuses to reopen a closed `mode="w"` handler, so every
record after `serve()` was dropped in silence while the handler stayed in
`root.handlers` looking healthy. Live logs ended mid-boot, a few lines before
the daemon's "listening" line, for every release since the daemon shipped.

The probe below runs in a subprocess on purpose. Exercising the unfixed case
in-process would close pytest's own logging handlers too and poison every test
that ran afterwards -- the same blast radius that makes the bug worth a guard.
"""
import pathlib
import subprocess
import sys

_PROBE = """
import json, logging, sys, uvicorn

path = sys.argv[1]
pass_log_config = sys.argv[2] == "fixed"

logging.basicConfig(level=logging.INFO)
fh = logging.FileHandler(path, mode="w", encoding="utf-8")
logging.getLogger().addHandler(fh)
logging.getLogger("probe").warning("before")

async def _app(scope, receive, send):
    pass

kwargs = dict(host="127.0.0.1", port=8799, log_level="warning")
if pass_log_config:
    kwargs["log_config"] = None
uvicorn.Config(_app, **kwargs).configure_logging()

logging.getLogger("probe").warning("after")
print(json.dumps({"stream_open": fh.stream is not None}))
"""


def _probe(tmp_path, variant: str) -> tuple[bool, str]:
    """Return (handler still open, what actually landed in the file)."""
    log_path = tmp_path / f"{variant}.log"
    result = subprocess.run(
        [sys.executable, "-c", _PROBE, str(log_path), variant],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    import json
    stream_open = json.loads(result.stdout.strip().splitlines()[-1])["stream_open"]
    return stream_open, log_path.read_text(encoding="utf-8")


def test_uvicorns_default_logging_config_still_closes_our_file_handler(tmp_path):
    """PROOF-OF-FAILURE: this is the bug, reproduced.

    If this ever starts passing, uvicorn changed and the guard in server.py may
    no longer be load-bearing -- but do not remove it on that evidence alone;
    a pinned dependency can regress back.
    """
    stream_open, contents = _probe(tmp_path, "unfixed")
    assert not stream_open, "uvicorn no longer closes foreign handlers"
    assert "before" in contents
    assert "after" not in contents, "the record should have been silently dropped"


def test_passing_log_config_none_keeps_the_file_handler_writing(tmp_path):
    stream_open, contents = _probe(tmp_path, "fixed")
    assert stream_open
    assert "before" in contents
    assert "after" in contents, "records after configure_logging must still land"


def test_the_daemon_passes_log_config_none(tmp_path):
    """The property above only protects us if serve() actually asks for it."""
    source = pathlib.Path("assistant/io/api/server.py").read_text(encoding="utf-8")
    assert "log_config=None" in source, (
        "serve() lets uvicorn reconfigure process-wide logging; debug.log will "
        "stop recording the moment the daemon starts"
    )
