"""TENKA — first-run setup wizard.

Stdlib-only by design: this script runs *before* `pip install`, so it
cannot import anything from `assistant/` or from any third-party package.

Usage:
    python scripts/setup.py            # interactive
    python scripts/setup.py --force    # ignore the setup marker, re-do everything
    python scripts/setup.py --no-launch  # never offer to launch at the end
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path


# ─── Paths and constants ────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = REPO_ROOT / ".env"
ENV_EXAMPLE_PATH = REPO_ROOT / ".env.example"
MARKER_PATH = REPO_ROOT / ".tenka_setup.json"
REQUIREMENTS_PATH = REPO_ROOT / "requirements.txt"

MIN_PY = (3, 11)
MAX_PY = (3, 11)
MARKER_SCHEMA_VERSION = 1
TOTAL_STEPS = 7


# ─── Provider table — data, not branches ────────────────────────────
# Adding a provider is a row append. There is no app-specific code path.
PROVIDERS = [
    {
        "key": "GEMINI_API_KEY",
        "name": "Gemini",
        "url": "https://aistudio.google.com/apikey",
        "tier": "primary",
        "blurb": "Routes intent, planning, synthesis, vision. Free tier is generous.",
    },
    {
        "key": "GROQ_API_KEY",
        "name": "Groq",
        "url": "https://console.groq.com/",
        "tier": "fallback",
        "blurb": "Fast 70b fallback. Free tier ~1K req/day.",
    },
    {
        "key": "CEREBRAS_API_KEY",
        "name": "Cerebras",
        "url": "https://cloud.cerebras.ai/",
        "tier": "fallback",
        "blurb": "Synthesis fallback (gpt-oss-120b). Free tier.",
    },
    {
        "key": "TAVILY_API_KEY",
        "name": "Tavily",
        "url": "https://app.tavily.com/",
        "tier": "optional",
        "blurb": "Web search. Skip if you don't need the web_search intent.",
    },
    {
        "key": "JINA_API_KEY",
        "name": "Jina",
        "url": "https://jina.ai/api-dashboard/",
        "tier": "optional",
        "blurb": "Reranker for memory retrieval. Skip for local-only reranking.",
    },
    {
        "key": "HF_TOKEN",
        "name": "Hugging Face",
        "url": "https://huggingface.co/settings/tokens",
        "tier": "optional",
        "blurb": "Only needed for gated HF models (most TENKA defaults are open).",
    },
]


# ─── Terminal styling (ANSI; Windows 10+ understands these) ─────────
def _supports_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if not sys.stdout.isatty():
        return False
    if sys.platform == "win32":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        except Exception:
            return False
    return True


_COLOR = _supports_color()


def _c(code: str, text: str) -> str:
    return f"\x1b[{code}m{text}\x1b[0m" if _COLOR else text


def bold(t: str) -> str: return _c("1", t)
def dim(t: str) -> str: return _c("2", t)
def green(t: str) -> str: return _c("32", t)
def yellow(t: str) -> str: return _c("33", t)
def red(t: str) -> str: return _c("31", t)
def cyan(t: str) -> str: return _c("36", t)


def heading(n: int, total: int, title: str) -> None:
    print(f"\n{bold(cyan(f'[{n}/{total}]'))} {bold(title)}")


def info(msg: str) -> None: print(f"  {msg}")
def ok(msg: str) -> None: print(f"  {green('OK')} {msg}")
def warn(msg: str) -> None: print(f"  {yellow('!!')} {msg}")
def fail(msg: str) -> None: print(f"  {red('XX')} {msg}")


# ─── Setup marker (idempotence + schema versioning) ─────────────────
def load_marker() -> dict:
    if not MARKER_PATH.exists():
        return {}
    try:
        data = json.loads(MARKER_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if data.get("version") != MARKER_SCHEMA_VERSION:
        return {}
    return data


def save_marker(marker: dict) -> None:
    marker["version"] = MARKER_SCHEMA_VERSION
    marker["updated"] = datetime.now().isoformat(timespec="seconds")
    _atomic_write(MARKER_PATH, json.dumps(marker, indent=2, sort_keys=True))


# ─── Atomic write (temp + rename) ───────────────────────────────────
def _atomic_write(path: Path, content: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(content)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def _file_sha(path: Path) -> str:
    h = hashlib.sha256()
    try:
        h.update(path.read_bytes())
    except Exception:
        return ""
    return h.hexdigest()


# ─── .env merge (read-modify-write, preserves existing values) ──────
_ENV_LINE_RE = re.compile(r"^\s*([A-Z_][A-Z0-9_]*)\s*=\s*(.*)$")


def parse_env(text: str) -> dict:
    out: dict[str, str] = {}
    for line in text.splitlines():
        m = _ENV_LINE_RE.match(line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def merge_env_text(existing_text: str, updates: dict, template_text: str = "") -> str:
    """Return new .env text:
      - keeps all lines from existing_text in order (comments, blanks, formatting)
      - replaces values for keys in `updates` if those keys already appear
      - appends missing `updates` keys under a "Added by setup wizard" header
      - if existing_text is empty AND template_text is given, uses the template as base
    """
    base = existing_text if existing_text.strip() else template_text
    seen: set[str] = set()
    out_lines: list[str] = []
    for line in base.splitlines():
        m = _ENV_LINE_RE.match(line)
        if m and m.group(1) in updates:
            key = m.group(1)
            out_lines.append(f"{key}={updates[key]}")
            seen.add(key)
        else:
            out_lines.append(line)
    missing = [(k, v) for k, v in updates.items() if k not in seen]
    if missing:
        if out_lines and out_lines[-1].strip():
            out_lines.append("")
        out_lines.append("# ─── Added by setup wizard ───────────────────────────")
        for k, v in missing:
            out_lines.append(f"{k}={v}")
    return "\n".join(out_lines) + "\n"


# ─── Region / timezone auto-detect ──────────────────────────────────
def autodetect_region() -> str:
    """Two-letter ISO 3166-1 alpha-2 country code from the OS, or empty string.

    Tries, in order:
      1. POSIX-style locale ('en_US', 'ja_JP') via locale.getlocale().
      2. Windows Win32 GetUserDefaultGeoName (Windows 10 1709+).
      3. Env vars LANG / LC_ALL / LC_CTYPE ('en_US.UTF-8').
    """
    try:
        import locale
        loc = locale.getlocale()[0] or ""
        m = re.search(r"[_-]([A-Za-z]{2})$", loc)
        if m:
            return m.group(1).upper()
    except Exception:
        pass
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            fn = kernel32.GetUserDefaultGeoName
            fn.argtypes = [wintypes.LPWSTR, ctypes.c_int]
            fn.restype = ctypes.c_int
            buf = ctypes.create_unicode_buffer(16)
            if fn(buf, 16) > 0 and buf.value:
                return buf.value.upper()
        except (AttributeError, OSError):
            pass
    for var in ("LANG", "LC_ALL", "LC_CTYPE"):
        v = os.environ.get(var, "")
        m = re.search(r"[_-]([A-Za-z]{2})(?:[.@]|$)", v)
        if m:
            return m.group(1).upper()
    return ""


# Older IANA zone names that are still kept as Links in tzdata's `backward`
# file. tzlocal / Windows mappings often resolve to the legacy form; we
# normalise to the current canonical name so the .env is future-proof.
# Source: https://data.iana.org/time-zones/ (zone1970.tab + backward).
_TZ_ALIAS_TO_CANONICAL = {
    "Asia/Calcutta": "Asia/Kolkata",
    "Asia/Katmandu": "Asia/Kathmandu",
    "Asia/Rangoon": "Asia/Yangon",
    "Asia/Saigon": "Asia/Ho_Chi_Minh",
    "Asia/Thimbu": "Asia/Thimphu",
    "Europe/Kiev": "Europe/Kyiv",
    "America/Buenos_Aires": "America/Argentina/Buenos_Aires",
    "Africa/Asmera": "Africa/Asmara",
    "Atlantic/Faeroe": "Atlantic/Faroe",
    "Pacific/Ponape": "Pacific/Pohnpei",
    "Pacific/Truk": "Pacific/Chuuk",
}


def _canonicalize_tz(name: str) -> str:
    """Normalise legacy IANA aliases to their modern canonical form."""
    return _TZ_ALIAS_TO_CANONICAL.get(name, name)


def autodetect_timezone() -> str:
    """IANA timezone name (e.g. 'Asia/Kolkata') from the OS, or empty string.

    Tries, in order:
      1. datetime.now().astimezone().tzinfo  (Linux/macOS — gives the IANA name).
      2. tzlocal.get_localzone()  (handles Windows registry → IANA mapping).

    Results are passed through `_canonicalize_tz` so legacy IANA aliases
    (e.g. 'Asia/Calcutta') become their modern names (e.g. 'Asia/Kolkata').
    """
    try:
        tz = datetime.now().astimezone().tzinfo
        name = str(tz) if tz else ""
        if "/" in name:
            return _canonicalize_tz(name)
    except Exception:
        pass
    try:
        import tzlocal
        z = tzlocal.get_localzone()
        name = getattr(z, "key", None) or getattr(z, "zone", None) or str(z)
        if name:
            return _canonicalize_tz(name)
    except Exception:
        pass
    return ""


def _prompt_region(default: str) -> str:
    while True:
        try:
            v = input(f"  Country code (2 letters) [{default}]: ").strip().upper() or default
        except (KeyboardInterrupt, EOFError):
            print()
            sys.exit(1)
        if not v:
            return ""
        if re.fullmatch(r"[A-Z]{2}", v):
            return v
        warn("Use a 2-letter ISO code like US, IN, GB, JP.")


def _prompt_timezone(default: str) -> str:
    try:
        v = input(
            f"  IANA timezone (e.g. America/New_York, Asia/Kolkata) [{default}]: "
        ).strip() or default
    except (KeyboardInterrupt, EOFError):
        print()
        sys.exit(1)
    return v


# ─── Steps ──────────────────────────────────────────────────────────
def step_python_version(marker: dict, args) -> None:
    heading(1, TOTAL_STEPS, "Python version check")
    v = sys.version_info
    info(f"Detected Python {v.major}.{v.minor}.{v.micro}")
    pair = (v.major, v.minor)
    if pair < MIN_PY:
        fail(f"TENKA requires Python {MIN_PY[0]}.{MIN_PY[1]}+. Please upgrade.")
        sys.exit(2)
    if pair > MAX_PY:
        warn(
            f"Python {v.major}.{v.minor} is newer than the tested version "
            f"({MAX_PY[0]}.{MAX_PY[1]}). Some deps (faiss, torch) can lag on "
            f"newer Python. Continuing."
        )
    else:
        ok(f"Python {v.major}.{v.minor} is supported.")
    marker.setdefault("steps", {})["python_version"] = {
        "version": f"{v.major}.{v.minor}.{v.micro}", "ok": True,
    }


def step_pip_install(marker: dict, args) -> None:
    heading(2, TOTAL_STEPS, "Install Python dependencies")
    req_hash = _file_sha(REQUIREMENTS_PATH)
    prev = marker.get("steps", {}).get("pip_install", {})
    if prev.get("requirements_sha") == req_hash and prev.get("ok") and not args.force:
        ok("requirements.txt unchanged since last run — skipping.")
        return
    info(f"Running: pip install -r {REQUIREMENTS_PATH.name}")
    info(dim("(several minutes — torch is ~2 GB)"))
    cmd = [sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS_PATH)]
    res = subprocess.run(cmd)
    if res.returncode != 0:
        fail("pip install failed. Re-run after resolving the errors above.")
        sys.exit(res.returncode)
    ok("Dependencies installed.")
    marker.setdefault("steps", {})["pip_install"] = {
        "requirements_sha": req_hash, "ok": True,
    }


def step_playwright_chromium(marker: dict, args) -> None:
    heading(3, TOTAL_STEPS, "Install Playwright Chromium")
    prev = marker.get("steps", {}).get("playwright_chromium", {})
    if prev.get("ok") and not args.force:
        ok("Chromium already installed — skipping.")
        return
    info("Running: playwright install chromium")
    res = subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"])
    if res.returncode != 0:
        warn(
            "Playwright install failed. Browser automation will not work until "
            "you run `python -m playwright install chromium` manually."
        )
        marker.setdefault("steps", {})["playwright_chromium"] = {"ok": False}
        return
    ok("Chromium installed.")
    marker.setdefault("steps", {})["playwright_chromium"] = {"ok": True}


def step_api_keys(marker: dict, args) -> dict:
    heading(4, TOTAL_STEPS, "API keys")
    existing = parse_env(ENV_PATH.read_text(encoding="utf-8")) if ENV_PATH.exists() else {}
    collected: dict[str, str] = {}
    info("For each provider, paste a key — or press Enter to skip.")
    info(dim("Existing values are kept unless you provide a new one."))
    print()
    for p in PROVIDERS:
        cur = existing.get(p["key"], "").strip()
        tier_paint = green if p["tier"] == "primary" else yellow if p["tier"] == "fallback" else dim
        print(f"  {bold(p['name'])} {tier_paint('(' + p['tier'] + ')')}")
        print(f"    {dim(p['blurb'])}")
        print(f"    {dim('Get a key: ' + p['url'])}")
        if cur:
            print(f"    {dim('(value already set — press Enter to keep)')}")
        try:
            val = input("    > ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            sys.exit(1)
        if val:
            collected[p["key"]] = val
        print()
    cloud_keys = {p["key"] for p in PROVIDERS if p["tier"] in ("primary", "fallback")}
    has_any_cloud = any(
        (collected.get(k) or existing.get(k, "").strip()) for k in cloud_keys
    )
    if not has_any_cloud:
        warn("No cloud LLM key set. TENKA will only run with a local Ollama daemon.")
    marker.setdefault("steps", {})["api_keys"] = {
        "providers_offered": [p["key"] for p in PROVIDERS], "ok": True,
    }
    return collected


def step_region_timezone(marker: dict, args) -> dict:
    heading(5, TOTAL_STEPS, "Region and timezone")
    existing = parse_env(ENV_PATH.read_text(encoding="utf-8")) if ENV_PATH.exists() else {}
    region = autodetect_region()
    tz = autodetect_timezone()
    info(f"Auto-detected region:   {bold(region or '(unknown)')}")
    info(f"Auto-detected timezone: {bold(tz or '(unknown)')}")
    print()
    if region and tz:
        try:
            ans = input("  Use these? [Y/n] ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print()
            sys.exit(1)
        if ans in ("", "y", "yes"):
            chosen_region, chosen_tz = region, tz
        else:
            chosen_region = _prompt_region(existing.get("USER_REGION", "") or region)
            chosen_tz = _prompt_timezone(existing.get("USER_TIMEZONE", "") or tz)
    else:
        info(dim("Couldn't auto-detect — please enter manually."))
        chosen_region = _prompt_region(existing.get("USER_REGION", "") or region)
        chosen_tz = _prompt_timezone(existing.get("USER_TIMEZONE", "") or tz)
    out: dict[str, str] = {}
    if chosen_region:
        out["USER_REGION"] = chosen_region
    if chosen_tz:
        out["USER_TIMEZONE"] = chosen_tz
    marker.setdefault("steps", {})["region_timezone"] = {
        "region": chosen_region, "timezone": chosen_tz, "ok": True,
    }
    return out


def step_write_env(marker: dict, args, updates: dict) -> None:
    heading(6, TOTAL_STEPS, "Write .env")
    existing_text = ENV_PATH.read_text(encoding="utf-8") if ENV_PATH.exists() else ""
    template_text = ENV_EXAMPLE_PATH.read_text(encoding="utf-8") if ENV_EXAMPLE_PATH.exists() else ""
    new_text = merge_env_text(existing_text, updates, template_text)
    _atomic_write(ENV_PATH, new_text)
    ok(f"Wrote {ENV_PATH.name} ({len(updates)} value(s) set/updated; existing preserved).")
    marker.setdefault("steps", {})["write_env"] = {
        "keys_written": sorted(updates.keys()), "ok": True,
    }


def step_done(marker: dict, args) -> None:
    heading(7, TOTAL_STEPS, "Done — next steps")
    print()
    print(f"  {green('TENKA is set up.')} To start her:")
    print(f"    {bold('Windows:')}  start_assistant.bat")
    print(f"    {bold('Any OS:')}   python -m assistant.main")
    print()
    print(f"  Re-run the wizard later: {bold('python scripts/setup.py')}")
    print(f"  Edit settings directly:  {bold('.env')} at the repo root")
    print()
    if args.no_launch:
        return
    try:
        ans = input("  Launch TENKA now? [y/N] ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        print()
        return
    if ans in ("y", "yes"):
        info("Launching: python -m assistant.main")
        try:
            subprocess.run([sys.executable, "-m", "assistant.main"])
        except KeyboardInterrupt:
            # Ctrl+C reaches both processes; the child shuts down on its own.
            # Swallow here so we don't print a traceback after a clean exit.
            pass


# ─── Entry point ────────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="TENKA — first-run setup wizard.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--force", action="store_true",
                        help="Re-run all steps even if marker says they're done.")
    parser.add_argument("--no-launch", action="store_true",
                        help="Skip the 'launch TENKA now?' prompt at the end.")
    args = parser.parse_args(argv)

    print(bold(cyan("\nTENKA — setup wizard\n")))
    print(dim(f"  repo: {REPO_ROOT}"))

    marker = load_marker()
    step_python_version(marker, args)
    step_pip_install(marker, args)
    step_playwright_chromium(marker, args)
    api_updates = step_api_keys(marker, args) or {}
    geo_updates = step_region_timezone(marker, args) or {}
    step_write_env(marker, args, {**api_updates, **geo_updates})
    save_marker(marker)
    step_done(marker, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
