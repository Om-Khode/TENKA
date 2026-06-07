@echo off
chcp 65001 >nul
REM ─────────────────────────────────────────────────────
REM  TENKA — first-run setup wizard
REM
REM  Checks Python version, installs dependencies, downloads
REM  Playwright Chromium, prompts for API keys, writes .env.
REM
REM  Re-running is safe; completed steps are skipped via
REM  .tenka_setup.json marker. Use --force to redo everything.
REM ─────────────────────────────────────────────────────

cd /d "%~dp0"

python scripts\setup.py %*

if errorlevel 1 (
    echo.
    echo [ERROR] Setup did not complete cleanly. See messages above.
    echo.
    pause
)
