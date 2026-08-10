@echo off
chcp 65001 >nul
REM ─────────────────────────────────────────────────────
REM  TENKA — Voice Assistant Launcher
REM
REM  This script starts the Python voice assistant.
REM  Place this in the root of your TENKA project.
REM
REM  First-time setup:
REM    1. Install Python 3.10+ from python.org
REM    2. Open a terminal in this folder and run:
REM         pip install -r requirements.txt
REM    3. Set your Groq API key (optional, for cloud LLM):
REM         set GROQ_API_KEY=your_key_here
REM       Or create a .env file with: GROQ_API_KEY=your_key
REM ─────────────────────────────────────────────────────

cd /d "%~dp0"

REM Load .env if present.
REM
REM This was a bare `set "%%a=%%b"`, which trims nothing and strips nothing.
REM A line written `KEY = "value"` -- legal-looking, and how most people write
REM it -- created a variable literally named "KEY " (trailing space) holding
REM ` "value"` (leading space, quotes kept), so os.getenv("KEY") returned None
REM and nothing said why. That cost a real debugging session on 2026-08-10:
REM STUDIO_API_ENABLED was set in .env and the daemon still would not start.
REM
REM :setenv trims whitespace either side of the `=` and strips one layer of
REM surrounding quotes. `eol=#` skips comment lines.
if exist .env (
    for /f "usebackq eol=# tokens=1,* delims==" %%a in (".env") do call :setenv "%%a" "%%b"
)
goto :env_loaded

:setenv
set "_k=%~1"
set "_v=%~2"
REM Double quotes go FIRST, before any comparison. `if "%_v:~0,1%"==" "` with a
REM value that starts with a quote expands to `if """==" "`, which cmd cannot
REM parse -- the trim below would die on exactly the `KEY = "value"` line this
REM whole routine exists to handle. Removal (not a comparison) is parse-safe,
REM and no .env value legitimately contains a double quote.
if defined _v set "_v=%_v:"=%"
:setenv_trim_key
if not defined _k goto :eof
if "%_k:~-1%"==" " set "_k=%_k:~0,-1%" & goto :setenv_trim_key
:setenv_trim_val
if not defined _v goto :setenv_assign
if "%_v:~0,1%"==" " set "_v=%_v:~1%" & goto :setenv_trim_val
if "%_v:~-1%"==" " set "_v=%_v:~0,-1%" & goto :setenv_trim_val
REM Single quotes are safe to compare, so those are stripped only when they
REM actually wrap the whole value.
if defined _v if "%_v:~0,1%"=="'" if "%_v:~-1%"=="'" set "_v=%_v:~1,-1%"
:setenv_assign
set "%_k%=%_v%"
goto :eof

:env_loaded

echo.
echo  ╔══════════════════════════════════════════════╗
echo  ║       TENKA — Voice Assistant (Python)       ║
echo  ╚══════════════════════════════════════════════╝
echo.

python -m assistant.main

if errorlevel 1 (
    echo.
    echo [ERROR] Python exited with an error.
    echo Make sure Python 3.10+ is installed and requirements are met:
    echo   pip install -r requirements.txt
    echo.
)

pause
