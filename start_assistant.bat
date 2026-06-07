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

REM Load .env file if it exists (simple key=value parsing)
if exist .env (
    for /f "usebackq tokens=1,* delims==" %%a in (".env") do (
        set "%%a=%%b"
    )
)

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
