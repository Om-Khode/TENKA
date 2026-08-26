"""Manual smoke check: can Playwright start a driver and launch Chromium?

Not a test, and it was in `tests/` until 2026-08-25. pytest imported it,
`asyncio.run(main())` ran at module scope, and a real Chromium launched -- 14.6
seconds of it during a baseline run -- while pytest collected nothing and
reported the file EMPTY. A file that looks like coverage, provides none, and
starts a browser process.

Run by hand when the browser stack is suspect:

    py -3.11 tools/playwright_smoke.py
"""
import asyncio, os
from pathlib import Path
os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(Path.home() / "TENKA" / "browser-cache"))
from playwright.async_api import async_playwright

async def main():
    print("Starting driver...")
    pw = await async_playwright().start()
    print("Driver started. Launching chromium...")
    b = await pw.chromium.launch(headless=True)
    print("Chromium launched. Closing.")
    await b.close()
    await pw.stop()
    print("OK")

asyncio.run(main())