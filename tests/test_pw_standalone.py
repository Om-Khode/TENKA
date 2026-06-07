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