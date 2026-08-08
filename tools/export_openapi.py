"""Write the daemon's OpenAPI schema to a file.

Studio generates its TypeScript types from this, so the Pydantic models stay
the single source of truth for both sides. Run from the repo root:

    py -3.11 tools/export_openapi.py openapi.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assistant.io.api.app import create_app          # noqa: E402
from assistant.io.api.vault import TokenVault        # noqa: E402
from tests.fakes.studio_runtime import build_fake_runtime  # noqa: E402


def main() -> int:
    destination = Path(sys.argv[1] if len(sys.argv) > 1 else "openapi.json")
    app = create_app(build_fake_runtime(), TokenVault(Path.cwd() / ".openapi-tmp"),
                     origins=["http://localhost:3000"])
    app.openapi_url = "/openapi.json"   # off in the running daemon, on to export
    schema = app.openapi()
    destination.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    print(f"wrote {destination} — {len(schema['paths'])} paths")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
