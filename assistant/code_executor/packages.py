"""packages.py — Package metadata constants for the code_executor router.

Extracted from routing.py to break the circular import:
  routing → prompts → router_examples → routing.

This module is a leaf — imports stdlib only. Both routing.py and
router_examples.py import from here.
"""

# ─── Tier 2 allowlist ──────────────────────────────────────────────────────

TIER2_ALLOWED_PACKAGES = frozenset({
    "requests", "beautifulsoup4", "httpx",
    "spotipy",
    "psutil",
    "pandas", "openpyxl",
    "python-docx",
    "google-api-python-client", "google-auth-oauthlib",
    "pycaw",
    "neonize",
    "kociemba",
    "opencv-python", "numpy",
})

# ─── Package name → import name (when they differ) ────────────────────────

_PACKAGE_IMPORT_NAMES = {
    "beautifulsoup4": "bs4",
    "google-api-python-client": "googleapiclient",
    "google-auth-oauthlib": "google_auth_oauthlib",
    "python-docx": "docx",
    "opencv-python": "cv2",
}

# ─── Inverse map: import name → pip package name ──────────────────────────
# _PACKAGE_IMPORT_NAMES covers mismatches; the loop below fills identity
# entries for packages whose import name == pip name (after normalisation).

_IMPORT_TO_PACKAGE = {v: k for k, v in _PACKAGE_IMPORT_NAMES.items()}
for _pkg in TIER2_ALLOWED_PACKAGES:
    _import_name = _PACKAGE_IMPORT_NAMES.get(_pkg, _pkg).replace("-", "_")
    if _import_name not in _IMPORT_TO_PACKAGE:
        _IMPORT_TO_PACKAGE[_import_name] = _pkg
