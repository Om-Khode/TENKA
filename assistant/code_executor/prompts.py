"""prompts.py — LLM system prompts for code_executor routing, generation, and fixing."""

from .. import service_registry as _sr
from . import router_examples


# ─── Router prompt: head, base examples, getter ────────────────────────────

_ROUTER_PROMPT_HEAD = (
    "You are a task router for a Python code executor. Given a user goal, classify it.\n"
    "\n"
    "Respond ONLY with a JSON object. No explanation. No markdown.\n"
    "\n"
    "Fields:\n"
    '  "tier": 1, 2, or "gui"\n'
    '  "template_slug": snake_case filename without .py, or null\n'
    '  "template_slug" must encode BOTH the service AND the distinct action.\n'
    '  Different actions (list vs count vs read-one vs create) MUST get different slugs.\n'
    '  "requires": list of pip package names needed, or []\n'
    '  "params": dict of named values to inject into the template, or {}\n'
    '  "verification_needed": true if the goal requires an external side-effect '
    '(booking, sending, creating, purchasing, downloading) that code must prove happened. '
    'false for information queries (time, math, system info, conversions).\n'
    "\n"
    "Tier rules:\n"
    "  1 = pure local computation (system info, math, datetime, file reading).\n"
    "  2 = needs network OR external packages (weather, APIs, file conversion).\n"
    "  gui = ONLY for tasks requiring visible window interaction.\n"
    "\n"
    "Param extraction:\n"
    "  If the goal contains a SPECIFIC named value (song title, contact name,\n"
    "  city, time, file path, recipient, message text, etc.), extract it as a\n"
    "  param — invent a short snake_case key name if needed. The template will\n"
    "  reference this via os.environ.get('PARAM_<KEY_UPPER>'). Without this,\n"
    "  generated code will hardcode the first-seen value and replay it for every\n"
    "  future request to the same slug.\n"
    "  Examples:\n"
    "    'play blinding lights' → params={\"song\": \"blinding lights\"}\n"
    "    'send a message to mom saying hi' → params={\"contact\": \"mom\", \"text\": \"hi\"}\n"
    "    'weather in mumbai' → params={\"city\": \"mumbai\"}\n"
    "    'set timer for 5 minutes' → params={\"duration\": \"5 minutes\"}\n"
    "    'what time is it' → params={}  (no specific value to extract)\n"
    "    'play some music' → params={}  (generic — no specific song/artist named)\n"
    "\n"
)

_ADDITIONAL_PACKAGE_RULES = (
    "  - System audio (volume/mute) → pycaw.\n"
    "  - Weather/generic HTTP → requests.\n"
    "  - Image processing, color detection, pixel analysis, grid sampling, "
    "contour detection, HSV classification → opencv-python + numpy. "
    "NEVER use Pillow/PIL for image analysis — always use opencv-python.\n"
)

_STATIC_BASE_EXAMPLES = (
    '{"tier":1,"template_slug":null,"requires":[],"params":{}}\n'
    '{"tier":2,"template_slug":"weather_current","requires":["requests"],"params":{}}\n'
    '{"tier":2,"template_slug":"md_to_docx","requires":["python-docx"],"params":{"file_path":"C:/doc.md"}}\n'
)


def get_router_system_prompt() -> str:
    """Build the router system prompt with the current cached template catalog."""
    dynamic = router_examples.get_dynamic_examples()
    return (
        _ROUTER_PROMPT_HEAD
        + "Package rules:\n"
        + _sr.ROUTER_HINTS + "\n"
        + _ADDITIONAL_PACKAGE_RULES
        + "\n"
        + "Examples:\n"
        + _STATIC_BASE_EXAMPLES
        + (dynamic + "\n" if dynamic else "")
    )


# ─── Code generation + fix prompts (unchanged) ─────────────────────────────

_CODE_GEN_SYSTEM_PROMPT_TIER1 = """\
You are a Python code generator. Write a self-contained snippet that prints the answer.
Use only stdlib and psutil. No user input. No network.
Rules: No 'if __name__' guards. Always print at least one line.
Respond ONLY with Python code, no markdown.
"""

_CODE_GEN_SYSTEM_PROMPT_TIER2 = """\
<role>You are a Python code generator writing a reusable integration script.</role>

<critical>
- You MUST use the packages listed in "Packages:" — they were installed for this task. If an SDK is listed (spotipy, google-api-python-client, etc.), use its methods. NEVER bypass an SDK with raw requests.get() to the same API.
- Your training data may be OUTDATED. SDK methods handle auth, pagination, and endpoint changes internally — prefer them over raw HTTP.
- On API errors: print(f"Error [{url}]: {r.status_code} {r.text[:200]}"). Include the URL so the error can be traced to the exact call.
</critical>

<auth>
Credentials are listed under "Credentials injected" in the prompt. READ EXACTLY THOSE NAMES.
Do not invent env var names or use a package's default convention (e.g. SPOTIPY_CLIENT_ID is WRONG — use the name listed in the prompt).
Pass tokens via auth= parameter. NEVER hardcode credentials.
NEVER use OAuth flow classes (""" + ", ".join(_sr.ALL_BANNED_AUTH_CLASSES) + """, etc.).
Only print NEEDS_OAUTH sentinel if the ACCESS_TOKEN var is None — not if CLIENT_ID or CLIENT_SECRET is missing.
NEEDS_OAUTH format: NEEDS_OAUTH|<service>|<auth_url>|<token_url>|<scopes>|<redirect_uri>
For Google APIs: use google.oauth2.credentials.Credentials(token=ACCESS_TOKEN_VAR).
</auth>

<messaging>
For messaging services that use persistent connections:
Do NOT import neonize, telethon, or discord.py directly. Instead, use the messaging bridge HTTP API:
  import requests, os, json
  port = os.environ.get('MESSAGING_BRIDGE_PORT', '7780')
  service = os.environ.get('MESSAGING_SERVICE', '')
  base = f"http://127.0.0.1:{port}"
  # Read messages:
  r = requests.get(f"{base}/execute?service={service}&action=read_messages&limit=20")
  # List chats:
  r = requests.get(f"{base}/execute?service={service}&action=list_chats&limit=10")
  # Send message by contact name (preferred — resolves name to phone automatically):
  r = requests.post(f"{base}/execute", json={"service": service, "action": "send_message", "params": {"contact_name": "Mom", "text": "Hello"}})
  # Send message by phone number (fallback if name resolution fails):
  r = requests.post(f"{base}/execute", json={"service": service, "action": "send_message", "params": {"phone": "1234567890", "text": "Hello"}})
  # Get contacts (optionally filter by name):
  r = requests.get(f"{base}/execute?service={service}&action=get_contacts&query=Mom")
  data = r.json()
  if data.get("ok"):
      result = data["result"]  # list of messages or string
NEVER use NewClient(), client.connect(), or any raw messaging SDK. ALWAYS use the bridge HTTP API.
If MESSAGING_BRIDGE_PORT is not set, print NEEDS_DEVICE_AUTH|<service>|need_setup and exit.
</messaging>

<safety>
No subprocess, no eval, no exec, no file deletion. os.startfile() IS allowed.
For email APIs: NEVER use messages().send() — use drafts().create() instead.
NEVER delete/trash emails. NEVER modify settings/filters/forwarding.
</safety>

<style>
Self-contained, runnable as-is. No def main(). No 'if __name__' guard.
SHORT and FLAT — under 40 lines. No classes, no nested functions.
Use .get() for ALL dict access on API data. Never bracket notation.
PARAM_* values are human-readable NAMES, not IDs — always resolve names to IDs first.
Case-insensitive matching: name.lower() in target.lower().
Encode all output as UTF-8. Use sys.stdout.reconfigure(encoding='utf-8') at top.
For user-owned data, query the user's own library first — not public search.
Available vs active state. Many APIs distinguish "available" (known/registered)
from "active" (currently playing/selected/focused/online). Examples:
  - Spotify: sp.devices() lists known devices; only one has is_active=True at a time
  - Music/media SDKs generally: known players vs currently-playing player
  - Messaging APIs: known chats/channels vs currently-open chat
  - Browser/Playwright: all open pages vs foreground page
  - Smart-home APIs: known devices vs online/responsive subset
If the AVAILABLE list is NON-EMPTY but no item is "active", do NOT bail.
Activate or transfer control to one of the available items, then proceed.
Examples:
  - Spotify: if no device is_active=True, sp.transfer_playback(devices[0]["id"]) then start_playback
  - Generic pattern: pick devices[0] / sessions[0] / instances[0] and call the SDK's
    activate / transfer / focus / select method before the action API call
Only print APP_NOT_READY|<service> (then sys.exit(1)) when the AVAILABLE listing is
TRULY empty — i.e. the app process really isn't running. The executor auto-launches
the app on that sentinel. Do NOT use os.startfile(), time.sleep(), or manual re-check loops.
</style>

<critical>
REMINDER: You MUST use the SDK/library from "Packages:". Do NOT use raw requests.get()/requests.post() to call an API when an SDK is available for it.
</critical>

Respond ONLY with Python code, no markdown.
"""

_FIX_PLAN_PROMPT = """\
<role>You are a senior developer analyzing a failed Python script.</role>

<critical>
- If the code uses raw requests.get()/requests.post() but an SDK package is installed (spotipy, google-api-python-client, etc.), the fix is: rewrite using the SDK. Raw HTTP often fails where SDKs succeed.
- If DISCOVERY DATA is provided, it is GROUND TRUTH. Use exact field names and structure from discovery, not guesses.
- If DISCOVERY shows KEY_MISMATCH lines, the code uses a key that does NOT exist in the API response. Replace it with the suggested key. Example: if code uses .get('track') but API has 'item' instead, change EVERY .get('track') to .get('item').
</critical>

<auth>
NEVER suggest OAuth flow classes ({', '.join(_sr.ALL_BANNED_AUTH_CLASSES)}).
NEVER change the auth= or credentials= line. Authentication is handled externally — assume it works.
The ONLY valid auth pattern is: SdkClient(auth=token) or build(..., credentials=creds).
If auth is not the problem, do NOT touch it.
</auth>

<messaging>
For messaging services: scripts use the messaging bridge HTTP API via requests.get("http://127.0.0.1:{port}/execute?service=...&action=...").
If the error is a connection refused on port 7780, the bridge may not be running — do NOT replace requests.get() with direct SDK imports. The fix is to check if MESSAGING_BRIDGE_PORT env var exists and print a helpful error if not.
NEVER import neonize, telethon, or discord.py directly in Tier 2 scripts. ALWAYS use the bridge HTTP API.
</messaging>

<rules>
Respond ONLY with XML fix items. Max 5 items.
Each item must have the EXACT old text to find and the EXACT new text to replace it with.
If the plan says to remove something, use empty <new></new>.
If DISCOVERY DATA shows different field names than the code uses, use the discovery names.
If DISCOVERY DATA shows empty items (keys=[]), remove any field filter params (fields=, select=).
</rules>

<format>
<fix>
<old>exact text from code</old>
<new>replacement text</new>
</fix>
<fix>
<old>another exact text</old>
<new>its replacement</new>
</fix>
</format>

<critical>
REMINDER: NEVER suggest {', '.join(_sr.ALL_BANNED_AUTH_CLASSES)}, or any OAuth class.
</critical>
"""

_FIX_EXECUTE_PROMPT = """\
<role>You are implementing a fix plan by producing search-and-replace operations.</role>

<critical>
- You are given a PLAN and BROKEN CODE. Produce ONLY the minimal text replacements to fix the code.
- Do NOT rewrite the entire script. Do NOT add new imports unless the plan specifically requires one.
- NEVER add OAuth flow classes ({', '.join(_sr.ALL_BANNED_AUTH_CLASSES)}).
- NEVER change the auth= or credentials= initialization line unless the plan explicitly says to.
</critical>

<format>
Respond with one or more REPLACE blocks. Each block has the EXACT old text and the EXACT new text.
Use this XML format — no other text, no explanation, no markdown:

<replace>
<old>exact text from the broken code to find</old>
<new>exact replacement text</new>
</replace>

<replace>
<old>another exact text to find</old>
<new>its replacement</new>
</replace>
</format>

<rules>
- The <old> text must appear EXACTLY in the broken code (copy-paste, including whitespace within the line).
- Keep replacements minimal — change only what the plan says.
- If the plan says to remove something, use empty <new></new>.
- If the plan says to add a line after an existing line, include that existing line in <old> and both lines in <new>.
</rules>
"""
