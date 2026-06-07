"""
messaging_bridge.py — Generic persistent-connection messaging bridge.

Manages long-lived connections to messaging services (WhatsApp, Telegram,
Discord, etc.) that can't run in Tier 2's 30-second subprocess model.

Architecture:
  - One background thread per connected service
  - Tiny HTTP server on localhost for Tier 2 scripts to call
  - Convention-based adapter loading from assistant/adapters/
  - Session data persisted in SANDBOX_DIR/service_data/{service}/

Adding a new service:
  1. Create assistant/adapters/{service}_adapter.py implementing the adapter interface
  2. Add the package to TIER2_ALLOWED_PACKAGES in code_executor.py
  3. Add entry to _DEVICE_AUTH_PACKAGE_MAP in code_executor.py
  That's it. No other code changes needed.

Adapter interface (each adapter must implement):
  connect(session_path: str) -> None     — Connect (blocking, runs in thread)
  disconnect() -> None                    — Clean disconnect
  is_connected() -> bool                  — Connection status
  execute(action: str, params: dict) -> dict  — Run a command, return result
  get_client() -> object                  — Return the raw client (for advanced use)
"""

import importlib
import json
import logging
import os
import threading
import time
import queue as _queue
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs

logger = logging.getLogger("messaging_bridge")

# ═══════════════════════════════════════════════════════════════════════════════
#  ADAPTER REGISTRY — convention-based, maps service name to adapter module
# ═══════════════════════════════════════════════════════════════════════════════

# Maps service names to their adapter module names in assistant/io/adapters/
# To add a new service: just add one entry here.
_ADAPTER_MAP: dict[str, str] = {
    "whatsapp": "whatsapp",
    # "telegram": "telegram",
    # "discord": "discord",
}


# ═══════════════════════════════════════════════════════════════════════════════
#  BRIDGE STATE
# ═══════════════════════════════════════════════════════════════════════════════

_adapters: dict[str, object] = {}          # service_name → loaded adapter instance
_threads: dict[str, threading.Thread] = {} # service_name → connection thread
_server: HTTPServer | None = None
_server_thread: threading.Thread | None = None
_BRIDGE_PORT = 7780  # localhost only — not exposed to network
_running = False
_upgrade_attempted: dict[str, bool] = {}  # max 1 upgrade per service per session


# ═══════════════════════════════════════════════════════════════════════════════
#  NOTIFICATION QUEUE — adapters push incoming message notifications here
#  main.py drains this queue each loop cycle
# ═══════════════════════════════════════════════════════════════════════════════

_notification_queue: _queue.Queue = _queue.Queue()



def _get_session_dir(service: str) -> Path:
    """Get the session data directory for a service. Creates if needed."""
    from .. import config
    d = config.SANDBOX_DIR / "service_data" / service
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_adapter(service: str):
    """
    Load an adapter for a service.

    Checks the channel_registry first (populated by adapter self-registration
    on import).  Falls back to dynamic importlib loading so that adapters
    which have not yet been imported still work.
    """
    from .channels import channel_registry
    existing = channel_registry.get(service)
    if existing is not None:
        logger.info(f"[BRIDGE] Found pre-registered adapter for '{service}' in channel_registry")
        return existing

    # Dynamic import fallback — adapter modules that register themselves will
    # populate the registry as a side-effect; others return a fresh Adapter().
    if service not in _ADAPTER_MAP:
        raise ValueError(f"No adapter registered for service '{service}'")

    module_name = _ADAPTER_MAP[service]
    try:
        mod = importlib.import_module(f".adapters.{module_name}", package="assistant.io")
        # After import the module may have registered itself; prefer that instance.
        registered = channel_registry.get(service)
        if registered is not None:
            logger.info(f"[BRIDGE] Loaded adapter for '{service}' (via registry after import)")
            return registered
        adapter = mod.Adapter()
        logger.info(f"[BRIDGE] Loaded adapter for '{service}' (fresh instance)")
        return adapter
    except Exception as e:
        logger.error(f"[BRIDGE] Failed to load adapter for '{service}': {e}")
        raise


def _connect_service(service: str, packages: list[str] | None = None) -> str | None:
    """
    Connect to a service. Runs the adapter's connect() in a background thread.
    Returns None on success, or a NEEDS_DEVICE_AUTH sentinel if pairing is needed.
    
    Re-uses an existing adapter instance if one exists (avoids killing event
    handlers registered by a previous connection).
    """
    # Already connected — nothing to do
    if service in _adapters and _adapters[service].is_connected():
        logger.info(f"[BRIDGE] '{service}' already connected")
        return None

    # Re-use existing adapter if present, only load fresh if none exists
    if service in _adapters:
        adapter = _adapters[service]
        logger.debug(f"[BRIDGE] Re-using existing adapter for '{service}'")
    else:
        adapter = _load_adapter(service)
        _adapters[service] = adapter

    session_dir = _get_session_dir(service)
    session_path = str(session_dir / "session.db")

    # Check if session exists (previously paired)
    has_session = os.path.exists(session_path) and os.path.getsize(session_path) > 0

    if not has_session:
        # No session — need device auth (QR code pairing)
        return f"NEEDS_DEVICE_AUTH|{service}|{session_path}"

    # Session exists — connect in background thread
    def _run():
        try:
            logger.info(f"[BRIDGE] Connecting '{service}' with existing session...")
            adapter.connect(session_path)
        except Exception as e:
            logger.error(f"[BRIDGE] Connection failed for '{service}': {e}")

    t = threading.Thread(target=_run, name=f"bridge_{service}", daemon=True)
    t.start()
    _threads[service] = t

    # Wait briefly for connection to establish
    for _ in range(30):  # 15 seconds max
        time.sleep(0.5)
        if adapter.is_connected():
            logger.info(f"[BRIDGE] '{service}' connected successfully")
            return None
        if getattr(adapter, '_client_outdated', False):
            return _handle_client_outdated(service, adapter, session_path, packages)

    logger.warning(f"[BRIDGE] '{service}' connection timed out after 15s")
    return f"CONNECT_TIMEOUT|{service}"


def _handle_client_outdated(
    service: str, adapter, session_path: str,
    packages: list[str] | None = None,
) -> str | None:
    """Attempt auto-upgrade on 405, then reconnect. Falls back to sentinel."""
    if service in _upgrade_attempted:
        logger.error(f"[BRIDGE] '{service}' still outdated after upgrade attempt")
        return f"CLIENT_OUTDATED|{service}"

    _upgrade_attempted[service] = True
    pkgs = packages or []
    if not pkgs:
        logger.error(f"[BRIDGE] No packages found for '{service}' — cannot auto-upgrade")
        return f"CLIENT_OUTDATED|{service}"

    pkg = pkgs[0]
    logger.info(f"[BRIDGE] '{service}' client outdated — attempting auto-upgrade of '{pkg}'...")

    from ..core.package_upgrade import upgrade_package
    result = upgrade_package(pkg)

    if not result.success:
        logger.error(f"[BRIDGE] Auto-upgrade of '{pkg}' failed: {result.error_msg}")
        _push_upgrade_notification(service, "fail", result)
        return f"CLIENT_OUTDATED|{service}"

    if result.old_version == result.new_version:
        logger.warning(f"[BRIDGE] '{pkg}' already at {result.old_version} — no newer version available")
        _push_upgrade_notification(service, "same_version", result)
        return f"CLIENT_OUTDATED|{service}"

    logger.info(f"[BRIDGE] '{pkg}' upgraded {result.old_version} -> {result.new_version}, reconnecting...")

    try:
        adapter.disconnect()
    except Exception as e:
        logger.debug(f"[BRIDGE] Disconnect before reconnect failed: {e}")
    del _adapters[service]

    new_adapter = _load_adapter(service)
    _adapters[service] = new_adapter

    def _run():
        try:
            new_adapter.connect(session_path)
        except Exception as e:
            logger.error(f"[BRIDGE] Reconnect after upgrade failed: {e}")

    t = threading.Thread(target=_run, name=f"bridge_{service}_upgrade", daemon=True)
    t.start()
    _threads[service] = t

    for _ in range(30):
        time.sleep(0.5)
        if new_adapter.is_connected():
            logger.info(f"[BRIDGE] '{service}' reconnected after upgrade")
            _push_upgrade_notification(service, "success", result)
            return None
        if getattr(new_adapter, '_client_outdated', False):
            logger.error(f"[BRIDGE] '{service}' still outdated after upgrade to {result.new_version}")
            _push_upgrade_notification(service, "still_outdated", result)
            return f"CLIENT_OUTDATED|{service}"

    logger.warning(f"[BRIDGE] '{service}' reconnect timed out after upgrade")
    return f"CONNECT_TIMEOUT|{service}"


def _push_upgrade_notification(service: str, status: str, result) -> None:
    """Push a TTS notification about the upgrade result."""
    name = service.title()
    if status == "success":
        msg = f"{name} was outdated. I upgraded and reconnected automatically."
    elif status == "fail":
        msg = f"{name} is outdated but the upgrade failed. Check the terminal."
    elif status == "same_version":
        msg = f"{name} is outdated but already on the latest version. No new release yet."
    elif status == "still_outdated":
        msg = f"{name} upgraded to {result.new_version} but is still rejected. A newer release may be needed."
    else:
        return
    push_notification(service, "upgrade_status", {"message": msg})


def connect_for_pairing(service: str) -> object:
    """
    Connect a service for first-time QR code pairing.
    Called by actions.py during the device auth flow.
    Returns the adapter (which will print QR to terminal).
    """
    adapter = _load_adapter(service) if service not in _adapters else _adapters[service]
    _adapters[service] = adapter
    session_dir = _get_session_dir(service)
    session_path = str(session_dir / "session.db")

    def _run():
        try:
            logger.info(f"[BRIDGE] Starting '{service}' pairing (QR will appear in terminal)...")
            adapter.connect(session_path)
        except Exception as e:
            logger.error(f"[BRIDGE] Pairing connection failed for '{service}': {e}")

    t = threading.Thread(target=_run, name=f"bridge_{service}", daemon=True)
    t.start()
    _threads[service] = t
    return adapter


def pair_phone(service: str, phone: str) -> str:
    """
    Generate a phone-number pairing code for a service (alternative to QR).
    Requires the adapter to already have an active WebSocket (connect_for_pairing called first).
    Returns the 8-char code the user enters in WhatsApp → Linked Devices.
    """
    if service not in _adapters:
        raise ValueError(f"No adapter loaded for '{service}'")
    adapter = _adapters[service]
    if not hasattr(adapter, 'pair_phone'):
        raise ValueError(f"Adapter for '{service}' does not support phone pairing")
    return adapter.pair_phone(phone)


def is_client_outdated(service: str) -> bool:
    """Check if a service was rejected as outdated (405)."""
    if service in _adapters:
        return getattr(_adapters[service], '_client_outdated', False)
    return False


def is_connected(service: str) -> bool:
    """Check if a service is currently connected."""
    if service in _adapters:
        return _adapters[service].is_connected()
    return False


def execute(service: str, action: str, params: dict | None = None) -> dict:
    """
    Execute a command on a connected service.
    Returns a dict with at least {"ok": bool, "result": ...}
    """
    if service not in _adapters or not _adapters[service].is_connected():
        # Try to auto-connect
        sentinel = _connect_service(service)
        if sentinel:
            if sentinel.startswith("CONNECT_TIMEOUT|"):
                return {"ok": False, "error": f"Connection to {service} timed out. Try again in a moment."}
            return {"ok": False, "error": sentinel}
        if not is_connected(service):
            return {"ok": False, "error": f"Service '{service}' is not connected"}

    try:
        result = _adapters[service].execute(action, params or {})
        return {"ok": True, "result": result}
    except Exception as e:
        logger.error(f"[BRIDGE] Execute failed for '{service}.{action}': {e}")
        return {"ok": False, "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
#  HTTP SERVER — Tier 2 scripts call this via requests.get()
# ═══════════════════════════════════════════════════════════════════════════════

class _BridgeHandler(BaseHTTPRequestHandler):
    """Handle HTTP requests from Tier 2 scripts."""

    def do_GET(self):
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)

        if parsed.path == "/execute":
            service = query.get("service", [""])[0]
            action = query.get("action", [""])[0]
            # Parse params from query string
            params = {}
            for k, v in query.items():
                if k not in ("service", "action"):
                    params[k] = v[0]

            if not service or not action:
                self._respond(400, {"ok": False, "error": "Missing 'service' or 'action' parameter"})
                return

            result = execute(service, action, params)
            self._respond(200, result)

        elif parsed.path == "/status":
            service = query.get("service", [""])[0]
            if service:
                self._respond(200, {"ok": True, "connected": is_connected(service)})
            else:
                status = {s: _adapters[s].is_connected() for s in _adapters}
                self._respond(200, {"ok": True, "services": status})

        else:
            self._respond(404, {"ok": False, "error": "Unknown endpoint"})

    def do_POST(self):
        """Handle POST requests for commands with JSON body."""
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length).decode('utf-8') if content_length else '{}'
        
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self._respond(400, {"ok": False, "error": "Invalid JSON"})
            return

        parsed = urlparse(self.path)
        if parsed.path == "/execute":
            service = data.get("service", "")
            action = data.get("action", "")
            params = data.get("params", {})

            if not service or not action:
                self._respond(400, {"ok": False, "error": "Missing 'service' or 'action'"})
                return

            result = execute(service, action, params)
            self._respond(200, result)
        else:
            self._respond(404, {"ok": False, "error": "Unknown endpoint"})

    def _respond(self, status: int, data: dict):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False, default=str).encode("utf-8"))

    def log_message(self, format, *args):
        """Suppress default HTTP logging — use our logger instead."""
        logger.debug(f"[BRIDGE HTTP] {args[0] if args else ''}")


# ═══════════════════════════════════════════════════════════════════════════════
#  LIFECYCLE — start/stop called from main.py
# ═══════════════════════════════════════════════════════════════════════════════

def auto_connect_services() -> None:
    """
    Auto-connect messaging services that have saved sessions.
    Called once at startup from main.py. Non-blocking — connections happen
    in background threads.
    
    Only connects services listed in MESSAGING_AUTO_CONNECT env var
    (comma-separated, e.g. "whatsapp,telegram"). If the env var is not set,
    defaults to all services in _ADAPTER_MAP that have a saved session.
    """
    from .. import config

    # Parse auto-connect list from env
    auto_connect_env = os.environ.get("MESSAGING_AUTO_CONNECT", "").strip()
    if auto_connect_env:
        services = [s.strip().lower() for s in auto_connect_env.split(",") if s.strip()]
    else:
        # Default: all registered services
        services = list(_ADAPTER_MAP.keys())

    for service in services:
        if service not in _ADAPTER_MAP:
            logger.warning(f"[BRIDGE] Unknown service in MESSAGING_AUTO_CONNECT: '{service}'")
            continue

        session_dir = _get_session_dir(service)
        session_path = str(session_dir / "session.db")
        has_session = os.path.exists(session_path) and os.path.getsize(session_path) > 0

        if not has_session:
            logger.debug(f"[BRIDGE] No saved session for '{service}' — skipping auto-connect")
            continue

        # Load adapter and start connection in background
        try:
            adapter = _load_adapter(service)
            _adapters[service] = adapter

            def _run(svc=service, adp=adapter, sp=session_path):
                try:
                    logger.info(f"[BRIDGE] Auto-connecting '{svc}' at startup...")
                    adp.connect(sp)
                except Exception as e:
                    logger.error(f"[BRIDGE] Auto-connect failed for '{svc}': {e}")

            t = threading.Thread(target=_run, name=f"bridge_{service}", daemon=True)
            t.start()
            _threads[service] = t
            logger.info(f"[BRIDGE] Auto-connect started for '{service}'")
        except Exception as e:
            logger.error(f"[BRIDGE] Failed to load adapter for auto-connect '{service}': {e}")


def start():
    """Start the messaging bridge HTTP server."""
    global _server, _server_thread, _running

    if _running:
        logger.info("[BRIDGE] Already running")
        return

    try:
        _server = HTTPServer(("127.0.0.1", _BRIDGE_PORT), _BridgeHandler)
        _server.timeout = 1  # short timeout so shutdown isn't blocked

        def _serve():
            logger.info(f"[BRIDGE] HTTP server listening on 127.0.0.1:{_BRIDGE_PORT}")
            while _running:
                _server.handle_request()
            logger.info("[BRIDGE] HTTP server stopped")

        _running = True
        _server_thread = threading.Thread(target=_serve, name="bridge_http", daemon=True)
        _server_thread.start()
        logger.info("[BRIDGE] Messaging bridge started")
    except OSError as e:
        logger.error(f"[BRIDGE] Failed to start HTTP server: {e}")
        _running = False


def stop():
    """Stop the bridge and disconnect all services."""
    global _server, _server_thread, _running

    _running = False

    # Disconnect all adapters
    for service, adapter in _adapters.items():
        try:
            logger.info(f"[BRIDGE] Disconnecting '{service}'...")
            adapter.disconnect()
        except Exception as e:
            logger.warning(f"[BRIDGE] Error disconnecting '{service}': {e}")

    _adapters.clear()
    _threads.clear()

    # Clear pending notifications
    while not _notification_queue.empty():
        try:
            _notification_queue.get_nowait()
        except _queue.Empty:
            break

    if _server:
        try:
            _server.server_close()
        except Exception:
            pass
        _server = None

    if _server_thread and _server_thread.is_alive():
        _server_thread.join(timeout=5)
    _server_thread = None

    logger.info("[BRIDGE] Messaging bridge stopped")


def get_port() -> int:
    """Return the bridge HTTP port. Used by code_executor for env injection."""
    return _BRIDGE_PORT


def push_notification(service: str, event_type: str, data: dict) -> None:
    """
    Push a notification from an adapter. Called from adapter background threads.
    
    Args:
        service:    e.g. "whatsapp"
        event_type: e.g. "incoming_message" (future: "call", "typing", etc.)
        data:       event-specific payload dict
    """
    notification = {
        "service": service,
        "type": event_type,
        **data,
    }
    _notification_queue.put(notification)
    logger.debug(f"[BRIDGE] Notification queued: {service}/{event_type}")


def drain_notifications() -> list[dict]:
    """
    Drain all pending notifications. Called by main.py each loop cycle.
    Returns a list of notification dicts (may be empty).
    """
    items = []
    while True:
        try:
            items.append(_notification_queue.get_nowait())
        except _queue.Empty:
            break
    return items


def has_pending_notifications() -> bool:
    """Check if there are pending notifications without draining them."""
    return not _notification_queue.empty()
