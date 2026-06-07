"""Tiny mock server for testing scheduler scheduler notify modes.

Run:   python test_signal_server.py
Check: curl http://localhost:9999/status
Set:   curl http://localhost:9999/set?value=ALERT
Reset: curl http://localhost:9999/set?value=OK

TENKA usage:
  "Check http://localhost:9999/status every minute, only tell me if it says ALERT"
  "Check http://localhost:9999/status every minute, tell me if anything changes"
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

_current_value = "OK"


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        global _current_value
        parsed = urlparse(self.path)

        if parsed.path == "/status":
            self._respond(200, _current_value)
        elif parsed.path == "/set":
            new_val = parse_qs(parsed.query).get("value", ["OK"])[0]
            _current_value = new_val
            self._respond(200, f"Set to: {_current_value}")
        else:
            self._respond(404, "Not found. Use /status or /set?value=X")

    def _respond(self, code, body):
        self.send_response(code)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(body.encode())

    def log_message(self, fmt, *args):
        print(f"  [{self.log_date_time_string()}] {fmt % args}")


if __name__ == "__main__":
    port = 9999
    server = HTTPServer(("127.0.0.1", port), Handler)
    print(f"Signal server running on http://localhost:{port}")
    print(f"  /status  → returns current value (now: '{_current_value}')")
    print(f"  /set?value=ALERT  → change value")
    print(f"  /set?value=OK     → reset")
    print()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
