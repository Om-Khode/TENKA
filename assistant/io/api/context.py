# assistant/io/api/context.py
"""Per-request identifiers, threaded through a contextvar rather than a
parameter every route would otherwise have to accept and forward.

`app.py`'s `audit_and_tag` middleware sets `request_id_var` before it calls
into the router; `schemas.py`'s `Meta` reads it back via a default_factory
when a route builds its `Envelope`. Neither module imports the other --
schemas.py can't import app.py (app.py imports the route modules that import
schemas.py), so the shared value has to live somewhere both can reach.
"""
from __future__ import annotations

import contextvars

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "studio_request_id", default=""
)
