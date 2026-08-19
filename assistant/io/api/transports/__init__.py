# assistant/io/api/transports/__init__.py
"""Transport registry -- one provider per module, self-registering.

Milestone 6b exposes the Studio daemon over two transports (`tailnet`,
`funnel`), each its own listener with its own capability ceiling
(`..policy.POLICIES`). A third, `quick` (a Cloudflare tunnel), shipped in the
same milestone and was removed from it -- no device could ever authenticate
over it (`..policy`'s module docstring has the full argument). Adding a new
one is a new module under this package and nothing else -- there is no
transport-specific branching anywhere outside it. This is the same
self-registration shape `llm/providers/` uses for `provider_registry` and
`actions/` uses for `tool_registry`: a registry that exists once, and modules
that call `transport_registry.register(...)` on import rather than being
wired in by name from the outside.

Layering: `io/api/` may import `core/` and `config` only. Importing
`..policy` from here is legal and expected -- `POLICIES` is what `register()`
checks a name against.
"""
from __future__ import annotations

from ....core.registry import RegistryBase
from ..policy import POLICIES
from .base import HOSTNAME_TIMEOUT_SECONDS, TransportAdapter, TransportSession

__all__ = [
    "HOSTNAME_TIMEOUT_SECONDS",
    "TransportAdapter",
    "TransportSession",
    "TransportRegistry",
    "transport_registry",
]


# ─── Transport Registry ──────────────────────────────────────────────────────

class TransportRegistry(RegistryBase[TransportAdapter]):
    """`RegistryBase` plus the two refusals a transport needs that a bare
    string-keyed registry does not: a name that is not a policy, and the one
    name that is a policy but must never be a transport.
    """

    def __init__(self) -> None:
        super().__init__("transport")

    def register(self, name: str, adapter: TransportAdapter) -> TransportAdapter:
        """Register *adapter* under *name*. Refuses three things:

        - *name* not a key in `POLICIES` -- nothing declared what such a
          transport would be trusted to carry, so it cannot be given a port.
        - *name* == `"local"` -- a KI-17 sibling. `local` is the loopback
          listener bound directly by `server.py`; it has no tunnel and no
          adapter. An adapter claiming that name would be handed local's own
          port by `port_for("local", base_port)` and inherit `POLICIES["local"]`
          in full -- admin, bearer, every capability including `EXECUTE` --
          which is Milestone 6a.5's landmine (KI-17) reached through the
          registry instead of through a stray tunnel.
        - a duplicate *name* -- delegated to `RegistryBase.register`, which
          already refuses re-registration under the same key.
        """
        if name == "local":
            raise ValueError(
                "transport 'local' is refused: 'local' is the loopback "
                "listener and has no tunnel -- an adapter registering under "
                "that name would be handed local's port and its full "
                "policy (KI-17)"
            )
        if name not in POLICIES:
            raise ValueError(
                f"transport '{name}' is not a policy name in POLICIES "
                f"({sorted(POLICIES)}) -- nothing declares what it may carry"
            )
        return super().register(name, adapter)

    def names(self) -> list[str]:
        """Sorted, stable snapshot of registered transport names. Sorted
        because the set of registered adapters is small and read for display
        (e.g. a future `/v1/status` row); stable so two calls in the same
        process never disagree about order."""
        return sorted(self.keys())


transport_registry = TransportRegistry()

# Provider modules import here, at the bottom, so registering with
# `transport_registry` happens as a side effect of importing this package --
# the same shape `llm/providers/__init__.py` uses for `gemini`, `groq`,
# `cerebras`, `ollama`.
#
# `tailscale.py` registers `tailnet` and `funnel` (Task 7). A `cloudflare.py`
# registering `quick` (Task 8) existed through Milestone 6b and was deleted
# in the same milestone -- no device could ever authenticate over that
# transport (`..policy`'s module docstring). A new provider adds its own
# `from . import <module>  # noqa: E402, F401` line here and nothing else --
# importing a module that is not yet written would fail the whole package
# import, so a line is only ever added once its module lands.
from . import tailscale  # noqa: E402, F401
