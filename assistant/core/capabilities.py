# assistant/core/capabilities.py
"""What a device is allowed to ask for.

Lifted out of `io/api/vault.py` in Milestone 6a.5. The enum is read by
`actions/` -- the single dispatch choke point compares a turn's grant set
against the capability its intent requires -- and `actions/ -> io/api/` is not
a legal edge. `actions/ -> core/` already is, and `core/` importing nothing is
exactly the rule `core/` exists to satisfy, so the enum lives here and
`vault.py` re-exports it. No `ignore_imports` entry was added and none is
permitted: if the enforcement point needs one, the enforcement point is in the
wrong layer.

This module imports `enum` and nothing else. A test asserts that.
"""

import enum


class Capability(str, enum.Enum):
    """What a device is allowed to ask for. Granted per device, never implied."""

    # Watching her work: status, telemetry, the live /v1/events stream, and
    # the routes that describe how she is configured (settings, personality,
    # the command catalogue, whether backups run). Everything here is about
    # the assistant herself, and none of it is something a user told her.
    OBSERVE = "observe"
    # Reading what she stored: conversation transcripts, the knowledge graph,
    # preferences, taught procedures, the names of the people she recognises.
    #
    # Split out of the old `CHAT`, which meant both of these at once. That
    # ambiguity let the `quick` ceiling -- the Cloudflare tunnel, where a
    # third party terminates TLS and reads the plaintext -- look like
    # "observation only" while actually admitting the entire knowledge graph
    # and every transcript. `read_screen` and `camera_look` are intents, so
    # her narration of what was on screen lands in a transcript: excluding
    # SCREEN from that ceiling while admitting RECALL was excluding the
    # photograph and shipping the description.
    #
    # Neither implies the other. A wall display may watch without reading a
    # word she was told; an archive tool may read history without a live view.
    RECALL = "recall"
    # POST /v1/chat hands text to the same pipeline voice uses, so it reaches
    # every intent -- code_executor, file_task, shutdown, manage_backup --
    # not just conversation. Neither read capability may carry that: both gate
    # routes a device should be able to hold without being able to drive her.
    # Split so a device can be trusted to read a transcript without being
    # trusted to act on the machine through one.
    CHAT_SEND = "chat_send"
    SCREEN = "screen"
    FILES = "files"
    SYSTEM_CONTROL = "system_control"
