"""
core/intent_scopes.py — Intent scope constants.

Pure data module, zero imports. Defines which intents are available in each
system state scope, and which intents are always available regardless of scope.
"""

SCOPES = {
    "browser_mode": {
        "browser_cdp_setup", "browse_url", "find_and_click", "read_screen",
    },
    "recording_mode": {
        "start_recording", "stop_recording", "get_recording", "summarize_recording",
    },
    "camera_mode": {
        "camera_look", "meet_face", "recognize_face", "forget_face",
    },
}

ALWAYS_AVAILABLE = {
    "small_talk", "unknown", "get_time", "memory_query", "store_memory",
    "hide_avatar", "show_avatar", "web_search", "code_executor",
    "computer_task", "planner", "open_browser", "create_note",
    "file_task", "set_reminder", "cancel_reminder", "manage_shortcut",
    "manage_procedure", "enroll_voice", "forget_voice", "shutdown",
    "manage_schedule", "manage_monitor",
}
