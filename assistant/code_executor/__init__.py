"""
code_executor — Sandboxed Python code execution for TENKA.

Three execution tiers:
  Tier 1 — Restricted in-process sandbox. Pure compute only.
  Tier 2 — Subprocess sandbox. Network allowed, whitelisted packages.
  Tier 3 — Unrestricted subprocess. Only when CODE_EXECUTOR_POWER_MODE=true.
"""

from .orchestrator import execute_code_task
from .sandbox import run_code
from .routing import detect_service_from_packages, get_oauth_env_map
from ._utils import _needs_retry, _detect_app_not_running
from .retry import pop_pending_knowledge, _pending_knowledge_queue

GUI_HANDOFF_SIGNAL = "__NEEDS_GUI__"
PLANNER_ESCALATION_SIGNAL = "__ESCALATE_PLANNER__"
