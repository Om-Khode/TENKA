"""
planner/ — Multi-step goal planning.

**Planning, not running.** §17.P8 split the package: what a plan *is* is
decided here, and running one is `brain/plan_runner.py`. `execute_plan`,
`resume_plan` and the suspension helpers used to be re-exported from this line
and are deliberately absent -- an `actions` module importing them would be
reaching for a layer above it, and the import would not resolve.

    from assistant.actions.planner import needs_planning, TOOL_MANIFEST
"""

from .planner import (
    needs_planning,
    PlanStep,
    Plan,
    TOOL_MANIFEST,
)

__all__ = [
    "needs_planning",
    "PlanStep",
    "Plan",
    "TOOL_MANIFEST",
]
