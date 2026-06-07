"""
planner/ — Multi-step goal orchestration package.

Public API re-exported here so callers can do:
    from assistant.actions.planner import execute_plan, needs_planning, ...
or:
    from assistant.actions import planner
    planner.execute_plan(...)
"""

from .planner import (
    execute_plan,
    resume_plan,
    needs_planning,
    has_suspended_plan,
    clear_suspended_plan,
    PlanStep,
    Plan,
    TOOL_MANIFEST,
)

__all__ = [
    "execute_plan",
    "resume_plan",
    "needs_planning",
    "has_suspended_plan",
    "clear_suspended_plan",
    "PlanStep",
    "Plan",
    "TOOL_MANIFEST",
]
