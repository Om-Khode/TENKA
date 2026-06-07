"""
procedures.py — Procedure storage facade over storage/repos/procedure.py.
"""

from typing import Optional

from .storage.db import get_db, init_db
from .storage.repos.procedure import _MAX_STEPS  # noqa: F401 — re-exported for teaching.py
from . import config

_repo = None


def _get_repo():
    global _repo
    if _repo is None:
        from .storage.repos.procedure import ProcedureRepo
        db = get_db()
        if db is None:
            raise RuntimeError(
                "procedures not initialized — call init_procedure_db() first"
            )
        _repo = ProcedureRepo(db, assistant_name_lower=config.ASSISTANT_NAME_LOWER)
    return _repo


def init_procedure_db() -> None:
    """Initialize the shared database if needed, then bind the repo."""
    global _repo
    db = get_db()
    if db is None:
        db_path = config.SANDBOX_DIR / "memory" / "tenka.db"
        db = init_db(db_path)
    from .storage.repos.procedure import ProcedureRepo
    _repo = ProcedureRepo(db, assistant_name_lower=config.ASSISTANT_NAME_LOWER)


def create_procedure(
    trigger: str,
    name: str,
    steps: list[dict],
    backend: str = "auto",
    description: str = "",
) -> int:
    return _get_repo().create_procedure(trigger, name, steps, backend, description)


def get_procedure(trigger: str) -> Optional[dict]:
    return _get_repo().get_procedure(trigger)


def get_procedure_by_id(proc_id: int) -> Optional[dict]:
    return _get_repo().get_procedure_by_id(proc_id)


def update_procedure(
    proc_id: int,
    steps: Optional[list[dict]] = None,
    name: Optional[str] = None,
    trigger: Optional[str] = None,
    description: Optional[str] = None,
    backend: Optional[str] = None,
) -> bool:
    return _get_repo().update_procedure(
        proc_id, steps=steps, name=name, trigger=trigger,
        description=description, backend=backend,
    )


def delete_procedure(proc_id: int) -> bool:
    return _get_repo().delete_procedure(proc_id)


def list_procedures(enabled_only: bool = True) -> list[dict]:
    return _get_repo().list_procedures(enabled_only)


def match_trigger(text: str) -> Optional[dict]:
    return _get_repo().match_trigger(text)


def record_usage(proc_id: int) -> None:
    _get_repo().record_usage(proc_id)


def check_trigger_conflict(trigger: str) -> Optional[str]:
    return _get_repo().check_trigger_conflict(trigger)


def find_by_name_or_trigger(text: str, enabled_only: bool = True) -> Optional[dict]:
    return _get_repo().find_by_name_or_trigger(text, enabled_only)


def subsequence_remainder(trigger: str, text: str) -> str:
    from .storage.repos.procedure import ProcedureRepo
    return ProcedureRepo.subsequence_remainder(trigger, text)


def step_count_warning(steps: list[dict]) -> Optional[str]:
    from .storage.repos.procedure import ProcedureRepo
    return ProcedureRepo.step_count_warning(steps)
