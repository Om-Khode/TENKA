"""The daemon's dependencies must be declared, not merely installed."""
import pathlib


def test_requirements_declares_fastapi_and_uvicorn():
    req = pathlib.Path("requirements.txt").read_text(encoding="utf-8").lower()
    assert "fastapi" in req
    assert "uvicorn" in req


def test_fastapi_and_uvicorn_import():
    import fastapi  # noqa: F401
    import uvicorn  # noqa: F401
