"""Runtime environment loading shared by both planes."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv


def load_runtime_env() -> None:
    """Load repo-root ``.env`` values without overriding shell env vars."""
    repo_root = Path(__file__).resolve().parent.parent
    load_dotenv(repo_root / ".env", override=False)


__all__ = ["load_runtime_env"]
