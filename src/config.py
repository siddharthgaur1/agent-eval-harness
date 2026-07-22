"""Environment-validated settings.

Loaded once at import time so a misconfigured deployment fails at startup with a
readable error instead of halfway through a suite run.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Every knob the harness reads from the environment."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    openai_api_key: str = ""
    judge_model: str = "gpt-4o"
    cheap_model: str = "gpt-4o-mini"
    judge_max_retries: int = 3

    db_path: Path = Path("data/harness.db")
    runs_dir: Path = Path("data/runs")
    reports_dir: Path = Path("data/reports")

    # Regression thresholds. A dimension must drop by more than this AND clear the
    # noise band (see compare/regression.py) before it counts as a hard regression.
    regression_threshold: float = 0.05
    overall_threshold: float = 0.03
    drift_window: int = 5
    drift_threshold: float = 0.08

    # Runner
    repeats: int = 3
    workers: int = 4
    loop_repeat_limit: int = 3

    @field_validator("regression_threshold", "overall_threshold", "drift_threshold")
    @classmethod
    def _in_unit_range(cls, v: float) -> float:
        if not 0.0 < v < 1.0:
            raise ValueError("thresholds must be in (0, 1)")
        return v

    @field_validator("repeats", "workers")
    @classmethod
    def _positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("must be >= 1")
        return v

    def require_openai(self) -> str:
        """Fail loudly at the point an LLM judge is actually needed."""
        if not self.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Deterministic scorers still run with "
                "--no-llm; LLM judges require a key."
            )
        return self.openai_api_key

    def ensure_dirs(self) -> None:
        """Create the data directories the store and reporters write into."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()


settings = get_settings()
