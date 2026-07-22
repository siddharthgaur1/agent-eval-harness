"""The scorer contract.

Every scorer returns a score in [0, 1], prose reasoning, and the step indices it
based that on. Evidence is not optional: a dimension that drops without naming
the steps responsible is a number nobody can act on, so `ScoreResult` rejects a
non-perfect score that cites nothing.
"""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, Field, model_validator

from ..suites.schema import TaskDef
from ..trace.schema import Trajectory


class ScoreResult(BaseModel):
    """One dimension's verdict on one trajectory."""

    scorer: str
    score: float = Field(ge=0.0, le=1.0)
    reasoning: str
    evidence: list[int] = Field(default_factory=list)
    details: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _evidence_required(self) -> "ScoreResult":
        # A perfect score legitimately has nothing to point at ("no failures
        # occurred"). Anything less must cite the steps that cost it.
        if self.score < 1.0 and not self.evidence and not self.details:
            raise ValueError(
                f"{self.scorer}: score {self.score} cites no evidence — "
                "every deduction must name the steps responsible"
            )
        return self


class Scorer(Protocol):
    """Structural type for all scorers."""

    name: str

    def score(self, traj: Trajectory, task: TaskDef) -> ScoreResult: ...


DETERMINISTIC: dict[str, Scorer] = {}
LLM_JUDGES: dict[str, Scorer] = {}


def register(kind: str = "deterministic"):
    """Class decorator that adds a scorer to the right registry."""

    def decorator(cls):
        registry = DETERMINISTIC if kind == "deterministic" else LLM_JUDGES
        registry[cls.name] = cls()
        return cls

    return decorator


def all_scorers(include_llm: bool = True) -> dict[str, Scorer]:
    """Every registered scorer, deterministic first."""
    scorers = dict(DETERMINISTIC)
    if include_llm:
        scorers.update(LLM_JUDGES)
    return scorers
