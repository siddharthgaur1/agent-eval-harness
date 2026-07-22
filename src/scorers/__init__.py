"""Scoring dimensions.

Importing this package registers every scorer. `score_trajectory` is the only
entry point the runner needs.
"""

from __future__ import annotations

from typing import Any

from ..persistence.store import Store
from ..suites.schema import TaskDef
from ..trace.schema import Trajectory
from . import deterministic, judge  # noqa: F401  (import registers the scorers)
from .base import DETERMINISTIC, LLM_JUDGES, ScoreResult, Scorer, all_scorers, register

__all__ = [
    "DETERMINISTIC",
    "LLM_JUDGES",
    "ScoreResult",
    "Scorer",
    "all_scorers",
    "register",
    "score_trajectory",
]


def score_trajectory(
    traj: Trajectory,
    task: TaskDef,
    *,
    include_llm: bool = True,
    store: Store | None = None,
    client: Any = None,
) -> dict[str, ScoreResult]:
    """Run every scorer over one trajectory.

    A judge that fails after its retries yields a 0.0 with the error as its
    reasoning rather than aborting the suite — one flaky API call should not
    discard a run that took minutes to produce.
    """
    results: dict[str, ScoreResult] = {}

    for name, scorer in DETERMINISTIC.items():
        results[name] = scorer.score(traj, task)

    if include_llm:
        for name, scorer in LLM_JUDGES.items():
            try:
                results[name] = scorer.score(traj, task, store=store, client=client)
            except Exception as exc:
                results[name] = ScoreResult(
                    scorer=name,
                    score=0.0,
                    reasoning=f"judge unavailable: {exc}",
                    details={"error": str(exc), "unscored": True},
                )

    return results
