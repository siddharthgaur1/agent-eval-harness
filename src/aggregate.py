"""Aggregation across repeats and tasks.

Agents are stochastic. A single run is an anecdote, so everything here carries a
spread alongside its mean, and the regression detector refuses to act on a delta
it cannot distinguish from that spread.
"""

from __future__ import annotations

import statistics
from typing import Any

from pydantic import BaseModel, Field

from .scorers.base import ScoreResult

# A dimension whose scores vary by more than this across repeats of the same task
# is reported as unstable — its mean is not measuring anything you can regress on.
UNSTABLE_STDEV = 0.15


class TaskRun(BaseModel):
    """One execution of one task, scored."""

    task_id: str
    repeat: int
    trajectory_id: str
    category: str = "happy_path"
    weight: float = 1.0
    scores: dict[str, ScoreResult] = Field(default_factory=dict)
    error: str | None = None

    @property
    def overall(self) -> float:
        """Unweighted mean across dimensions for this single execution."""
        if not self.scores:
            return 0.0
        return sum(s.score for s in self.scores.values()) / len(self.scores)


class Stat(BaseModel):
    """Mean and spread of one measurement across repeats."""

    mean: float
    stdev: float
    n: int
    min: float
    max: float

    @property
    def unstable(self) -> bool:
        return self.stdev > UNSTABLE_STDEV

    @classmethod
    def of(cls, values: list[float]) -> "Stat":
        values = values or [0.0]
        return cls(
            mean=round(statistics.fmean(values), 4),
            stdev=round(statistics.stdev(values) if len(values) > 1 else 0.0, 4),
            n=len(values),
            min=round(min(values), 4),
            max=round(max(values), 4),
        )


class TaskSummary(BaseModel):
    """One task, aggregated over its repeats."""

    task_id: str
    category: str
    weight: float
    overall: Stat
    dimensions: dict[str, Stat]
    passed: bool

    @property
    def unstable_dimensions(self) -> list[str]:
        return sorted(k for k, v in self.dimensions.items() if v.unstable)


class RunAggregate(BaseModel):
    """The whole suite, aggregated."""

    overall: Stat
    dimensions: dict[str, Stat]
    tasks: dict[str, TaskSummary]

    @property
    def unstable_dimensions(self) -> list[str]:
        return sorted(k for k, v in self.dimensions.items() if v.unstable)


# A task counts as passing when its weakest signal still clears this. Completion
# is separated out because a task that did not complete has not passed, whatever
# its trajectory-quality scores say.
PASS_OVERALL = 0.7
PASS_COMPLETION = 0.6


def aggregate(task_runs: list[TaskRun]) -> RunAggregate:
    """Roll individual executions up into per-task and per-dimension statistics."""
    by_task: dict[str, list[TaskRun]] = {}
    for tr in task_runs:
        by_task.setdefault(tr.task_id, []).append(tr)

    summaries: dict[str, TaskSummary] = {}
    for task_id, runs in by_task.items():
        dimension_names = sorted({d for r in runs for d in r.scores})
        dimensions = {
            name: Stat.of([r.scores[name].score for r in runs if name in r.scores])
            for name in dimension_names
        }
        overall = Stat.of([r.overall for r in runs])
        completion = dimensions.get("task_completion")
        summaries[task_id] = TaskSummary(
            task_id=task_id,
            category=runs[0].category,
            weight=runs[0].weight,
            overall=overall,
            dimensions=dimensions,
            passed=(
                overall.mean >= PASS_OVERALL
                and (completion is None or completion.mean >= PASS_COMPLETION)
            ),
        )

    all_dimensions = sorted({d for s in summaries.values() for d in s.dimensions})
    suite_dimensions = {
        name: _weighted_stat(summaries, name) for name in all_dimensions
    }

    total_weight = sum(s.weight for s in summaries.values()) or 1.0
    suite_overall = Stat(
        mean=round(
            sum(s.overall.mean * s.weight for s in summaries.values()) / total_weight, 4
        ),
        stdev=round(
            statistics.fmean([s.overall.stdev for s in summaries.values()] or [0.0]), 4
        ),
        n=len(task_runs),
        min=round(min((s.overall.min for s in summaries.values()), default=0.0), 4),
        max=round(max((s.overall.max for s in summaries.values()), default=0.0), 4),
    )

    return RunAggregate(
        overall=suite_overall, dimensions=suite_dimensions, tasks=summaries
    )


def _weighted_stat(summaries: dict[str, TaskSummary], dimension: str) -> Stat:
    """Suite-level stat for a dimension, weighted by task weight.

    The stdev reported is the mean of per-task stdevs — within-task run-to-run
    noise, which is what the regression detector needs. Spread *between* tasks is
    a property of the suite, not of the agent, and would mask real regressions if
    folded in here.
    """
    entries = [
        (s.weight, s.dimensions[dimension]) for s in summaries.values() if dimension in s.dimensions
    ]
    if not entries:
        return Stat(mean=0.0, stdev=0.0, n=0, min=0.0, max=0.0)

    total_weight = sum(w for w, _ in entries) or 1.0
    return Stat(
        mean=round(sum(w * st.mean for w, st in entries) / total_weight, 4),
        stdev=round(statistics.fmean([st.stdev for _, st in entries]), 4),
        n=sum(st.n for _, st in entries),
        min=round(min(st.min for _, st in entries), 4),
        max=round(max(st.max for _, st in entries), 4),
    )


def to_run_record(
    run_id: str,
    suite: str,
    agent_version: str,
    created_at: str,
    task_runs: list[TaskRun],
    agg: RunAggregate,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The persisted run file: everything needed to compare or re-render later."""
    return {
        "run_id": run_id,
        "suite": suite,
        "agent_version": agent_version,
        "created_at": created_at,
        "task_runs": [tr.model_dump(mode="json") for tr in task_runs],
        "aggregate": agg.model_dump(mode="json"),
        **(extra or {}),
    }
