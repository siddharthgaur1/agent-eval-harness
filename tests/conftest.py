"""Fixture trajectories with known-correct scores.

Every deterministic scorer is tested against a trajectory hand-built to fail it
in a specific way. If a looping trajectory ever scores well on loop detection,
that is the test that catches it.
"""

from __future__ import annotations

import pytest

from src.suites.schema import Assertion, Budget, TaskDef
from src.trace.schema import Step, StepType, TerminalState, Trajectory


def make_trajectory(steps: list[Step], **kwargs) -> Trajectory:
    """Build a trajectory with sane defaults, re-indexing the steps."""
    for i, step in enumerate(steps):
        step.index = i
    defaults = dict(
        run_id="t1",
        task_id="demo",
        agent_version="test",
        terminal_state=TerminalState.COMPLETED,
        final_output="done",
    )
    defaults.update(kwargs)
    return Trajectory(steps=steps, **defaults).finalize()


def call(tool: str, **inp) -> Step:
    return Step(index=0, step_type=StepType.TOOL_CALL, tool_name=tool, tool_input=inp or {"a": 1})


def result(tool: str, output: str = "ok", error: str | None = None, tokens: int = 100) -> Step:
    return Step(
        index=0,
        step_type=StepType.TOOL_RESULT,
        tool_name=tool,
        tool_output=None if error else output,
        error=error,
        tokens=tokens,
    )


@pytest.fixture
def task() -> TaskDef:
    """A task expecting four tools in twelve steps, under budget."""
    return TaskDef(
        task_id="demo",
        description="demo task",
        goal="do the demo task",
        expected_tools=["load", "clean", "train", "evaluate"],
        optimal_steps=4,
        budget=Budget(max_tokens=10_000, max_cost_usd=1.0, max_seconds=100),
        success_assertions=[
            Assertion(terminal_state=TerminalState.COMPLETED),
            Assertion(metric_present="roc_auc"),
        ],
    )


@pytest.fixture
def perfect(task: TaskDef) -> Trajectory:
    """Calls every expected tool once, succeeds, stays in budget."""
    steps = []
    for tool in task.expected_tools:
        steps += [call(tool), result(tool)]
    return make_trajectory(
        steps,
        total_tokens=4000,
        total_cost=0.4,
        wall_clock_seconds=40,
        metadata={"metrics": {"roc_auc": 0.9}, "artifacts": ["model.pkl"]},
    )


@pytest.fixture
def looping() -> Trajectory:
    """Calls the same tool with the same input eight times."""
    steps = []
    for _ in range(8):
        steps += [call("train", seed=1), result("train")]
    return make_trajectory(steps, total_tokens=8000, wall_clock_seconds=80)


@pytest.fixture
def gave_up() -> Trajectory:
    """Fails on `train` and stops without ever recovering."""
    steps = [
        call("load"),
        result("load"),
        call("train"),
        result("train", error="OOM"),
    ]
    return make_trajectory(
        steps, terminal_state=TerminalState.FAILED, total_tokens=2000, wall_clock_seconds=20
    )


@pytest.fixture
def recovered() -> Trajectory:
    """Fails once on `train`, retries, and succeeds."""
    steps = [
        call("load"),
        result("load"),
        call("train"),
        result("train", error="transient network error"),
        Step(index=0, step_type=StepType.RETRY, tool_name="train"),
        call("train", seed=2),
        result("train"),
        call("evaluate"),
        result("evaluate"),
    ]
    return make_trajectory(steps, total_tokens=3000, wall_clock_seconds=30)


@pytest.fixture
def over_budget() -> Trajectory:
    """Well over every declared cap."""
    steps = [call("load"), result("load", tokens=25_000)]
    return make_trajectory(
        steps,
        terminal_state=TerminalState.BUDGET_EXCEEDED,
        total_tokens=25_000,
        total_cost=3.0,
        wall_clock_seconds=400,
    )
