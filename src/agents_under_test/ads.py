"""Driver for the `autonomous-data-scientist` LangGraph agent.

Kept out of the harness core on purpose: the harness must never import a specific
agent. This module is the seam — it knows about ADS, ADS knows nothing about the
harness, and the only thing crossing between them is a `Trajectory`.

Point AGENT_UNDER_TEST_PATH at the ADS checkout, e.g.
    AGENT_UNDER_TEST_PATH=C:/Users/munis/projects/autonomous-data-scientist
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from ..suites.schema import TaskDef
from ..trace.langgraph import TrajectoryCallbackHandler, trajectory_from_state
from ..trace.schema import TerminalState, Trajectory


def _load_ads():
    """Import ADS's runner from wherever the checkout lives."""
    root = os.environ.get("AGENT_UNDER_TEST_PATH")
    if not root:
        raise RuntimeError(
            "AGENT_UNDER_TEST_PATH is not set — point it at the "
            "autonomous-data-scientist checkout."
        )
    path = Path(root).resolve()
    if not (path / "src" / "graph" / "runner.py").exists():
        raise RuntimeError(f"{path} does not look like an autonomous-data-scientist checkout")
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    from src.graph.runner import execute_run, prepare_run  # type: ignore

    return prepare_run, execute_run


def run_task(task: TaskDef, repeat: int = 0) -> Trajectory:
    """Execute one ADS run and return its trajectory.

    Capture is belt-and-braces: the callback handler records tool and LLM events
    live, and the final state is used to fill in terminal state, artifacts and
    metrics. Either alone loses something — the callbacks miss the artifact log,
    and the state misses per-step latency.
    """
    prepare_run, execute_run = _load_ads()
    version = os.environ.get("AGENT_UNDER_TEST_VERSION", "ads")

    goal = task.goal or task.description
    csv_path = task.input.get("csv_path") or task.input.get("dataset")
    if not csv_path:
        raise ValueError(f"{task.task_id}: task.input needs a csv_path")

    handler = TrajectoryCallbackHandler(task.task_id, version, model=os.environ.get("MODEL", ""))
    started = time.perf_counter()

    state = prepare_run(goal, csv_path)
    try:
        final = execute_run(state)
    except Exception as exc:
        return handler.finalize(
            TerminalState.FAILED, final_output=f"{type(exc).__name__}: {exc}"
        )

    live = handler.trajectory
    reconstructed = trajectory_from_state(
        dict(final), task.task_id, version, run_id=live.run_id
    )

    # Prefer the live step log when the callbacks actually saw anything.
    if len(live.steps) >= len(reconstructed.steps):
        reconstructed.steps = live.steps
    reconstructed.wall_clock_seconds = round(time.perf_counter() - started, 2)
    return reconstructed.finalize()
