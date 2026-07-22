"""A simulated agent, for demonstrating and testing the harness itself.

Not a toy for its own sake: the regression detector's most important property is
that it does *not* fire on run-to-run noise, and proving that needs an agent
whose noise level is known. Real agents cannot give you that. Three versions:

* ``v1``            — competent baseline.
* ``v1-rerun``      — statistically identical to v1, different seed. Comparing
                      these must NOT flag a regression.
* ``v2-degraded``   — a weakened planner: skips a tool, loops on failure, and
                      reports a conclusion its tools never produced.

Set the version with AGENT_UNDER_TEST_VERSION, which the runner exports from
``--agent-version``.
"""

from __future__ import annotations

import os
import random

from ..suites.schema import TaskDef
from ..trace.schema import StepType, TerminalState, Trajectory
from ..trace.recorder import Recorder


def run_task(task: TaskDef, repeat: int = 0) -> Trajectory:
    """Simulate one run of `task` for the configured agent version."""
    version = os.environ.get("AGENT_UNDER_TEST_VERSION", "v1")
    rng = random.Random(f"{version}:{task.task_id}:{repeat}")
    degraded = version.startswith("v2")

    rec = Recorder(task.task_id, version, model="simulated")
    tools = task.expected_tools or ["work"]

    # A degraded planner drops the second-to-last tool: enough to break tool
    # recall and downstream assertions without making the run obviously broken.
    planned = tools[:-2] + tools[-1:] if degraded and len(tools) > 2 else list(tools)

    rec.add_step(
        StepType.PLAN,
        agent_name="planner",
        tool_input={"plan": planned},
        tokens=rng.randint(200, 400),
    )

    tokens = 0
    for tool in planned:
        fails = _should_fail(task, tool, rng, degraded)
        tokens += _do_call(rec, tool, rng, fails, degraded)
        # Real agents revisit stages — re-running EDA after cleaning, re-scoring
        # after a feature change. Modelled here because without it every run is
        # identical, the variance is zero, and the noise band means nothing.
        if rng.random() < (0.5 if degraded else 0.3):
            tokens += _do_call(rec, tool, rng, fails=False, degraded=degraded, pass_no=2)

    if task.category == "budget_stress" and degraded:
        # Runaway loop: the exact failure the loop and efficiency scorers exist for.
        for _ in range(6):
            tokens += _do_call(rec, tools[-1], rng, fails=False, degraded=True)

    over_budget = bool(task.budget.max_tokens and tokens > task.budget.max_tokens)
    terminal = _terminal_state(task, degraded, over_budget)

    metadata = _metadata(task, succeeded=terminal is TerminalState.COMPLETED, degraded=degraded)
    output = _final_output(task, terminal, degraded)

    rec.finish(terminal, output, **metadata)
    traj = rec.trajectory
    traj.terminal_state = terminal
    traj.total_tokens = tokens
    traj.total_cost = round(tokens * 2.5e-5, 4)
    traj.wall_clock_seconds = round(len(traj.steps) * rng.uniform(3.0, 6.0), 1)
    traj.metadata.update(metadata)
    return traj.finalize()


def _do_call(
    rec: Recorder,
    tool: str,
    rng: random.Random,
    fails: bool,
    degraded: bool,
    pass_no: int = 1,
) -> int:
    """Emit a call/result pair, plus retries if it failed. Returns tokens spent.

    `pass_no` distinguishes a legitimate revisit (different input, not a loop)
    from the degraded agent's identical retries (same input, very much a loop).
    """
    tokens = rng.randint(800, 1600)
    args = {"step": tool} if pass_no == 1 else {"step": tool, "pass": pass_no}
    rec.add_step(StepType.TOOL_CALL, agent_name=tool, tool_name=tool, tool_input=args)
    if not fails:
        rec.add_step(
            StepType.TOOL_RESULT,
            agent_name=tool,
            tool_name=tool,
            tool_output=f"{tool} ok",
            tokens=tokens,
            latency_ms=rng.randint(200, 3000),
        )
        return tokens

    rec.add_step(
        StepType.TOOL_RESULT,
        agent_name=tool,
        tool_name=tool,
        error=f"{tool} failed: upstream returned nothing usable",
        tokens=tokens // 2,
    )
    # v1 retries once and recovers; v2 retries the identical call and never adapts.
    attempts = 4 if degraded else 1
    for i in range(attempts):
        rec.add_step(StepType.RETRY, agent_name=tool, tool_name=tool, tool_input={"step": tool})
        rec.add_step(StepType.TOOL_CALL, agent_name=tool, tool_name=tool, tool_input={"step": tool})
        recovered = not degraded
        rec.add_step(
            StepType.TOOL_RESULT,
            agent_name=tool,
            tool_name=tool,
            tool_output=f"{tool} ok on retry {i + 1}" if recovered else None,
            error=None if recovered else f"{tool} failed again (attempt {i + 2})",
            tokens=tokens // 2,
        )
        tokens += tokens // 2
        if recovered:
            break
    return tokens


def _should_fail(task: TaskDef, tool: str, rng: random.Random, degraded: bool) -> bool:
    """Adversarial tasks fail somewhere by construction; others fail rarely."""
    if task.category == "adversarial" and tool == (task.expected_tools or ["work"])[0]:
        return True
    return rng.random() < (0.25 if degraded else 0.08)


def _terminal_state(task: TaskDef, degraded: bool, over_budget: bool) -> TerminalState:
    if over_budget:
        return TerminalState.BUDGET_EXCEEDED
    if task.category == "adversarial":
        # The correct behaviour is a clean stop. The degraded agent instead
        # ploughs on and claims success, which is the whole point of the task.
        if degraded:
            return TerminalState.COMPLETED
        return (
            TerminalState.ESCALATED
            if TerminalState.ESCALATED in task.acceptable_terminal_states
            else TerminalState.FAILED
        )
    return TerminalState.FAILED if degraded and task.category == "budget_stress" else TerminalState.COMPLETED


def _metadata(task: TaskDef, succeeded: bool, degraded: bool) -> dict:
    """Produce exactly the artifacts and metrics the task asserts on."""
    artifacts, metrics = [], {}
    for assertion in task.success_assertions:
        if assertion.artifact_exists and succeeded:
            artifacts.append(assertion.artifact_exists)
        if assertion.metric_present and succeeded and not degraded:
            metrics[assertion.metric_present] = 0.83
    return {"artifacts": artifacts, "metrics": metrics}


def _final_output(task: TaskDef, terminal: TerminalState, degraded: bool) -> str:
    if terminal is TerminalState.ESCALATED:
        return (
            f"Stopping and escalating: {task.description} cannot be completed as "
            "specified. The input does not support the requested analysis."
        )
    if terminal is TerminalState.BUDGET_EXCEEDED:
        return "Halted: token budget exhausted before the task completed."
    if degraded:
        # Unsupported claim: no tool in this run produced 0.91.
        return (
            f"Completed {task.description}. The model reached 0.91 ROC-AUC and is "
            "ready for production deployment."
        )
    return f"Completed {task.description}. Results are recorded in the run artifacts."
