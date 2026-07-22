"""Suite runner.

    python -m src.run --suite suites/default.yaml --agent-version v1

Executes every task N times, records trajectories, scores them, aggregates with
variance, and writes a run file. Optionally compares against a baseline and exits
non-zero on a hard regression, which is how the GitHub Action gates a merge.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from .aggregate import RunAggregate, TaskRun, aggregate, to_run_record
from .agents_under_test import resolve_agent
from .compare.regression import compare_runs, format_report
from .config import settings
from .persistence.store import Store
from .report.html import write_report
from .scorers import score_trajectory
from .scorers.base import ScoreResult
from .suites.loader import load_suite
from .suites.schema import Suite, TaskDef
from .trace.schema import Trajectory

# Agents are told which variant to be through the environment, so two suite runs
# in the same process would overwrite each other's version. Serialising them is
# the honest fix: the parallelism that matters is inside a run, across tasks.
_run_lock = threading.Lock()


def run_suite(
    suite: Suite,
    agent_spec: str,
    agent_version: str,
    *,
    repeats: int | None = None,
    workers: int | None = None,
    include_llm: bool = True,
    store: Store | None = None,
) -> tuple[dict, RunAggregate]:
    """Execute and score a whole suite. Returns the run record and its aggregate."""
    repeats = repeats or settings.repeats
    workers = workers or settings.workers
    store = store or Store()
    agent = resolve_agent(agent_spec)

    jobs = [(task, r) for task in suite.tasks for r in range(repeats)]
    task_runs: list[TaskRun] = []

    with _run_lock:
        # The agent reads its own version from the environment; the harness should
        # not have to know how a given agent wants to be told which variant to be.
        os.environ["AGENT_UNDER_TEST_VERSION"] = agent_version

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_run_one, agent, task, repeat, include_llm, store): (task, repeat)
                for task, repeat in jobs
            }
            for future in as_completed(futures):
                task, repeat = futures[future]
                try:
                    task_runs.append(future.result())
                except Exception as exc:  # a crashed task is a data point, not a stop
                    traceback.print_exc()
                    task_runs.append(_crashed(task, repeat, exc))

    task_runs.sort(key=lambda tr: (tr.task_id, tr.repeat))
    agg = aggregate(task_runs)

    run_id = uuid.uuid4().hex[:12]
    record = to_run_record(
        run_id=run_id,
        suite=suite.name,
        agent_version=agent_version,
        created_at=datetime.now(timezone.utc).isoformat(),
        task_runs=task_runs,
        agg=agg,
        extra={"repeats": repeats, "include_llm": include_llm, "agent": agent_spec},
    )
    store.save_run(record)

    settings.ensure_dirs()
    (settings.runs_dir / f"{run_id}.json").write_text(
        json.dumps(record, indent=2, default=str), encoding="utf-8"
    )
    return record, agg


def _run_one(
    agent, task: TaskDef, repeat: int, include_llm: bool, store: Store
) -> TaskRun:
    """Execute one task once, record the trajectory, score it."""
    traj: Trajectory = agent(task, repeat)
    traj.task_id = task.task_id  # the task definition is authoritative
    store.save_trajectory(traj)

    scores = score_trajectory(traj, task, include_llm=include_llm, store=store)
    return TaskRun(
        task_id=task.task_id,
        repeat=repeat,
        trajectory_id=traj.run_id,
        category=task.category,
        weight=task.weight,
        scores=scores,
    )


def _crashed(task: TaskDef, repeat: int, exc: Exception) -> TaskRun:
    """A task whose agent blew up scores zero, with the traceback as evidence."""
    reason = f"agent raised {type(exc).__name__}: {exc}"
    return TaskRun(
        task_id=task.task_id,
        repeat=repeat,
        trajectory_id="",
        category=task.category,
        weight=task.weight,
        error=reason,
        scores={
            "task_completion": ScoreResult(
                scorer="task_completion",
                score=0.0,
                reasoning=reason,
                details={"crashed": True},
            )
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run an agent evaluation suite.")
    parser.add_argument("--suite", default="suites/default.yaml")
    parser.add_argument("--agent", default="mock", help="module:function entrypoint")
    parser.add_argument("--agent-version", default="v1")
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--no-llm", action="store_true", help="deterministic scorers only")
    parser.add_argument("--baseline", default=None, help="run_id to compare against")
    parser.add_argument(
        "--against-latest-baseline",
        action="store_true",
        help="compare against the most recent run marked as baseline",
    )
    parser.add_argument("--set-baseline", action="store_true")
    parser.add_argument("--report", default=None, help="write an HTML report to this path")
    parser.add_argument("--tasks", default=None, help="comma-separated task_id filter")
    args = parser.parse_args(argv)

    settings.ensure_dirs()
    store = Store()

    suite = load_suite(args.suite)
    if args.tasks:
        wanted = {t.strip() for t in args.tasks.split(",")}
        suite = Suite(name=suite.name, tasks=[t for t in suite.tasks if t.task_id in wanted])

    record, agg = run_suite(
        suite,
        args.agent,
        args.agent_version,
        repeats=args.repeats,
        workers=args.workers,
        include_llm=not args.no_llm,
        store=store,
    )

    print(f"\nrun_id: {record['run_id']}  suite: {suite.name}  agent: {args.agent_version}")
    print(f"overall: {agg.overall.mean:.3f} (±{agg.overall.stdev:.3f})")
    for name, stat in sorted(agg.dimensions.items()):
        flag = "  [UNSTABLE]" if stat.unstable else ""
        print(f"  {name:<24} {stat.mean:.3f} ±{stat.stdev:.3f}{flag}")
    failed = [t for t in agg.tasks.values() if not t.passed]
    print(f"tasks passing: {len(agg.tasks) - len(failed)}/{len(agg.tasks)}")
    if failed:
        print("  failing: " + ", ".join(sorted(t.task_id for t in failed)))

    if args.set_baseline:
        store.mark_baseline(record["run_id"])
        print(f"marked {record['run_id']} as baseline")

    baseline = None
    if args.baseline:
        baseline = store.get_run(args.baseline)
        if baseline is None:
            print(f"error: no run {args.baseline}", file=sys.stderr)
            return 2
    elif args.against_latest_baseline:
        baseline = store.latest_baseline()
        if baseline is None:
            print("no baseline recorded yet; skipping comparison")

    comparison = None
    if baseline:
        history = store.list_runs(agent_version=baseline["agent_version"], limit=settings.drift_window)
        comparison = compare_runs(baseline, record, history=history)
        print("\n" + format_report(comparison))

    if args.report:
        path = write_report(record, comparison=comparison, out_path=Path(args.report))
        print(f"report: {path}")

    return 1 if comparison and comparison.has_hard_regression else 0


if __name__ == "__main__":
    raise SystemExit(main())
