"""FastAPI surface.

Ingest trajectories recorded elsewhere, kick off suite runs, read results, and
fetch a rendered report. Evaluation runs are backgrounded — a suite takes minutes
and an HTTP client should not be holding a socket open through it.
"""

from __future__ import annotations

import uuid
from typing import Any, Literal

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from ..compare.regression import compare_runs, format_report
from ..config import settings
from ..persistence.store import Store
from ..report.html import render_report
from ..run import run_suite
from ..suites.loader import load_suite
from ..trace.importer import parse_trajectory

app = FastAPI(
    title="Agent Evaluation Harness",
    description="Trajectory-level scoring and regression detection for multi-step agents.",
    version="1.0.0",
)

store = Store()

# In-process job table. A single-node harness with a handful of concurrent suite
# runs does not need a broker; if this ever outgrows one process, the state to
# move is this dict and nothing else.
_jobs: dict[str, dict[str, Any]] = {}


class EvaluateRequest(BaseModel):
    suite: str = "suites/default.yaml"
    agent: str = "mock"
    agent_version: str = "v1"
    repeats: int | None = None
    include_llm: bool = True
    baseline_run_id: str | None = None
    set_baseline: bool = False


class EvaluateResponse(BaseModel):
    job_id: str
    status: Literal["queued", "running", "done", "error"]
    run_id: str | None = None


class TrajectoryResponse(BaseModel):
    run_id: str
    task_id: str
    agent_version: str
    steps: int
    terminal_state: str


@app.get("/health")
def health() -> dict[str, Any]:
    """Liveness plus enough config to debug a bad deployment."""
    return {
        "status": "ok",
        "db": str(settings.db_path),
        "llm_judges": bool(settings.openai_api_key),
        "judge_model": settings.judge_model,
    }


@app.post("/trajectories", response_model=TrajectoryResponse, status_code=201)
def ingest_trajectory(payload: dict[str, Any]) -> TrajectoryResponse:
    """Store a trajectory recorded by someone else's agent."""
    try:
        traj = parse_trajectory(payload)
    except Exception as exc:
        raise HTTPException(422, f"invalid trajectory: {exc}") from exc
    store.save_trajectory(traj)
    return TrajectoryResponse(
        run_id=traj.run_id,
        task_id=traj.task_id,
        agent_version=traj.agent_version,
        steps=len(traj.steps),
        terminal_state=traj.terminal_state.value,
    )


@app.get("/trajectories/{run_id}")
def get_trajectory(run_id: str) -> dict[str, Any]:
    traj = store.get_trajectory(run_id)
    if traj is None:
        raise HTTPException(404, f"no trajectory {run_id}")
    return traj.model_dump(mode="json")


@app.get("/trajectories")
def list_trajectories(
    task_id: str | None = None, agent_version: str | None = None, limit: int = 50
) -> list[TrajectoryResponse]:
    return [
        TrajectoryResponse(
            run_id=t.run_id,
            task_id=t.task_id,
            agent_version=t.agent_version,
            steps=len(t.steps),
            terminal_state=t.terminal_state.value,
        )
        for t in store.list_trajectories(task_id, agent_version, limit)
    ]


@app.post("/evaluate", response_model=EvaluateResponse, status_code=202)
def evaluate(req: EvaluateRequest, background: BackgroundTasks) -> EvaluateResponse:
    """Start a suite run in the background."""
    try:
        suite = load_suite(req.suite)
    except Exception as exc:
        raise HTTPException(400, f"cannot load suite: {exc}") from exc

    job_id = uuid.uuid4().hex[:12]
    _jobs[job_id] = {"status": "queued", "run_id": None, "error": None}
    background.add_task(_run_job, job_id, suite, req)
    return EvaluateResponse(job_id=job_id, status="queued")


def _run_job(job_id: str, suite, req: EvaluateRequest) -> None:
    _jobs[job_id]["status"] = "running"
    try:
        record, _ = run_suite(
            suite,
            req.agent,
            req.agent_version,
            repeats=req.repeats,
            include_llm=req.include_llm,
            store=store,
        )
        if req.set_baseline:
            store.mark_baseline(record["run_id"])
        _jobs[job_id].update(status="done", run_id=record["run_id"])
    except Exception as exc:
        _jobs[job_id].update(status="error", error=str(exc))


@app.get("/jobs/{job_id}", response_model=EvaluateResponse)
def job_status(job_id: str) -> EvaluateResponse:
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"no job {job_id}")
    return EvaluateResponse(job_id=job_id, status=job["status"], run_id=job["run_id"])


@app.get("/runs")
def list_runs(agent_version: str | None = None, limit: int = 25) -> list[dict[str, Any]]:
    """Run headers only — the full records are large."""
    return [
        {
            "run_id": r["run_id"],
            "suite": r["suite"],
            "agent_version": r["agent_version"],
            "created_at": r["created_at"],
            "overall": r["aggregate"]["overall"]["mean"],
        }
        for r in store.list_runs(agent_version, limit)
    ]


@app.get("/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    run = store.get_run(run_id)
    if run is None:
        raise HTTPException(404, f"no run {run_id}")
    return run


@app.get("/runs/{run_id}/report.html", response_class=HTMLResponse)
def get_report(run_id: str, baseline: str | None = None) -> HTMLResponse:
    """Rendered report, optionally diffed against a baseline run."""
    run = store.get_run(run_id)
    if run is None:
        raise HTTPException(404, f"no run {run_id}")

    comparison = None
    if baseline:
        base = store.get_run(baseline)
        if base is None:
            raise HTTPException(404, f"no baseline run {baseline}")
        comparison = compare_runs(base, run, history=store.list_runs(base["agent_version"]))

    return HTMLResponse(render_report(run, comparison))


class CompareRequest(BaseModel):
    baseline_run_id: str
    candidate_run_id: str


@app.post("/compare")
def compare(req: CompareRequest) -> dict[str, Any]:
    """Diff two stored runs. `hard_regression` is what CI keys off."""
    base, cand = store.get_run(req.baseline_run_id), store.get_run(req.candidate_run_id)
    if base is None or cand is None:
        raise HTTPException(404, "baseline or candidate run not found")
    result = compare_runs(base, cand, history=store.list_runs(base["agent_version"]))
    return {
        "hard_regression": result.has_hard_regression,
        "text": format_report(result),
        **result.model_dump(mode="json"),
    }


@app.get("/judge-calls/{trajectory_id}")
def judge_calls(trajectory_id: str) -> list[dict[str, Any]]:
    """Raw judge exchanges for one trajectory — the LLM scores' audit trail."""
    return store.judge_calls(trajectory_id)
