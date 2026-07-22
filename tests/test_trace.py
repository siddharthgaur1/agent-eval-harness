"""Schema, recorders, adapters, and the JSON importer."""

from __future__ import annotations

import json

import pytest

from src.trace.importer import load_trajectory, parse_trajectory
from src.trace.langgraph import TrajectoryCallbackHandler, trajectory_from_state
from src.trace.recorder import Recorder, record_step
from src.trace.schema import StepType, TerminalState


# -- generic recorder --------------------------------------------------------


def test_recorder_builds_a_well_formed_trajectory():
    with Recorder("t1", "v1", model="gpt-4o") as rec:
        rec.add_step(StepType.PLAN, tool_input={"plan": ["a"]})
        rec.add_step(StepType.TOOL_CALL, tool_name="load", tokens=100)
        rec.finish(TerminalState.COMPLETED, "all done", artifacts=["m.pkl"])

    traj = rec.trajectory
    assert [s.index for s in traj.steps] == [0, 1]
    assert traj.terminal_state is TerminalState.COMPLETED
    assert traj.final_output == "all done"
    assert traj.total_tokens == 100
    assert traj.metadata["artifacts"] == ["m.pkl"]
    assert traj.ended_at is not None


def test_an_escaping_exception_marks_the_run_failed():
    rec = Recorder("t1", "v1")
    with pytest.raises(ValueError):
        with rec:
            rec.add_step(StepType.TOOL_CALL, tool_name="load")
            raise ValueError("kaboom")

    assert rec.trajectory.terminal_state is TerminalState.FAILED
    assert "kaboom" in rec.trajectory.steps[-1].error


def test_decorator_records_calls_and_reraises():
    @record_step(tool_name="fetch")
    def fetch(url: str) -> str:
        if url == "bad":
            raise RuntimeError("404")
        return "payload"

    with Recorder("t1", "v1") as rec:
        assert fetch("good") == "payload"
        with pytest.raises(RuntimeError):
            fetch("bad")

    steps = rec.trajectory.steps
    assert [s.tool_name for s in steps] == ["fetch", "fetch"]
    assert steps[0].tool_output == "payload"
    assert "404" in steps[1].error


def test_decorator_is_a_noop_without_a_recorder():
    @record_step()
    def add(a: int, b: int) -> int:
        return a + b

    assert add(2, 3) == 5  # no active recorder, no crash


# -- schema helpers ----------------------------------------------------------


def test_call_signature_distinguishes_inputs():
    from tests.conftest import call

    assert call("t", a=1).call_signature() == call("t", a=1).call_signature()
    assert call("t", a=1).call_signature() != call("t", a=2).call_signature()


def test_tools_used_preserves_order_and_duplicates(perfect):
    assert perfect.tools_used() == [
        "load", "load", "clean", "clean", "train", "train", "evaluate", "evaluate"
    ]


# -- LangGraph adapter -------------------------------------------------------


def test_fake_langgraph_run_produces_a_well_formed_trajectory():
    handler = TrajectoryCallbackHandler("churn", "v1", model="gpt-4o")

    handler.on_chain_start({"name": "supervisor"}, {}, name="supervisor")
    handler.on_tool_start({"name": "load_data"}, "churn.csv", inputs={"path": "churn.csv"})
    handler.on_tool_end("12 columns loaded")
    handler.on_chain_end({})

    handler.on_chain_start({"name": "training"}, {}, name="training")
    handler.on_tool_start({"name": "train_model"}, "{}", inputs={})
    handler.on_tool_error(RuntimeError("singular matrix"))
    handler.on_chain_end({})

    traj = handler.finalize(TerminalState.COMPLETED, "model trained", artifacts=["m.pkl"])

    assert traj.task_id == "churn"
    assert traj.tools_used() == ["load_data", "load_data", "train_model", "train_model"]
    assert [s.index for s in traj.steps] == list(range(len(traj.steps)))
    assert traj.steps[1].agent_name == "supervisor"
    assert len(traj.failed_steps()) == 1
    assert "singular matrix" in traj.failed_steps()[0].error
    assert traj.terminal_state is TerminalState.COMPLETED


def test_internal_langgraph_chains_are_not_recorded_as_nodes():
    handler = TrajectoryCallbackHandler("t", "v1")
    handler.on_chain_start({"name": "RunnableSequence"}, {}, name="RunnableSequence")
    assert handler.trajectory.steps == []


def test_reconstruction_from_a_finished_state():
    state = {
        "run_id": "abc123",
        "status": "completed",
        "narrative": "Churn is driven by contract length.",
        "messages": [
            {"agent": "supervisor", "content": "planning", "level": "info"},
            {"agent": "cleaning", "content": "dropped 3 nulls", "level": "info"},
            {"agent": "eda", "content": "read failed", "level": "error"},
            {"agent": "evaluation", "content": "roc_auc 0.87", "level": "info"},
        ],
        "artifacts": [{"kind": "model", "path": "model.pkl"}],
        "eval_metrics": {"metrics": {"roc_auc": 0.87}},
        "token_usage": {"cleaning": {"prompt_tokens": 100, "completion_tokens": 50}},
    }
    traj = trajectory_from_state(state, "churn_baseline", "ads-v1")

    assert traj.terminal_state is TerminalState.COMPLETED
    assert traj.metadata["metrics"] == {"roc_auc": 0.87}
    assert traj.metadata["artifacts"] == ["model.pkl"]
    assert "clean" in traj.tools_used()
    assert len(traj.failed_steps()) == 1
    assert traj.total_tokens == 150


def test_awaiting_human_reconstructs_as_escalated():
    traj = trajectory_from_state(
        {"status": "awaiting_human", "needs_human": True, "messages": []}, "t", "v"
    )
    assert traj.terminal_state is TerminalState.ESCALATED


def test_unrecognised_state_degrades_instead_of_raising():
    traj = trajectory_from_state({"nothing": "useful"}, "t", "v")
    assert traj.terminal_state is TerminalState.FAILED
    assert traj.steps == []


# -- JSON importer -----------------------------------------------------------


def test_importer_reindexes_and_normalises(tmp_path):
    payload = {
        "run_id": "x1",
        "task_id": "t",
        "agent_version": "v",
        "terminal_state": "success",  # alias
        "steps": [
            {"index": 7, "step_type": "tool_call", "tool_name": "a", "tool_input": "raw string"},
            {"index": 7, "step_type": "tool_result", "tool_name": "a"},
        ],
    }
    path = tmp_path / "traj.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    traj = load_trajectory(path)
    assert traj.terminal_state is TerminalState.COMPLETED
    assert [s.index for s in traj.steps] == [0, 1], "evidence needs unique, gapless indices"
    assert traj.steps[0].tool_input == {"input": "raw string"}


def test_importer_rejects_a_bad_payload():
    with pytest.raises(Exception):
        parse_trajectory({"task_id": "t"})  # no run_id
