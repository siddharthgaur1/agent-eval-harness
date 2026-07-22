"""Suite loading, aggregation, persistence, and the API surface."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.aggregate import TaskRun, aggregate
from src.persistence.store import Store
from src.scorers.base import ScoreResult
from src.suites.loader import load_suite
from src.suites.schema import Assertion, TaskDef
from src.trace.schema import TerminalState


# -- suite loading -----------------------------------------------------------


def test_default_suite_loads_and_covers_all_three_categories():
    suite = load_suite("suites/default.yaml")
    assert len(suite.tasks) >= 15
    categories = {t.category for t in suite.tasks}
    assert categories == {"happy_path", "adversarial", "budget_stress"}


def test_adversarial_tasks_do_not_all_demand_completion():
    """Graceful failure is the correct answer for these; asserting completed is a bug."""
    suite = load_suite("suites/default.yaml")
    adversarial = [t for t in suite.tasks if t.category == "adversarial"]
    assert adversarial
    assert any(
        TerminalState.ESCALATED in t.acceptable_terminal_states for t in adversarial
    )


def test_every_task_input_file_exists():
    """A task pointing at a missing CSV is unrunnable against a real agent.

    The mock never opens these files, so nothing else in the suite would notice
    until someone pointed the harness at a live agent and lost the run.
    """
    from pathlib import Path

    missing = [
        (t.task_id, t.input["csv_path"])
        for t in load_suite("suites/default.yaml").tasks
        if t.input.get("csv_path") and not Path(t.input["csv_path"]).exists()
    ]
    assert not missing, f"tasks reference files that do not exist: {missing}"


def test_duplicate_task_ids_are_rejected(tmp_path):
    path = tmp_path / "dupes.yaml"
    path.write_text("- task_id: a\n- task_id: a\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate task_id"):
        load_suite(path)


def test_an_assertion_must_set_exactly_one_field():
    with pytest.raises(ValueError, match="exactly one field"):
        Assertion(metric_present="x", output_contains="y")
    with pytest.raises(ValueError, match="exactly one field"):
        Assertion()


def test_assertions_check_against_a_trajectory(perfect):
    assert Assertion(metric_present="roc_auc").check(perfect)
    assert not Assertion(metric_present="f1").check(perfect)
    assert Assertion(artifact_exists="model.pkl").check(perfect)
    assert Assertion(tool_called="train").check(perfect)
    assert Assertion(tool_not_called="tune").check(perfect)
    assert Assertion(terminal_state=TerminalState.COMPLETED).check(perfect)


# -- aggregation -------------------------------------------------------------


def _task_run(task_id: str, repeat: int, value: float, weight: float = 1.0) -> TaskRun:
    return TaskRun(
        task_id=task_id,
        repeat=repeat,
        trajectory_id=f"{task_id}{repeat}",
        weight=weight,
        scores={
            "task_completion": ScoreResult(
                scorer="task_completion", score=value, reasoning="f", details={"f": 1}
            )
        },
    )


def test_aggregate_reports_mean_and_spread():
    agg = aggregate([_task_run("t", i, v) for i, v in enumerate([0.6, 0.8, 1.0])])
    stat = agg.tasks["t"].dimensions["task_completion"]
    assert stat.mean == pytest.approx(0.8)
    assert stat.stdev > 0
    assert stat.n == 3


def test_a_wide_spread_marks_a_dimension_unstable():
    agg = aggregate([_task_run("t", i, v) for i, v in enumerate([0.1, 0.9, 0.5, 1.0])])
    assert "task_completion" in agg.unstable_dimensions


def test_suite_overall_is_weighted_by_task_weight():
    runs = [_task_run("cheap", 0, 0.0, weight=1.0), _task_run("heavy", 0, 1.0, weight=3.0)]
    assert aggregate(runs).overall.mean == pytest.approx(0.75)


def test_a_task_that_did_not_complete_does_not_pass():
    assert not aggregate([_task_run("t", 0, 0.5)]).tasks["t"].passed
    assert aggregate([_task_run("t", 0, 0.95)]).tasks["t"].passed


# -- persistence -------------------------------------------------------------


def test_trajectories_round_trip(tmp_path, perfect):
    store = Store(tmp_path / "t.db")
    store.save_trajectory(perfect)
    loaded = store.get_trajectory(perfect.run_id)
    assert loaded is not None
    assert len(loaded.steps) == len(perfect.steps)
    assert loaded.metadata == perfect.metadata


def test_saving_the_same_run_id_twice_replaces_it(tmp_path, perfect):
    store = Store(tmp_path / "t.db")
    store.save_trajectory(perfect)
    perfect.final_output = "changed"
    store.save_trajectory(perfect)
    assert len(store.list_trajectories()) == 1
    assert store.get_trajectory(perfect.run_id).final_output == "changed"


def test_baseline_marking(tmp_path):
    store = Store(tmp_path / "t.db")
    for rid in ("r1", "r2"):
        store.save_run(
            {"run_id": rid, "suite": "s", "agent_version": "v", "created_at": rid, "aggregate": {}}
        )
    assert store.latest_baseline() is None
    store.mark_baseline("r1")
    assert store.latest_baseline()["run_id"] == "r1"


# -- API ---------------------------------------------------------------------


@pytest.fixture
def client(tmp_path, monkeypatch):
    import src.api.app as app_module

    monkeypatch.setattr(app_module, "store", Store(tmp_path / "api.db"))
    return TestClient(app_module.app)


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"


def test_ingest_and_read_back_a_trajectory(client, perfect):
    response = client.post("/trajectories", json=perfect.model_dump(mode="json"))
    assert response.status_code == 201
    assert response.json()["steps"] == len(perfect.steps)

    fetched = client.get(f"/trajectories/{perfect.run_id}")
    assert fetched.status_code == 200
    assert fetched.json()["task_id"] == perfect.task_id


def test_ingest_rejects_a_malformed_trajectory(client):
    assert client.post("/trajectories", json={"task_id": "no run id"}).status_code == 422


def test_unknown_ids_are_404(client):
    assert client.get("/trajectories/nope").status_code == 404
    assert client.get("/runs/nope").status_code == 404
    assert client.get("/runs/nope/report.html").status_code == 404
