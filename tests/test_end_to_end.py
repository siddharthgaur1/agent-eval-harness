"""End to end: run a suite, score it, compare versions, render a report.

The acceptance criteria the harness exists to satisfy, as tests.
"""

from __future__ import annotations

import pytest

from src.compare.regression import compare_runs
from src.persistence.store import Store
from src.report.html import markdown_summary, render_report
from src.run import run_suite
from src.suites.loader import load_suite
from src.suites.schema import Suite


@pytest.fixture(scope="module")
def suite() -> Suite:
    """A four-task slice: one per category, plus a second happy path."""
    full = load_suite("suites/default.yaml")
    wanted = {"churn_baseline", "eda_only", "missing_target_column", "unbounded_exploration"}
    return Suite(name="e2e", tasks=[t for t in full.tasks if t.task_id in wanted])


@pytest.fixture(scope="module")
def runs(tmp_path_factory, suite):
    """Baseline, a statistically identical rerun, and a degraded version."""
    store = Store(tmp_path_factory.mktemp("e2e") / "e2e.db")
    kwargs = dict(repeats=4, workers=4, include_llm=False, store=store)
    baseline, _ = run_suite(suite, "mock", "v1", **kwargs)
    rerun, _ = run_suite(suite, "mock", "v1-rerun", **kwargs)
    degraded, _ = run_suite(suite, "mock", "v2-degraded", **kwargs)
    return baseline, rerun, degraded


def test_a_run_produces_a_real_scorecard(runs):
    baseline, _, _ = runs
    agg = baseline["aggregate"]

    assert set(agg["dimensions"]) == {
        "task_completion",
        "tool_selection",
        "step_efficiency",
        "error_recovery",
        "budget_adherence",
        "loop_detection",
    }
    assert 0.0 <= agg["overall"]["mean"] <= 1.0
    assert len(baseline["task_runs"]) == 16  # 4 tasks x 4 repeats


def test_every_deduction_in_a_real_run_cites_steps(runs):
    baseline, _, _ = runs
    for task_run in baseline["task_runs"]:
        for name, score in task_run["scores"].items():
            if score["score"] < 1.0:
                assert score["evidence"] or score["details"], (
                    f"{task_run['task_id']}/{name} deducted without evidence"
                )


def test_repeats_produce_a_variance_estimate(runs):
    baseline, _, _ = runs
    assert all(s["n"] > 1 for s in baseline["aggregate"]["dimensions"].values())


def test_a_rerun_of_the_same_agent_is_not_a_regression(runs):
    """Acceptance criterion 3: noise must not turn CI red."""
    baseline, rerun, _ = runs
    assert not compare_runs(baseline, rerun).has_hard_regression


def test_degrading_the_agent_produces_a_flagged_regression(runs):
    """Acceptance criterion 2: a real degradation must be caught and named."""
    baseline, _, degraded = runs
    result = compare_runs(baseline, degraded)

    assert result.has_hard_regression
    assert result.regressed_dimensions
    assert result.newly_failing_tasks, "the report must name which tasks broke"
    # The degraded agent loops and skips a tool; those dimensions must move.
    moved = {d.dimension for d in result.regressed_dimensions}
    assert {"loop_detection", "step_efficiency"} & moved


def test_the_degraded_agent_is_caught_ploughing_through_an_adversarial_task(runs):
    """It reports success on a task whose correct answer is a clean stop."""
    baseline, _, degraded = runs

    def completion(record):
        return [
            tr["scores"]["task_completion"]["score"]
            for tr in record["task_runs"]
            if tr["task_id"] == "missing_target_column"
        ]

    good, bad = completion(baseline), completion(degraded)
    assert good and bad
    assert max(bad) < min(good), (
        "an agent that returns a confident answer to an impossible task must "
        "score below one that stops cleanly"
    )


def test_report_renders_with_and_without_a_comparison(runs):
    baseline, _, degraded = runs
    result = compare_runs(baseline, degraded)

    plain = render_report(baseline)
    assert "<table" in plain and baseline["run_id"] in plain

    diffed = render_report(degraded, result)
    assert "HARD REGRESSION" in diffed
    assert "noise band" in diffed


def test_markdown_summary_is_pr_comment_shaped(runs):
    baseline, _, degraded = runs
    body = markdown_summary(degraded, compare_runs(baseline, degraded))
    assert body.startswith("### Agent eval")
    assert "| dimension | score | spread |" in body
    assert "Hard regression" in body
