"""Each deterministic scorer against a trajectory built to fail it."""

from __future__ import annotations

import pytest

from src.scorers.base import DETERMINISTIC, ScoreResult
from src.suites.schema import Assertion, TaskDef
from src.trace.schema import TerminalState

from .conftest import call, make_trajectory, result


def score(name: str, traj, task) -> ScoreResult:
    return DETERMINISTIC[name].score(traj, task)


# -- task completion ---------------------------------------------------------


def test_completion_perfect(perfect, task):
    assert score("task_completion", perfect, task).score == 1.0


def test_completion_missing_metric_partially_credits(perfect, task):
    perfect.metadata["metrics"] = {}
    r = score("task_completion", perfect, task)
    assert r.score == pytest.approx(0.7)  # terminal ok, 1 of 2 assertions
    assert "metric_present=roc_auc" in r.reasoning


def test_completion_wrong_terminal_state(gave_up, task):
    r = score("task_completion", gave_up, task)
    assert r.score == 0.0
    assert r.evidence, "a failed completion must cite the step it failed at"


def test_escalation_is_success_for_adversarial_tasks(gave_up):
    task = TaskDef(
        task_id="adv",
        category="adversarial",
        acceptable_terminal_states=[TerminalState.FAILED, TerminalState.ESCALATED],
        success_assertions=[Assertion(tool_not_called="tune")],
    )
    assert score("task_completion", gave_up, task).score == 1.0


# -- tool selection ----------------------------------------------------------


def test_tool_selection_perfect(perfect, task):
    assert score("tool_selection", perfect, task).score == 1.0


def test_tool_selection_penalises_missing_and_extra(task):
    traj = make_trajectory([call("load"), result("load"), call("gossip"), result("gossip")])
    r = score("tool_selection", traj, task)
    assert r.score < 0.5
    assert r.details["missing"] == ["clean", "evaluate", "train"]
    assert r.details["unexpected"] == ["gossip"]
    assert r.evidence, "unexpected tool calls must be cited"


def test_forbidden_tool_halves_the_score(task):
    task.forbidden_tools = ["train"]
    traj = make_trajectory([call(t) for t in task.expected_tools])
    r = score("tool_selection", traj, task)
    assert r.score == pytest.approx(0.5)
    assert "FORBIDDEN" in r.reasoning


# -- step efficiency ---------------------------------------------------------


def test_efficiency_at_optimum(perfect, task):
    assert score("step_efficiency", perfect, task).score == 1.0


def test_efficiency_decays_with_bloat(looping, task):
    r = score("step_efficiency", looping, task)
    assert r.score == 0.0  # 8 calls vs optimal 4 is 2x
    assert r.details["actual"] == 8


def test_efficiency_zero_when_nothing_was_called(task):
    traj = make_trajectory([result("load")])
    assert score("step_efficiency", traj, task).score == 0.0


# -- error recovery ----------------------------------------------------------


def test_no_failures_scores_perfect(perfect, task):
    assert score("error_recovery", perfect, task).score == 1.0


def test_gave_up_scores_zero(gave_up, task):
    r = score("error_recovery", gave_up, task)
    assert r.score == 0.0
    assert r.details["outcomes"] == {"gave_up": 1}
    assert r.evidence == [3]


def test_recovered_scores_full(recovered, task):
    r = score("error_recovery", recovered, task)
    assert r.score == 1.0
    assert r.details["outcomes"] == {"recovered": 1}


def test_looping_on_a_failure_is_worse_than_recovering(task):
    steps = [call("train"), result("train", error="boom")]
    for _ in range(4):
        steps += [call("train"), result("train", error="boom")]
    traj = make_trajectory(steps, terminal_state=TerminalState.FAILED)
    r = score("error_recovery", traj, task)
    assert r.score < 0.5


# -- budget ------------------------------------------------------------------


def test_within_budget(perfect, task):
    assert score("budget_adherence", perfect, task).score == 1.0


def test_over_budget_scores_low_and_explains(over_budget, task):
    r = score("budget_adherence", over_budget, task)
    assert r.score <= 0.5
    assert "over budget" in r.reasoning
    assert r.details["breaches"]


def test_no_declared_budget_is_not_scored(perfect):
    task = TaskDef(task_id="nobudget")
    assert score("budget_adherence", perfect, task).score == 1.0


# -- loop detection ----------------------------------------------------------


def test_loop_detection_flags_repeats(looping, task):
    r = score("loop_detection", looping, task)
    assert r.score < 0.2, "a trajectory that loops 8x must score badly"
    assert r.details["loops"] == {"train": 8}
    assert len(r.evidence) == 8


def test_same_tool_different_input_is_not_a_loop(task):
    steps = []
    for i in range(6):
        steps += [call("train", seed=i), result("train")]
    traj = make_trajectory(steps)
    assert score("loop_detection", traj, task).score == 1.0


def test_clean_run_has_no_loops(perfect, task):
    assert score("loop_detection", perfect, task).score == 1.0


# -- the evidence contract ---------------------------------------------------


def test_every_deduction_cites_evidence(looping, gave_up, over_budget, task):
    """A score below 1.0 with neither evidence nor details is rejected outright."""
    for traj in (looping, gave_up, over_budget):
        for name in DETERMINISTIC:
            r = score(name, traj, task)
            if r.score < 1.0:
                assert r.evidence or r.details, f"{name} deducted without evidence"


def test_score_result_rejects_evidence_free_deduction():
    with pytest.raises(ValueError, match="cites no evidence"):
        ScoreResult(scorer="bogus", score=0.3, reasoning="because I said so")
