"""The regression detector's two jobs: flag real drops, ignore noise."""

from __future__ import annotations

from src.aggregate import Stat, TaskRun, aggregate
from src.compare.regression import compare_runs, detect_drift, noise_band
from src.scorers.base import ScoreResult

DIMENSIONS = ["task_completion", "tool_selection", "step_efficiency"]


def _run(run_id: str, scores_by_repeat: list[dict[str, float]], created_at: str = "2026-01-01") -> dict:
    """Build a run record from raw per-repeat dimension scores."""
    task_runs = [
        TaskRun(
            task_id="t1",
            repeat=i,
            trajectory_id=f"{run_id}-{i}",
            scores={
                name: ScoreResult(
                    scorer=name, score=value, reasoning="fixture", details={"fixture": True}
                )
                for name, value in scores.items()
            },
        )
        for i, scores in enumerate(scores_by_repeat)
    ]
    agg = aggregate(task_runs)
    return {
        "run_id": run_id,
        "suite": "test",
        "agent_version": "v",
        "created_at": created_at,
        "task_runs": [tr.model_dump(mode="json") for tr in task_runs],
        "aggregate": agg.model_dump(mode="json"),
    }


def _spread(base: float, jitter: float, n: int = 5) -> list[dict[str, float]]:
    """n repeats jittering around `base` by ±jitter."""
    offsets = [-jitter, -jitter / 2, 0.0, jitter / 2, jitter][:n]
    return [{d: min(1.0, max(0.0, base + o)) for d in DIMENSIONS} for o in offsets]


# -- the thing that must not happen ------------------------------------------


def test_within_noise_difference_is_not_flagged():
    """A noisy agent scoring 0.02 lower is not a regression. This is the whole point."""
    baseline = _run("base", _spread(0.80, 0.10))
    candidate = _run("cand", _spread(0.78, 0.10))
    result = compare_runs(baseline, candidate)

    assert not result.has_hard_regression
    assert all(d.verdict == "within_noise" for d in result.dimensions)


def test_identical_runs_are_not_flagged():
    scores = _spread(0.9, 0.05)
    result = compare_runs(_run("a", scores), _run("b", scores))
    assert not result.has_hard_regression
    assert result.overall.delta == 0.0


# -- the thing that must happen ----------------------------------------------


def test_clear_regression_is_flagged():
    baseline = _run("base", _spread(0.90, 0.02))
    candidate = _run("cand", _spread(0.55, 0.02))
    result = compare_runs(baseline, candidate)

    assert result.has_hard_regression
    assert {d.dimension for d in result.regressed_dimensions} == set(DIMENSIONS)
    assert result.overall.delta < -0.3


def test_regression_names_the_affected_task():
    baseline = _run("base", _spread(0.95, 0.01))
    candidate = _run("cand", _spread(0.30, 0.01))
    result = compare_runs(baseline, candidate)

    failing = result.newly_failing_tasks
    assert [t.task_id for t in failing] == ["t1"]
    assert failing[0].dimensions, "a failing task must break down by dimension"


def test_a_large_drop_survives_a_wide_noise_band():
    """Genuine collapse must still fire even when the agent is very noisy."""
    baseline = _run("base", _spread(0.90, 0.20))
    candidate = _run("cand", _spread(0.20, 0.20))
    assert compare_runs(baseline, candidate).has_hard_regression


def test_improvement_is_not_a_regression():
    result = compare_runs(_run("base", _spread(0.50, 0.02)), _run("cand", _spread(0.95, 0.02)))
    assert not result.has_hard_regression
    assert all(d.verdict == "improvement" for d in result.dimensions)


# -- noise band --------------------------------------------------------------


def test_noise_band_widens_with_variance():
    tight = noise_band(Stat.of([0.9] * 5), Stat.of([0.9] * 5))
    wide = noise_band(Stat.of([0.5, 0.9, 0.3, 0.8, 0.6]), Stat.of([0.5, 0.9, 0.3, 0.8, 0.6]))
    assert wide > tight


def test_noise_band_has_a_floor():
    """Zero observed variance must not make every microscopic delta significant."""
    assert noise_band(Stat.of([0.9]), Stat.of([0.9])) >= 0.01


# -- slow drift --------------------------------------------------------------


def test_slow_drift_is_caught_even_though_no_single_step_trips_the_threshold():
    runs = [
        _run(f"r{i}", _spread(level, 0.01), created_at=f"2026-01-0{i + 1}")
        for i, level in enumerate([0.95, 0.925, 0.90, 0.875, 0.85])
    ]
    # No adjacent pair differs by more than the 0.05 threshold...
    for a, b in zip(runs, runs[1:]):
        assert not compare_runs(a, b).has_hard_regression
    # ...but end to end the decay is real.
    drift = detect_drift(runs)
    assert [d.dimension for d in drift] == sorted(DIMENSIONS)
    assert all(d.total_drift <= -0.08 for d in drift)


def test_no_drift_reported_for_a_stable_series():
    runs = [_run(f"r{i}", _spread(0.9, 0.01), created_at=f"2026-01-0{i + 1}") for i in range(5)]
    assert detect_drift(runs) == []


def test_drift_needs_at_least_three_runs():
    runs = [_run("a", _spread(0.9, 0.0)), _run("b", _spread(0.1, 0.0))]
    assert detect_drift(runs) == []
