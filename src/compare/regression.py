"""Regression detection between two runs.

The hard part is not spotting a drop, it is refusing to shout about one that is
indistinguishable from run-to-run randomness. An eval harness that cries wolf
gets muted within a week, at which point it is worse than nothing — so a delta
must clear both the configured threshold *and* the noise band derived from the
two runs' own variance before it counts.
"""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..aggregate import RunAggregate, Stat
from ..config import settings

# Two standard errors ≈ 95% of run-to-run variation. Below this, a difference is
# noise wearing a regression's clothes.
NOISE_SIGMA = 2.0
# Floor on the noise band: two runs that each happened to be perfectly consistent
# report stdev 0, and without a floor any microscopic delta would look significant.
MIN_NOISE_BAND = 0.01


class DimensionDelta(BaseModel):
    """One dimension, before and after."""

    dimension: str
    baseline: float
    candidate: float
    delta: float
    noise_band: float
    verdict: Literal["regression", "within_noise", "improvement", "stable"]

    @property
    def is_regression(self) -> bool:
        return self.verdict == "regression"


class TaskDelta(BaseModel):
    """One task, before and after."""

    task_id: str
    baseline: float
    candidate: float
    delta: float
    was_passing: bool
    now_passing: bool
    newly_failing: bool
    dimensions: list[DimensionDelta] = Field(default_factory=list)


class DriftSignal(BaseModel):
    """Gradual decay across the last N runs, invisible to any single diff."""

    dimension: str
    first: float
    last: float
    total_drift: float
    n_runs: int


class Comparison(BaseModel):
    """The full verdict on a candidate run."""

    baseline_run_id: str
    candidate_run_id: str
    overall: DimensionDelta
    dimensions: list[DimensionDelta]
    tasks: list[TaskDelta]
    drift: list[DriftSignal] = Field(default_factory=list)
    unstable_dimensions: list[str] = Field(default_factory=list)

    @property
    def regressed_dimensions(self) -> list[DimensionDelta]:
        return [d for d in self.dimensions if d.is_regression]

    @property
    def newly_failing_tasks(self) -> list[TaskDelta]:
        return [t for t in self.tasks if t.newly_failing]

    @property
    def has_hard_regression(self) -> bool:
        """What makes the CI check red."""
        return bool(
            self.regressed_dimensions
            or self.newly_failing_tasks
            or self.overall.is_regression
        )


def noise_band(baseline: Stat, candidate: Stat) -> float:
    """Half-width of the band inside which a delta means nothing.

    Standard error of the difference of two means, at NOISE_SIGMA sigma.
    """
    def se(stat: Stat) -> float:
        return stat.stdev / math.sqrt(stat.n) if stat.n > 1 else stat.stdev

    combined = math.sqrt(se(baseline) ** 2 + se(candidate) ** 2)
    return max(NOISE_SIGMA * combined, MIN_NOISE_BAND)


def _classify(
    dimension: str, baseline: Stat, candidate: Stat, threshold: float
) -> DimensionDelta:
    # Rounded before comparison: means are reported to 4dp, and an unrounded
    # 0.030000000000000027 must not read as "over the 0.03 threshold" when the
    # printed report says the delta is exactly 0.03.
    delta = round(candidate.mean - baseline.mean, 4)
    band = noise_band(baseline, candidate)

    if abs(delta) <= band:
        verdict = "within_noise"
    elif delta > 0:
        verdict = "improvement" if delta > threshold else "stable"
    elif -delta > threshold:
        verdict = "regression"
    else:
        verdict = "stable"

    return DimensionDelta(
        dimension=dimension,
        baseline=round(baseline.mean, 4),
        candidate=round(candidate.mean, 4),
        delta=round(delta, 4),
        noise_band=round(band, 4),
        verdict=verdict,
    )


def compare_runs(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    history: list[dict[str, Any]] | None = None,
) -> Comparison:
    """Compare two run records, optionally checking history for slow drift."""
    base_agg = RunAggregate.model_validate(baseline["aggregate"])
    cand_agg = RunAggregate.model_validate(candidate["aggregate"])

    overall = _classify("overall", base_agg.overall, cand_agg.overall, settings.overall_threshold)

    dimensions = [
        _classify(name, base_agg.dimensions[name], stat, settings.regression_threshold)
        for name, stat in sorted(cand_agg.dimensions.items())
        if name in base_agg.dimensions
    ]

    tasks: list[TaskDelta] = []
    for task_id, cand_task in sorted(cand_agg.tasks.items()):
        base_task = base_agg.tasks.get(task_id)
        if base_task is None:
            continue  # a new task has nothing to regress against
        per_dim = [
            _classify(name, base_task.dimensions[name], stat, settings.regression_threshold)
            for name, stat in sorted(cand_task.dimensions.items())
            if name in base_task.dimensions
        ]
        tasks.append(
            TaskDelta(
                task_id=task_id,
                baseline=base_task.overall.mean,
                candidate=cand_task.overall.mean,
                delta=round(cand_task.overall.mean - base_task.overall.mean, 4),
                was_passing=base_task.passed,
                now_passing=cand_task.passed,
                newly_failing=base_task.passed and not cand_task.passed,
                dimensions=per_dim,
            )
        )

    return Comparison(
        baseline_run_id=baseline["run_id"],
        candidate_run_id=candidate["run_id"],
        overall=overall,
        dimensions=dimensions,
        tasks=tasks,
        drift=detect_drift((history or []) + [candidate]),
        unstable_dimensions=cand_agg.unstable_dimensions,
    )


def detect_drift(runs: list[dict[str, Any]]) -> list[DriftSignal]:
    """Cumulative decay over the last N runs.

    Catches the case each individual diff is designed to miss: a dimension that
    loses two points per release and a tenth of its value over a quarter, without
    any single comparison ever tripping the threshold.
    """
    window = sorted(runs, key=lambda r: r.get("created_at", ""))[-settings.drift_window :]
    if len(window) < 3:
        return []

    aggs = [RunAggregate.model_validate(r["aggregate"]) for r in window]
    signals: list[DriftSignal] = []
    for dimension in aggs[-1].dimensions:
        series = [a.dimensions[dimension].mean for a in aggs if dimension in a.dimensions]
        if len(series) < 3:
            continue
        drift = series[-1] - series[0]
        if -drift >= settings.drift_threshold:
            signals.append(
                DriftSignal(
                    dimension=dimension,
                    first=round(series[0], 4),
                    last=round(series[-1], 4),
                    total_drift=round(drift, 4),
                    n_runs=len(series),
                )
            )
    return signals


def format_report(cmp: Comparison) -> str:
    """Plain-text summary, used by the CLI and the GitHub Action comment."""
    lines = [
        f"Comparison: {cmp.baseline_run_id} (baseline) -> {cmp.candidate_run_id}",
        f"overall {cmp.overall.baseline:.3f} -> {cmp.overall.candidate:.3f} "
        f"({cmp.overall.delta:+.3f}, noise band ±{cmp.overall.noise_band:.3f}) "
        f"[{cmp.overall.verdict}]",
        "",
        "Dimensions:",
    ]
    for d in cmp.dimensions:
        marker = {"regression": "FAIL", "improvement": "UP", "within_noise": "noise"}.get(
            d.verdict, "ok"
        )
        lines.append(
            f"  {d.dimension:<24} {d.baseline:.3f} -> {d.candidate:.3f} "
            f"({d.delta:+.3f} ±{d.noise_band:.3f})  {marker}"
        )

    if cmp.newly_failing_tasks:
        lines += ["", "Newly failing tasks:"]
        lines += [
            f"  {t.task_id}: {t.baseline:.3f} -> {t.candidate:.3f} ({t.delta:+.3f})"
            for t in cmp.newly_failing_tasks
        ]

    regressed = [t for t in cmp.tasks if t.delta < -settings.regression_threshold]
    if regressed:
        lines += ["", "Tasks with the largest drops:"]
        for t in sorted(regressed, key=lambda x: x.delta)[:10]:
            worst = min(t.dimensions, key=lambda d: d.delta, default=None)
            detail = f" (worst: {worst.dimension} {worst.delta:+.3f})" if worst else ""
            lines.append(f"  {t.task_id}: {t.delta:+.3f}{detail}")

    if cmp.drift:
        lines += ["", "Slow drift over recent runs:"]
        lines += [
            f"  {d.dimension}: {d.first:.3f} -> {d.last:.3f} ({d.total_drift:+.3f} over {d.n_runs} runs)"
            for d in cmp.drift
        ]

    if cmp.unstable_dimensions:
        lines += [
            "",
            "High-variance dimensions (wide spread across repeats — the noise "
            "band already accounts for it, but the mean is a weak summary): "
            + ", ".join(cmp.unstable_dimensions),
        ]

    lines += ["", "HARD REGRESSION" if cmp.has_hard_regression else "No hard regression."]
    return "\n".join(lines)
