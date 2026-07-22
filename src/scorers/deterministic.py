"""The six deterministic scorers.

No LLM calls: these are fast, free, and give identical answers on identical
input, which is what makes the regression detector's noise band meaningful. If a
dimension can be computed from the trajectory alone, it belongs here rather than
in a judge.
"""

from __future__ import annotations

from collections import Counter

from ..config import settings
from ..suites.schema import TaskDef
from ..trace.schema import StepType, TerminalState, Trajectory
from .base import ScoreResult, register


@register()
class TaskCompletion:
    """Did the run end acceptably and satisfy its declared assertions?

    Split 40/60 between reaching an acceptable terminal state and passing the
    assertions, because an agent that stops cleanly but produces nothing is not
    half-right — it is mostly wrong, and the assertions are what say so.
    """

    name = "task_completion"

    def score(self, traj: Trajectory, task: TaskDef) -> ScoreResult:
        terminal_ok = traj.terminal_state in task.acceptable_terminal_states
        results = [(a, a.check(traj)) for a in task.success_assertions]
        passed = [a for a, ok in results if ok]
        failed = [a for a, ok in results if not ok]

        assertion_score = len(passed) / len(results) if results else float(terminal_ok)
        score = 0.4 * float(terminal_ok) + 0.6 * assertion_score

        evidence: list[int] = []
        if not terminal_ok and traj.steps:
            evidence.append(traj.steps[-1].index)
        if failed:
            evidence.extend(s.index for s in traj.failed_steps())
            if not evidence and traj.steps:
                evidence.append(traj.steps[-1].index)

        reason = (
            f"terminal_state={traj.terminal_state.value} "
            f"({'acceptable' if terminal_ok else 'not acceptable'}); "
            f"{len(passed)}/{len(results)} assertions passed"
        )
        if failed:
            reason += f"; failed: {', '.join(a.label for a in failed)}"

        return ScoreResult(
            scorer=self.name,
            score=round(score, 4),
            reasoning=reason,
            evidence=sorted(set(evidence)),
            details={
                "terminal_ok": terminal_ok,
                "failed_assertions": [a.label for a in failed],
            },
        )


@register()
class ToolSelection:
    """Precision and recall of the tools called against the expected set.

    Scored as F1 so an agent cannot game it either by calling everything
    (recall 1, precision floor) or by calling one correct tool and stopping.
    Forbidden tools are a hard multiplicative penalty, not a subtracted point:
    calling a tool the task explicitly banned is a different class of error.
    """

    name = "tool_selection"

    def score(self, traj: Trajectory, task: TaskDef) -> ScoreResult:
        if not task.expected_tools:
            return ScoreResult(
                scorer=self.name,
                score=1.0,
                reasoning="task declares no expected tools; nothing to check",
            )

        used_order = traj.tools_used()
        used = set(used_order)
        expected = set(task.expected_tools)

        hits = used & expected
        missing = expected - used
        extra = used - expected

        precision = len(hits) / len(used) if used else 0.0
        recall = len(hits) / len(expected)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        forbidden_used = used & set(task.forbidden_tools)
        if forbidden_used:
            f1 *= 0.5

        evidence = [
            s.index
            for s in traj.steps
            if s.tool_name and (s.tool_name in extra or s.tool_name in forbidden_used)
        ]
        if missing and traj.steps and not evidence:
            evidence.append(traj.steps[-1].index)

        reason = (
            f"precision={precision:.2f} recall={recall:.2f} f1={f1:.2f}; "
            f"missing={sorted(missing) or 'none'}; unexpected={sorted(extra) or 'none'}"
        )
        if forbidden_used:
            reason += f"; FORBIDDEN tools called: {sorted(forbidden_used)} (score halved)"

        return ScoreResult(
            scorer=self.name,
            score=round(f1, 4),
            reasoning=reason,
            evidence=sorted(set(evidence)),
            details={
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "missing": sorted(missing),
                "unexpected": sorted(extra),
                "forbidden": sorted(forbidden_used),
            },
        )


@register()
class StepEfficiency:
    """Actual steps against the task's optimal count.

    Only tool calls count. Ratios below 1.0 are not rewarded — finishing in
    fewer steps than the optimum usually means skipping work, and the completion
    scorer is what should notice that, not this one.
    """

    name = "step_efficiency"

    def score(self, traj: Trajectory, task: TaskDef) -> ScoreResult:
        calls = [s for s in traj.steps if s.step_type is StepType.TOOL_CALL]
        actual = len(calls)
        optimal = task.optimal_steps

        if actual == 0:
            return ScoreResult(
                scorer=self.name,
                score=0.0,
                reasoning="agent made no tool calls at all",
                evidence=[s.index for s in traj.steps[:1]],
                details={"actual": 0, "optimal": optimal},
            )

        ratio = actual / optimal
        # Linear decay: at 2x the optimal step count the score is 0.
        score = 1.0 if ratio <= 1.0 else max(0.0, 2.0 - ratio)

        redundant = _redundant_calls(calls)
        evidence = [s.index for s in calls[optimal:]] if actual > optimal else []
        evidence += [s.index for s in redundant]

        return ScoreResult(
            scorer=self.name,
            score=round(score, 4),
            reasoning=(
                f"{actual} tool calls vs optimal {optimal} (ratio {ratio:.2f}); "
                f"{len(redundant)} redundant repeat(s)"
            ),
            evidence=sorted(set(evidence)),
            details={"actual": actual, "optimal": optimal, "ratio": round(ratio, 3)},
        )


def _redundant_calls(calls: list) -> list:
    """Every call after the first with an identical tool+input signature."""
    seen: set[str] = set()
    out = []
    for step in calls:
        sig = step.call_signature()
        if sig in seen:
            out.append(step)
        seen.add(sig)
    return out


@register()
class ErrorRecovery:
    """For every failure, did the agent recover — and how expensively?

    Classifies each failed step as recovered, gave_up, or looped. A trajectory
    with no failures scores 1.0; this measures resilience, not reliability, and
    penalising an agent for never failing would be backwards.
    """

    name = "error_recovery"

    def score(self, traj: Trajectory, task: TaskDef) -> ScoreResult:
        failures = traj.failed_steps()
        if not failures:
            return ScoreResult(
                scorer=self.name,
                score=1.0,
                reasoning="no failed steps to recover from",
            )

        outcomes: list[str] = []
        for failure in failures:
            outcomes.append(self._classify(traj, failure))

        weights = {"recovered": 1.0, "recovered_slowly": 0.6, "looped": 0.2, "gave_up": 0.0}
        score = sum(weights[o] for o in outcomes) / len(outcomes)
        counts = Counter(outcomes)

        return ScoreResult(
            scorer=self.name,
            score=round(score, 4),
            reasoning=(
                f"{len(failures)} failed step(s): "
                + ", ".join(f"{n} {k}" for k, n in counts.items())
            ),
            evidence=[f.index for f in failures],
            details={"outcomes": dict(counts)},
        )

    def _classify(self, traj: Trajectory, failure) -> str:
        """Look at what happened after a failure, on the same tool."""
        after = traj.steps[failure.index + 1 :]
        if not after:
            return "gave_up"

        # Only results carry an outcome. Counting the tool_call half of a
        # retry pair as "didn't fail" would score an agent that retried the
        # same broken call forever as having recovered every single time.
        same_tool = [
            s
            for s in after
            if s.tool_name == failure.tool_name and s.step_type is StepType.TOOL_RESULT
        ]
        attempts = 0
        for step in same_tool:
            attempts += 1
            if not step.failed:
                return "recovered" if attempts <= 2 else "recovered_slowly"

        if attempts >= settings.loop_repeat_limit:
            return "looped"
        # It moved on to other work and finished: that counts as routing around
        # the failure, which is a recovery, just not of the same tool.
        if traj.terminal_state in {TerminalState.COMPLETED, TerminalState.ESCALATED}:
            return "recovered_slowly"
        return "gave_up"


@register()
class BudgetAdherence:
    """Tokens, cost and wall clock against the task's declared caps.

    Each declared axis is scored independently and averaged. An axis with no cap
    is skipped rather than scored 1.0, so adding a cap to a task can only make
    its score more informative, never inflate it.
    """

    name = "budget_adherence"

    def score(self, traj: Trajectory, task: TaskDef) -> ScoreResult:
        axes = [
            ("tokens", traj.total_tokens, task.budget.max_tokens),
            ("cost_usd", traj.total_cost, task.budget.max_cost_usd),
            ("seconds", traj.wall_clock_seconds, task.budget.max_seconds),
        ]
        declared = [(n, actual, cap) for n, actual, cap in axes if cap]

        if not declared:
            return ScoreResult(
                scorer=self.name, score=1.0, reasoning="task declares no budget caps"
            )

        parts, breaches = [], []
        for name, actual, cap in declared:
            usage = actual / cap
            # Under budget is a pass. Over budget decays to 0 at 2x the cap.
            parts.append(1.0 if usage <= 1.0 else max(0.0, 2.0 - usage))
            if usage > 1.0:
                breaches.append(f"{name} {actual:.2f}/{cap} ({usage:.0%})")

        score = sum(parts) / len(parts)
        if traj.terminal_state is TerminalState.BUDGET_EXCEEDED:
            breaches.append("run terminated as budget_exceeded")
            score = min(score, 0.5)

        evidence = [traj.steps[-1].index] if breaches and traj.steps else []
        return ScoreResult(
            scorer=self.name,
            score=round(score, 4),
            reasoning=(
                "within all declared budgets"
                if not breaches
                else "over budget: " + "; ".join(breaches)
            ),
            evidence=evidence,
            details={
                "usage": {n: round(a / c, 3) for n, a, c in declared},
                "breaches": breaches,
            },
        )


@register()
class LoopDetection:
    """Flags the same tool called with the same input more than N times.

    Distinct from step efficiency: an agent can be inefficient without looping
    (too many *different* calls), and can loop while still landing near the
    optimal step count. They fail differently, so they score separately.
    """

    name = "loop_detection"

    def score(self, traj: Trajectory, task: TaskDef) -> ScoreResult:
        limit = settings.loop_repeat_limit
        calls = [s for s in traj.steps if s.step_type is StepType.TOOL_CALL and s.tool_name]

        by_sig: dict[str, list[int]] = {}
        for step in calls:
            by_sig.setdefault(step.call_signature(), []).append(step.index)

        loops = {sig: idxs for sig, idxs in by_sig.items() if len(idxs) >= limit}
        if not loops:
            return ScoreResult(
                scorer=self.name,
                score=1.0,
                reasoning=f"no tool call repeated {limit}+ times with identical input",
            )

        # Score by how much of the run was spent looping, not by loop count: one
        # 40-step loop is worse than four 3-step ones.
        looped_steps = sum(len(idxs) for idxs in loops.values())
        share = looped_steps / max(len(calls), 1)
        score = max(0.0, 1.0 - share * 1.5)

        names = sorted({sig.split(":", 1)[0] for sig in loops})
        return ScoreResult(
            scorer=self.name,
            score=round(score, 4),
            reasoning=(
                f"{len(loops)} repeated call signature(s) on {names}; "
                f"{looped_steps}/{len(calls)} calls were repeats"
            ),
            evidence=sorted(i for idxs in loops.values() for i in idxs),
            details={"loops": {s.split(":", 1)[0]: len(i) for s, i in loops.items()}},
        )
