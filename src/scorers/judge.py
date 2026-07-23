"""LLM-judge scorers.

Three dimensions that cannot be computed from the trajectory alone: whether the
plan made sense, whether the sequence of decisions was reasonable, and whether
the final answer is actually supported by what the tools returned.

Every call is temperature 0, uses structured output, retries with backoff, and
persists the raw exchange to the store. An LLM score nobody can go back and read
the judge's own words for is not evidence, it is a rumour.
"""

from __future__ import annotations

import json
import time
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from ..config import settings
from ..persistence.store import Store
from ..suites.schema import TaskDef
from ..trace.schema import StepType, Trajectory
from .base import ScoreResult, register

MAX_STEPS_IN_PROMPT = 60


class JudgeVerdict(BaseModel):
    """The shape every judge must return."""

    score: float = Field(ge=0.0, le=1.0)
    reasoning: str
    evidence_steps: list[int] = Field(default_factory=list)


class JudgeError(RuntimeError):
    """Raised when the judge could not be reached or parsed after all retries."""


def _client():
    """Lazily construct the OpenAI client so imports stay cheap and key-free."""
    from openai import OpenAI

    # Explicit timeout: the SDK default is 10 minutes, long enough for one wedged
    # judge call to stall a whole suite run behind it.
    return OpenAI(api_key=settings.require_openai(), timeout=60.0, max_retries=0)


def call_judge(
    system_prompt: str,
    user_prompt: str,
    *,
    scorer: str,
    trajectory_id: str,
    model: str | None = None,
    store: Store | None = None,
    client: Any = None,
) -> JudgeVerdict:
    """Run one structured judge call with retries, persisting the raw response.

    `client` is injectable so tests can stub the judge without patching imports.
    """
    model = model or settings.judge_model
    client = client or _client()
    last_error: Exception | None = None

    for attempt in range(settings.judge_max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            raw = response.choices[0].message.content or ""
            if store is not None:
                store.save_judge_call(trajectory_id, scorer, model, user_prompt, raw)
            return JudgeVerdict.model_validate(json.loads(raw))
        except (json.JSONDecodeError, ValidationError, KeyError, IndexError) as exc:
            last_error = exc  # bad output: retrying may well fix it
        except Exception as exc:  # transport / rate limit
            last_error = exc
        time.sleep(2**attempt)

    raise JudgeError(
        f"{scorer}: judge failed after {settings.judge_max_retries} attempts: {last_error}"
    )


def render_steps(traj: Trajectory, limit: int = MAX_STEPS_IN_PROMPT) -> str:
    """Compact step log for the prompt.

    Long runs are truncated in the middle rather than the end: the last steps are
    where a run goes wrong, so they are the ones the judge most needs to see.
    """
    steps = traj.steps
    if len(steps) > limit:
        head, tail = steps[: limit // 2], steps[-(limit // 2) :]
        omitted = len(steps) - len(head) - len(tail)
        rendered = (
            [_render_step(s) for s in head]
            + [f"... [{omitted} steps omitted] ..."]
            + [_render_step(s) for s in tail]
        )
    else:
        rendered = [_render_step(s) for s in steps]
    return "\n".join(rendered)


def _render_step(step) -> str:
    bits = [f"[{step.index}] {step.step_type.value}"]
    if step.agent_name:
        bits.append(f"agent={step.agent_name}")
    if step.tool_name:
        bits.append(f"tool={step.tool_name}")
    if step.tool_input:
        bits.append(f"input={json.dumps(step.tool_input, default=str)[:300]}")
    if step.error:
        bits.append(f"ERROR={step.error[:300]}")
    elif step.tool_output is not None:
        bits.append(f"output={str(step.tool_output)[:300]}")
    return " | ".join(bits)


class _BaseJudge:
    """Shared plumbing: build a prompt, call the judge, wrap the verdict."""

    name: str
    system_prompt: str
    model_setting = "judge"

    def build_user_prompt(self, traj: Trajectory, task: TaskDef) -> str:
        raise NotImplementedError

    def score(
        self, traj: Trajectory, task: TaskDef, store: Store | None = None, client: Any = None
    ) -> ScoreResult:
        model = settings.judge_model if self.model_setting == "judge" else settings.cheap_model
        verdict = call_judge(
            self.system_prompt,
            self.build_user_prompt(traj, task),
            scorer=self.name,
            trajectory_id=traj.run_id,
            model=model,
            store=store,
            client=client,
        )
        valid = {s.index for s in traj.steps}
        evidence = sorted(i for i in verdict.evidence_steps if i in valid)
        return ScoreResult(
            scorer=self.name,
            score=verdict.score,
            reasoning=verdict.reasoning,
            evidence=evidence,
            details={"model": model, "judge_cited": verdict.evidence_steps},
        )


_JSON_CONTRACT = (
    'Respond with JSON only: {"score": <float 0-1>, "reasoning": "<2-4 sentences>", '
    '"evidence_steps": [<step indices you based this on>]}. '
    "evidence_steps must not be empty unless the score is 1.0."
)


@register("llm")
class PlanCoherence(_BaseJudge):
    """Was the plan sensible for the goal, and did execution follow it?"""

    name = "plan_coherence"
    system_prompt = (
        "You evaluate AI agent planning. Given a goal and an execution log, judge "
        "whether the agent's plan was a sensible route to the goal AND whether the "
        "steps it actually executed followed that plan. A good plan abandoned "
        "mid-run scores low; an improvised route that adapted sensibly to real "
        "obstacles scores high. Ignore output quality — judge the plan only. "
        + _JSON_CONTRACT
    )

    def build_user_prompt(self, traj: Trajectory, task: TaskDef) -> str:
        plans = [s for s in traj.steps if s.step_type is StepType.PLAN]
        plan_text = "\n".join(_render_step(s) for s in plans) or "(no explicit plan steps)"
        return (
            f"GOAL: {task.goal}\n\n"
            f"STATED PLAN STEPS:\n{plan_text}\n\n"
            f"FULL EXECUTION LOG:\n{render_steps(traj)}\n\n"
            f"TERMINAL STATE: {traj.terminal_state.value}"
        )


@register("llm")
class TrajectoryReasoning(_BaseJudge):
    """Judge the sequence of decisions, not the final answer."""

    name = "trajectory_reasoning"
    system_prompt = (
        "You evaluate the quality of an AI agent's decision-making over a whole "
        "run. Judge the SEQUENCE: was each step a reasonable thing to do given "
        "what the agent knew at that point? Penalise steps that ignore what a "
        "previous step returned, repeated work, and unjustified jumps. Do not "
        "reward a lucky correct answer reached by poor reasoning. " + _JSON_CONTRACT
    )

    def build_user_prompt(self, traj: Trajectory, task: TaskDef) -> str:
        return (
            f"GOAL: {task.goal}\n\n"
            f"EXECUTION LOG:\n{render_steps(traj)}\n\n"
            f"TERMINAL STATE: {traj.terminal_state.value}\n"
            f"FINAL OUTPUT: {traj.final_output[:2000]}"
        )


@register("llm")
class OutputFaithfulness(_BaseJudge):
    """Is the final output supported by what the tools actually returned?

    The one that catches an agent confidently reporting a conclusion its own
    tools never produced — the failure mode single-turn output eval is blind to,
    because the answer reads perfectly well on its own.
    """

    name = "output_faithfulness"
    system_prompt = (
        "You check whether an AI agent's final answer is supported by evidence "
        "its own tools returned during the run. For each factual claim in the "
        "final output — numbers especially — find the tool result that supports "
        "it. Claims with no supporting tool output are hallucinations and must "
        "drive the score down hard, even if they sound plausible or are likely "
        "true. Cite the steps you checked against. " + _JSON_CONTRACT
    )

    def build_user_prompt(self, traj: Trajectory, task: TaskDef) -> str:
        observations = [
            _render_step(s)
            for s in traj.steps
            if s.step_type is StepType.TOOL_RESULT and s.tool_output is not None
        ]
        obs_text = "\n".join(observations) or "(the agent's tools returned nothing)"
        return (
            f"GOAL: {task.goal}\n\n"
            f"TOOL RESULTS THE AGENT OBSERVED:\n{obs_text}\n\n"
            f"FINAL OUTPUT TO VERIFY:\n{traj.final_output[:4000]}"
        )
