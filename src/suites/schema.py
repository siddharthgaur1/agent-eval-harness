"""Golden task definitions.

Tasks live in YAML so the suite is editable by anyone with an opinion about what
the agent should do, without touching Python.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from ..trace.schema import TerminalState, Trajectory


class Budget(BaseModel):
    """Declared caps. `None` means "no cap on this axis"."""

    max_tokens: int | None = None
    max_cost_usd: float | None = None
    max_seconds: float | None = None


class Assertion(BaseModel):
    """One checkable claim about a finished run.

    Exactly one field is set. Kept as a single model rather than a union because
    the YAML is nicer to write (`- metric_present: roc_auc`) and the check is a
    three-line dispatch either way.
    """

    artifact_exists: str | None = None
    metric_present: str | None = None
    terminal_state: TerminalState | None = None
    output_contains: str | None = None
    tool_called: str | None = None
    tool_not_called: str | None = None

    @model_validator(mode="after")
    def _exactly_one(self) -> "Assertion":
        set_fields = [k for k, v in self.model_dump().items() if v is not None]
        if len(set_fields) != 1:
            raise ValueError(
                f"an assertion must set exactly one field, got {set_fields or 'none'}"
            )
        return self

    @property
    def label(self) -> str:
        """Human-readable form, used in scorer reasoning."""
        k, v = next((k, v) for k, v in self.model_dump().items() if v is not None)
        return f"{k}={getattr(v, 'value', v)}"

    def check(self, traj: Trajectory) -> bool:
        """Evaluate against a finished trajectory."""
        if self.terminal_state is not None:
            return traj.terminal_state is self.terminal_state
        if self.artifact_exists is not None:
            artifacts = traj.metadata.get("artifacts") or []
            return any(self.artifact_exists in str(a) for a in artifacts)
        if self.metric_present is not None:
            metrics = traj.metadata.get("metrics") or {}
            return self.metric_present in metrics
        if self.output_contains is not None:
            return self.output_contains.lower() in traj.final_output.lower()
        if self.tool_called is not None:
            return self.tool_called in traj.tools_used()
        if self.tool_not_called is not None:
            return self.tool_not_called not in traj.tools_used()
        return False


class TaskDef(BaseModel):
    """One golden task."""

    task_id: str
    description: str = ""
    category: Literal["happy_path", "adversarial", "budget_stress"] = "happy_path"
    input: dict[str, Any] = Field(default_factory=dict)

    expected_tools: list[str] = Field(default_factory=list)
    forbidden_tools: list[str] = Field(default_factory=list)
    optimal_steps: int = 10
    budget: Budget = Field(default_factory=Budget)
    success_assertions: list[Assertion] = Field(default_factory=list)

    # For adversarial tasks the right answer is often a clean stop, not a result.
    acceptable_terminal_states: list[TerminalState] = Field(
        default_factory=lambda: [TerminalState.COMPLETED]
    )
    weight: float = 1.0
    goal: str = ""

    @model_validator(mode="after")
    def _defaults(self) -> "TaskDef":
        if not self.goal:
            self.goal = self.description
        if self.optimal_steps < 1:
            raise ValueError("optimal_steps must be >= 1")
        return self


class Suite(BaseModel):
    """A named collection of tasks."""

    name: str
    tasks: list[TaskDef]

    def get(self, task_id: str) -> TaskDef | None:
        return next((t for t in self.tasks if t.task_id == task_id), None)
