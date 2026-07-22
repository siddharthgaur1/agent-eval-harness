"""The framework-agnostic trajectory schema.

This is the contract. Adapters write it, scorers read it, and nothing downstream
of here knows or cares whether the agent was LangGraph, CrewAI, or a for-loop
someone wrote on a Friday. Keeping the schema narrow is what makes the harness
portable; every field below has to be recoverable from any agent framework.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


def _now() -> datetime:
    return datetime.now(timezone.utc)


class StepType(str, Enum):
    """What kind of thing happened at this step."""

    PLAN = "plan"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    LLM_MESSAGE = "llm_message"
    RETRY = "retry"
    ESCALATION = "escalation"


class TerminalState(str, Enum):
    """How the run ended. `escalated` is a success for some adversarial tasks."""

    COMPLETED = "completed"
    FAILED = "failed"
    ESCALATED = "escalated"
    TIMEOUT = "timeout"
    BUDGET_EXCEEDED = "budget_exceeded"


class Step(BaseModel):
    """One observable action in the agent's run."""

    index: int
    agent_name: str = ""
    step_type: StepType

    tool_name: str | None = None
    tool_input: dict[str, Any] = Field(default_factory=dict)
    tool_output: Any = None
    error: str | None = None

    tokens: int = 0
    latency_ms: int = 0
    timestamp: datetime = Field(default_factory=_now)

    @property
    def failed(self) -> bool:
        """True when this step recorded an error, whatever its type."""
        return self.error is not None

    def call_signature(self) -> str:
        """Stable identity for loop detection: same tool, same input."""
        import json

        payload = json.dumps(self.tool_input, sort_keys=True, default=str)
        return f"{self.tool_name}:{payload}"


class Trajectory(BaseModel):
    """A complete recorded agent run."""

    run_id: str
    task_id: str
    agent_version: str
    model: str = ""
    started_at: datetime = Field(default_factory=_now)
    ended_at: datetime | None = None

    steps: list[Step] = Field(default_factory=list)
    final_output: str = ""
    terminal_state: TerminalState = TerminalState.FAILED

    total_tokens: int = 0
    total_cost: float = 0.0
    wall_clock_seconds: float = 0.0

    # Free-form: artifacts produced, metrics emitted. Success assertions read it.
    metadata: dict[str, Any] = Field(default_factory=dict)

    def tools_used(self) -> list[str]:
        """Tool names in call order, duplicates kept."""
        return [s.tool_name for s in self.steps if s.tool_name]

    def failed_steps(self) -> list[Step]:
        """Every step that recorded an error."""
        return [s for s in self.steps if s.failed]

    def finalize(self) -> "Trajectory":
        """Backfill the totals that adapters can derive rather than track."""
        if self.ended_at is None:
            self.ended_at = _now()
        if not self.total_tokens:
            self.total_tokens = sum(s.tokens for s in self.steps)
        if not self.wall_clock_seconds:
            self.wall_clock_seconds = (self.ended_at - self.started_at).total_seconds()
        return self
