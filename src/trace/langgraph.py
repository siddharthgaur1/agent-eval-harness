"""LangGraph / LangChain adapter.

Two entry points, because LangGraph runs give up their information in two
different places:

* `TrajectoryCallbackHandler` — live capture of tool calls, LLM messages, token
  usage and errors while the graph executes.
* `trajectory_from_state` — post-hoc reconstruction from a finished run's state
  dict, for agents (like `autonomous-data-scientist`) that already keep an
  append-only message and artifact log.

Neither imports langchain at module scope, so the harness installs without it.
"""

from __future__ import annotations

import time
from typing import Any
from uuid import UUID

from .recorder import Recorder
from .schema import StepType, TerminalState, Trajectory

try:  # pragma: no cover - exercised only when langchain is installed
    from langchain_core.callbacks import BaseCallbackHandler
except ImportError:  # pragma: no cover

    class BaseCallbackHandler:  # type: ignore[no-redef]
        """Stand-in so the module imports without langchain installed."""


class TrajectoryCallbackHandler(BaseCallbackHandler):
    """Records a LangGraph run into a `Trajectory` as it happens.

    Pass as `config={"callbacks": [handler]}` to `graph.invoke`. Node transitions
    arrive as chain start/end events; LangGraph names each node, so `agent_name`
    on every step is the node that produced it.
    """

    def __init__(
        self, task_id: str, agent_version: str, model: str = "", run_id: str | None = None
    ) -> None:
        self.recorder = Recorder(task_id, agent_version, model=model, run_id=run_id)
        self._starts: dict[UUID, float] = {}
        self._node_stack: list[str] = []

    @property
    def trajectory(self) -> Trajectory:
        """The trajectory built so far. Call `finalize()` when the run ends."""
        return self.recorder.trajectory

    # -- graph nodes -------------------------------------------------------

    def on_chain_start(
        self, serialized: dict[str, Any], inputs: dict[str, Any], **kwargs: Any
    ) -> None:
        name = _node_name(serialized, kwargs)
        if not name:
            return
        self._node_stack.append(name)
        self.recorder.add_step(
            StepType.PLAN, agent_name=name, tool_input={"node": name}
        )

    def on_chain_end(self, outputs: dict[str, Any], **kwargs: Any) -> None:
        if self._node_stack:
            self._node_stack.pop()

    def on_chain_error(self, error: BaseException, **kwargs: Any) -> None:
        self.recorder.add_step(
            StepType.RETRY,
            agent_name=self._current_node,
            error=f"{type(error).__name__}: {error}",
        )

    # -- tools -------------------------------------------------------------

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if run_id is not None:
            self._starts[run_id] = time.perf_counter()
        self.recorder.add_step(
            StepType.TOOL_CALL,
            agent_name=self._current_node,
            tool_name=(serialized or {}).get("name", "unknown_tool"),
            tool_input=inputs or {"input": input_str},
        )

    def on_tool_end(
        self, output: Any, *, run_id: UUID | None = None, **kwargs: Any
    ) -> None:
        self.recorder.add_step(
            StepType.TOOL_RESULT,
            agent_name=self._current_node,
            tool_name=self._last_tool_name,
            tool_output=str(output)[:2000],
            latency_ms=self._elapsed_ms(run_id),
        )

    def on_tool_error(
        self, error: BaseException, *, run_id: UUID | None = None, **kwargs: Any
    ) -> None:
        self.recorder.add_step(
            StepType.TOOL_RESULT,
            agent_name=self._current_node,
            tool_name=self._last_tool_name,
            error=f"{type(error).__name__}: {error}",
            latency_ms=self._elapsed_ms(run_id),
        )

    # -- llm ---------------------------------------------------------------

    def on_llm_start(
        self, serialized: dict[str, Any], prompts: list[str], *, run_id: UUID | None = None, **kwargs: Any
    ) -> None:
        if run_id is not None:
            self._starts[run_id] = time.perf_counter()

    def on_llm_end(self, response: Any, *, run_id: UUID | None = None, **kwargs: Any) -> None:
        usage = _token_usage(response)
        text = _first_text(response)
        self.recorder.add_step(
            StepType.LLM_MESSAGE,
            agent_name=self._current_node,
            tool_output=text[:2000],
            tokens=usage,
            latency_ms=self._elapsed_ms(run_id),
        )
        self.trajectory.total_tokens += usage

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        self.recorder.add_step(
            StepType.LLM_MESSAGE,
            agent_name=self._current_node,
            error=f"{type(error).__name__}: {error}",
        )

    # -- finishing ---------------------------------------------------------

    def finalize(
        self,
        terminal_state: TerminalState,
        final_output: str = "",
        **metadata: Any,
    ) -> Trajectory:
        """Close the trajectory out. Safe to call exactly once."""
        self.recorder.finish(terminal_state, final_output, **metadata)
        traj = self.trajectory
        traj.terminal_state = terminal_state
        return traj.finalize()

    # -- internals ---------------------------------------------------------

    @property
    def _current_node(self) -> str:
        return self._node_stack[-1] if self._node_stack else ""

    @property
    def _last_tool_name(self) -> str | None:
        for step in reversed(self.trajectory.steps):
            if step.step_type is StepType.TOOL_CALL:
                return step.tool_name
        return None

    def _elapsed_ms(self, run_id: UUID | None) -> int:
        start = self._starts.pop(run_id, None) if run_id is not None else None
        return int((time.perf_counter() - start) * 1000) if start else 0


# ---------------------------------------------------------------------------
# Post-hoc reconstruction
# ---------------------------------------------------------------------------

# Maps the message levels used by autonomous-data-scientist / research-debate-agent
# onto step types. Anything unrecognised becomes an llm_message, which is the
# honest default: we know the agent said something, not what kind of thing.
_STAGE_TOOLS = {
    "supervisor": "plan",
    "cleaning": "clean",
    "eda": "eda",
    "features": "engineer_features",
    "model_selection": "select_model",
    "tuning": "tune",
    "evaluation": "evaluate",
    "narrative": "narrate",
    "report": "report",
    "reviewer": "review",
    "human": "escalate",
}


def trajectory_from_state(
    state: dict[str, Any],
    task_id: str,
    agent_version: str,
    model: str = "",
    run_id: str | None = None,
) -> Trajectory:
    """Rebuild a `Trajectory` from a finished LangGraph state dict.

    Expects the append-only `messages` log convention (`agent`, `content`,
    `level`) that both agent repos use. Unknown shapes degrade to a single-step
    trajectory rather than raising — a thin trajectory scores badly, which is the
    correct signal, whereas a crash loses the run entirely.
    """
    rec = Recorder(task_id, agent_version, model=model, run_id=run_id or state.get("run_id"))
    usage = state.get("token_usage") or {}

    for msg in state.get("messages") or []:
        agent = _get(msg, "agent", "")
        level = _get(msg, "level", "info")
        content = _get(msg, "content", "")
        tokens = _tokens_for(usage, agent)
        if level == "error":
            rec.add_step(
                StepType.TOOL_RESULT,
                agent_name=agent,
                tool_name=_STAGE_TOOLS.get(agent),
                error=content,
                tokens=tokens,
            )
        elif agent == "human" or "escalat" in content.lower():
            rec.add_step(StepType.ESCALATION, agent_name=agent, tool_output=content)
        elif agent in _STAGE_TOOLS:
            rec.add_step(
                StepType.TOOL_CALL,
                agent_name=agent,
                tool_name=_STAGE_TOOLS[agent],
                tool_input={"stage": agent},
            )
            rec.add_step(
                StepType.TOOL_RESULT,
                agent_name=agent,
                tool_name=_STAGE_TOOLS[agent],
                tool_output=content,
                tokens=tokens,
            )
        else:
            rec.add_step(
                StepType.LLM_MESSAGE, agent_name=agent, tool_output=content, tokens=tokens
            )

    traj = rec.trajectory
    traj.terminal_state = _terminal_from_state(state)
    traj.final_output = state.get("narrative") or state.get("final_output") or ""
    traj.total_cost = sum(_get(u, "cost_usd", 0.0) for u in usage.values())
    traj.metadata = {
        "artifacts": [_get(a, "path", "") for a in state.get("artifacts") or []],
        "metrics": _metrics_from_state(state),
        "retry_counts": state.get("retry_counts") or {},
    }
    return traj.finalize()


def _terminal_from_state(state: dict[str, Any]) -> TerminalState:
    status = str(state.get("status", "")).lower()
    if "awaiting_human" in status or state.get("needs_human"):
        return TerminalState.ESCALATED
    if "completed" in status:
        return TerminalState.COMPLETED
    if "budget" in str(state.get("error", "")).lower():
        return TerminalState.BUDGET_EXCEEDED
    return TerminalState.FAILED


def _metrics_from_state(state: dict[str, Any]) -> dict[str, Any]:
    evaluation = state.get("eval_metrics")
    metrics = _get(evaluation, "metrics", {}) if evaluation else {}
    return dict(metrics or {})


def _tokens_for(usage: dict[str, Any], agent: str) -> int:
    entry = usage.get(agent)
    if entry is None:
        return 0
    return int(_get(entry, "prompt_tokens", 0)) + int(_get(entry, "completion_tokens", 0))


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read a field from either a dict or a pydantic/dataclass-ish object."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _node_name(serialized: dict[str, Any] | None, kwargs: dict[str, Any]) -> str:
    tags = kwargs.get("tags") or []
    for tag in tags:
        if isinstance(tag, str) and tag.startswith("graph:step:"):
            continue
    name = kwargs.get("name") or (serialized or {}).get("name", "")
    # LangGraph emits internal chains too; only named nodes are interesting.
    return "" if name in {"", "LangGraph", "RunnableSequence", "__start__"} else str(name)


def _token_usage(response: Any) -> int:
    output = getattr(response, "llm_output", None) or {}
    usage = output.get("token_usage") or output.get("usage") or {}
    if usage:
        return int(usage.get("total_tokens", 0))
    # newer langchain puts usage on the message itself
    try:
        gen = response.generations[0][0]
        meta = getattr(gen.message, "usage_metadata", None) or {}
        return int(meta.get("total_tokens", 0))
    except (AttributeError, IndexError, TypeError):
        return 0


def _first_text(response: Any) -> str:
    try:
        return str(response.generations[0][0].text)
    except (AttributeError, IndexError, TypeError):
        return ""
