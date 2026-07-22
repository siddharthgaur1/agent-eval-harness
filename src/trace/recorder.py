"""Generic recorder: a context manager plus a decorator.

Any Python agent can produce a well-formed `Trajectory` with these two, without
importing a framework. The LangGraph adapter is built on top of this rather than
beside it, so there is only one place that knows how to append a step.
"""

from __future__ import annotations

import contextvars
import functools
import time
import uuid
from datetime import datetime, timezone
from types import TracebackType
from typing import Any, Callable, Iterator

from .schema import Step, StepType, TerminalState, Trajectory

# The recorder the @record_step decorator writes into. A ContextVar rather than a
# module global so concurrent suite workers do not cross-contaminate.
_active: contextvars.ContextVar["Recorder | None"] = contextvars.ContextVar(
    "active_recorder", default=None
)


class Recorder:
    """Accumulates steps into a `Trajectory`.

    Use as a context manager. Terminal state defaults to `completed` on a clean
    exit and `failed` if an exception escapes, unless explicitly set.
    """

    def __init__(
        self,
        task_id: str,
        agent_version: str,
        model: str = "",
        run_id: str | None = None,
    ) -> None:
        self.trajectory = Trajectory(
            run_id=run_id or uuid.uuid4().hex[:12],
            task_id=task_id,
            agent_version=agent_version,
            model=model,
        )
        self._explicit_terminal: TerminalState | None = None
        self._token: contextvars.Token | None = None

    # -- lifecycle ---------------------------------------------------------

    def __enter__(self) -> "Recorder":
        self._token = _active.set(self)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        if self._token is not None:
            _active.reset(self._token)
        if self._explicit_terminal is not None:
            self.trajectory.terminal_state = self._explicit_terminal
        elif exc is not None:
            self.trajectory.terminal_state = TerminalState.FAILED
            self.add_step(StepType.TOOL_RESULT, error=f"{exc_type.__name__}: {exc}")
        else:
            self.trajectory.terminal_state = TerminalState.COMPLETED
        self.trajectory.ended_at = datetime.now(timezone.utc)
        self.trajectory.finalize()
        return False  # never swallow

    # -- recording ---------------------------------------------------------

    def add_step(
        self,
        step_type: StepType,
        *,
        agent_name: str = "",
        tool_name: str | None = None,
        tool_input: dict[str, Any] | None = None,
        tool_output: Any = None,
        error: str | None = None,
        tokens: int = 0,
        latency_ms: int = 0,
    ) -> Step:
        """Append one step and return it."""
        step = Step(
            index=len(self.trajectory.steps),
            agent_name=agent_name,
            step_type=step_type,
            tool_name=tool_name,
            tool_input=tool_input or {},
            tool_output=tool_output,
            error=error,
            tokens=tokens,
            latency_ms=latency_ms,
        )
        self.trajectory.steps.append(step)
        return step

    def finish(
        self,
        terminal_state: TerminalState,
        final_output: str = "",
        **metadata: Any,
    ) -> None:
        """Declare how the run ended. Overrides the exit-time default."""
        self._explicit_terminal = terminal_state
        if final_output:
            self.trajectory.final_output = final_output
        self.trajectory.metadata.update(metadata)


def current_recorder() -> Recorder | None:
    """The recorder for the current context, if any."""
    return _active.get()


def record_step(
    step_type: StepType = StepType.TOOL_CALL,
    *,
    tool_name: str | None = None,
    agent_name: str = "",
) -> Callable:
    """Decorator that records a function call as a step.

    A no-op when there is no active `Recorder`, so instrumented agent code runs
    unchanged in production.
    """

    def decorator(fn: Callable) -> Callable:
        name = tool_name or fn.__name__

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            rec = current_recorder()
            if rec is None:
                return fn(*args, **kwargs)
            start = time.perf_counter()
            try:
                result = fn(*args, **kwargs)
            except Exception as exc:
                rec.add_step(
                    step_type,
                    agent_name=agent_name,
                    tool_name=name,
                    tool_input=_safe_input(args, kwargs),
                    error=f"{type(exc).__name__}: {exc}",
                    latency_ms=int((time.perf_counter() - start) * 1000),
                )
                raise
            rec.add_step(
                step_type,
                agent_name=agent_name,
                tool_name=name,
                tool_input=_safe_input(args, kwargs),
                tool_output=_truncate(result),
                latency_ms=int((time.perf_counter() - start) * 1000),
            )
            return result

        return wrapper

    return decorator


def _safe_input(args: tuple, kwargs: dict) -> dict[str, Any]:
    """Serialize call arguments without choking on unserializable objects."""
    out: dict[str, Any] = {f"arg{i}": _truncate(a) for i, a in enumerate(args)}
    out.update({k: _truncate(v) for k, v in kwargs.items()})
    return out


def _truncate(value: Any, limit: int = 2000) -> Any:
    """Keep tool payloads small enough to store and show in a viewer."""
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    text = value if isinstance(value, str) else repr(value)
    return text if len(text) <= limit else text[:limit] + "…[truncated]"


def steps_of(trajectory: Trajectory) -> Iterator[Step]:
    """Convenience iterator, mostly for readability at call sites."""
    yield from trajectory.steps
