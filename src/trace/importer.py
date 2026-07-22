"""JSON importer for trajectories recorded elsewhere.

Pydantic already validates the canonical shape, so this module's only real job is
accepting the sloppier shapes people actually export: steps without indices,
timestamps as epoch floats, terminal states spelled differently.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schema import TerminalState, Trajectory

_TERMINAL_ALIASES = {
    "success": TerminalState.COMPLETED,
    "ok": TerminalState.COMPLETED,
    "done": TerminalState.COMPLETED,
    "error": TerminalState.FAILED,
    "failure": TerminalState.FAILED,
    "human": TerminalState.ESCALATED,
    "awaiting_human": TerminalState.ESCALATED,
    "timed_out": TerminalState.TIMEOUT,
    "over_budget": TerminalState.BUDGET_EXCEEDED,
}


def load_trajectory(path: str | Path) -> Trajectory:
    """Load one trajectory from a JSON file."""
    return parse_trajectory(json.loads(Path(path).read_text(encoding="utf-8")))


def load_trajectories(path: str | Path) -> list[Trajectory]:
    """Load a file holding either one trajectory or a list of them."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [parse_trajectory(d) for d in (data if isinstance(data, list) else [data])]


def parse_trajectory(data: dict[str, Any]) -> Trajectory:
    """Normalize a loosely-shaped dict, then validate it."""
    data = dict(data)

    raw_terminal = str(data.get("terminal_state", "")).lower()
    if raw_terminal in _TERMINAL_ALIASES:
        data["terminal_state"] = _TERMINAL_ALIASES[raw_terminal].value

    steps = []
    for i, step in enumerate(data.get("steps") or []):
        step = dict(step)
        step.setdefault("index", i)
        # Re-index unconditionally: scorers cite step indices as evidence, and an
        # imported file with duplicate or gapped indices makes evidence unusable.
        step["index"] = i
        if not isinstance(step.get("tool_input"), dict):
            step["tool_input"] = (
                {} if step.get("tool_input") is None else {"input": step["tool_input"]}
            )
        steps.append(step)
    data["steps"] = steps

    return Trajectory.model_validate(data).finalize()
