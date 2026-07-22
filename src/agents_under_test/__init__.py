"""Agents the harness can drive.

The runner takes an entrypoint string (`module:function`) resolving to a callable
`(task: TaskDef, repeat: int) -> Trajectory`. Nothing here is privileged — any
importable callable with that signature works, which is the point of a
framework-agnostic schema.
"""

from __future__ import annotations

import importlib
from typing import Callable

from ..suites.schema import TaskDef
from ..trace.schema import Trajectory

AgentEntrypoint = Callable[[TaskDef, int], Trajectory]


def resolve_agent(spec: str) -> AgentEntrypoint:
    """Import `module:function`. Bare names resolve inside this package."""
    if ":" not in spec:
        spec = f"src.agents_under_test.{spec}:run_task"
    module_name, _, func_name = spec.partition(":")
    module = importlib.import_module(module_name)
    fn = getattr(module, func_name, None)
    if not callable(fn):
        raise ValueError(f"{spec}: not a callable agent entrypoint")
    return fn
