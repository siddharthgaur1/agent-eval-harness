"""Load suites from YAML."""

from __future__ import annotations

from pathlib import Path

import yaml

from .schema import Suite, TaskDef


def load_suite(path: str | Path) -> Suite:
    """Read a suite file.

    Accepts either a bare list of tasks or a mapping with `name` and `tasks`.
    Duplicate task ids are an error — they would silently overwrite each other in
    every aggregation downstream.
    """
    p = Path(path)
    data = yaml.safe_load(p.read_text(encoding="utf-8"))

    if isinstance(data, list):
        name, raw_tasks = p.stem, data
    elif isinstance(data, dict):
        name, raw_tasks = data.get("name", p.stem), data.get("tasks", [])
    else:
        raise ValueError(f"{p}: expected a list of tasks or a mapping with 'tasks'")

    tasks = [TaskDef.model_validate(t) for t in raw_tasks]
    ids = [t.task_id for t in tasks]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"{p}: duplicate task_id(s): {sorted(dupes)}")
    if not tasks:
        raise ValueError(f"{p}: suite contains no tasks")

    return Suite(name=name, tasks=tasks)
