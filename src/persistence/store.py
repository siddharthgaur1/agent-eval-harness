"""SQLite store for trajectories, evaluation runs, and raw judge responses.

Deliberately three tables of JSON blobs with indexed keys rather than a
normalized schema. The trajectory schema is the thing that has to be stable;
storage is an implementation detail, and a blob store means a schema addition
never needs a migration.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from ..config import settings
from ..trace.schema import Trajectory

_SCHEMA = """
CREATE TABLE IF NOT EXISTS trajectories (
    run_id        TEXT PRIMARY KEY,
    task_id       TEXT NOT NULL,
    agent_version TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    payload       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_traj_task ON trajectories(task_id, agent_version);

CREATE TABLE IF NOT EXISTS runs (
    run_id        TEXT PRIMARY KEY,
    suite         TEXT NOT NULL,
    agent_version TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    is_baseline   INTEGER NOT NULL DEFAULT 0,
    payload       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_runs_agent ON runs(agent_version, created_at);

CREATE TABLE IF NOT EXISTS judge_calls (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    trajectory_id TEXT NOT NULL,
    scorer        TEXT NOT NULL,
    model         TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    prompt        TEXT NOT NULL,
    response      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_judge_traj ON judge_calls(trajectory_id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Store:
    """Thin persistence layer. One connection per call — SQLite is fine with it."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = Path(db_path or settings.db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # -- trajectories ------------------------------------------------------

    def save_trajectory(self, traj: Trajectory) -> None:
        """Upsert, keyed by run_id. Re-recording a run replaces it."""
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO trajectories "
                "(run_id, task_id, agent_version, created_at, payload) VALUES (?,?,?,?,?)",
                (
                    traj.run_id,
                    traj.task_id,
                    traj.agent_version,
                    _now(),
                    traj.model_dump_json(),
                ),
            )

    def get_trajectory(self, run_id: str) -> Trajectory | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT payload FROM trajectories WHERE run_id = ?", (run_id,)
            ).fetchone()
        return Trajectory.model_validate_json(row["payload"]) if row else None

    def list_trajectories(
        self, task_id: str | None = None, agent_version: str | None = None, limit: int = 100
    ) -> list[Trajectory]:
        sql = "SELECT payload FROM trajectories WHERE 1=1"
        params: list[Any] = []
        if task_id:
            sql += " AND task_id = ?"
            params.append(task_id)
        if agent_version:
            sql += " AND agent_version = ?"
            params.append(agent_version)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [Trajectory.model_validate_json(r["payload"]) for r in rows]

    # -- evaluation runs ---------------------------------------------------

    def save_run(self, run: dict[str, Any], is_baseline: bool = False) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO runs "
                "(run_id, suite, agent_version, created_at, is_baseline, payload) "
                "VALUES (?,?,?,?,?,?)",
                (
                    run["run_id"],
                    run.get("suite", ""),
                    run.get("agent_version", ""),
                    run.get("created_at", _now()),
                    int(is_baseline),
                    json.dumps(run, default=str),
                ),
            )

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT payload FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        return json.loads(row["payload"]) if row else None

    def list_runs(self, agent_version: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        sql = "SELECT payload FROM runs"
        params: list[Any] = []
        if agent_version:
            sql += " WHERE agent_version = ?"
            params.append(agent_version)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [json.loads(r["payload"]) for r in rows]

    def latest_baseline(self, agent_version: str | None = None) -> dict[str, Any] | None:
        """Most recent run marked as a baseline, optionally for one agent version."""
        sql = "SELECT payload FROM runs WHERE is_baseline = 1"
        params: list[Any] = []
        if agent_version:
            sql += " AND agent_version = ?"
            params.append(agent_version)
        sql += " ORDER BY created_at DESC LIMIT 1"
        with self._conn() as conn:
            row = conn.execute(sql, params).fetchone()
        return json.loads(row["payload"]) if row else None

    def mark_baseline(self, run_id: str) -> None:
        with self._conn() as conn:
            conn.execute("UPDATE runs SET is_baseline = 1 WHERE run_id = ?", (run_id,))

    # -- judge audit trail -------------------------------------------------

    def save_judge_call(
        self, trajectory_id: str, scorer: str, model: str, prompt: str, response: str
    ) -> None:
        """Persist the raw judge exchange. Without this, an LLM score is unauditable."""
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO judge_calls "
                "(trajectory_id, scorer, model, created_at, prompt, response) "
                "VALUES (?,?,?,?,?,?)",
                (trajectory_id, scorer, model, _now(), prompt, response),
            )

    def judge_calls(self, trajectory_id: str) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT scorer, model, created_at, prompt, response FROM judge_calls "
                "WHERE trajectory_id = ? ORDER BY id",
                (trajectory_id,),
            ).fetchall()
        return [dict(r) for r in rows]
