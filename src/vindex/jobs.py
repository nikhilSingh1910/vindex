"""Per-(video, stage) job tracking — the whole of round 1's resumability. A stage marks
itself done only after its outputs are committed, so a crash resumes at the failed stage.
"""

from __future__ import annotations

import sqlite3
from enum import Enum


class Status(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


def get_status(conn: sqlite3.Connection, video_id: str, stage: str) -> Status:
    row = conn.execute(
        "SELECT status FROM jobs WHERE video_id = ? AND stage = ?", (video_id, stage)
    ).fetchone()
    return Status(row["status"]) if row else Status.PENDING


def set_status(
    conn: sqlite3.Connection,
    video_id: str,
    stage: str,
    status: Status,
    error: str | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO jobs (video_id, stage, status, error, updated_at)
        VALUES (?, ?, ?, ?, datetime('now'))
        ON CONFLICT(video_id, stage) DO UPDATE SET
            status=excluded.status, error=excluded.error, updated_at=excluded.updated_at
        """,
        (video_id, stage, status.value, error),
    )
    conn.commit()


def is_done(conn: sqlite3.Connection, video_id: str, stage: str) -> bool:
    return get_status(conn, video_id, stage) == Status.DONE
