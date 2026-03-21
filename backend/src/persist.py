from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Optional


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def init_db(db_path: Path) -> None:
    with _connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
              run_id TEXT PRIMARY KEY,
              created_at INTEGER NOT NULL,
              created_by TEXT NOT NULL,
              summary_json TEXT,
              rows_json TEXT,
              final_rows_json TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_last_run (
              user_id TEXT PRIMARY KEY,
              run_id TEXT NOT NULL,
              updated_at INTEGER NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_created_by ON runs(created_by)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(created_at)")
        conn.commit()


def _now() -> int:
    return int(time.time())


def upsert_run(
    db_path: Path,
    *,
    run_id: str,
    created_by: str,
    summary: Dict[str, Any],
    rows: list[dict[str, Any]],
    final_rows: list[dict[str, Any]],
) -> None:
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO runs (run_id, created_at, created_by, summary_json, rows_json, final_rows_json)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
              summary_json=excluded.summary_json,
              rows_json=excluded.rows_json,
              final_rows_json=excluded.final_rows_json
            """,
            (
                run_id,
                _now(),
                created_by,
                json.dumps(summary or {}),
                json.dumps(rows or []),
                json.dumps(final_rows or []),
            ),
        )
        conn.commit()


def set_last_run(db_path: Path, *, user_id: str, run_id: str) -> None:
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO user_last_run (user_id, run_id, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
              run_id=excluded.run_id,
              updated_at=excluded.updated_at
            """,
            (user_id, run_id, _now()),
        )
        conn.commit()


def clear_last_run(db_path: Path, *, user_id: str) -> None:
    with _connect(db_path) as conn:
        conn.execute("DELETE FROM user_last_run WHERE user_id = ?", (user_id,))
        conn.commit()


def get_last_run_id(db_path: Path, *, user_id: str) -> Optional[str]:
    with _connect(db_path) as conn:
        cur = conn.execute("SELECT run_id FROM user_last_run WHERE user_id = ?", (user_id,))
        row = cur.fetchone()
        return row[0] if row else None


def get_last_run(db_path: Path, *, user_id: str) -> Optional[Dict[str, Any]]:
    with _connect(db_path) as conn:
        cur = conn.execute(
            "SELECT run_id, updated_at FROM user_last_run WHERE user_id = ?",
            (user_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return {"run_id": row[0], "updated_at": row[1]}


def get_run_payload(db_path: Path, *, run_id: str) -> Optional[Dict[str, Any]]:
    with _connect(db_path) as conn:
        cur = conn.execute(
            """
            SELECT run_id, summary_json, rows_json, final_rows_json
            FROM runs
            WHERE run_id = ?
            """,
            (run_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return {
            "run_id": row[0],
            "summary": json.loads(row[1] or "{}"),
            "rows": json.loads(row[2] or "[]"),
            "final_rows": json.loads(row[3] or "[]"),
        }