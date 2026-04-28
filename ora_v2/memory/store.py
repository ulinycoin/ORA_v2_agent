"""Async SQLite-backed memory store."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite

from ora_v2.memory.models import (
    AgentRun, Document, Fact, Profile, Session, Turn, UserSecret,
)

logger = logging.getLogger(__name__)

_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS sessions (
    chat_id      INTEGER PRIMARY KEY,
    user_id      INTEGER,
    mode         TEXT NOT NULL DEFAULT 'chat',
    active_task  TEXT DEFAULT '',
    summary      TEXT DEFAULT '',
    agent_status TEXT NOT NULL DEFAULT 'idle',
    updated_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS turns (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER NOT NULL REFERENCES sessions(chat_id),
    role    TEXT NOT NULL,
    kind    TEXT NOT NULL DEFAULT 'chat',
    text    TEXT NOT NULL,
    ts      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS profiles (
    user_id      INTEGER PRIMARY KEY,
    username     TEXT,
    display_name TEXT,
    summary      TEXT DEFAULT '',
    updated_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS facts (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES profiles(user_id),
    text    TEXT NOT NULL,
    ts      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id      INTEGER NOT NULL,
    name         TEXT NOT NULL,
    size_bytes   INTEGER NOT NULL,
    storage_path TEXT NOT NULL,
    ts           TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS secrets (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    key     TEXT NOT NULL,
    value   TEXT NOT NULL,
    ts      TEXT NOT NULL,
    UNIQUE(user_id, key)
);

CREATE TABLE IF NOT EXISTS agent_runs (
    run_id      TEXT PRIMARY KEY,
    chat_id     INTEGER,
    goal        TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'running',
    iterations  INTEGER DEFAULT 0,
    started_at  TEXT NOT NULL,
    finished_at TEXT
);
"""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class MemoryStore:
    def __init__(self, db_path: str = "ora_v2.db") -> None:
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def init(self) -> None:
        if self._db is not None:
            return
        self._db = await aiosqlite.connect(self._db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(_SCHEMA)
        await self._db.commit()

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    @property
    def _conn(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("MemoryStore.init() must be awaited before use")
        return self._db

    # ── Sessions ────────────────────────────────────────────────────────────

    async def get_session(self, chat_id: int) -> Session:
        async with self._conn.execute(
            "SELECT * FROM sessions WHERE chat_id = ?", (chat_id,)
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return Session(chat_id=chat_id)
        return Session(**dict(row))

    async def save_session(self, session: Session) -> None:
        session.updated_at = _utc_now()
        await self._conn.execute(
            """
            INSERT INTO sessions (chat_id, user_id, mode, active_task, summary, agent_status, updated_at)
            VALUES (:chat_id, :user_id, :mode, :active_task, :summary, :agent_status, :updated_at)
            ON CONFLICT(chat_id) DO UPDATE SET
                user_id      = excluded.user_id,
                mode         = excluded.mode,
                active_task  = excluded.active_task,
                summary      = excluded.summary,
                agent_status = excluded.agent_status,
                updated_at   = excluded.updated_at
            """,
            session.model_dump(),
        )
        await self._conn.commit()

    # ── Turns ────────────────────────────────────────────────────────────────

    async def append_turn(self, chat_id: int, turn: Turn) -> None:
        # Ensure session row exists to satisfy FK constraint
        await self._conn.execute(
            "INSERT OR IGNORE INTO sessions (chat_id, updated_at) VALUES (?, ?)",
            (chat_id, _utc_now()),
        )
        await self._conn.execute(
            "INSERT INTO turns (chat_id, role, kind, text, ts) VALUES (?, ?, ?, ?, ?)",
            (chat_id, turn.role, turn.kind, turn.text, turn.ts),
        )
        await self._conn.commit()

    async def get_turns(self, chat_id: int, limit: int = 40) -> list[Turn]:
        async with self._conn.execute(
            "SELECT * FROM turns WHERE chat_id = ? ORDER BY id DESC LIMIT ?",
            (chat_id, limit),
        ) as cur:
            rows = await cur.fetchall()
        return [Turn(**dict(r)) for r in reversed(rows)]

    async def clear_turns(self, chat_id: int) -> None:
        await self._conn.execute("DELETE FROM turns WHERE chat_id = ?", (chat_id,))
        await self._conn.commit()

    # ── Profiles ─────────────────────────────────────────────────────────────

    async def get_profile(self, user_id: int) -> Profile:
        async with self._conn.execute(
            "SELECT * FROM profiles WHERE user_id = ?", (user_id,)
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return Profile(user_id=user_id)
        return Profile(**dict(row))

    async def save_profile(self, profile: Profile) -> None:
        profile.updated_at = _utc_now()
        await self._conn.execute(
            """
            INSERT INTO profiles (user_id, username, display_name, summary, updated_at)
            VALUES (:user_id, :username, :display_name, :summary, :updated_at)
            ON CONFLICT(user_id) DO UPDATE SET
                username     = excluded.username,
                display_name = excluded.display_name,
                summary      = excluded.summary,
                updated_at   = excluded.updated_at
            """,
            profile.model_dump(),
        )
        await self._conn.commit()

    # ── Facts ────────────────────────────────────────────────────────────────

    async def add_fact(self, fact: Fact) -> None:
        await self._conn.execute(
            "INSERT OR IGNORE INTO profiles (user_id, updated_at) VALUES (?, ?)",
            (fact.user_id, _utc_now()),
        )
        await self._conn.execute(
            "INSERT INTO facts (user_id, text, ts) VALUES (?, ?, ?)",
            (fact.user_id, fact.text, fact.ts),
        )
        await self._conn.commit()

    async def get_facts(self, user_id: int, limit: int = 100) -> list[Fact]:
        async with self._conn.execute(
            "SELECT * FROM facts WHERE user_id = ? ORDER BY id DESC LIMIT ?",
            (user_id, limit),
        ) as cur:
            rows = await cur.fetchall()
        return [Fact(**dict(r)) for r in reversed(rows)]

    async def delete_fact(self, fact_id: int) -> bool:
        """Delete a fact by id. Returns True if a row was deleted."""
        async with self._conn.execute(
            "DELETE FROM facts WHERE id = ?", (fact_id,)
        ) as cur:
            deleted = cur.rowcount > 0
        await self._conn.commit()
        return deleted

    # ── Secrets ──────────────────────────────────────────────────────────────

    async def set_secret(self, secret: UserSecret) -> None:
        await self._conn.execute(
            """
            INSERT INTO secrets (user_id, key, value, ts)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id, key) DO UPDATE SET value = excluded.value, ts = excluded.ts
            """,
            (secret.user_id, secret.key.lower(), secret.value, secret.ts),
        )
        await self._conn.commit()

    async def get_secret(self, user_id: int, key: str) -> str | None:
        async with self._conn.execute(
            "SELECT value FROM secrets WHERE user_id = ? AND key = ?",
            (user_id, key.lower()),
        ) as cur:
            row = await cur.fetchone()
        return row[0] if row else None

    async def list_secret_keys(self, user_id: int) -> list[str]:
        """Return key names only — never values."""
        async with self._conn.execute(
            "SELECT key FROM secrets WHERE user_id = ? ORDER BY key",
            (user_id,),
        ) as cur:
            rows = await cur.fetchall()
        return [r[0] for r in rows]

    async def delete_secret(self, user_id: int, key: str) -> bool:
        async with self._conn.execute(
            "DELETE FROM secrets WHERE user_id = ? AND key = ?",
            (user_id, key.lower()),
        ) as cur:
            deleted = cur.rowcount > 0
        await self._conn.commit()
        return deleted

    # ── Documents ────────────────────────────────────────────────────────────

    async def store_document(self, chat_id: int, name: str, data: bytes) -> Document:
        path = Path(self._db_path).parent / "documents" / str(chat_id)
        path.mkdir(parents=True, exist_ok=True)
        ts = _utc_now().replace(":", "-").replace(".", "-")
        file_path = path / f"{ts}_{name}"
        file_path.write_bytes(data)
        doc = Document(
            chat_id=chat_id,
            name=name,
            size_bytes=len(data),
            storage_path=str(file_path),
        )
        await self._conn.execute(
            "INSERT INTO documents (chat_id, name, size_bytes, storage_path, ts) VALUES (?, ?, ?, ?, ?)",
            (doc.chat_id, doc.name, doc.size_bytes, doc.storage_path, doc.ts),
        )
        await self._conn.commit()
        return doc

    async def get_documents(self, chat_id: int) -> list[Document]:
        async with self._conn.execute(
            "SELECT * FROM documents WHERE chat_id = ? ORDER BY id DESC LIMIT 5",
            (chat_id,),
        ) as cur:
            rows = await cur.fetchall()
        return [Document(**dict(r)) for r in rows]

    # ── Agent runs ───────────────────────────────────────────────────────────

    async def record_agent_run(self, run: AgentRun) -> None:
        await self._conn.execute(
            """
            INSERT INTO agent_runs (run_id, chat_id, goal, status, iterations, started_at, finished_at)
            VALUES (:run_id, :chat_id, :goal, :status, :iterations, :started_at, :finished_at)
            ON CONFLICT(run_id) DO UPDATE SET
                status      = excluded.status,
                iterations  = excluded.iterations,
                finished_at = excluded.finished_at
            """,
            run.model_dump(),
        )
        await self._conn.commit()
