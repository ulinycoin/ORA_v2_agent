"""Task queue: local asyncio.Queue backend (Redis backend is V3)."""
from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Awaitable, Callable


@dataclass
class AgentTask:
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    chat_id: int | None = None
    user_id: int | None = None
    goal: str = ""
    context: str = ""
    on_progress: Callable[[str], Awaitable[None]] | None = None
    on_finish: Callable[[str, str], Awaitable[None]] | None = None  # (run_id, report)


class LocalTaskQueue:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[AgentTask] = asyncio.Queue()

    async def enqueue(self, task: AgentTask) -> str:
        await self._queue.put(task)
        return task.task_id

    async def dequeue(self) -> AgentTask:
        return await self._queue.get()

    def task_done(self) -> None:
        self._queue.task_done()

    def qsize(self) -> int:
        return self._queue.qsize()
