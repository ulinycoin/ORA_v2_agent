"""WorkerPool: N async workers consuming from a LocalTaskQueue."""
from __future__ import annotations

import asyncio
import logging

from ora_v2.agent.orchestrator import OrchestratorAgent
from ora_v2.agent.task_queue import AgentTask, LocalTaskQueue
from ora_v2.memory.store import MemoryStore
from ora_v2.skills.base import SkillRegistry

logger = logging.getLogger(__name__)


class WorkerPool:
    def __init__(
        self,
        queue: LocalTaskQueue,
        store: MemoryStore,
        registry: SkillRegistry,
        n_workers: int = 3,
        max_iterations: int = 20,
        subagent_timeout: int = 120,
        subagent_max_steps: int = 8,
        require_skill_approval: bool = False,
        client=None,
    ) -> None:
        self._queue = queue
        self._store = store
        self._registry = registry
        self._n_workers = n_workers
        self._max_iterations = max_iterations
        self._subagent_timeout = subagent_timeout
        self._subagent_max_steps = subagent_max_steps
        self._require_skill_approval = require_skill_approval
        self._client = client
        self._tasks: list[asyncio.Task] = []
        self._running = False

    async def start(self) -> None:
        self._running = True
        for i in range(self._n_workers):
            t = asyncio.create_task(self._worker(i), name=f"worker-{i}")
            self._tasks.append(t)
        logger.info("WorkerPool started with %d workers", self._n_workers)

    async def stop(self) -> None:
        self._running = False
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        logger.info("WorkerPool stopped")

    async def _worker(self, worker_id: int) -> None:
        logger.debug("Worker %d started", worker_id)
        while self._running:
            try:
                task: AgentTask = await asyncio.wait_for(self._queue.dequeue(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            logger.info("Worker %d picked up task %s: %s", worker_id, task.task_id, task.goal[:60])
            try:
                agent = OrchestratorAgent(
                    store=self._store,
                    registry=self._registry,
                    max_iterations=self._max_iterations,
                    subagent_timeout=self._subagent_timeout,
                    subagent_max_steps=self._subagent_max_steps,
                    require_skill_approval=self._require_skill_approval,
                    client=self._client,
                    user_id=task.user_id,
                )
                result = await agent.run(
                    goal=task.goal,
                    context=task.context,
                    chat_id=task.chat_id,
                    on_progress=task.on_progress,
                )
                if task.on_finish:
                    await task.on_finish(result.run_id, result.report)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Worker %d error on task %s: %s", worker_id, task.task_id, exc)
                if task.on_finish:
                    try:
                        await task.on_finish("", f"ERROR: {exc}")
                    except Exception:
                        pass
            finally:
                self._queue.task_done()
