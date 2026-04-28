"""Integration tests: orchestrator and subagent with mock LLM."""
from __future__ import annotations
import asyncio
import pytest
from ora_v2.llm.schemas import OrchestratorPlan, SubagentDecision
from ora_v2.agent.orchestrator import OrchestratorAgent
from ora_v2.agent import subagent as subagent_mod
from ora_v2.skills.base import SkillRegistry
from ora_v2.tests.fixtures import InMemoryMemoryStore, MockLLMClient, MockSkill


@pytest.mark.asyncio
async def test_orchestrator_completes_in_two_iterations():
    store = InMemoryMemoryStore()
    await store.init()

    registry = SkillRegistry()
    registry.register(MockSkill("web_search", response="found: Python is great"))

    iter1_plan = OrchestratorPlan(
        thought="I need to search",
        summary_for_user="Searching...",
        spawn_subagents=[{"task": "find info about Python", "context": ""}],
        done=False,
    )
    subagent_search = SubagentDecision(
        decision_rationale="searching",
        confidence=0.8,
        action="web_search",
        args={"query": "Python"},
    )
    subagent_finish = SubagentDecision(
        decision_rationale="done",
        confidence=0.9,
        action="finish",
        args={},
        final_answer="Python is great",
    )
    iter2_plan = OrchestratorPlan(
        thought="Have enough info",
        summary_for_user="Found answer.",
        spawn_subagents=[],
        done=True,
        final_report="Python is a great language.",
    )

    mock_client = MockLLMClient([iter1_plan, subagent_search, subagent_finish, iter2_plan])

    agent = OrchestratorAgent(
        store=store,
        registry=registry,
        max_iterations=5,
        client=mock_client,
    )
    result = await agent.run(goal="Tell me about Python", chat_id=1)

    assert result.status == "completed"
    assert "Python" in result.report
    assert result.iterations == 2

    await store.close()


@pytest.mark.asyncio
async def test_subagent_timeout():
    registry = SkillRegistry()

    class HangingSkill(MockSkill):
        def run(self, **kwargs):
            # Block the executor thread for a long time
            import time
            time.sleep(999)
            return "never"

    registry.register(HangingSkill("slow_skill"))

    subagent_action = SubagentDecision(
        decision_rationale="using slow skill",
        confidence=0.5,
        action="slow_skill",
        args={},
    )
    mock_client = MockLLMClient([subagent_action] * 10)

    result = await subagent_mod.run(
        task="do something slow",
        registry=registry,
        timeout_seconds=1,
        client=mock_client,
    )
    assert result.timed_out is True


@pytest.mark.asyncio
async def test_worker_pool_parallel():
    """Three tasks enqueued, all complete."""
    from ora_v2.agent.task_queue import AgentTask, LocalTaskQueue
    from ora_v2.agent.worker_pool import WorkerPool
    import unittest.mock as mock

    store = InMemoryMemoryStore()
    await store.init()
    registry = SkillRegistry()
    queue = LocalTaskQueue()
    finished: list[str] = []

    done_plan = OrchestratorPlan(
        thought="done",
        summary_for_user="ok",
        spawn_subagents=[],
        done=True,
        final_report="report",
    )

    async def _mock_call_json(prompt, schema, *, tier, timeout, caller, **kwargs):
        return done_plan

    mock_client = mock.MagicMock()
    mock_client.call_json = _mock_call_json
    mock_client.call = mock.AsyncMock(return_value="report")

    pool = WorkerPool(queue=queue, store=store, registry=registry, n_workers=3, client=mock_client)
    await pool.start()

    async def _finish(run_id, report):
        finished.append(run_id)

    for i in range(3):
        task = AgentTask(goal=f"task {i}", chat_id=i, on_finish=_finish)
        await queue.enqueue(task)

    for _ in range(50):
        await asyncio.sleep(0.1)
        if len(finished) >= 3:
            break
    else:
        await pool.stop()
        await store.close()
        pytest.fail("Worker pool test timed out — only %d/3 tasks finished" % len(finished))

    await pool.stop()
    assert len(finished) == 3
    await store.close()
