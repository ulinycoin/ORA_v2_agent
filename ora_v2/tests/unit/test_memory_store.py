"""Unit tests: MemoryStore with SQLite in-memory."""
import pytest
import pytest_asyncio
from ora_v2.memory.models import Session, Turn, Profile, Fact, AgentRun
from ora_v2.tests.fixtures import InMemoryMemoryStore


@pytest_asyncio.fixture
async def store():
    s = InMemoryMemoryStore()
    await s.init()
    yield s
    await s.close()


@pytest.mark.asyncio
async def test_save_and_load_session(store):
    session = Session(chat_id=1, mode="agent", active_task="research X")
    await store.save_session(session)
    loaded = await store.get_session(1)
    assert loaded.mode == "agent"
    assert loaded.active_task == "research X"


@pytest.mark.asyncio
async def test_get_missing_session_returns_default(store):
    session = await store.get_session(9999)
    assert session.chat_id == 9999
    assert session.mode == "chat"


@pytest.mark.asyncio
async def test_append_and_get_turns(store):
    await store.save_session(Session(chat_id=10))
    await store.append_turn(10, Turn(chat_id=10, role="user", text="hello"))
    await store.append_turn(10, Turn(chat_id=10, role="assistant", text="hi"))
    turns = await store.get_turns(10)
    assert len(turns) == 2
    assert turns[0].role == "user"
    assert turns[1].role == "assistant"


@pytest.mark.asyncio
async def test_get_turns_limit(store):
    await store.save_session(Session(chat_id=20))
    for i in range(10):
        await store.append_turn(20, Turn(chat_id=20, role="user", text=f"msg {i}"))
    turns = await store.get_turns(20, limit=3)
    assert len(turns) == 3


@pytest.mark.asyncio
async def test_save_and_load_profile(store):
    profile = Profile(user_id=100, username="alice", display_name="Alice")
    await store.save_profile(profile)
    loaded = await store.get_profile(100)
    assert loaded.username == "alice"
    assert loaded.display_name == "Alice"


@pytest.mark.asyncio
async def test_add_and_get_facts(store):
    await store.save_profile(Profile(user_id=200))
    await store.add_fact(Fact(user_id=200, text="likes Python"))
    await store.add_fact(Fact(user_id=200, text="works remote"))
    facts = await store.get_facts(200)
    assert len(facts) == 2
    texts = [f.text for f in facts]
    assert "likes Python" in texts


@pytest.mark.asyncio
async def test_record_agent_run(store):
    run = AgentRun(run_id="run-001", chat_id=1, goal="test goal")
    await store.record_agent_run(run)
    run.status = "completed"
    run.iterations = 5
    await store.record_agent_run(run)
