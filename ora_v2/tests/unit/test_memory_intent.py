"""Unit tests: memory intent detection (remember / forget)."""
import pytest
import pytest_asyncio
from unittest.mock import patch, AsyncMock

from ora_v2.memory.models import Profile, Fact
from ora_v2.chat.memory_intent import try_handle_memory_intent, _looks_like_remember, _looks_like_forget
from ora_v2.tests.fixtures import InMemoryMemoryStore


@pytest_asyncio.fixture
async def store():
    s = InMemoryMemoryStore()
    await s.init()
    yield s
    await s.close()


# ── Pattern detection ──────────────────────────────────────────────────────────

def test_looks_like_remember_triggers():
    assert _looks_like_remember("запомни, что я люблю Python") is True
    assert _looks_like_remember("Сохрани это для меня") is True
    assert _looks_like_remember("не забудь: встреча в пятницу") is True


def test_looks_like_remember_no_match():
    assert _looks_like_remember("расскажи о Python") is False
    assert _looks_like_remember("что ты помнишь обо мне?") is False


def test_looks_like_forget_triggers():
    assert _looks_like_forget("забудь про встречу") is True
    assert _looks_like_forget("удали из памяти мой email") is True
    assert _looks_like_forget("больше не помни мой адрес") is True


def test_looks_like_forget_no_match():
    assert _looks_like_forget("что ты помнишь?") is False
    assert _looks_like_forget("расскажи мне о себе") is False


# ── Remember intent ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_remember_saves_fact(store):
    import ora_v2.chat.memory_intent as mi_mod
    with patch.object(mi_mod, "get_client") as mock_get:
        mock_client = AsyncMock()
        mock_client.call = AsyncMock(return_value="Пользователь предпочитает Python.")
        mock_get.return_value = mock_client

        reply = await try_handle_memory_intent(store, user_id=1, text="запомни, что я люблю Python")

    assert reply == "Запомнил."
    facts = await store.get_facts(1)
    assert any("Python" in f.text for f in facts)


@pytest.mark.asyncio
async def test_remember_skip_when_vague(store):
    import ora_v2.chat.memory_intent as mi_mod
    with patch.object(mi_mod, "get_client") as mock_get:
        mock_client = AsyncMock()
        mock_client.call = AsyncMock(return_value="SKIP")
        mock_get.return_value = mock_client

        reply = await try_handle_memory_intent(store, user_id=1, text="запомни это")

    assert reply is not None
    assert "распознать" in reply or "точнее" in reply
    facts = await store.get_facts(1)
    assert facts == []


# ── Forget intent ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_forget_deletes_correct_fact(store):
    await store.save_profile(Profile(user_id=5))
    await store.add_fact(Fact(user_id=5, text="Пользователь работает в Яндексе."))
    facts_before = await store.get_facts(5)
    target_id = facts_before[0].id

    import ora_v2.chat.memory_intent as mi_mod
    with patch.object(mi_mod, "get_client") as mock_get:
        mock_client = AsyncMock()
        mock_client.call = AsyncMock(return_value=str(target_id))
        mock_get.return_value = mock_client

        reply = await try_handle_memory_intent(store, user_id=5, text="забудь про Яндекс")

    assert reply == "Забыл."
    facts_after = await store.get_facts(5)
    assert facts_after == []


@pytest.mark.asyncio
async def test_forget_no_facts(store):
    reply = await try_handle_memory_intent(store, user_id=99, text="забудь про всё")
    assert reply is not None
    assert "нет" in reply.lower() or "факт" in reply.lower()


@pytest.mark.asyncio
async def test_forget_no_match(store):
    await store.save_profile(Profile(user_id=6))
    await store.add_fact(Fact(user_id=6, text="Пользователь любит кофе."))

    import ora_v2.chat.memory_intent as mi_mod
    with patch.object(mi_mod, "get_client") as mock_get:
        mock_client = AsyncMock()
        mock_client.call = AsyncMock(return_value="NONE")
        mock_get.return_value = mock_client

        reply = await try_handle_memory_intent(store, user_id=6, text="забудь про мой адрес")

    assert reply is not None
    assert "нашёл" in reply or "факт" in reply


# ── Non-memory text ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_neutral_text_returns_none(store):
    reply = await try_handle_memory_intent(store, user_id=1, text="что такое нейросети?")
    assert reply is None
