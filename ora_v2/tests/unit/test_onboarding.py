"""Unit tests: onboarding flow."""
import pytest
import pytest_asyncio
from unittest.mock import patch, AsyncMock
from ora_v2.memory.models import Profile, Session, Fact
from ora_v2.chat.onboarding import (
    _is_new_user, _onboarding_stage, _set_stage, _clear_onboarding,
    maybe_start_onboarding, handle_onboarding,
)
from ora_v2.tests.fixtures import InMemoryMemoryStore


@pytest_asyncio.fixture
async def store():
    s = InMemoryMemoryStore()
    await s.init()
    yield s
    await s.close()


def test_is_new_user_no_facts():
    profile = Profile(user_id=1)
    assert _is_new_user(profile, []) is True


def test_is_new_user_has_facts():
    profile = Profile(user_id=1)
    facts = [Fact(user_id=1, text="likes Python")]
    assert _is_new_user(profile, facts) is False


def test_is_new_user_has_summary():
    profile = Profile(user_id=1, summary="developer")
    assert _is_new_user(profile, []) is False


def test_onboarding_stage_not_set():
    session = Session(chat_id=1)
    assert _onboarding_stage(session) is None


def test_onboarding_stage_set():
    session = Session(chat_id=1)
    _set_stage(session, "name")
    assert _onboarding_stage(session) == "name"


def test_clear_onboarding():
    session = Session(chat_id=1)
    _set_stage(session, "occupation")
    _clear_onboarding(session)
    assert _onboarding_stage(session) is None


@pytest.mark.asyncio
async def test_maybe_start_onboarding_new_user(store):
    session = Session(chat_id=1)
    await store.save_session(session)

    greeting = await maybe_start_onboarding(store, session, user_id=1, display_name="Алексей")

    assert greeting is not None
    assert "Алексей" in greeting or "привет" in greeting.lower()
    assert _onboarding_stage(session) == "name"


@pytest.mark.asyncio
async def test_maybe_start_onboarding_existing_user(store):
    # User has facts — skip onboarding
    await store.save_profile(Profile(user_id=2))
    await store.add_fact(Fact(user_id=2, text="developer"))
    session = Session(chat_id=2)
    await store.save_session(session)

    greeting = await maybe_start_onboarding(store, session, user_id=2, display_name=None)
    assert greeting is None


@pytest.mark.asyncio
async def test_handle_onboarding_saves_fact(store):
    session = Session(chat_id=10)
    _set_stage(session, "name")
    await store.save_session(session)

    mock_llm = AsyncMock(side_effect=[
        "Пользователь представился как Алексей.",  # fact extraction
        "Приятно познакомиться, Алексей! Чем ты занимаешься?",  # next question
    ])

    import ora_v2.llm.client as llm_mod
    with patch.object(llm_mod, "get_client") as mock_get:
        mock_client = AsyncMock()
        mock_client.call = mock_llm
        mock_get.return_value = mock_client

        reply, done = await handle_onboarding(store, session, user_id=10, text="Меня зовут Алексей")

    assert done is False
    assert _onboarding_stage(session) == "occupation"
    facts = await store.get_facts(10)
    assert any("Алексей" in f.text for f in facts)


@pytest.mark.asyncio
async def test_handle_onboarding_completes(store):
    """After last stage, onboarding finishes and profile summary is set."""
    # Pre-populate two facts so last stage closes it out
    await store.save_profile(Profile(user_id=20))
    await store.add_fact(Fact(user_id=20, text="Пользователь — разработчик."))
    await store.add_fact(Fact(user_id=20, text="Пользователь интересуется AI."))

    session = Session(chat_id=20)
    _set_stage(session, "interests")  # last stage
    await store.save_session(session)

    mock_calls = AsyncMock(side_effect=[
        "Пользователь интересуется машинным обучением.",  # fact extraction
        "Отлично! Я запомнил всё о тебе. Можем начинать!",  # closing message
        "Разработчик, интересующийся AI.",  # profile summary
    ])

    import ora_v2.llm.client as llm_mod
    with patch.object(llm_mod, "get_client") as mock_get:
        mock_client = AsyncMock()
        mock_client.call = mock_calls
        mock_get.return_value = mock_client

        reply, done = await handle_onboarding(store, session, user_id=20, text="мне интересен ML")

    assert done is True
    assert _onboarding_stage(session) is None
    profile = await store.get_profile(20)
    assert profile.summary != ""
