"""MessageRouter: routes IncomingMessage to chat or agent."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from ora_v2.agent.task_queue import AgentTask, LocalTaskQueue
from ora_v2.memory.models import Session, Turn
from ora_v2.memory.store import MemoryStore
from ora_v2.router.classifier import classify
from ora_v2.router.commands import parse as parse_cmd

logger = logging.getLogger(__name__)


@dataclass
class IncomingMessage:
    chat_id: int
    user_id: int | None
    text: str
    username: str | None = None
    message_id: int | None = None
    raw: dict = field(default_factory=dict)


@dataclass
class RouteDecision:
    replies: list[str] = field(default_factory=list)
    start_agent: bool = False
    agent_goal: str = ""
    agent_context: str = ""
    stop_agent: bool = False


class MessageRouter:
    def __init__(
        self,
        store: MemoryStore,
        queue: LocalTaskQueue,
        on_agent_start: Callable[[int, str], Awaitable[None]] | None = None,
        on_agent_stop: Callable[[int], Awaitable[None]] | None = None,
    ) -> None:
        self._store = store
        self._queue = queue
        self._on_agent_start = on_agent_start
        self._on_agent_stop = on_agent_stop

    async def route(
        self,
        message: IncomingMessage,
        on_progress: Callable[[str], Awaitable[None]] | None = None,
        on_finish: Callable[[str, str], Awaitable[None]] | None = None,
    ) -> RouteDecision:
        text = message.text.strip()
        chat_id = message.chat_id
        session = await self._store.get_session(chat_id)

        cmd = parse_cmd(text)
        if cmd:
            return await self._handle_command(cmd.name, cmd.args, message, session, on_progress, on_finish)

        # If agent is running, forward input
        if session.agent_status == "running":
            return RouteDecision(replies=[])

        user_id = message.user_id or 0
        raw_from = message.raw.get("from", {})
        display_name = " ".join(
            p for p in [raw_from.get("first_name"), raw_from.get("last_name")] if p
        ).strip() or None

        # Onboarding: check if in progress
        from ora_v2.chat.onboarding import _onboarding_stage, handle_onboarding, maybe_start_onboarding
        onboarding_stage = _onboarding_stage(session)

        if onboarding_stage is not None:
            # Continue existing onboarding flow
            reply, done = await handle_onboarding(self._store, session, user_id, text)
            await self._store.append_turn(chat_id, Turn(chat_id=chat_id, role="user", text=text))
            await self._store.append_turn(chat_id, Turn(chat_id=chat_id, role="assistant", text=reply))
            return RouteDecision(replies=[reply])

        # Check if user is new — start onboarding
        greeting = await maybe_start_onboarding(self._store, session, user_id, display_name)
        if greeting is not None:
            await self._store.append_turn(chat_id, Turn(chat_id=chat_id, role="assistant", text=greeting))
            return RouteDecision(replies=[greeting])

        # Check for explicit remember/forget intent before general classify
        from ora_v2.chat.memory_intent import try_handle_memory_intent
        mem_reply = await try_handle_memory_intent(self._store, user_id, text)
        if mem_reply is not None:
            await self._store.append_turn(chat_id, Turn(chat_id=chat_id, role="user", text=text))
            await self._store.append_turn(chat_id, Turn(chat_id=chat_id, role="assistant", text=mem_reply))
            return RouteDecision(replies=[mem_reply])

        # Auto-classify
        turns = await self._store.get_turns(chat_id, limit=8)
        turn_dicts = [{"role": t.role, "text": t.text} for t in turns]
        mode, reason = await classify(text, turn_dicts)
        logger.info("Classifier: %r → %s (%s)", text[:60], mode, reason)

        if mode == "agent":
            return await self._start_agent(text, "", message, session, on_progress, on_finish)

        # Chat reply
        reply = await self._chat_reply(text, session, message)
        await self._store.append_turn(chat_id, Turn(chat_id=chat_id, role="user", text=text))
        await self._store.append_turn(chat_id, Turn(chat_id=chat_id, role="assistant", text=reply))
        return RouteDecision(replies=[reply])

    async def _handle_command(
        self,
        cmd: str,
        args: str,
        message: IncomingMessage,
        session: Session,
        on_progress,
        on_finish,
    ) -> RouteDecision:
        chat_id = message.chat_id

        if cmd == "help":
            return RouteDecision(replies=[self._help_text()])

        if cmd == "start":
            user_id = message.user_id or 0
            raw_from = message.raw.get("from", {})
            display_name = " ".join(
                p for p in [raw_from.get("first_name"), raw_from.get("last_name")] if p
            ).strip() or None
            from ora_v2.chat.onboarding import maybe_start_onboarding
            greeting = await maybe_start_onboarding(self._store, session, user_id, display_name)
            if greeting:
                return RouteDecision(replies=[greeting])
            return RouteDecision(replies=[self._help_text()])

        if cmd in ("task", "agent"):
            goal = args.strip()
            if not goal:
                return RouteDecision(replies=["Usage: /task <goal>"])
            return await self._start_agent(goal, "", message, session, on_progress, on_finish)

        if cmd == "stop":
            if self._on_agent_stop:
                await self._on_agent_stop(chat_id)
            session.agent_status = "idle"
            await self._store.save_session(session)
            return RouteDecision(replies=["Agent stopped."])

        if cmd == "status":
            return RouteDecision(replies=[
                f"Mode: {session.mode}\n"
                f"Task: {session.active_task or '(none)'}\n"
                f"Agent: {session.agent_status}"
            ])

        if cmd == "memory":
            turns = await self._store.get_turns(chat_id, limit=10)
            lines = [f"[summary] {session.summary}"] if session.summary else []
            lines += [f"- {t.role}: {t.text[:200]}" for t in turns]
            return RouteDecision(replies=["\n".join(lines) or "(empty)"])

        if cmd == "profile":
            profile = await self._store.get_profile(message.user_id or 0)
            facts = await self._store.get_facts(profile.user_id, limit=10)
            lines = [f"User: {profile.display_name or profile.username or profile.user_id}"]
            if profile.summary:
                lines.append(f"[summary] {profile.summary}")
            lines += [f"- {f.text}" for f in facts]
            return RouteDecision(replies=["\n".join(lines)])

        if cmd == "remember":
            fact_text = args.strip()
            if not fact_text:
                return RouteDecision(replies=["Usage: /remember <fact>"])
            from ora_v2.memory.models import Fact
            uid = message.user_id or 0
            await self._store.add_fact(Fact(user_id=uid, text=fact_text))
            return RouteDecision(replies=["Запомнил."])

        if cmd == "forget":
            query = args.strip()
            if not query:
                return RouteDecision(replies=["Usage: /forget <описание факта>"])
            uid = message.user_id or 0
            from ora_v2.chat.memory_intent import try_handle_memory_intent
            reply = await try_handle_memory_intent(
                self._store, uid, f"забудь про {query}"
            )
            return RouteDecision(replies=[reply or "Не нашёл подходящего факта."])

        if cmd == "setkey":
            # /setkey <name> <value>  — store an API key under a short name
            parts = args.split(None, 1)
            if len(parts) != 2:
                return RouteDecision(replies=["Usage: /setkey <name> <api_key>\nExample: /setkey openweathermap abc123"])
            key_name, key_value = parts[0].lower(), parts[1].strip()
            uid = message.user_id or 0
            from ora_v2.memory.models import UserSecret
            await self._store.set_secret(UserSecret(user_id=uid, key=key_name, value=key_value))
            return RouteDecision(replies=[f"Ключ «{key_name}» сохранён."])

        if cmd == "delkey":
            key_name = args.strip().lower()
            if not key_name:
                return RouteDecision(replies=["Usage: /delkey <name>"])
            uid = message.user_id or 0
            deleted = await self._store.delete_secret(uid, key_name)
            return RouteDecision(replies=[f"Ключ «{key_name}» удалён." if deleted else f"Ключ «{key_name}» не найден."])

        if cmd == "keys":
            uid = message.user_id or 0
            keys = await self._store.list_secret_keys(uid)
            if not keys:
                return RouteDecision(replies=["Нет сохранённых ключей."])
            return RouteDecision(replies=["Сохранённые ключи:\n" + "\n".join(f"• {k}" for k in keys)])

        if cmd == "reset":
            session.mode = "chat"
            session.active_task = ""
            session.summary = ""
            session.agent_status = "idle"
            await self._store.save_session(session)
            await self._store.clear_turns(chat_id)
            return RouteDecision(replies=["Session cleared."])

        return RouteDecision(replies=[f"Unknown command: /{cmd}"])

    async def _start_agent(
        self,
        goal: str,
        context: str,
        message: IncomingMessage,
        session: Session,
        on_progress,
        on_finish,
    ) -> RouteDecision:
        chat_id = message.chat_id
        session.mode = "agent"
        session.active_task = goal
        session.agent_status = "running"
        await self._store.save_session(session)

        task = AgentTask(
            chat_id=chat_id,
            user_id=message.user_id,
            goal=goal,
            context=context,
            on_progress=on_progress,
            on_finish=on_finish,
        )
        await self._queue.enqueue(task)

        if self._on_agent_start:
            await self._on_agent_start(chat_id, goal)

        return RouteDecision(
            replies=["Запускаю исследование..."],
            start_agent=True,
            agent_goal=goal,
        )

    async def _chat_reply(self, text: str, session: Session, message: IncomingMessage) -> str:
        from ora_v2.chat.engine import ChatEngine
        engine = ChatEngine()
        profile = await self._store.get_profile(message.user_id or 0)
        turns = await self._store.get_turns(message.chat_id, limit=8)
        return await engine.reply(text, session=session, profile=profile, turns=turns)

    def _help_text(self) -> str:
        return (
            "ORA — умный ассистент-исследователь.\n\n"
            "/task <задача> — запустить исследование\n"
            "/stop — остановить агента\n"
            "/status — статус агента\n"
            "/memory — память сессии\n"
            "/profile — профиль пользователя\n"
            "/remember <факт> — запомнить факт\n"
            "/forget <факт> — удалить факт из памяти\n"
            "/reset — очистить сессию\n"
            "/setkey <name> <key> — сохранить API-ключ\n"
            "/delkey <name> — удалить API-ключ\n"
            "/keys — список сохранённых ключей"
        )
