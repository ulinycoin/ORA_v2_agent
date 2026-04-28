"""ORA V2 entry point."""
from __future__ import annotations

import asyncio
import logging
import signal
from pathlib import Path

from ora_v2.agent.task_queue import LocalTaskQueue
from ora_v2.agent.worker_pool import WorkerPool
from ora_v2.config import get_settings
from ora_v2.memory.store import MemoryStore
from ora_v2.router.router import IncomingMessage, MessageRouter
from ora_v2.skills.writer import load_builtins
from ora_v2.transport.rate_limiter import RateLimiter
from ora_v2.transport.telegram import TelegramTransport
from ora_v2.report_server import ReportServer, save_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def main() -> None:
    s = get_settings()

    logging.getLogger().setLevel(s.log_level.upper())

    store = MemoryStore(db_path=s.db_path)
    await store.init()

    load_builtins()

    from ora_v2.skills.base import get_registry
    registry = get_registry()

    queue = LocalTaskQueue()
    pool = WorkerPool(
        queue=queue,
        store=store,
        registry=registry,
        n_workers=s.worker_pool_size,
        max_iterations=s.agent_max_iterations,
        subagent_timeout=s.subagent_timeout_seconds,
        subagent_max_steps=s.subagent_max_steps,
        require_skill_approval=s.skill_writer_require_approval,
    )
    await pool.start()

    # Report server for downloadable HTML reports
    report_server: ReportServer | None = None
    if s.report_server_enabled:
        report_server = ReportServer(port=s.report_server_port)
        report_server.start()

    rl = RateLimiter(max_requests=s.telegram_rate_limit_per_hour, window_seconds=3600)
    allowed = set(s.telegram_allowed_chat_ids)

    send_queues: dict[int, asyncio.Queue[str]] = {}

    async def _send(chat_id: int, text: str) -> None:
        await transport.send_message(chat_id, text)

    async def _on_message(chat_id: int, user_id: int | None, text: str, raw: dict) -> None:
        if user_id and not rl.check(user_id):
            await _send(chat_id, "Слишком много задач. Попробуй через час.")
            return

        async def _progress(msg: str) -> None:
            await _send(chat_id, msg)

        async def _finish(run_id: str, report: str) -> None:
            await store.get_session(chat_id)
            sess = await store.get_session(chat_id)
            sess.agent_status = "completed"
            await store.save_session(sess)

            # Save as HTML report and send download link
            goal = sess.active_task or "исследование"
            server_url = report_server.url if report_server else ""
            if report_server and len(report) > 200:
                url = save_report(run_id, goal, report, server_url)
                await _send(chat_id, f"Исследование завершено.\nСкачать отчёт: {url}")
            else:
                await _send(chat_id, "Исследование завершено.")

            await _send(chat_id, report)

        msg = IncomingMessage(
            chat_id=chat_id,
            user_id=user_id,
            text=text,
            raw=raw,
        )
        router = MessageRouter(store=store, queue=queue)
        decision = await router.route(msg, on_progress=_progress, on_finish=_finish)
        for reply in decision.replies:
            if reply:
                await _send(chat_id, reply)

    async def _finish_doc(chat_id: int, run_id: str, report: str) -> None:
        await _finish(run_id, report)

    async def _on_document(
        chat_id: int, user_id: int | None, name: str, data: bytes, caption: str
    ) -> None:
        doc = await store.store_document(chat_id, name, data)
        question = caption or "Проанализируй этот документ и кратко изложи основное содержание."

        # Extract text in executor to avoid blocking the event loop on large files
        from ora_v2.skills.builtins.read_document import run as read_doc
        loop = asyncio.get_event_loop()
        text = await loop.run_in_executor(None, lambda: read_doc(path=doc.storage_path))

        if text.startswith("ERROR"):
            await _send(chat_id, f"Не удалось прочитать документ: {text}")
            return

        session = await store.get_session(chat_id)

        # Route caption through classifier — may be a research task, not just chat
        from ora_v2.router.classifier import classify
        turns = await store.get_turns(chat_id, limit=6)
        turn_dicts = [{"role": t.role, "text": t.text} for t in turns]
        mode, _ = await classify(question, turn_dicts)

        if mode == "agent":
            goal = f"{question}\n[документ: {name}]\n\n{text[:3000]}"
            msg = IncomingMessage(chat_id=chat_id, user_id=user_id, text=goal, raw={})
            router = MessageRouter(store=store, queue=queue)
            decision = await router.route(
                msg,
                on_progress=lambda m: _send(chat_id, m),
                on_finish=lambda rid, report: _finish_doc(chat_id, rid, report),
            )
            for reply in decision.replies:
                if reply:
                    await _send(chat_id, reply)
        else:
            from ora_v2.chat.engine import ChatEngine
            reply = await ChatEngine().reply(
                question, session=session, document_text=text, document_name=name
            )
            await _send(chat_id, reply)

    transport = TelegramTransport(
        token=s.telegram_bot_token,
        allowed_chat_ids=allowed,
        rate_limiter=rl,
        on_message=_on_message,
        on_document=_on_document,
    )

    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _handle_signal() -> None:
        logger.info("Shutdown signal received")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _handle_signal)

    await transport.start()
    logger.info("ORA V2 started")
    await stop_event.wait()

    logger.info("Shutting down...")
    await transport.stop()
    await pool.stop()
    if report_server:
        report_server.stop()
    await store.close()
    from ora_v2.llm.client import _client as llm_client
    if llm_client:
        await llm_client.close()
    logger.info("ORA V2 stopped")


if __name__ == "__main__":
    asyncio.run(main())
