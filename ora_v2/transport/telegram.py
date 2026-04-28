"""TelegramTransport: python-telegram-bot async adapter."""
from __future__ import annotations

import asyncio
import logging
import tempfile
import os
from pathlib import Path
from typing import Awaitable, Callable

from telegram import Update, Document as TGDocument
from telegram.ext import Application, ContextTypes, MessageHandler, filters

from ora_v2.transport.base import TransportAdapter
from ora_v2.transport.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

_SUPPORTED_MIME = {
    "application/pdf",
    "text/plain",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/msword",
}
_SUPPORTED_EXT = {".pdf", ".txt", ".md", ".docx", ".doc"}
_MAX_MESSAGE_LEN = 3900


def _split_message(text: str) -> list[str]:
    if len(text) <= _MAX_MESSAGE_LEN:
        return [text]
    return [text[i: i + _MAX_MESSAGE_LEN] for i in range(0, len(text), _MAX_MESSAGE_LEN)]


class TelegramTransport(TransportAdapter):
    def __init__(
        self,
        token: str,
        allowed_chat_ids: set[int],
        rate_limiter: RateLimiter,
        on_message: Callable[[int, int | None, str, dict], Awaitable[None]],
        on_document: Callable[[int, int | None, str, bytes, str], Awaitable[None]] | None = None,
    ) -> None:
        self._token = token
        self._allowed = allowed_chat_ids
        self._rl = rate_limiter
        self._on_message = on_message
        self._on_document = on_document
        self._app: Application | None = None
        self._send_lock: dict[int, asyncio.Lock] = {}

    async def start(self) -> None:
        self._app = Application.builder().token(self._token).build()
        self._app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_text))
        self._app.add_handler(MessageHandler(filters.COMMAND, self._handle_text))
        self._app.add_handler(MessageHandler(filters.Document.ALL, self._handle_document))
        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(drop_pending_updates=True)
        logger.info("TelegramTransport started")

    async def stop(self) -> None:
        if self._app:
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()
        logger.info("TelegramTransport stopped")

    async def send_message(self, chat_id: int, text: str) -> None:
        if not self._app:
            return
        if chat_id not in self._send_lock:
            self._send_lock[chat_id] = asyncio.Lock()
        async with self._send_lock[chat_id]:
            for chunk in _split_message(text):
                try:
                    await self._app.bot.send_message(chat_id=chat_id, text=chunk)
                    await asyncio.sleep(1.0)  # Telegram rate limit: 1 msg/s per chat
                except Exception as exc:
                    logger.error("send_message error chat_id=%d: %s", chat_id, exc)

    async def _handle_text(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message or not update.effective_chat:
            return
        chat_id = update.effective_chat.id
        user = update.effective_user
        user_id = user.id if user else None
        text = (update.message.text or "").strip()

        if self._allowed and chat_id not in self._allowed:
            await self.send_message(chat_id, f"Chat {chat_id} is not allowed.")
            return

        if not text:
            return

        await self._on_message(chat_id, user_id, text, update.to_dict())

    async def _handle_document(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message or not self._on_document:
            return
        chat_id = update.effective_chat.id if update.effective_chat else 0
        user = update.effective_user
        user_id = user.id if user else None
        doc: TGDocument | None = update.message.document
        if not doc:
            return

        ext = Path(doc.file_name or "").suffix.lower()
        mime = doc.mime_type or ""
        if ext not in _SUPPORTED_EXT and mime not in _SUPPORTED_MIME:
            await self.send_message(chat_id, "Формат не поддерживается. Отправь PDF, TXT или DOCX.")
            return

        await self.send_message(chat_id, "Читаю документ…")
        tmp_path: str | None = None
        try:
            file = await ctx.bot.get_file(doc.file_id)
            with tempfile.NamedTemporaryFile(suffix=ext or ".bin", delete=False) as tmp:
                tmp_path = tmp.name
            await file.download_to_drive(tmp_path)
            data = Path(tmp_path).read_bytes()
        except Exception as exc:
            await self.send_message(chat_id, f"Ошибка загрузки файла: {exc}")
            return
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

        caption = (update.message.caption or "").strip()
        await self._on_document(chat_id, user_id, doc.file_name or "document", data, caption)
