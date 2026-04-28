"""TransportAdapter ABC."""
from __future__ import annotations
from abc import ABC, abstractmethod


class TransportAdapter(ABC):
    @abstractmethod
    async def start(self) -> None: ...

    @abstractmethod
    async def stop(self) -> None: ...

    @abstractmethod
    async def send_message(self, chat_id: int, text: str) -> None: ...
