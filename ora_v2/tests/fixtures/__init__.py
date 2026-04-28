"""Test fixtures: MockLLMClient, InMemoryMemoryStore, MockSkill."""
from __future__ import annotations
from typing import Any
from pydantic import BaseModel
from ora_v2.llm.client import LLMClient, ModelTier
from ora_v2.memory.store import MemoryStore
from ora_v2.skills.base import Skill


class MockLLMClient(LLMClient):
    """Returns pre-programmed responses in sequence."""

    def __init__(self, responses: list[str | BaseModel]) -> None:
        self._responses = list(responses)
        self._index = 0

    def _next(self) -> str | BaseModel:
        if self._index >= len(self._responses):
            raise RuntimeError("MockLLMClient ran out of responses")
        val = self._responses[self._index]
        self._index += 1
        return val

    async def call(self, prompt: str, *, tier: ModelTier = ModelTier.FLASH, timeout: int = 60, caller: str = "", **kwargs) -> str:
        val = self._next()
        if isinstance(val, str):
            return val
        return val.model_dump_json()

    async def call_json(self, prompt: str, schema: type, *, tier: ModelTier = ModelTier.FLASH, timeout: int = 60, caller: str = "", **kwargs) -> Any:
        import json
        val = self._next()
        if isinstance(val, BaseModel):
            return val
        data = json.loads(val) if isinstance(val, str) else val
        return schema.model_validate(data)


class InMemoryMemoryStore(MemoryStore):
    def __init__(self) -> None:
        super().__init__(db_path=":memory:")


class MockSkill(Skill):
    def __init__(self, name: str, response: str = "ok") -> None:
        from pydantic import BaseModel as BM

        class _Args(BM):
            pass

        self.name = name
        self.description = f"mock skill {name}"
        self.args_schema = _Args
        self._response = response

    def run(self, **kwargs: Any) -> str:
        return self._response
