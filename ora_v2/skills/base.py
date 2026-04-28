"""Skill ABC and registry."""
from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class Skill(ABC):
    name: str
    description: str
    args_schema: type[BaseModel]

    @abstractmethod
    def run(self, **kwargs: Any) -> str: ...

    def describe(self) -> str:
        schema = self.args_schema.model_json_schema()
        props = schema.get("properties", {})
        args_str = ", ".join(
            f'"{k}": "{v.get("description", v.get("type", ""))}"'
            for k, v in props.items()
        )
        return f'- name: "{self.name}"\n  description: "{self.description}"\n  args: {{{args_str}}}'


class SkillRegistry:
    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}
        self._lock = threading.RLock()

    def register(self, skill: Skill) -> None:
        with self._lock:
            self._skills[skill.name] = skill
            logger.debug("Registered skill: %s", skill.name)

    def get(self, name: str) -> Skill | None:
        with self._lock:
            return self._skills.get(name)

    def list_all(self) -> list[Skill]:
        with self._lock:
            return list(self._skills.values())

    def describe_for_llm(self) -> str:
        with self._lock:
            return "\n".join(s.describe() for s in self._skills.values())

    def reload(self, skill: Skill) -> None:
        """Hot-replace a skill by name."""
        with self._lock:
            self._skills[skill.name] = skill
            logger.info("Hot-reloaded skill: %s", skill.name)


_registry = SkillRegistry()


def get_registry() -> SkillRegistry:
    return _registry
