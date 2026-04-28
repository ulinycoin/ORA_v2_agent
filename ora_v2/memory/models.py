from __future__ import annotations
from datetime import datetime, timezone
from pydantic import BaseModel, Field


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Turn(BaseModel):
    id: int | None = None
    chat_id: int
    role: str
    kind: str = "chat"
    text: str
    ts: str = Field(default_factory=_utc_now)


class Session(BaseModel):
    chat_id: int
    user_id: int | None = None
    mode: str = "chat"
    active_task: str = ""
    summary: str = ""
    agent_status: str = "idle"
    updated_at: str = Field(default_factory=_utc_now)


class Profile(BaseModel):
    user_id: int
    username: str | None = None
    display_name: str | None = None
    summary: str = ""
    updated_at: str = Field(default_factory=_utc_now)


class Fact(BaseModel):
    id: int | None = None
    user_id: int
    text: str
    ts: str = Field(default_factory=_utc_now)


class Document(BaseModel):
    id: int | None = None
    chat_id: int
    name: str
    size_bytes: int
    storage_path: str
    ts: str = Field(default_factory=_utc_now)


class UserSecret(BaseModel):
    id: int | None = None
    user_id: int
    key: str        # e.g. "openweathermap", "alpha_vantage"
    value: str      # the actual API key — never logged or included in prompts
    ts: str = Field(default_factory=_utc_now)


class AgentRun(BaseModel):
    run_id: str
    chat_id: int | None = None
    goal: str
    status: str = "running"
    iterations: int = 0
    started_at: str = Field(default_factory=_utc_now)
    finished_at: str | None = None
