from __future__ import annotations
from typing import Literal
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # LLM
    deepseek_api_key: str
    llm_model_flash: str = "deepseek-chat"
    llm_model_pro: str = "deepseek-reasoner"
    llm_timeout: int = 60
    llm_max_retries: int = 3
    llm_circuit_breaker_threshold: int = 5

    # Telegram
    telegram_bot_token: str
    telegram_allowed_chat_ids: list[int] = []
    telegram_rate_limit_per_hour: int = 0

    # Agent
    agent_max_iterations: int = 20
    subagent_max_steps: int = 8
    subagent_timeout_seconds: int = 120
    worker_pool_size: int = 3

    # Skills
    skill_writer_require_approval: bool = False
    skill_writer_enabled: bool = True

    # Storage
    db_path: str = "ora_v2.db"
    queue_backend: Literal["local", "redis"] = "local"
    redis_url: str = "redis://localhost:6379"

    # Search
    brave_api_key: str

    # Skill HTTP allowlist — if non-empty, generated skills may only contact these hostnames.
    # Exact match or suffix match (e.g. "openweathermap.org" covers "api.openweathermap.org").
    # Empty list means no restriction (useful for local dev; set a list in production).
    skill_allowed_hosts: list[str] = []

    @field_validator("skill_allowed_hosts", mode="before")
    @classmethod
    def parse_allowed_hosts(cls, v: object) -> list[str]:
        if isinstance(v, str):
            return [h.strip().lower() for h in v.split(",") if h.strip()]
        return [h.lower() for h in v] if v else []

    # Observability
    log_level: str = "INFO"
    metrics_enabled: bool = False
    metrics_port: int = 9090

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("telegram_allowed_chat_ids", mode="before")
    @classmethod
    def parse_chat_ids(cls, v: object) -> list[int]:
        if isinstance(v, str):
            return [int(x.strip()) for x in v.split(",") if x.strip()]
        return list(v) if v else []


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
