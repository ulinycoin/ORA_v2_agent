"""LLM client with retry, circuit breaker, and two-tier model routing."""
from __future__ import annotations

import asyncio
import json
import logging
import time
from enum import Enum
from typing import TYPE_CHECKING, Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_TYPE_EXAMPLES: dict[str, str] = {
    "string": '"example text"',
    "number": "0.9",
    "integer": "1",
    "boolean": "true",
    "array": "[]",
    "object": "{}",
    "null": "null",
}


def _build_schema_hint(schema: type[BaseModel]) -> str:
    """Return a compact field list with types and example values instead of full JSON Schema."""
    js = schema.model_json_schema()
    props = js.get("properties", {})
    required = set(js.get("required", []))
    lines = []
    for field_name, field_info in props.items():
        ftype = field_info.get("type", "any")
        if isinstance(ftype, list):
            ftype = " | ".join(ftype)
        example = _TYPE_EXAMPLES.get(ftype, '"..."')
        req_marker = " (required)" if field_name in required else " (optional)"
        lines.append(f'  "{field_name}": {example}  // {ftype}{req_marker}')
    return "{\n" + ",\n".join(lines) + "\n}"

_BACKOFF = [1, 4, 16]  # seconds between retries


class ModelTier(str, Enum):
    FLASH = "flash"
    PRO = "pro"


class CircuitOpenError(Exception):
    pass


class _CircuitBreaker:
    def __init__(self, threshold: int, reset_seconds: int = 60) -> None:
        self._threshold = threshold
        self._reset = reset_seconds
        self._failures = 0
        self._opened_at: float | None = None

    def is_open(self) -> bool:
        if self._opened_at is None:
            return False
        if time.monotonic() - self._opened_at >= self._reset:
            self._failures = 0
            self._opened_at = None
            return False
        return True

    def record_success(self) -> None:
        self._failures = 0
        self._opened_at = None

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self._threshold:
            self._opened_at = time.monotonic()
            logger.warning("LLM circuit breaker opened after %d failures", self._failures)


class LLMClient:
    def __init__(
        self,
        api_key: str,
        model_flash: str,
        model_pro: str,
        api_url: str = "https://api.deepseek.com/chat/completions",
        max_retries: int = 3,
        circuit_breaker_threshold: int = 5,
    ) -> None:
        self._api_key = api_key
        self._model_flash = model_flash
        self._model_pro = model_pro
        self._api_url = api_url
        self._max_retries = max_retries
        self._cb = _CircuitBreaker(circuit_breaker_threshold)
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        """Lazy-init shared client for connection pooling across all calls."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
            )
        return self._client

    def _model_for(self, tier: ModelTier) -> str:
        return self._model_pro if tier == ModelTier.PRO else self._model_flash

    async def _post(self, payload: dict[str, Any], timeout: int, *, system: str | None = None) -> str:
        if self._cb.is_open():
            raise CircuitOpenError("LLM circuit breaker is open — not sending request")

        if system:
            payload["messages"] = [
                {"role": "system", "content": system},
                *payload.get("messages", []),
            ]

        client = self._get_client()
        t0 = time.monotonic()
        try:
            resp = await client.post(
                self._api_url,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self._api_key}",
                },
                timeout=timeout,
            )
            resp.raise_for_status()
            result = resp.json()
            content = result["choices"][0]["message"]["content"]
            latency = int((time.monotonic() - t0) * 1000)
            usage = result.get("usage", {})
            logger.debug(
                "LLM call ok model=%s latency=%dms prompt_tokens=%s completion_tokens=%s",
                payload.get("model"),
                latency,
                usage.get("prompt_tokens", "?"),
                usage.get("completion_tokens", "?"),
            )
            self._cb.record_success()
            return content
        except httpx.HTTPStatusError as e:
            body = e.response.text
            self._cb.record_failure()
            raise RuntimeError(f"DeepSeek API error {e.response.status_code}: {body}") from e
        except httpx.TimeoutException as e:
            self._cb.record_failure()
            raise RuntimeError(f"LLM API request timed out after {timeout}s") from e
        except httpx.ConnectError as e:
            self._cb.record_failure()
            raise RuntimeError(f"Network error reaching LLM API: {e}") from e
        except (KeyError, IndexError) as e:
            self._cb.record_failure()
            raise RuntimeError(f"Unexpected API response structure: {e}") from e

    async def call(
        self,
        prompt: str,
        *,
        tier: ModelTier = ModelTier.FLASH,
        timeout: int = 60,
        caller: str = "",
        system: str | None = None,
    ) -> str:
        model = self._model_for(tier)
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
        }
        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                return await self._post(payload, timeout, system=system)
            except CircuitOpenError:
                raise
            except Exception as exc:
                last_exc = exc
                if attempt < self._max_retries - 1:
                    await asyncio.sleep(_BACKOFF[attempt])
                    logger.warning("LLM retry %d/%d caller=%s error=%s", attempt + 1, self._max_retries, caller, exc)
        raise RuntimeError(f"LLM call failed after {self._max_retries} attempts: {last_exc}") from last_exc

    async def call_json(
        self,
        prompt: str,
        schema: type[T],
        *,
        tier: ModelTier = ModelTier.FLASH,
        timeout: int = 60,
        caller: str = "",
        system: str | None = None,
    ) -> T:
        schema_hint = _build_schema_hint(schema)
        base_prompt = (
            prompt
            + f"\n\nRespond with a single JSON object (no markdown fences, no explanation).\n"
            f"Fill in actual values — do NOT return the schema or type descriptions.\n"
            f"Required fields and their types:\n{schema_hint}"
        )

        last_raw = ""
        last_exc: Exception | None = None
        for parse_attempt in range(3):
            effective_tier = ModelTier.PRO if parse_attempt == 2 else tier
            if parse_attempt > 0:
                logger.warning("call_json parse retry %d caller=%s tier=%s", parse_attempt, caller, effective_tier)
                full_prompt = (
                    f"Your previous response could not be parsed as valid JSON or was missing required fields.\n"
                    f"Error: {last_exc}\n"
                    f"Previous response:\n{last_raw}\n\n"
                    f"Original request:\n{prompt}\n\n"
                    f"Return a single flat JSON object with these fields filled with real values:\n{schema_hint}"
                )
            else:
                full_prompt = base_prompt

            raw = await self.call(full_prompt, tier=effective_tier, timeout=timeout, caller=caller, system=system)
            last_raw = raw

            clean = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            try:
                data = json.loads(clean)
                return schema.model_validate(data)
            except (json.JSONDecodeError, ValidationError) as exc:
                last_exc = exc
                if parse_attempt == 2:
                    raise RuntimeError(f"call_json failed after 3 attempts: {exc}\nLast raw: {raw[:500]}") from exc

        raise RuntimeError("call_json: unreachable")

    async def close(self) -> None:
        """Close the shared HTTP client. Idempotent. Call on shutdown."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None


_client: LLMClient | None = None


def get_client() -> LLMClient:
    global _client
    if _client is None:
        from ora_v2.config import get_settings
        s = get_settings()
        _client = LLMClient(
            api_key=s.deepseek_api_key,
            model_flash=s.llm_model_flash,
            model_pro=s.llm_model_pro,
            max_retries=s.llm_max_retries,
            circuit_breaker_threshold=s.llm_circuit_breaker_threshold,
        )
    return _client
