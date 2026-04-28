"""Unit tests: LLM client retry and circuit breaker."""
from __future__ import annotations
import asyncio
import pytest
from unittest.mock import patch, MagicMock
from ora_v2.llm.client import LLMClient, ModelTier, CircuitOpenError


def _make_client(**kwargs) -> LLMClient:
    defaults = dict(
        api_key="test-key",
        model_flash="flash",
        model_pro="pro",
        max_retries=3,
        circuit_breaker_threshold=3,
    )
    defaults.update(kwargs)
    return LLMClient(**defaults)


@pytest.mark.asyncio
async def test_retry_on_http_error():
    client = _make_client()
    call_count = 0

    def _bad_post(payload, timeout, **kwargs):
        nonlocal call_count
        call_count += 1
        raise RuntimeError("HTTP 500")

    with patch.object(client, "_post", side_effect=_bad_post):
        with patch("asyncio.sleep", return_value=None):
            with pytest.raises(RuntimeError, match="3 attempts"):
                await client.call("hello", tier=ModelTier.FLASH, caller="test")

    assert call_count == 3


@pytest.mark.asyncio
async def test_retry_succeeds_on_second_attempt():
    client = _make_client()
    attempts = []

    def _flaky(payload, timeout, **kwargs):
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("HTTP 429")
        return "hello"

    with patch.object(client, "_post", side_effect=_flaky):
        with patch("asyncio.sleep", return_value=None):
            result = await client.call("ping", tier=ModelTier.FLASH, caller="test")

    assert result == "hello"
    assert len(attempts) == 2


@pytest.mark.asyncio
async def test_circuit_breaker_opens():
    client = _make_client(circuit_breaker_threshold=2)

    def _fail(payload, timeout, **kwargs):
        raise RuntimeError("HTTP 500")

    with patch.object(client, "_post", side_effect=_fail):
        with patch("asyncio.sleep", return_value=None):
            with pytest.raises(RuntimeError):
                await client.call("x", caller="t1")

    # Circuit should be open now
    with pytest.raises(CircuitOpenError):
        await client.call("x", caller="t2")


@pytest.mark.asyncio
async def test_pro_tier_uses_pro_model():
    client = _make_client()
    used_models = []

    def _capture(payload, timeout, **kwargs):
        used_models.append(payload["model"])
        return "ok"

    with patch.object(client, "_post", side_effect=_capture):
        await client.call("x", tier=ModelTier.PRO, caller="test")

    assert used_models == ["pro"]


@pytest.mark.asyncio
async def test_flash_tier_uses_flash_model():
    client = _make_client()
    used_models = []

    def _capture(payload, timeout, **kwargs):
        used_models.append(payload["model"])
        return "ok"

    with patch.object(client, "_post", side_effect=_capture):
        await client.call("x", tier=ModelTier.FLASH, caller="test")

    assert used_models == ["flash"]
