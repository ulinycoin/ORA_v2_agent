"""Unit tests: call_json schema validation and retry escalation."""
from __future__ import annotations
import json
import pytest
from unittest.mock import patch
from pydantic import BaseModel
from ora_v2.llm.client import LLMClient, ModelTier


class _Schema(BaseModel):
    value: int
    label: str


def _make_client() -> LLMClient:
    return LLMClient(
        api_key="k",
        model_flash="flash",
        model_pro="pro",
        max_retries=1,
        circuit_breaker_threshold=10,
    )


@pytest.mark.asyncio
async def test_call_json_valid_response():
    client = _make_client()
    responses = [json.dumps({"value": 42, "label": "hello"})]
    idx = 0

    async def _mock_call(prompt, *, tier, timeout, caller, **kwargs):
        nonlocal idx
        r = responses[idx]
        idx += 1
        return r

    with patch.object(client, "call", side_effect=_mock_call):
        result = await client.call_json("test", _Schema, tier=ModelTier.FLASH)

    assert result.value == 42
    assert result.label == "hello"


@pytest.mark.asyncio
async def test_call_json_retries_on_bad_json():
    client = _make_client()
    call_count = 0
    tiers_used = []

    async def _mock_call(prompt, *, tier, timeout, caller, **kwargs):
        nonlocal call_count
        call_count += 1
        tiers_used.append(tier)
        if call_count < 3:
            return "not json at all"
        return json.dumps({"value": 1, "label": "ok"})

    with patch.object(client, "call", side_effect=_mock_call):
        result = await client.call_json("test", _Schema, tier=ModelTier.FLASH)

    assert result.value == 1
    assert call_count == 3
    # Third attempt must use PRO
    assert tiers_used[2] == ModelTier.PRO


@pytest.mark.asyncio
async def test_call_json_fails_after_3_attempts():
    client = _make_client()

    async def _bad(prompt, *, tier, timeout, caller, **kwargs):
        return "definitely not json"

    with patch.object(client, "call", side_effect=_bad):
        with pytest.raises(RuntimeError, match="3 attempts"):
            await client.call_json("test", _Schema, tier=ModelTier.FLASH)
