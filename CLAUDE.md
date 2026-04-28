# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ORA V2 is a complete rewrite of an autonomous research agent. The codebase does not yet exist — this repository contains only the technical specification (`ORA_V2_SPEC.md`). V2 fixes critical security and reliability issues from V1 and establishes a horizontally-scalable architecture.

## Implementation Order (from spec)

Follow this priority sequence when building out the system:

1. **P0** — `llm/client.py` (retry + circuit breaker) → unit tests
2. **P0** — `skills/sandbox.py` (AST auditor) → tests
3. **P0** — Prompt injection protection (`<user_input>` wrapping throughout)
4. **P1** — `memory/store.py` (SQLite via aiosqlite) + JSON→SQLite migration script
5. **P1** — Pydantic schemas for all LLM responses (`OrchestratorPlan`, `SubagentDecision`)
6. **P1** — `agent/orchestrator.py` + `agent/subagent.py` → integration tests with mock LLM
7. **P2** — `transport/telegram.py` + rate limiter
8. **P2** — `agent/task_queue.py` + `agent/worker_pool.py` (local backend first)
9. **P3** — Redis queue backend, metrics

## Module Structure

```
ora_v2/
├── transport/      # TransportAdapter ABC, TelegramTransport, CLITransport
├── router/         # MessageRouter, LLM-based classifier, CommandParser
├── agent/          # OrchestratorAgent, SubagentWorker, TaskQueue, WorkerPool
├── skills/         # Skill ABC + SkillRegistry, sandbox, skill_writer, builtins/
├── llm/            # LLMClient, providers (deepseek, openai), Pydantic schemas
├── memory/         # MemoryStore (SQLite/aiosqlite), Pydantic models, Alembic migrations
├── chat/           # ChatEngine (fast replies), SearchPolicy
├── config.py       # pydantic-settings Settings class
├── metrics.py      # Prometheus counters/histograms
└── tests/
    ├── unit/
    ├── integration/
    └── fixtures/   # MockLLMClient, MockSkillRegistry, InMemoryMemoryStore
```

## Architecture Principles

**Layer isolation:** Transport knows nothing about agents. Router knows nothing about LLM. Each layer communicates only through defined interfaces.

**All external calls go through LLMClient** — single entry point with retry/circuit-breaker. Never call LLM APIs directly.

**All persistent state lives in storage, not in process memory.** Use `MemoryStore` (SQLite) for all sessions, turns, profiles, facts, documents, agent runs.

**Message flow:**  
`Transport` → `IncomingMessage` → `MessageRouter` → (ChatEngine or TaskQueue) → `WorkerPool` → `OrchestratorAgent` → `SubagentWorker` → Skills

## Key Interfaces

### LLMClient
```python
async def call(self, prompt: str, *, tier: ModelTier = ModelTier.FLASH, timeout: int = 60, caller: str = "") -> str
async def call_json(self, prompt: str, schema: type[BaseModel], *, tier: ModelTier = ModelTier.FLASH, timeout: int = 60, caller: str = "") -> BaseModel
```
Retry: 3 attempts, backoff 1s/4s/16s. Circuit breaker: opens after 5 consecutive errors for 60s. `call_json` retries parsing up to 2 times; on the second failed parse it automatically escalates to `PRO` regardless of the passed `tier`.

**Model routing — never pass model strings directly, always use `tier`:**

| Tier | Model | Used for |
|------|-------|----------|
| `FLASH` (default) | `deepseek-v4-flash` | Router classification, chat replies, subagent steps, JSON formatting |
| `PRO` | `deepseek-v4-pro` | Orchestrator planning, final reports, `write_skill` code generation, `call_json` fallback after 2nd parse failure |

Config keys: `llm_model_flash = "deepseek-v4-flash"`, `llm_model_pro = "deepseek-v4-pro"`.

### MemoryStore
All async. Uses `aiosqlite`. Schema managed via Alembic migrations. Documents stored separately from session (own table, session holds only references).

### OrchestratorAgent
```python
async def run(self, goal: str, context: str = "", chat_id: int | None = None, on_progress: Callable[[str], Awaitable[None]] | None = None) -> AgentResult
```
Findings passed between subagents as `list[Finding]` (structured, with `confidence` and `sources`), not a single accumulated string.

### Skill ABC
Each skill must define `name` (snake_case), `description` (one line for LLM), and `args_schema` (Pydantic model for argument validation).

## Security Requirements (non-negotiable)

**Prompt injection:** All user input must be wrapped: `<user_input>{sanitized}</user_input>`. Prompts must explicitly instruct the model to ignore instructions inside this tag.

**Skill sandbox (two levels):**
1. AST static analysis before any LLM-generated code is written — block `os`, `sys`, `subprocess`, `socket`, `importlib`, `ctypes`, `eval`, `exec`, `__import__`, `open`, `os.environ`
2. Execution in subprocess with restricted `sys.path`, no parent env vars, 30s timeout

**SSRF:** `validate_url()` must block localhost, 127.x, 10.x, 172.16-31.x, 192.168.x, 169.254.x (AWS metadata), IPv6 loopback/private before any HTTP fetch.

**Rate limiting:** Default 10 agent tasks per user per hour.

## Configuration

Single `Settings` class in `config.py` using `pydantic-settings`. Load from `.env` or environment variables. The app must fail fast with a clear error at startup if required keys are missing — before any logic runs.

Required keys: `deepseek_api_key`, `telegram_bot_token`, `brave_api_key`.

## Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# All tests
pytest

# Single test file
pytest tests/unit/test_llm_retry.py -v

# With coverage
pytest --cov=ora_v2 --cov-report=term-missing
```

Tests use `pytest-asyncio` for async tests. The `InMemoryMemoryStore` fixture uses SQLite `":memory:"` — never write tests that hit the real database.

## What is Out of Scope for V2

Do not implement: parallel subagent execution, embedding-based memory retrieval, multi-modal input, Web UI / REST API, Kubernetes deployment. These are explicitly deferred to V3.
