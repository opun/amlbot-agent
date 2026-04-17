# Exec plan: Tracer performance optimization (completed)

## Objective

Reduce per-trace latency, CPU, and log volume: logging, HTTP tuning, optional parallelism, tx_list caps, prod reload/tracing gates.

## Changes shipped

| Area | Change |
|------|--------|
| [src/agent/base_tracer.py](../../src/agent/base_tracer.py) | `logger.propagate = False`; TXS/TXLIST DEBUG-only; OpenAI httpx keepalive + HTTP/2; `AGENT_PARALLEL_TOOL_CALLS` + `AGENT_MAX_CONCURRENT_TOOLS`; `AGENT_PARALLEL_HOPS` + `AGENT_HOP_FANOUT`; `MAX_TX_LIST` collection + `_cap_visualization_tx_lists` |
| [src/agent/api.py](../../src/agent/api.py) | `AGENT_DISABLE_OPENAI_TRACING` → `set_tracing_disabled`; `AGENT_RELOAD` for uvicorn |
| [src/agent/mcp_http_client.py](../../src/agent/mcp_http_client.py) | httpx pool limits + HTTP/2 |
| [pyproject.toml](../../pyproject.toml) | `h2` dependency |
| [tests/test_cap_visualization_tx_lists.py](../../tests/test_cap_visualization_tx_lists.py) | Caps helper regression tests |

## Environment variables

| Variable | Effect |
|----------|--------|
| `AGENT_DISABLE_OPENAI_TRACING=1` | Disables OpenAI Agents SDK trace ingest |
| `AGENT_RELOAD=false` | Disables uvicorn `--reload` (production) |
| `AGENT_PARALLEL_TOOL_CALLS=1` | Parallel orchestrator tool calls per turn (default off) |
| `AGENT_MAX_CONCURRENT_TOOLS` | Semaphore limit when parallel tools (default 3) |
| `AGENT_PARALLEL_HOPS=1` | Batch hop phase-1 MCP calls (default off) |
| `AGENT_HOP_FANOUT` | Max hops per scheduler iteration when parallel (default 3, max 16) |

## Verification

```bash
cd amlbot-agent && uv lock && uv run pytest tests/ -q
```

**Outcome:** `56 passed` (54 existing + 2 new cap tests), `uv run ruff check` clean on touched paths.

## Residual risk

- Parallel tool/hop paths need load testing against MCP server concurrency.
- HTTP/2 + `h2` adds a dependency; if deployment blocks it, set `http2=False` locally (not implemented as env flag).
