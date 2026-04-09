# AI Agent - Telco Customer Support (App Template Format)

## Overview

This is a telco customer support AI agent ported from the dbdemos `ai-agent` demo into the
Databricks `agent-langgraph-long-term-memory` app template format.

## Architecture

```
User Request -> Databricks App (OAuth) -> FastAPI (/invocations)
  -> @stream() handler -> AsyncDatabricksStore (Lakebase)
  -> create_agent() with ChatDatabricks + tools
  -> LangGraph execution (tool calls, LLM reasoning)
  -> process_agent_astream_events() -> ResponsesAgentStreamEvent
  -> SSE stream back to client
```

## Key Files

- `agent_server/agent.py` — Core agent definition with `@invoke()`/`@stream()` handlers
- `agent_server/utils.py` — Stream processing and auth utilities
- `agent_server/utils_memory.py` — Lakebase-backed memory tools (get/save/delete)
- `agent_server/start_server.py` — MLflow AgentServer entry point
- `agent_server/evaluate_agent.py` — Evaluation with MLflow scorers
- `app.yaml` — Databricks App configuration
- `databricks.yml` — DAB bundle config (app, experiment, Lakebase)
- `pyproject.toml` — Python dependencies and entry points

## Tools

1. **Unity Catalog Functions** — `get_customer_by_email`, `get_customer_billing_and_subscriptions`, `calculate_math_expression`
2. **Vector Search** — Knowledge base search over telco documentation (optional, set `VECTOR_SEARCH_INDEX`)
3. **Memory Tools** — `get_user_memory`, `save_user_memory`, `delete_user_memory` (Lakebase-backed)
4. **MCP Tools** — External MCP server integration (optional, set `MCP_SERVER_URLS`)
5. **get_current_time** — Simple utility tool

## Configuration

All configuration is via environment variables (see `.env.example`). Key settings:
- `LLM_ENDPOINT_NAME` — Model endpoint (default: `databricks-claude-sonnet-4-6`)
- `UC_TOOL_NAMES` — Comma-separated UC function patterns
- `VECTOR_SEARCH_INDEX` — Vector Search index name (empty to disable)
- `LAKEBASE_INSTANCE_NAME` or `LAKEBASE_AUTOSCALING_PROJECT`/`BRANCH` — Memory backend

## Deployment

```bash
# Deploy with Databricks Asset Bundles
databricks bundle deploy --target dev

# Or run locally
uv run start-server
```
