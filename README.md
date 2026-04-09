[![DBR](https://img.shields.io/badge/DBR-16.x-red?logo=databricks&style=for-the-badge)](https://docs.databricks.com/release-notes/runtime/16.0.html)
[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&style=for-the-badge)](https://www.python.org/)

# Telco Customer Support AI Agent

A production-ready **AI agent for telecom customer support** built on Databricks, featuring long-term memory, Genie-powered data queries, Unity Catalog tools, vector search RAG, and MLflow LLM-as-Judge evaluation — deployed as a **Databricks App** with a real-time streaming chat UI.

---

## Architecture

```
                          +------------------+
                          |   Databricks App |
                          |  (FastAPI + SSE) |
                          +--------+---------+
                                   |
                    +--------------+--------------+
                    |                             |
            +-------v-------+           +--------v--------+
            |  LangGraph    |           |    Lakebase      |
            |  Agent        |           |  (Chat History + |
            |  (Claude 3.7  |           |   Long-Term      |
            |   Sonnet)     |           |   Memory)        |
            +-------+-------+           +-----------------+
                    |
     +--------------+--------------+
     |              |              |
+----v----+  +------v------+  +---v-----------+
|  Genie  |  |   Vector    |  | UC Functions  |
| Agents  |  |   Search    |  | (Customer     |
| (Account|  | (Telco KB   |  |  Lookup,      |
| +Billing)|  |  RAG)      |  |  Billing,     |
+---------+  +-------------+  |  Math)        |
     |                        +---------------+
     |
+----v-----------+         +------------------+
| Unity Catalog  |         |  MLflow Tracing  |
| (Customer &    |         |  + LLM-as-Judge  |
|  Billing Data) |         |  (5 scorers)     |
+----------------+         +------------------+
```

---

## Features

- **6 tool types** — UC functions, Genie agents, vector search, semantic memory, MCP servers, utilities
- **Long-term memory** — per-user semantic memory via `AsyncDatabricksStore` (Lakebase + embeddings)
- **Genie integration** — natural language to SQL for account and billing data
- **MLflow observability** — end-to-end tracing with UC destination, 5 automated scorers, user feedback
- **Streaming UI** — dark-themed chat interface with thinking/planning visualization and assessment pills
- **DABs deployment** — single-command deploy with dev/prod targets

---

## Project Structure

```
ai-agent-apps/
├── agent_server/
│   ├── agent.py                # Core agent: tools, system prompt, @invoke/@stream handlers
│   ├── start_server.py         # FastAPI server + embedded chat UI + SSE streaming
│   ├── db.py                   # Lakebase chat history persistence
│   ├── scorers.py              # 5 MLflow scorers (3 built-in + 2 custom LLM judges)
│   ├── utils.py                # Stream processing, auth, session helpers
│   ├── utils_memory.py         # Lakebase-backed memory tools (get/save/delete)
│   └── evaluate_agent.py       # Offline batch evaluation CLI
│
├── notebooks/
│   ├── 01-create-tools/        # Create UC functions (SQL + Python)
│   ├── 02-agent-eval/          # Agent definition & evaluation
│   ├── 03-knowledge-base-rag/  # PDF parsing & vector search index setup
│   ├── 04-offline-eval/        # Batch evaluation runs
│   ├── 05-setup-uc-trace-location.py  # MLflow UC trace integration
│   ├── config.py               # Shared config (catalog, schema, endpoint)
│   └── _resources/             # Data generation & setup helpers
│
├── scripts/
│   └── start_app.py            # Combined frontend + backend launcher
│
├── databricks.yml              # DAB config (app, job, experiment, targets)
├── app.yaml                    # Databricks App runtime config + env vars
├── pyproject.toml              # Dependencies & entry points
└── .env.example                # Environment variable template
```

---

## Tools

| Tool | Type | Description |
|------|------|-------------|
| `get_customer_by_email` | UC Function | Look up customer profile by email |
| `get_customer_billing_and_subscriptions` | UC Function | Retrieve billing and subscription data |
| `calculate_math_expression` | UC Function | Evaluate math expressions |
| `account_genie` | Genie Agent | Natural language queries over customer accounts |
| `billing_genie` | Genie Agent | Natural language queries over billing data |
| `knowledge_base_search` | Vector Search | RAG over telco documentation (routers, error codes) |
| `get_user_memory` | Memory (Lakebase) | Semantic search over stored user preferences |
| `save_user_memory` | Memory (Lakebase) | Persist user context for future conversations |
| `delete_user_memory` | Memory (Lakebase) | Remove specific user memories |
| `get_current_time` | Utility | Current date/time awareness |

---

## MLflow Evaluation

### Real-time Scorers (background, per-request)

| Scorer | Type | What it Measures |
|--------|------|-----------------|
| **RetrievalGroundedness** | Built-in | Is the answer grounded in retrieved data? |
| **RelevanceToQuery** | Built-in | Does the answer address the question? |
| **Safety** | Built-in | No harmful or inappropriate content? |
| **steps_and_reasoning** | Custom LLM Judge | Does the response avoid exposing internal tool mechanics? |
| **retrieval_quality** | Custom LLM Judge | Is the response rich with specific retrieved data? |

### Additional Capabilities

- **User feedback** — thumbs up/down with optional comments, logged as MLflow assessments
- **Offline batch evaluation** — `uv run agent-evaluate` runs all 5 scorers across collected traces
- **UC trace destination** — traces stored in Unity Catalog for governance and lineage

---

## Getting Started

### Prerequisites

- Databricks workspace with Unity Catalog
- Databricks CLI configured with a workspace profile
- `uv` package manager (or `pip`)

### 1. Configure Environment

Copy `.env.example` to `.env` and set your values:

```env
LLM_ENDPOINT_NAME=databricks-claude-3-7-sonnet
UC_TOOL_NAMES=your_catalog.your_schema.*
VECTOR_SEARCH_INDEX=your_catalog.your_schema.knowledge_base_vs_index
EMBEDDING_ENDPOINT=databricks-gte-large-en
GENIE_ACCOUNT_SPACE_ID=<your-genie-space-id>
GENIE_BILLING_SPACE_ID=<your-genie-space-id>
LAKEBASE_AUTOSCALING_BRANCH=<your-lakebase-branch>
```

### 2. Deploy with DABs

```bash
# Deploy app + job + experiment
databricks bundle deploy --target dev

# Run the data setup pipeline (creates UC functions, eval, vector search)
databricks bundle run ai-agent-data-setup --target dev
```

### 3. Run Locally (Optional)

```bash
uv run start-server
```

---

## DABs Resources

| Resource | Name | Description |
|----------|------|-------------|
| **App** | `ai-agent-telco` | FastAPI + Uvicorn chat application |
| **Job** | `ai-agent-data-setup` | 3-task pipeline: UC tools -> Agent eval -> Knowledge base |
| **Experiment** | `ai-agent-telco-app` | MLflow experiment for tracing & evaluation |

Targets: `dev` (personalized, default) and `prod` (shared).

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Claude 3.7 Sonnet (Databricks Foundation Model API) |
| Agent Framework | LangGraph |
| Tools | Unity Catalog Functions, Genie Agents, Vector Search |
| Memory | AsyncDatabricksStore (Lakebase + GTE-Large-EN embeddings) |
| Observability | MLflow Tracing + LLM-as-Judge (UC destination) |
| App Server | FastAPI + Uvicorn (Databricks Apps) |
| Chat Persistence | Lakebase (managed PostgreSQL) |
| Deployment | Databricks Asset Bundles (DABs) |
| Streaming | Server-Sent Events (SSE) |

---

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `01-create-tools/01_create_first_billing_agent.py` | Create UC SQL/Python functions for customer lookup & billing |
| `02-agent-eval/02.1_agent_evaluation.py` | Agent creation, config, and evaluation |
| `03-knowledge-base-rag/03.1-pdf-rag-tool.py` | Parse PDFs, build knowledge base, create vector search index |
| `04-offline-eval/` | Batch evaluation with MLflow scorers |
| `05-setup-uc-trace-location.py` | Configure MLflow trace destination in Unity Catalog |

---

## Authors

- Tian Tan (tian.tan@databricks.com)

Based on the [dbdemos ai-agent demo](https://www.dbdemos.ai/) and the Databricks `agent-langgraph-long-term-memory` app template.
