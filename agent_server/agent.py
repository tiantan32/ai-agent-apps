import logging
import os
from datetime import datetime
from typing import Any, AsyncGenerator, Optional

import mlflow
from databricks.sdk import WorkspaceClient
from databricks_langchain import (
    AsyncDatabricksStore,
    ChatDatabricks,
    GenieAgent,
    UCFunctionToolkit,
    VectorSearchRetrieverTool,
    DatabricksFunctionClient,
    set_uc_function_client,
)
from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.store.base import BaseStore
from mlflow.genai.agent_server import invoke, stream
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
    to_chat_completions_input,
)

from agent_server.utils import (
    get_databricks_host_from_env,
    get_session_id,
    get_user_workspace_client,
    process_agent_astream_events,
)
from agent_server.utils_memory import (
    get_lakebase_access_error_message,
    get_user_id,
    memory_tools,
    resolve_lakebase_instance_name,
)

logger = logging.getLogger(__name__)
logging.getLogger("mlflow.utils.autologging_utils").setLevel(logging.ERROR)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
mlflow.langchain.autolog()

# Set the experiment so traces go to the right place
_experiment_id = os.getenv("MLFLOW_EXPERIMENT_ID", "")
if _experiment_id:
    try:
        mlflow.set_experiment(experiment_id=_experiment_id)
    except Exception as e:
        logger.warning(f"Failed to set experiment: {e}")

# Required to use Unity Catalog UDFs as tools
set_uc_function_client(DatabricksFunctionClient())

sp_workspace_client = WorkspaceClient()

# Ensure DATABRICKS_HOST has the https:// scheme for all SDK clients
import os as _os
_host = sp_workspace_client.config.host
if _host and not _host.startswith("http"):
    _os.environ["DATABRICKS_HOST"] = f"https://{_host}"

############################################
# Configuration
############################################
LLM_ENDPOINT_NAME = os.getenv("LLM_ENDPOINT_NAME", "databricks-claude-3-7-sonnet")
UC_TOOL_NAMES = os.getenv("UC_TOOL_NAMES", "ttan_demo_catalog_main.ai_agent_apps.*").split(",")
VECTOR_SEARCH_INDEX = os.getenv("VECTOR_SEARCH_INDEX", "")
VECTOR_SEARCH_ENDPOINT = os.getenv("VECTOR_SEARCH_ENDPOINT_NAME", "dbdemos_vs_endpoint")
MCP_SERVER_URLS = [u.strip() for u in os.getenv("MCP_SERVER_URLS", "").split(",") if u.strip()]
GENIE_ACCOUNT_SPACE_ID = os.getenv("GENIE_ACCOUNT_SPACE_ID", "01f0a40904311b03a400b5d37bca6eef")
GENIE_BILLING_SPACE_ID = os.getenv("GENIE_BILLING_SPACE_ID", "01f0a408f2d11efe8212bd582d0ee08d")

_LAKEBASE_INSTANCE_NAME_RAW = os.getenv("LAKEBASE_INSTANCE_NAME") or None
EMBEDDING_ENDPOINT = os.getenv("EMBEDDING_ENDPOINT", "databricks-gte-large-en")
EMBEDDING_DIMS = int(os.getenv("EMBEDDING_DIMS", "1024"))
LAKEBASE_AUTOSCALING_PROJECT = os.getenv("LAKEBASE_AUTOSCALING_PROJECT") or None
LAKEBASE_AUTOSCALING_BRANCH = os.getenv("LAKEBASE_AUTOSCALING_BRANCH") or None

_has_autoscaling = LAKEBASE_AUTOSCALING_BRANCH or (LAKEBASE_AUTOSCALING_PROJECT and LAKEBASE_AUTOSCALING_BRANCH)
MEMORY_ENABLED = bool(_LAKEBASE_INSTANCE_NAME_RAW or _has_autoscaling)
if not MEMORY_ENABLED:
    logger.warning(
        "No Lakebase config found. Long-term memory is disabled. "
        "Set LAKEBASE_INSTANCE_NAME or LAKEBASE_AUTOSCALING_PROJECT/BRANCH to enable."
    )

LAKEBASE_INSTANCE_NAME = (
    resolve_lakebase_instance_name(_LAKEBASE_INSTANCE_NAME_RAW)
    if _LAKEBASE_INSTANCE_NAME_RAW
    else None
)

SYSTEM_PROMPT = """You are a telco customer support assistant. You MUST use your tools to answer \
questions — never guess or make up information. Call the appropriate tool for every request.

Your tools:
- account_genie: Use for customer-related questions — account details, profile info, customer \
  segments, loyalty tiers, contact information, subscriptions, plans, and service types.
- billing_genie: Use for billing-related questions — payment history, invoices, outstanding \
  balances, billing cycles, late fees, payment methods, and charges.
- knowledge_base_search: Use this for ANY technical question about products, routers, modems, \
  firmware, error codes, troubleshooting, installation guides, or equipment documentation. \
  ALWAYS search the knowledge base before answering technical/product questions.
- Memory tools: Use get_user_memory to recall past interactions, save_user_memory to remember \
  user preferences, delete_user_memory to forget information when asked.

ROUTING RULES:
- Questions about who a customer is, their account status, subscriptions, or plans → account_genie
- Questions about payments, bills, charges, invoices, or balances → billing_genie
- If a question spans both (e.g. "tell me everything about this customer"), call both tools.
- For technical/product questions, ALWAYS call knowledge_base_search first. Never answer from \
  general knowledge — only use information returned by the tool.

IMPORTANT RULES:
- DO NOT mention any internal tool or reasoning steps in your final answer.
- Do not say "according to records" or imply that you are looking up information.
- Respond naturally as if you already know the answer.
"""


@tool
def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.now().isoformat()


def _build_uc_tools() -> list:
    """Load Unity Catalog function tools."""
    try:
        return UCFunctionToolkit(function_names=UC_TOOL_NAMES).tools
    except Exception as e:
        logger.warning(f"Failed to load UC tools: {e}")
        return []


# Build Genie agents once at module level (with tracing disabled to avoid startup traces)
mlflow.tracing.disable()
_GENIE_TOOLS = []
try:
    _account_genie_runnable = GenieAgent(
        genie_space_id=GENIE_ACCOUNT_SPACE_ID,
        genie_agent_name="account_genie",
        description="Query customer account data.",
        client=sp_workspace_client,
    )

    @tool
    def account_genie(query: str) -> str:
        """Query customer account data: profiles, segments, loyalty tiers, contact info, subscriptions, plans, and service types. Use for questions about who a customer is, their status, or what plans they have."""
        mlflow.tracing.disable()
        try:
            result = _account_genie_runnable.invoke({"messages": [{"role": "user", "content": query}]})
            msgs = result.get("messages", []) if isinstance(result, dict) else []
            return msgs[-1].content if msgs else str(result)
        finally:
            mlflow.tracing.enable()

    _GENIE_TOOLS.append(account_genie)
except Exception as e:
    logger.warning(f"Failed to load account Genie tool: {e}")

try:
    _billing_genie_runnable = GenieAgent(
        genie_space_id=GENIE_BILLING_SPACE_ID,
        genie_agent_name="billing_genie",
        description="Query billing data.",
        client=sp_workspace_client,
    )

    @tool
    def billing_genie(query: str) -> str:
        """Query billing data: payment history, invoices, outstanding balances, billing cycles, late fees, payment methods, and charges. Use for questions about payments, bills, or financial amounts."""
        mlflow.tracing.disable()
        try:
            result = _billing_genie_runnable.invoke({"messages": [{"role": "user", "content": query}]})
            msgs = result.get("messages", []) if isinstance(result, dict) else []
            return msgs[-1].content if msgs else str(result)
        finally:
            mlflow.tracing.enable()

    _GENIE_TOOLS.append(billing_genie)
except Exception as e:
    logger.warning(f"Failed to load billing Genie tool: {e}")
mlflow.tracing.enable()


def _build_vector_search_tool() -> list:
    """Build Vector Search retriever tool if configured."""
    if not VECTOR_SEARCH_INDEX:
        return []
    try:
        tool = VectorSearchRetrieverTool(
            index_name=VECTOR_SEARCH_INDEX,
            name="knowledge_base_search",
            description="Search the telco knowledge base for product documentation, "
            "troubleshooting guides, router/modem manuals, and error code references. "
            "Use this tool for ANY question about routers, modems, firmware, error codes, "
            "installation, troubleshooting, or technical support.",
            num_results=3,
            columns=["id", "content", "product_name", "title"],
            workspace_client=sp_workspace_client,
        )
        # Remove the @vector_search_retriever_tool_trace decorator that creates
        # a separate top-level MLflow trace on every call. Our parent @mlflow.trace
        # in the /stream endpoint already captures everything.
        original_run = tool._run.__wrapped__ if hasattr(tool._run, '__wrapped__') else None
        if original_run:
            import types
            tool._run = types.MethodType(original_run, tool)
        return [tool]
    except Exception as e:
        logger.warning(f"Failed to load Vector Search tool: {e}")
        return []


async def init_agent(store: Optional[BaseStore] = None, workspace_client=None):
    """Initialize the LangGraph agent with all tools."""
    tools = [get_current_time] + _build_uc_tools() + _GENIE_TOOLS + _build_vector_search_tool()
    if store and MEMORY_ENABLED:
        tools += memory_tools()

    return create_agent(
        model=ChatDatabricks(endpoint=LLM_ENDPOINT_NAME),
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        store=store,
    )


# Stash raw LangGraph messages so /stream can parse thinking/planning/tools
_last_invoke_raw_messages = []


@invoke()
async def invoke_handler(request: ResponsesAgentRequest) -> ResponsesAgentResponse:
    """Handle invocation — run the agent and return only the final text answer."""
    global _last_invoke_raw_messages
    from langchain_core.messages import AIMessage

    user_id = get_user_id(request)
    messages = {"messages": to_chat_completions_input([i.model_dump() for i in request.input])}

    # Build agent with optional memory
    store = None
    config = {"configurable": {"user_id": user_id}}
    ctx_manager = None

    if MEMORY_ENABLED:
        store_kwargs = {"embedding_endpoint": EMBEDDING_ENDPOINT, "embedding_dims": EMBEDDING_DIMS}
        if LAKEBASE_AUTOSCALING_BRANCH:
            store_kwargs["branch"] = LAKEBASE_AUTOSCALING_BRANCH
            if LAKEBASE_AUTOSCALING_PROJECT and not LAKEBASE_AUTOSCALING_BRANCH.startswith("projects/"):
                store_kwargs["project"] = LAKEBASE_AUTOSCALING_PROJECT
        elif LAKEBASE_INSTANCE_NAME:
            store_kwargs["instance_name"] = LAKEBASE_INSTANCE_NAME
        elif LAKEBASE_AUTOSCALING_PROJECT:
            store_kwargs["project"] = LAKEBASE_AUTOSCALING_PROJECT

        ctx_manager = AsyncDatabricksStore(**store_kwargs)
        store = await ctx_manager.__aenter__()
        await store.setup()
        config["configurable"]["store"] = store

    agent = await init_agent(store=store, workspace_client=sp_workspace_client)
    result = await agent.ainvoke(input=messages, config=config)

    if ctx_manager:
        await ctx_manager.__aexit__(None, None, None)

    # Stash raw messages for /stream endpoint
    _last_invoke_raw_messages = list(result.get("messages", []))

    # Extract the final AI message text
    final_text = ""
    output_items = []
    for msg in result.get("messages", []):
        if isinstance(msg, AIMessage):
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    output_items.append({
                        "type": "function_call",
                        "call_id": tc["id"],
                        "name": tc["name"],
                        "arguments": str(tc["args"]),
                    })
            elif msg.content and isinstance(msg.content, str):
                final_text = msg.content

    from uuid import uuid4
    output_items.append({
        "type": "message",
        "id": str(uuid4()),
        "role": "assistant",
        "content": [{"type": "output_text", "text": final_text}],
    })

    return ResponsesAgentResponse(output=output_items)


@stream()
async def stream_handler(
    request: ResponsesAgentRequest,
) -> AsyncGenerator[ResponsesAgentStreamEvent, None]:
    """Handle streaming invocation with optional Lakebase-backed memory."""
    user_id = get_user_id(request)
    messages = {"messages": to_chat_completions_input([i.model_dump() for i in request.input])}

    if MEMORY_ENABLED:
        # If branch is a full resource path (contains project), don't pass project separately
        store_kwargs = {
            "embedding_endpoint": EMBEDDING_ENDPOINT,
            "embedding_dims": EMBEDDING_DIMS,
        }
        if LAKEBASE_INSTANCE_NAME:
            store_kwargs["instance_name"] = LAKEBASE_INSTANCE_NAME
        if LAKEBASE_AUTOSCALING_BRANCH:
            store_kwargs["branch"] = LAKEBASE_AUTOSCALING_BRANCH
            # Only pass project if branch is NOT already a full path
            if LAKEBASE_AUTOSCALING_PROJECT and not LAKEBASE_AUTOSCALING_BRANCH.startswith("projects/"):
                store_kwargs["project"] = LAKEBASE_AUTOSCALING_PROJECT
        elif LAKEBASE_AUTOSCALING_PROJECT:
            store_kwargs["project"] = LAKEBASE_AUTOSCALING_PROJECT

        async with AsyncDatabricksStore(**store_kwargs) as store:
            await store.setup()
            config = {"configurable": {"store": store, "user_id": user_id}}
            agent = await init_agent(workspace_client=sp_workspace_client, store=store)
            async for event in process_agent_astream_events(
                agent.astream(input=messages, config=config, stream_mode=["updates", "messages"])
            ):
                yield event
    else:
        agent = await init_agent(workspace_client=sp_workspace_client)
        config = {"configurable": {"user_id": user_id}}
        async for event in process_agent_astream_events(
            agent.astream(input=messages, config=config, stream_mode=["updates", "messages"])
        ):
            yield event
