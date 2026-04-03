import json
import logging
import os
import re
from typing import Optional

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.store.base import BaseStore
from mlflow.types.responses import ResponsesAgentRequest

logger = logging.getLogger(__name__)


def resolve_lakebase_instance_name(raw_name: str) -> str:
    """Resolve a Lakebase hostname or instance name to the instance name only."""
    if not raw_name:
        return raw_name
    # If it looks like a hostname (contains dots), extract the instance name
    match = re.match(r"^([^.]+)\.", raw_name)
    if match:
        return match.group(1)
    return raw_name


def get_user_id(request: ResponsesAgentRequest) -> str:
    """Extract user_id from request custom_inputs or context."""
    if hasattr(request, "custom_inputs") and request.custom_inputs:
        uid = request.custom_inputs.get("user_id")
        if uid:
            return uid
    if hasattr(request, "context") and request.context:
        uid = request.context.get("user_id")
        if uid:
            return uid
    return "anonymous"


def get_lakebase_access_error_message() -> str:
    """Return a helpful error message for Lakebase connection failures."""
    return (
        "Could not connect to Lakebase. Please verify:\n"
        "1. LAKEBASE_INSTANCE_NAME or LAKEBASE_AUTOSCALING_PROJECT/BRANCH env vars are set\n"
        "2. The Lakebase instance exists and is running\n"
        "3. The service principal has access to the Lakebase instance\n"
        "Run 'uv run grant-lakebase-permissions' to fix permission issues."
    )


def memory_tools() -> list:
    """Return the list of memory tools for the agent."""

    @tool
    async def get_user_memory(query: str, config: RunnableConfig) -> str:
        """Search long-term memory for previously saved information about the user.

        Args:
            query: A natural language search query to find relevant memories.
        """
        store: BaseStore = config["configurable"]["store"]
        user_id: str = config["configurable"]["user_id"]
        namespace = ("user_memories", user_id)

        try:
            results = await store.asearch(namespace, query=query, limit=5)
            if not results:
                return "No memories found for this query."
            memories = []
            for item in results:
                memories.append(
                    f"[{item.key}]: {json.dumps(item.value)}"
                )
            return "\n".join(memories)
        except Exception as e:
            logger.error(f"Error searching memory: {e}")
            return f"Error searching memory: {e}"

    @tool
    async def save_user_memory(
        memory_key: str, memory_data_json: str, config: RunnableConfig
    ) -> str:
        """Save information to long-term memory about the user.

        Args:
            memory_key: A short, descriptive key for this memory (e.g. 'favorite_plan', 'billing_preference').
            memory_data_json: A JSON string with the data to remember (e.g. '{"plan": "Premium", "reason": "needs international calling"}').
        """
        store: BaseStore = config["configurable"]["store"]
        user_id: str = config["configurable"]["user_id"]
        namespace = ("user_memories", user_id)

        try:
            data = json.loads(memory_data_json)
            await store.aput(namespace, memory_key, data)
            return f"Memory saved with key '{memory_key}'."
        except json.JSONDecodeError:
            return "Error: memory_data_json must be valid JSON."
        except Exception as e:
            logger.error(f"Error saving memory: {e}")
            return f"Error saving memory: {e}"

    @tool
    async def delete_user_memory(memory_key: str, config: RunnableConfig) -> str:
        """Delete a specific memory by key.

        Args:
            memory_key: The key of the memory to delete.
        """
        store: BaseStore = config["configurable"]["store"]
        user_id: str = config["configurable"]["user_id"]
        namespace = ("user_memories", user_id)

        try:
            await store.adelete(namespace, memory_key)
            return f"Memory with key '{memory_key}' deleted."
        except Exception as e:
            logger.error(f"Error deleting memory: {e}")
            return f"Error deleting memory: {e}"

    return [get_user_memory, save_user_memory, delete_user_memory]
