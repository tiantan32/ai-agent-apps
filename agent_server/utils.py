import logging
import os
from typing import AsyncGenerator

from databricks.sdk import WorkspaceClient
from langchain_core.messages import AIMessageChunk
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentStreamEvent

logger = logging.getLogger(__name__)


def get_databricks_host_from_env() -> str:
    """Get the Databricks workspace host URL from environment."""
    host = os.getenv("DATABRICKS_HOST", "")
    if not host:
        try:
            w = WorkspaceClient()
            host = w.config.host
        except Exception:
            host = ""
    return host.rstrip("/")


def get_session_id(request: ResponsesAgentRequest) -> str:
    """Extract session/conversation ID from the request."""
    if hasattr(request, "custom_inputs") and request.custom_inputs:
        session_id = request.custom_inputs.get("session_id") or request.custom_inputs.get(
            "conversation_id"
        )
        if session_id:
            return session_id
    return "default"


def get_user_workspace_client(request: ResponsesAgentRequest) -> WorkspaceClient:
    """Create a WorkspaceClient using the forwarded user token (OBO auth)."""
    if hasattr(request, "context") and request.context:
        token = request.context.get("user_token")
        if token:
            host = get_databricks_host_from_env()
            return WorkspaceClient(host=host, token=token)
    return WorkspaceClient()


async def process_agent_astream_events(
    astream,
) -> AsyncGenerator[ResponsesAgentStreamEvent, None]:
    """Convert LangGraph stream events into ResponsesAgentStreamEvent objects."""
    async for event in astream:
        if event[0] == "updates":
            for node_data in event[1].values():
                if "messages" not in node_data:
                    continue
                for msg in node_data["messages"]:
                    msg_dict = msg.model_dump() if hasattr(msg, "model_dump") else msg
                    msg_type = msg_dict.get("type", "")

                    if msg_type == "ai":
                        tool_calls = msg_dict.get("tool_calls", [])
                        if tool_calls:
                            for tc in tool_calls:
                                import json

                                yield ResponsesAgentStreamEvent(
                                    type="response.output_item.done",
                                    item={
                                        "type": "function_call",
                                        "id": msg_dict.get("id", ""),
                                        "call_id": tc["id"],
                                        "name": tc["name"],
                                        "arguments": json.dumps(tc["args"]),
                                    },
                                )
                        elif msg_dict.get("content"):
                            yield ResponsesAgentStreamEvent(
                                type="response.output_item.done",
                                item={
                                    "type": "message",
                                    "id": msg_dict.get("id", ""),
                                    "role": "assistant",
                                    "content": [
                                        {
                                            "type": "output_text",
                                            "text": msg_dict["content"],
                                        }
                                    ],
                                },
                            )

                    elif msg_type == "tool":
                        yield ResponsesAgentStreamEvent(
                            type="response.output_item.done",
                            item={
                                "type": "function_call_output",
                                "call_id": msg_dict.get("tool_call_id", ""),
                                "output": msg_dict.get("content", ""),
                            },
                        )

        elif event[0] == "messages":
            try:
                chunk = event[1][0]
                if isinstance(chunk, AIMessageChunk) and chunk.content:
                    yield ResponsesAgentStreamEvent(
                        type="response.content_part.delta",
                        item_id=chunk.id,
                        content_index=0,
                        delta=chunk.content,
                    )
            except Exception:
                pass
