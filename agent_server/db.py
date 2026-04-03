"""
Lakebase persistence for chat history.
Stores full conversation data: question, thinking steps, planning, tool outputs, and answer.
"""

import json
import os
import traceback

LAKEBASE_BRANCH = os.getenv("LAKEBASE_AUTOSCALING_BRANCH", "")

_client = None


def _get_client():
    global _client
    if _client is None:
        from databricks_ai_bridge.lakebase import LakebaseClient
        _client = LakebaseClient(branch=LAKEBASE_BRANCH)
        print(f"[DB] LakebaseClient initialized (branch={LAKEBASE_BRANCH})", flush=True)
    return _client


def init_db():
    try:
        client = _get_client()
        client.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id SERIAL PRIMARY KEY,
                session_id TEXT,
                trace_id TEXT,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                thinking_steps TEXT,
                planning TEXT,
                tool_outputs TEXT,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        client.execute("CREATE INDEX IF NOT EXISTS idx_chat_created ON chat_history (created_at DESC)")
        print("[DB] chat_history table ready", flush=True)
    except Exception as e:
        print(f"[DB] Failed to init: {e}", flush=True)


def save_conversation(
    question: str,
    answer: str,
    session_id: str = "",
    trace_id: str = "",
    thinking_steps: list = None,
    planning: dict = None,
    tool_outputs: list = None,
):
    try:
        client = _get_client()
        client.execute(
            "INSERT INTO chat_history (session_id, trace_id, question, answer, thinking_steps, planning, tool_outputs) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (
                session_id,
                trace_id,
                question,
                answer,
                json.dumps(thinking_steps or []),
                json.dumps(planning or {}),
                json.dumps(tool_outputs or []),
            ),
        )
        print(f"[DB] Saved conversation (trace={trace_id})", flush=True)
    except Exception as e:
        print(f"[DB] Failed to save: {e}", flush=True)
        traceback.print_exc()


def get_conversations(limit: int = 30):
    try:
        client = _get_client()
        result = client.execute(
            "SELECT id, session_id, trace_id, question, answer, thinking_steps, planning, tool_outputs, created_at "
            "FROM chat_history ORDER BY created_at DESC LIMIT %s",
            (limit,),
        )

        if isinstance(result, int) or result is None:
            # Fallback: use pool directly
            try:
                pool = client.pool
                conn = pool.getconn()
                conn.autocommit = True
                cur = conn.cursor()
                cur.execute(
                    "SELECT id, session_id, trace_id, question, answer, thinking_steps, planning, tool_outputs, created_at "
                    "FROM chat_history ORDER BY created_at DESC LIMIT %s",
                    (limit,),
                )
                rows = cur.fetchall()
                cols = [desc[0] for desc in cur.description]
                pool.putconn(conn)
                result = [dict(zip(cols, row)) for row in rows]
            except Exception as pool_err:
                print(f"[DB] Pool fallback failed: {pool_err}", flush=True)
                return []

        if not result:
            return []

        conversations = []
        for r in result:
            if not isinstance(r, dict):
                continue
            ts = r.get("created_at")
            ts_ms = int(ts.timestamp() * 1000) if ts else 0

            # Parse JSON fields
            try:
                thinking = json.loads(r.get("thinking_steps") or "[]")
            except Exception:
                thinking = []
            try:
                planning = json.loads(r.get("planning") or "{}")
            except Exception:
                planning = {}
            try:
                tools = json.loads(r.get("tool_outputs") or "[]")
            except Exception:
                tools = []

            conversations.append({
                "session_id": r.get("session_id", ""),
                "title": (r.get("question", "") or "Conversation")[:80],
                "timestamp": ts_ms,
                "trace_id": r.get("trace_id", ""),
                "messages": [{
                    "question": r.get("question", ""),
                    "answer": r.get("answer", ""),
                    "trace_id": r.get("trace_id", ""),
                    "timestamp": ts_ms,
                    "thinking_steps": thinking,
                    "planning": planning,
                    "tool_outputs": tools,
                }],
            })
        return conversations
    except Exception as e:
        print(f"[DB] Failed to fetch: {e}", flush=True)
        traceback.print_exc()
        return []
