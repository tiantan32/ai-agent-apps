import json as _json
import logging
import os
import threading
import time
from typing import Optional

import mlflow
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from mlflow.genai.agent_server import AgentServer, setup_mlflow_git_based_version_tracking
from pydantic import BaseModel

import agent_server.agent  # noqa: F401 — registers @invoke/@stream handlers
from agent_server.db import init_db, save_conversation, get_conversations

logger = logging.getLogger(__name__)
agent_server = AgentServer("ResponsesAgent", enable_chat_proxy=False)
app = agent_server.app
setup_mlflow_git_based_version_tracking()

try:
    init_db()
except Exception as e:
    logger.warning(f"Lakebase DB init deferred: {e}")

MLFLOW_EXPERIMENT_ID = os.getenv("MLFLOW_EXPERIMENT_ID", "")
LLM_ENDPOINT_NAME = os.getenv("LLM_ENDPOINT_NAME", "databricks-claude-3-7-sonnet")


# ─── Feedback ────────────────────────────────────────────────────────
class FeedbackRequest(BaseModel):
    request_id: str
    score: int
    comment: Optional[str] = None


@app.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    try:
        mlflow.log_feedback(
            trace_id=feedback.request_id, name="user_feedback",
            value=feedback.score, rationale=feedback.comment,
        )
        return {"status": "ok"}
    except Exception as e:
        logger.warning(f"Failed to log feedback: {e}")
        try:
            client = mlflow.MlflowClient()
            client.set_trace_tag(feedback.request_id, "user_feedback",
                                 "positive" if feedback.score == 1 else "negative")
            if feedback.comment:
                client.set_trace_tag(feedback.request_id, "user_feedback_comment", feedback.comment)
            return {"status": "ok"}
        except Exception as e2:
            return {"status": "error", "detail": str(e2)}


# ─── History ─────────────────────────────────────────────────────────
@app.get("/history")
async def get_history():
    """Load conversations from Lakebase, enrich with MLflow assessments."""
    convos = get_conversations(limit=30)

    # Build trace_id -> assessments map from MLflow
    trace_assessments = {}
    if MLFLOW_EXPERIMENT_ID:
        try:
            client = mlflow.MlflowClient()
            traces = client.search_traces(
                experiment_ids=[MLFLOW_EXPERIMENT_ID], max_results=50,
                order_by=["timestamp_ms DESC"],
            )
            for t in traces:
                assessments = [
                    {"name": a.name, "value": a.feedback.value if a.feedback else None,
                     "rationale": a.rationale or ""}
                    for a in (t.info.assessments or [])
                ]
                if assessments:
                    trace_assessments[t.info.trace_id] = assessments
        except Exception as e:
            logger.warning(f"Failed to fetch MLflow assessments: {e}")

    # Enrich conversations with assessments
    for c in convos:
        trace_id = c.get("trace_id", "")
        c["assessments"] = trace_assessments.get(trace_id, [])
        for msg in c.get("messages", []):
            msg["assessments"] = trace_assessments.get(msg.get("trace_id", ""), [])

    return {"conversations": convos}


# ─── Stream SSE endpoint ─────────────────────────────────────────────
class StreamRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


@app.post("/stream")
async def stream_chat(req: StreamRequest):
    """Run agent to completion (one trace), then stream thinking/planning/answer via SSE."""
    import asyncio
    from langchain_core.messages import AIMessage

    async def event_stream():
        from agent_server.agent import invoke_handler
        import agent_server.agent as _agent_mod
        from agent_server.scorers import score_in_background
        from mlflow.tracing.constant import SpanAttributeKey

        # Ensure traces go to the correct experiment
        if MLFLOW_EXPERIMENT_ID:
            mlflow.set_experiment(experiment_id=MLFLOW_EXPERIMENT_ID)

        yield f"data: {_json.dumps({'type': 'thinking', 'text': 'Analyzing your question...'})}\n\n"
        await asyncio.sleep(0.05)

        try:
            yield f"data: {_json.dumps({'type': 'thinking', 'text': 'Running agent...'})}\n\n"

            # Wrap in mlflow.start_span to create ResponsesAgent-formatted root span
            # (same as AgentServer._handle_invoke_request)
            request_data = {"input": [{"role": "user", "content": req.message}]}
            from mlflow.types.responses import ResponsesAgentRequest
            agent_request = ResponsesAgentRequest(**request_data)

            with mlflow.start_span(name="invoke_handler") as span:
                span.set_inputs(request_data)
                response = await invoke_handler(agent_request)
                result = response.model_dump() if hasattr(response, "model_dump") else response
                span.set_attribute(SpanAttributeKey.MESSAGE_FORMAT, "openai")
                span.set_outputs(result)

            # Get raw messages stashed by invoke_handler for SSE parsing
            raw_messages = list(_agent_mod._last_invoke_raw_messages)
            _agent_mod._last_invoke_raw_messages = []

            _trace_id = mlflow.get_last_active_trace_id() or ""
            print(f"[TRACE] Agent trace ID: {_trace_id}", flush=True)

            # Parse result messages to extract thinking, planning, tool outputs, answer
            thinking_steps = ["Analyzing your question...", "Running agent..."]
            planning_data = {}
            tool_outputs = []
            final_text = ""

            for msg in raw_messages:
                if not isinstance(msg, AIMessage):
                    # Tool output messages
                    msg_dict = msg.model_dump() if hasattr(msg, "model_dump") else {}
                    if msg_dict.get("type") == "tool":
                        tool_outputs.append(msg_dict.get("content", ""))
                    continue

                if msg.tool_calls:
                    # AI message with tool calls = planning step
                    tool_names = [tc["name"].split("__")[-1] for tc in msg.tool_calls]
                    plan_text = f"I'll look up the information using: {', '.join(tool_names)}"
                    if msg.content and isinstance(msg.content, str) and len(msg.content) > 10:
                        plan_text = msg.content[:500]
                    planning_data = {"text": plan_text, "tools": tool_names}
                elif msg.content and isinstance(msg.content, str):
                    # Final AI answer (last AI message without tool_calls)
                    final_text = msg.content

            # Now stream everything to the frontend
            if planning_data:
                yield f"data: {_json.dumps({'type': 'planning', 'text': planning_data['text'], 'tools': planning_data.get('tools', [])})}\n\n"

            for t in tool_outputs:
                preview = t[:400] + ("..." if len(t) > 400 else "")
                yield f"data: {_json.dumps({'type': 'tool_output', 'text': preview})}\n\n"

            if final_text:
                yield f"data: {_json.dumps({'type': 'answer', 'text': final_text})}\n\n"

            yield f"data: {_json.dumps({'type': 'done'})}\n\n"

            # Background: score + save
            _save_thinking = list(thinking_steps)
            _save_planning = dict(planning_data)
            _save_tools = list(tool_outputs)
            _answer = final_text
            _context = "\n".join(tool_outputs)[:5000]

            def _bg():
                time.sleep(5)
                # Score
                if _trace_id:
                    try:
                        print(f"[SCORER] Running 5 MLflow scorers on trace {_trace_id}", flush=True)
                        score_in_background(
                            trace_id=_trace_id,
                            question=req.message,
                            answer=_answer,
                            context=_context,
                        )
                    except Exception as e:
                        print(f"[SCORER] Scoring failed: {e}", flush=True)
                # Save to Lakebase
                try:
                    save_conversation(
                        question=req.message,
                        answer=_answer,
                        session_id=req.session_id or "",
                        trace_id=_trace_id,
                        thinking_steps=_save_thinking,
                        planning=_save_planning,
                        tool_outputs=_save_tools,
                    )
                except Exception as e:
                    print(f"[DB] Save failed: {e}", flush=True)

            threading.Thread(target=_bg, daemon=True).start()

        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"data: {_json.dumps({'type': 'error', 'text': str(e)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ─── Favicon ─────────────────────────────────────────────────────────
@app.get("/favicon.ico")
async def favicon():
    return Response(content=b"", media_type="image/x-icon", status_code=204)


# ─── Chat Frontend ───────────────────────────────────────────────────
CHAT_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Telco AI Agent</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f1117; color: #e0e0e0; height: 100vh; display: flex; }

  #sidebar { width: 280px; min-width: 280px; background: #141620; border-right: 1px solid #2a2d3a; display: flex; flex-direction: column; transition: margin-left 0.2s; }
  #sidebar.collapsed { margin-left: -280px; }
  .sb-header { padding: 14px 16px; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #2a2d3a; }
  .sb-header span { font-size: 13px; font-weight: 600; color: #aaa; text-transform: uppercase; letter-spacing: 0.5px; }
  .sb-new { background: #2563eb; color: #fff; border: none; border-radius: 6px; padding: 5px 12px; font-size: 12px; cursor: pointer; }
  .sb-new:hover { background: #1d4ed8; }
  #sidebar-list { flex: 1; overflow-y: auto; padding: 8px; }
  .sb-item { padding: 10px 12px; border-radius: 8px; cursor: pointer; margin-bottom: 2px; transition: background 0.1s; }
  .sb-item:hover { background: #1e2130; }
  .sb-item.active { background: #1e2130; border-left: 3px solid #2563eb; }
  .sb-item .sb-q { font-size: 13px; color: #ddd; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .sb-item .sb-time { font-size: 11px; color: #555; margin-top: 2px; }
  .sb-item .sb-scores { display: flex; gap: 4px; margin-top: 4px; flex-wrap: wrap; }
  .sb-pill { font-size: 10px; padding: 1px 6px; border-radius: 3px; }
  .sb-pill.good { background: #166534; color: #22c55e; }
  .sb-pill.bad { background: #7f1d1d; color: #ef4444; }
  .sb-pill.neutral { background: #2a2d3a; color: #888; }
  .sb-empty { color: #444; text-align: center; padding: 30px 10px; font-size: 13px; }

  #main { flex: 1; display: flex; flex-direction: column; min-width: 0; }
  header { background: #1a1d29; padding: 12px 20px; border-bottom: 1px solid #2a2d3a; display: flex; align-items: center; gap: 10px; }
  .toggle-btn { background: none; border: 1px solid #2a2d3a; border-radius: 6px; padding: 6px 8px; cursor: pointer; color: #888; font-size: 16px; line-height: 1; }
  .toggle-btn:hover { color: #fff; border-color: #555; }
  header h1 { font-size: 16px; font-weight: 600; color: #fff; flex: 1; }
  header .badge { font-size: 11px; background: #2563eb; color: #fff; padding: 2px 8px; border-radius: 4px; }

  #chat { flex: 1; overflow-y: auto; padding: 24px; display: flex; flex-direction: column; gap: 16px; }
  .msg-wrapper { display: flex; flex-direction: column; max-width: 80%; }
  .msg-wrapper.user { align-self: flex-end; }
  .msg-wrapper.bot { align-self: flex-start; }
  .msg { padding: 12px 16px; border-radius: 12px; line-height: 1.5; white-space: pre-wrap; word-wrap: break-word; }
  .msg-wrapper.user .msg { background: #2563eb; color: #fff; border-bottom-right-radius: 4px; }
  .msg-wrapper.bot .msg { background: #1e2130; border: 1px solid #2a2d3a; border-bottom-left-radius: 4px; }

  .feedback-bar { display: flex; gap: 8px; margin-top: 6px; align-items: center; }
  .feedback-bar button { background: none; border: 1px solid #2a2d3a; border-radius: 6px; padding: 4px 10px; cursor: pointer; font-size: 14px; color: #888; transition: all 0.15s; }
  .feedback-bar button:hover { border-color: #555; color: #ccc; }
  .feedback-bar button.selected-up { background: #166534; border-color: #22c55e; color: #22c55e; }
  .feedback-bar button.selected-down { background: #7f1d1d; border-color: #ef4444; color: #ef4444; }
  .feedback-bar .status { font-size: 11px; color: #666; margin-left: 4px; }
  .comment-input { margin-top: 4px; display: none; gap: 6px; }
  .comment-input.show { display: flex; }
  .comment-input input { flex: 1; background: #0f1117; border: 1px solid #2a2d3a; border-radius: 6px; padding: 6px 10px; color: #e0e0e0; font-size: 12px; outline: none; }
  .comment-input button { background: #2563eb; color: #fff; border: none; border-radius: 6px; padding: 6px 12px; font-size: 12px; cursor: pointer; }

  .suggestions { display: flex; flex-wrap: wrap; gap: 8px; padding: 0 4px; }
  .suggestions button { background: #1e2130; border: 1px solid #2a2d3a; border-radius: 8px; padding: 10px 16px; color: #a0a0b0; font-size: 13px; cursor: pointer; text-align: left; transition: all 0.15s; }
  .suggestions button:hover { border-color: #2563eb; color: #e0e0e0; }

  #input-area { padding: 14px 20px; background: #1a1d29; border-top: 1px solid #2a2d3a; display: flex; gap: 12px; }
  #input-area textarea { flex: 1; background: #0f1117; border: 1px solid #2a2d3a; border-radius: 8px; padding: 12px; color: #e0e0e0; font-size: 14px; resize: none; outline: none; font-family: inherit; }
  #input-area textarea:focus { border-color: #2563eb; }
  #input-area button { background: #2563eb; color: #fff; border: none; border-radius: 8px; padding: 12px 24px; font-size: 14px; cursor: pointer; font-weight: 500; }
  #input-area button:hover { background: #1d4ed8; }
  #input-area button:disabled { opacity: 0.5; cursor: not-allowed; }
</style>
</head>
<body>

<div id="sidebar">
  <div class="sb-header"><span>History</span><button class="sb-new" onclick="newChat()">+ New Chat</button></div>
  <div id="sidebar-list"><div class="sb-empty">Loading...</div></div>
</div>

<div id="main">
  <header>
    <button class="toggle-btn" onclick="toggleSidebar()" title="Toggle history">&#9776;</button>
    <h1>Telco Customer Support Agent</h1>
    <span class="badge">AI Agent</span>
  </header>
  <div id="chat">
    <div class="msg-wrapper bot"><div class="msg">Welcome! I can help you with customer information, billing details, subscriptions, and technical support.</div></div>
    <div class="suggestions">
      <button onclick="askSuggestion(this)">Give me the information about john21@example.net</button>
      <button onclick="askSuggestion(this)">Summarize all subscriptions held by john21@example.net</button>
      <button onclick="askSuggestion(this)">What are the step-by-step instructions for updating firmware on my router?</button>
    </div>
  </div>
  <div id="input-area">
    <textarea id="user-input" rows="1" placeholder="Ask about customers, billing, subscriptions, or technical support..." onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();sendMessage()}"></textarea>
    <button id="send-btn" onclick="sendMessage()">Send</button>
  </div>
</div>

<script>
const chat = document.getElementById('chat');
const input = document.getElementById('user-input');
const btn = document.getElementById('send-btn');
let sessionId = crypto.randomUUID();

function toggleSidebar() { document.getElementById('sidebar').classList.toggle('collapsed'); }
function newChat() {
  sessionId = crypto.randomUUID();
  chat.innerHTML = `<div class="msg-wrapper bot"><div class="msg">Welcome! I can help you with customer information, billing details, subscriptions, and technical support.</div></div>
    <div class="suggestions">
      <button onclick="askSuggestion(this)">Give me the information about john21@example.net</button>
      <button onclick="askSuggestion(this)">Summarize all subscriptions held by john21@example.net</button>
      <button onclick="askSuggestion(this)">What are the step-by-step instructions for updating firmware on my router?</button>
    </div>`;
  document.querySelectorAll('.sb-item').forEach(i => i.classList.remove('active'));
}

function loadHistoryItem(session, el) {
  document.querySelectorAll('.sb-item').forEach(i => i.classList.remove('active'));
  el.classList.add('active');
  chat.innerHTML = '';
  for (const msg of (session.messages || [])) {
    appendMsg(msg.question, 'user');

    // Render thinking/planning/tools (collapsed by default)
    const thinking = msg.thinking_steps || [];
    const planning = msg.planning || {};
    const tools = msg.tool_outputs || [];
    const hasThinking = thinking.length > 0 || planning.text || tools.length > 0;

    if (hasThinking) {
      const thinkW = document.createElement('div');
      thinkW.className = 'msg-wrapper bot';

      const thinkDiv = document.createElement('div');
      thinkDiv.className = 'msg';
      thinkDiv.style.cssText = 'font-size:12px;color:#888;display:none;';

      // Thinking steps
      for (const step of thinking) {
        thinkDiv.innerHTML += '<div style="margin:2px 0;">\\u{1F9E0} ' + escHtml(step) + '</div>';
      }
      // Planning
      if (planning.text) {
        const toolBadges = (planning.tools||[]).map(t => '<span style="display:inline-block;background:#1a1d40;color:#7aa2f7;padding:1px 8px;border-radius:4px;font-size:11px;margin:0 2px;">' + escHtml(t) + '</span>').join(' ');
        thinkDiv.innerHTML += '<div style="margin:6px 0;padding:8px 12px;background:#0f1117;border-radius:8px;border-left:3px solid #2563eb;"><div style="color:#2563eb;font-weight:600;margin-bottom:4px;">\\u{1F4CB} Plan</div><div style="color:#bbb;">' + escHtml(planning.text) + '</div>' + (toolBadges ? '<div style="margin-top:6px;">Tools: ' + toolBadges + '</div>' : '') + '</div>';
      }
      // Tool outputs
      for (const t of tools) {
        const preview = t.length > 400 ? t.substring(0, 400) + '...' : t;
        thinkDiv.innerHTML += '<div style="margin:4px 0;padding:6px 10px;background:#0f1117;border-radius:6px;border-left:2px solid #22c55e;font-size:11px;"><span style="color:#22c55e;">\\u{1F4E6} Data retrieved</span><pre style="margin:2px 0;white-space:pre-wrap;color:#777;max-height:100px;overflow-y:auto;font-size:11px;">' + escHtml(preview) + '</pre></div>';
      }

      const toggle = document.createElement('div');
      toggle.style.cssText = 'font-size:12px;color:#555;cursor:pointer;padding:4px 0;user-select:none;';
      toggle.textContent = '\\u{25B6} Show thinking process';
      toggle.addEventListener('click', function() {
        if (thinkDiv.style.display === 'none') {
          thinkDiv.style.display = '';
          toggle.textContent = '\\u{25BC} Hide thinking process';
        } else {
          thinkDiv.style.display = 'none';
          toggle.textContent = '\\u{25B6} Show thinking process';
        }
      });

      thinkW.appendChild(toggle);
      thinkW.appendChild(thinkDiv);
      chat.appendChild(thinkW);
    }

    // Answer
    const w = appendMsg(msg.answer, 'bot');

    // Assessments
    const aa = msg.assessments || [];
    if (aa.length) {
      const bar = document.createElement('div');
      bar.style.cssText = 'display:flex;gap:4px;flex-wrap:wrap;margin-top:6px;';
      for (const a of aa) {
        const val = typeof a.value==='number' ? a.value+'/5' : a.value;
        const v = a.value; const cls = (v==='yes'||v===true||v>=4)?'good':(v==='no'||v===false||v<=2)?'bad':'neutral';
        bar.innerHTML += '<span class="sb-pill ' + cls + '" title="' + escHtml(a.rationale||'') + '">' + a.name.replace(/_/g,' ') + ': ' + val + '</span>';
      }
      w.appendChild(bar);
    }
    if (msg.trace_id) {
      const meta = document.createElement('div');
      meta.style.cssText = 'font-size:11px;color:#444;margin-top:4px;';
      meta.textContent = (msg.timestamp ? new Date(msg.timestamp).toLocaleString() : '') + '  |  trace: ' + msg.trace_id;
      w.appendChild(meta);
    }
  }
}

async function loadHistory() {
  const list = document.getElementById('sidebar-list');
  try {
    const res = await fetch('/history');
    const data = await res.json();
    const items = data.conversations || [];
    if (!items.length) { list.innerHTML = '<div class="sb-empty">No conversations yet</div>'; return; }
    list.innerHTML = '';
    for (const s of items) {
      const item = document.createElement('div');
      item.className = 'sb-item';
      const ts = s.timestamp ? new Date(s.timestamp).toLocaleString(undefined,{month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'}) : '';
      let pills = '';
      for (const a of (s.assessments||[])) {
        const val = typeof a.value==='number' ? a.value+'/5' : a.value;
        const v = a.value; const cls = (v==='yes'||v===true||v>=4)?'good':(v==='no'||v===false||v<=2)?'bad':'neutral';
        pills += `<span class="sb-pill ${cls}">${a.name.replace(/_/g,' ')}: ${val}</span>`;
      }
      item.innerHTML = `<div class="sb-q">${escHtml(s.title||'Conversation')}</div><div class="sb-time">${ts}</div>${pills?'<div class="sb-scores">'+pills+'</div>':''}`;
      item.onclick = () => loadHistoryItem(s, item);
      list.appendChild(item);
    }
  } catch(e) { list.innerHTML = `<div class="sb-empty">Error: ${e.message}</div>`; }
}

function askSuggestion(el) {
  document.querySelector('.suggestions')?.remove();
  input.value = el.textContent;
  sendMessage();
}

async function sendMessage() {
  const text = input.value.trim();
  if (!text) return;
  appendMsg(text, 'user');
  input.value = '';
  btn.disabled = true;

  // Thinking steps
  const thinkW = document.createElement('div');
  thinkW.className = 'msg-wrapper bot';
  thinkW.innerHTML = '<div class="msg" style="font-size:12px;color:#888;"></div>';
  chat.appendChild(thinkW);
  const thinkDiv = thinkW.querySelector('.msg');

  // Answer area
  const ansW = document.createElement('div');
  ansW.className = 'msg-wrapper bot';
  ansW.innerHTML = '<div class="msg"></div>';
  ansW.style.display = 'none';
  chat.appendChild(ansW);
  const ansDiv = ansW.querySelector('.msg');

  let fullAnswer = '';
  try {
    const res = await fetch('/stream', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({message: text, session_id: sessionId})
    });
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, {stream: true});
      const lines = buffer.split('\\n');
      buffer = lines.pop();
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const payload = line.slice(6);
        if (payload === '[DONE]') break;
        try {
          const evt = JSON.parse(payload);
          if (evt.type === 'thinking') {
            thinkDiv.innerHTML += `<div style="margin:2px 0;">\\u{1F9E0} ${escHtml(evt.text)}</div>`;
          } else if (evt.type === 'planning') {
            const toolBadges = (evt.tools||[]).map(t => `<span style="display:inline-block;background:#1a1d40;color:#7aa2f7;padding:1px 8px;border-radius:4px;font-size:11px;margin:0 2px;">${escHtml(t)}</span>`).join(' ');
            thinkDiv.innerHTML += `<div style="margin:6px 0;padding:8px 12px;background:#0f1117;border-radius:8px;border-left:3px solid #2563eb;">
              <div style="color:#2563eb;font-weight:600;margin-bottom:4px;">\\u{1F4CB} Plan</div>
              <div style="color:#bbb;">${escHtml(evt.text)}</div>
              ${toolBadges ? '<div style="margin-top:6px;">Tools: '+toolBadges+'</div>' : ''}
            </div>`;
          } else if (evt.type === 'tool_output') {
            thinkDiv.innerHTML += `<div style="margin:4px 0;padding:6px 10px;background:#0f1117;border-radius:6px;border-left:2px solid #22c55e;font-size:11px;">
              <span style="color:#22c55e;">\\u{1F4E6} Data retrieved</span>
              <pre style="margin:2px 0;white-space:pre-wrap;color:#777;max-height:100px;overflow-y:auto;font-size:11px;">${escHtml(evt.text)}</pre>
            </div>`;
          } else if (evt.type === 'answer') {
            if (ansW.style.display === 'none') {
              ansW.style.display = '';
              // Collapse thinking: hide content, add toggle via DOM
              thinkDiv.style.display = 'none';
              const toggle = document.createElement('div');
              toggle.style.cssText = 'font-size:12px;color:#555;cursor:pointer;padding:4px 0;user-select:none;';
              toggle.textContent = '\\u{25B6} Show thinking process';
              toggle.addEventListener('click', function() {
                if (thinkDiv.style.display === 'none') {
                  thinkDiv.style.display = '';
                  toggle.textContent = '\\u{25BC} Hide thinking process';
                } else {
                  thinkDiv.style.display = 'none';
                  toggle.textContent = '\\u{25B6} Show thinking process';
                }
              });
              thinkW.insertBefore(toggle, thinkDiv);
            }
            fullAnswer += evt.text;
            ansDiv.textContent = fullAnswer;
          } else if (evt.type === 'error') {
            ansDiv.textContent = 'Error: ' + evt.text;
            ansW.style.display = '';
          }
          chat.scrollTop = chat.scrollHeight;
        } catch(e) {}
      }
    }

    if (!fullAnswer) { ansDiv.textContent = 'No response'; ansW.style.display = ''; }

    // Scoring note
    const scoreNote = document.createElement('div');
    scoreNote.style.cssText = 'font-size:11px;color:#555;margin-top:4px;font-style:italic;';
    scoreNote.textContent = 'Scoring with 3 LLM judges in background... refresh history in ~15s to see scores.';
    ansW.appendChild(scoreNote);

    addFeedbackBar(ansW);
    setTimeout(loadHistory, 15000);
  } catch(e) {
    ansDiv.textContent = 'Error: ' + e.message;
    ansW.style.display = '';
  }
  btn.disabled = false;
  input.focus();
}

function appendMsg(text, cls) {
  const w = document.createElement('div');
  w.className = 'msg-wrapper ' + cls;
  const d = document.createElement('div');
  d.className = 'msg';
  d.textContent = text;
  w.appendChild(d);
  chat.appendChild(w);
  chat.scrollTop = chat.scrollHeight;
  return w;
}

function addFeedbackBar(wrapper) {
  const bar = document.createElement('div');
  bar.className = 'feedback-bar';
  bar.innerHTML = '<button class="thumb-up" title="Helpful">\\u{1F44D}</button><button class="thumb-down" title="Not helpful">\\u{1F44E}</button><span class="status"></span>';
  const commentRow = document.createElement('div');
  commentRow.className = 'comment-input';
  commentRow.innerHTML = '<input type="text" placeholder="Optional: tell us why..." /><button>Submit</button>';
  wrapper.appendChild(bar);
  wrapper.appendChild(commentRow);
  const up = bar.querySelector('.thumb-up'), dn = bar.querySelector('.thumb-down'), st = bar.querySelector('.status');
  let sel = null;
  // We'll use the latest trace_id from history for feedback
  up.addEventListener('click', () => { sel=1; up.className='thumb-up selected-up'; dn.className='thumb-down'; commentRow.classList.add('show'); st.textContent='Sending...'; fetchLatestTraceAndFeedback(1,null,st); });
  dn.addEventListener('click', () => { sel=0; dn.className='thumb-down selected-down'; up.className='thumb-up'; commentRow.classList.add('show'); st.textContent='Sending...'; fetchLatestTraceAndFeedback(0,null,st); });
  commentRow.querySelector('button').addEventListener('click', () => { const c=commentRow.querySelector('input').value.trim(); if(c&&sel!==null){fetchLatestTraceAndFeedback(sel,c,st);commentRow.querySelector('input').value='';} });
}

async function fetchLatestTraceAndFeedback(score, comment, statusEl) {
  try {
    // Get latest trace from history
    const hres = await fetch('/history');
    const hdata = await hres.json();
    const latest = (hdata.conversations||[])[0];
    if (!latest || !latest.trace_id) { statusEl.textContent = 'No trace found yet'; return; }
    const body = {request_id: latest.trace_id, score};
    if (comment) body.comment = comment;
    const res = await fetch('/feedback', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
    const data = await res.json();
    statusEl.textContent = data.status==='ok' ? 'Recorded on trace '+latest.trace_id.slice(0,8)+'...' : 'Failed';
  } catch(e) { statusEl.textContent = 'Error'; }
  setTimeout(() => { statusEl.textContent = ''; }, 5000);
}

function escHtml(s) { const d=document.createElement('div'); d.textContent=s; return d.innerHTML; }

loadHistory();
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return CHAT_HTML


def main():
    agent_server.run(app_import_string="agent_server.start_server:app")
