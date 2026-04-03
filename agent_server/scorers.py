"""
Evaluation scorers for the telco AI agent.

Real-time scoring (per-trace): 3 MLflow built-in + 2 custom LLM-as-judge.
Offline scoring (batch): Same 5 scorers via mlflow.genai.evaluate().

Scorers:
  1. RetrievalGroundedness (built-in) — Is the answer grounded in retrieved data?
  2. RelevanceToQuery (built-in)      — Does the answer address the user's question?
  3. Safety (built-in)                — Is the response free of harmful content?
  4. steps_and_reasoning (custom)     — Does the FINAL ANSWER hide internal tool/reasoning steps?
  5. retrieval_quality (custom)       — Did the retrieved data support the answer?
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor

import mlflow
from databricks_langchain import ChatDatabricks
from mlflow.genai.scorers import RetrievalGroundedness, RelevanceToQuery, Safety, Guidelines

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="scorer")


# ─── Custom LLM-as-judge prompts ────────────────────────────────────
# These evaluate the FINAL ANSWER only, not intermediate trace spans.

STEPS_AND_REASONING_PROMPT = """You are an evaluation judge. You are given the FINAL ANSWER that was shown to the end user. Rate whether this answer avoids exposing raw internal agent mechanics.

ACCEPTABLE (do NOT penalize):
- "Based on the information I found" or "Based on your account information" — this is natural
- "I found your account" or "I can see that" — conversational and fine
- Any natural phrasing that implies the agent has access to information

NOT ACCEPTABLE (penalize):
- Explicitly naming tools or functions (e.g. "I called get_customer_by_email")
- Mentioning "database query", "API call", "vector search", or "knowledge base tool"
- Showing raw function names, SQL queries, or JSON output
- Describing step-by-step internal reasoning ("First I need to look up X, then call Y")

FINAL ANSWER:
{answer}

Rate on a scale of 1-5:
1 = Explicitly names tools, functions, or shows raw technical output
2 = Mentions "database", "API", or describes internal multi-step process
3 = Minor reference to internal systems
4 = Natural with very minor phrasing concern
5 = Completely natural and conversational

Respond ONLY with JSON: {{"score": <1-5>, "rationale": "<brief explanation>"}}"""

RETRIEVAL_QUALITY_PROMPT = """You are an evaluation judge for a customer support AI agent. Given a user's question and the final answer, rate whether the answer demonstrates that the agent successfully retrieved and used relevant data.

IMPORTANT: Focus on the FINAL ANSWER quality, not the raw tool output. The agent may have retrieved more data than shown here — judge by whether the answer contains specific, accurate details that could only come from a data lookup (customer names, account numbers, billing amounts, product specifications, error code details, etc.)

Question: {question}
Final Answer: {answer}

Rate on a scale of 1-5:
1 = Answer is vague/generic with no specific data — agent likely did not retrieve useful information
2 = Answer has some specific details but is mostly generic filler
3 = Answer has a mix of specific retrieved data and generic content
4 = Answer is mostly grounded in specific retrieved data with minor generic filler
5 = Answer is rich with specific details (names, numbers, dates, technical specs) clearly from retrieved data

If the question is a greeting or doesn't need data retrieval, rate 5.

Respond ONLY with JSON: {{"score": <1-5>, "rationale": "<brief explanation>"}}"""


def _parse_judge_output(text: str) -> dict:
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        return json.loads(text[start:end])
    except Exception:
        return {"score": 0, "rationale": f"Parse error: {text[:200]}"}


def _run_judge(judge_llm, prompt: str) -> dict:
    resp = judge_llm.invoke(prompt)
    return _parse_judge_output(resp.content)


# ─── Real-time scoring (per-trace) ──────────────────────────────────

def evaluate_response(
    trace_id: str,
    question: str = "",
    answer: str = "",
    context: str = "",
    llm_endpoint: str = "",
):
    """Run all 5 scorers and log results as MLflow assessments on the trace."""
    mlflow.langchain.autolog(disable=True)
    mlflow.tracing.disable()

    client = mlflow.MlflowClient()

    # --- 1-3: MLflow built-in scorers (need trace object) ---
    try:
        trace = client.get_trace(trace_id)
    except Exception as e:
        print(f"[SCORER] Failed to fetch trace {trace_id}: {e}", flush=True)
        trace = None

    builtin_scorers = [
        ("retrieval_groundedness", RetrievalGroundedness()),
        ("relevance_to_query", RelevanceToQuery()),
        ("safety", Safety()),
    ]

    for name, scorer in builtin_scorers:
        if not trace:
            continue
        try:
            feedback = scorer.run(
                inputs={"question": question} if question else None,
                outputs=answer if answer else None,
                trace=trace,
            )
            feedbacks = feedback if isinstance(feedback, list) else ([feedback] if feedback else [])
            for fb in feedbacks:
                score_name = getattr(fb, "name", name) or name
                value = fb.value if hasattr(fb, "value") else None
                rationale = str(fb.rationale)[:250] if hasattr(fb, "rationale") and fb.rationale else ""
                if value is None:
                    continue
                print(f"[SCORER] {score_name} = {value} — {rationale[:80]}", flush=True)
                try:
                    mlflow.log_feedback(trace_id=trace_id, name=score_name, value=value, rationale=rationale)
                except Exception as e1:
                    print(f"[SCORER] log_feedback failed for {score_name}: {e1}", flush=True)
        except Exception as e:
            print(f"[SCORER] {name} scoring failed: {e}", flush=True)

    # --- 4-5: Custom LLM-as-judge (evaluate FINAL ANSWER only) ---
    llm_ep = llm_endpoint or "databricks-claude-3-7-sonnet"
    judge = ChatDatabricks(endpoint=llm_ep, temperature=0.0)

    custom_metrics = {
        "steps_and_reasoning": STEPS_AND_REASONING_PROMPT.format(answer=answer),
        "retrieval_quality": RETRIEVAL_QUALITY_PROMPT.format(
            question=question, answer=answer
        ),
    }

    for name, prompt in custom_metrics.items():
        try:
            result = _run_judge(judge, prompt)
            score = result.get("score", 0)
            rationale = result.get("rationale", "")
            print(f"[SCORER] {name} = {score}/5 — {rationale[:80]}", flush=True)
            try:
                mlflow.log_feedback(trace_id=trace_id, name=name, value=score, rationale=rationale[:250])
            except Exception as e1:
                print(f"[SCORER] log_feedback failed for {name}: {e1}", flush=True)
        except Exception as e:
            print(f"[SCORER] {name} scoring failed: {e}", flush=True)

    mlflow.tracing.enable()
    mlflow.langchain.autolog()


def score_in_background(trace_id: str, question: str = "", answer: str = "", context: str = "", llm_endpoint: str = ""):
    """Submit scoring to background thread pool — fire and forget."""
    _executor.submit(evaluate_response, trace_id, question, answer, context, llm_endpoint)


# ─── Offline scoring (batch evaluation) ─────────────────────────────

def get_offline_scorers():
    """Return scorers for offline batch evaluation via mlflow.genai.evaluate()."""
    return [
        RetrievalGroundedness(),
        RelevanceToQuery(),
        Safety(),
        Guidelines(
            guidelines="""
            Evaluate ONLY the final answer shown to the user, not intermediate steps.
            The final answer must read naturally without:
            - Mentioning tools, functions, databases, or "looking up" information
            - Saying "according to records", "based on the data/results"
            - Describing reasoning steps or internal processes
            """,
            name="steps_and_reasoning",
        ),
        Guidelines(
            guidelines="""
            Evaluate whether the agent's answer is well-supported by the data it retrieved.
            For technical questions (routers, firmware, error codes), the agent should have
            used the knowledge base. For customer questions, it should have looked up customer data.
            If the question is a greeting or doesn't need retrieval, it passes automatically.
            The answer should not fabricate customer names, IDs, or billing amounts.
            """,
            name="retrieval_quality",
        ),
    ]
