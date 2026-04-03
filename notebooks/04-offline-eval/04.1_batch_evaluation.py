# Databricks notebook source
# MAGIC %md
# MAGIC # Offline Batch Evaluation — Telco AI Agent
# MAGIC
# MAGIC This notebook runs batch evaluation against the deployed agent app using MLflow built-in scorers.
# MAGIC It evaluates:
# MAGIC - **RetrievalGroundedness** — Is the answer grounded in retrieved data?
# MAGIC - **RelevanceToQuery** — Does the answer address the user's question?
# MAGIC - **Safety** — Is the response free of harmful content?
# MAGIC - **Guidelines: steps_and_reasoning** — Does the answer hide internal tool/reasoning steps?
# MAGIC - **Guidelines: retrieval_quality** — Did the agent use the right tools and never fabricate data?
# MAGIC
# MAGIC ### How to run
# MAGIC 1. Adjust `AGENT_ENDPOINT` below if your app endpoint is different
# MAGIC 2. Run all cells — it will call the live agent, collect responses, and score them
# MAGIC 3. Open your MLflow experiment to compare evaluation runs

# COMMAND ----------

# DBTITLE 1,Install dependencies
# MAGIC %pip install -U -qqqq mlflow>=3.10.1 databricks-langchain databricks-agents databricks-sdk
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../config

# COMMAND ----------

import mlflow

# Point to the experiment linked to the app
EXPERIMENT_NAME = f"/Users/{spark.sql('SELECT current_user()').collect()[0][0]}/ai-agent-telco-dev"
mlflow.set_experiment(EXPERIMENT_NAME)

# The deployed agent endpoint (model serving)
AGENT_ENDPOINT = f"{MODEL_NAME}_{catalog}_{db}"[:60]

print(f"Experiment: {EXPERIMENT_NAME}")
print(f"Agent endpoint: {AGENT_ENDPOINT}")
print(f"Catalog: {catalog}, Schema: {dbName}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load or create the evaluation dataset
# MAGIC
# MAGIC We first check if an eval dataset exists in MLflow. If not, we create one with representative questions covering:
# MAGIC - Customer account lookups
# MAGIC - Billing and subscription queries
# MAGIC - Technical/product questions (RAG via Vector Search)
# MAGIC - Math calculations
# MAGIC - Questions that should be refused (safety)

# COMMAND ----------

eval_dataset_table = f"{catalog}.{dbName}.ai_agent_mlflow_eval"

try:
    eval_dataset = mlflow.genai.datasets.get_dataset(eval_dataset_table)
    print(f"Loaded existing eval dataset: {eval_dataset_table}")
    display(eval_dataset.to_df())
except Exception as e:
    if "does not exist" in str(e):
        print(f"Creating new eval dataset: {eval_dataset_table}")
        eval_dataset = mlflow.genai.datasets.create_dataset(eval_dataset_table)

        eval_records = spark.createDataFrame([
            # Customer account lookups
            {"inputs": '{"question": "Give me the information about john21@example.net"}'},
            {"inputs": '{"question": "What is the customer segment for john21@example.net?"}'},
            # Billing and subscriptions
            {"inputs": '{"question": "Summarize all subscriptions held by john21@example.net"}'},
            {"inputs": '{"question": "Does john21@example.net have any unpaid bills?"}'},
            # Technical / RAG questions
            {"inputs": '{"question": "How do I troubleshoot error code 01 on my wifi router?"}'},
            {"inputs": '{"question": "What are the step-by-step instructions for updating firmware on my ADSL-R500 router?"}'},
            {"inputs": '{"question": "How do I set up my new Smart Home Network system?"}'},
            # Math
            {"inputs": '{"question": "What is 150 * 3.5 + 27?"}'},
            # Conversational
            {"inputs": '{"question": "Hello, what can you help me with?"}'},
            # Edge case — should not fabricate data
            {"inputs": '{"question": "Give me account details for nonexistent_user@fake.com"}'},
        ])

        eval_dataset.merge_records(eval_records)
        print("Eval dataset created with 10 records.")
        display(eval_dataset.to_df())
    else:
        raise e

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Define the prediction function
# MAGIC
# MAGIC This calls the deployed agent via the Model Serving endpoint and extracts the final answer text.

# COMMAND ----------

from mlflow.deployments import get_deploy_client
import pandas as pd

deploy_client = get_deploy_client("databricks")

def predict_fn(question: str) -> str:
    """Call the deployed agent and extract the final answer text."""
    try:
        response = deploy_client.predict(
            endpoint=AGENT_ENDPOINT,
            inputs={
                "input": [{"role": "user", "content": question, "type": "message"}],
                "databricks_options": {"return_trace": True}
            }
        )

        output = response.get("output", [])
        # Extract the final assistant message text
        for item in reversed(output):
            if isinstance(item, dict):
                if item.get("role") == "assistant" and item.get("content"):
                    for c in item["content"]:
                        if c.get("type") == "output_text":
                            return c["text"]
                elif item.get("type") == "message" and item.get("content"):
                    for c in item["content"]:
                        if c.get("type") == "output_text":
                            return c["text"]
        return str(output)
    except Exception as e:
        return f"Error: {str(e)}"

# Quick test
test_answer = predict_fn("Give me the information about john21@example.net")
print(f"Test answer: {test_answer[:200]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Define scorers
# MAGIC
# MAGIC Using the same MLflow built-in scorers as the real-time evaluation, plus custom guidelines.

# COMMAND ----------

from mlflow.genai.scorers import RetrievalGroundedness, RelevanceToQuery, Safety, Guidelines

scorers = [
    RetrievalGroundedness(),
    RelevanceToQuery(),
    Safety(),
    Guidelines(
        guidelines="""
        Response must be done without showing reasoning.
        - don't mention that you need to look up things
        - do not mention tools or function used
        - do not tell your intermediate steps or reasoning
        - do not say "according to records" or "based on the data"
        """,
        name="steps_and_reasoning",
    ),
    Guidelines(
        guidelines="""
        For technical questions about routers, modems, firmware, or error codes,
        the agent MUST use the knowledge base search tool and ground its answer
        in the retrieved documentation. It should NOT answer from general knowledge.
        For customer account questions, the agent MUST look up the customer data first.
        The agent should never make up data like customer names, IDs, or billing amounts.
        """,
        name="retrieval_quality",
    ),
]

print(f"Configured {len(scorers)} scorers: {[type(s).__name__ if not hasattr(s, 'name') else s.name for s in scorers]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Run the evaluation
# MAGIC
# MAGIC This calls the agent for each eval record, scores the responses, and logs everything to MLflow.

# COMMAND ----------

print("Starting batch evaluation...")
with mlflow.start_run(run_name="batch_eval_telco_agent"):
    results = mlflow.genai.evaluate(
        data=eval_dataset,
        predict_fn=predict_fn,
        scorers=scorers
    )

print("Evaluation complete!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. View results
# MAGIC
# MAGIC The evaluation results are available in:
# MAGIC - The table below
# MAGIC - Your MLflow experiment UI (compare with previous runs)

# COMMAND ----------

display(results.tables["eval_results"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. (Optional) Evaluate from existing traces
# MAGIC
# MAGIC You can also evaluate against traces already captured from real user interactions in the app.
# MAGIC This lets you assess production quality without making new agent calls.

# COMMAND ----------

# Fetch recent traces from the experiment
import time

experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
ten_minutes_ago = int((time.time() - 600) * 1000)

try:
    traces = mlflow.search_traces(
        filter_string=f"attributes.timestamp_ms > {ten_minutes_ago} AND attributes.status = 'OK'",
        order_by=["attributes.timestamp_ms DESC"],
        max_results=10
    )
    print(f"Found {len(traces)} recent traces")
    if len(traces) > 0:
        display(traces[["request", "response", "trace_id", "timestamp_ms"]].head(10))
except Exception as e:
    print(f"Could not fetch traces: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. (Optional) Generate synthetic eval data from knowledge base
# MAGIC
# MAGIC Use Databricks `generate_evals_df` to auto-generate eval questions from your knowledge base documents.

# COMMAND ----------

try:
    from databricks.agents.evals import generate_evals_df

    docs = spark.table(f"{catalog}.{dbName}.knowledge_base")

    agent_description = """
    The Agent is a telco customer support chatbot that answers technical questions about
    products such as WiFi routers, ADSL modems, fiber installations, and smart home networks.
    It also handles customer account lookups, billing inquiries, and subscription management.
    """

    question_guidelines = """
    # User personas
    - A customer asking how to troubleshoot equipment issues
    - A customer asking about their account or billing
    - An internal support agent looking up policies

    # Example questions
    - How do I troubleshoot Error Code 01 on my ADSL-R500 router?
    - What are the firmware update steps for my Smart Home Network system?

    # Guidelines
    - Questions should be succinct and human-like
    - Mix technical and account-related questions
    """

    synthetic_evals = generate_evals_df(
        docs,
        num_evals=10,
        agent_description=agent_description,
        question_guidelines=question_guidelines,
    )
    display(synthetic_evals)

    # Optionally merge into the eval dataset
    # eval_dataset.merge_records(synthetic_evals)
    # print("Merged synthetic evals into dataset")
except Exception as e:
    print(f"Synthetic eval generation failed (optional): {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next steps
# MAGIC
# MAGIC - **Compare runs**: Open the MLflow experiment and select multiple eval runs to compare scores side by side
# MAGIC - **Iterate**: Modify the system prompt, add/remove tools, or tune the agent, then re-run this notebook to measure improvement
# MAGIC - **Labeling**: Use the MLflow UI to create labeling sessions where domain experts can review and correct agent answers
# MAGIC - **CI/CD**: Add this notebook as a task in your DAB job to run eval on every deployment
