"""Offline agent evaluation using MLflow scorers.

Aligned with 02.1_agent_evaluation notebook scorers plus custom retrieval_quality.

Run with: uv run agent-evaluate
"""

import mlflow

from agent_server.scorers import get_offline_scorers


def main():
    """Run offline evaluation against test cases."""
    scorers = get_offline_scorers()

    eval_dataset_table = "ttan_demo_catalog_main.ai_agent_apps.ai_agent_mlflow_eval"

    try:
        eval_dataset = mlflow.genai.datasets.get_dataset(eval_dataset_table)
        print(f"Loaded eval dataset: {eval_dataset_table}")
    except Exception:
        print(f"Dataset {eval_dataset_table} not found. Using inline test cases.")
        eval_dataset = [
            {
                "inputs": {"question": "Give me the information about john21@example.net"},
            },
            {
                "inputs": {"question": "Summarize all subscriptions held by john21@example.net"},
            },
            {
                "inputs": {"question": "What are the step-by-step instructions for updating the firmware on my ADSL-R500 router?"},
            },
            {
                "inputs": {"question": "How do I troubleshoot error code 01 on my wifi router?"},
            },
            {
                "inputs": {"question": "What is 150 * 3.5 + 27?"},
            },
        ]

    print(f"Running evaluation with {len(scorers)} scorers...")
    with mlflow.start_run(run_name="telco_agent_offline_eval"):
        results = mlflow.genai.evaluate(
            data=eval_dataset,
            scorers=scorers,
        )
        print("Evaluation complete.")
        print(results.tables["eval_results"])


if __name__ == "__main__":
    main()
