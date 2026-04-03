# Databricks notebook source
# MAGIC %md
# MAGIC # One-time Setup: Link MLflow Experiment to Unity Catalog for Tracing
# MAGIC
# MAGIC This notebook links a **new** MLflow experiment to a Unity Catalog schema so that
# MAGIC all traces are stored in UC instead of MLflow experiment storage.
# MAGIC
# MAGIC **Requirement:** The experiment must have no existing traces — hence we create a fresh one.
# MAGIC After running, copy the printed experiment ID into `app.yaml` → `MLFLOW_EXPERIMENT_ID`.

# COMMAND ----------

# MAGIC %pip install "mlflow[databricks]>=3.9.0,<3.11" --upgrade --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
import mlflow
from mlflow.entities import UCSchemaLocation
from mlflow.tracing.enablement import set_experiment_trace_location

mlflow.set_tracking_uri("databricks")

# ── Configuration ────────────────────────────────────────────────────────────
CATALOG_NAME      = "ttan_demo_catalog_main"
SCHEMA_NAME       = "ai_agent_apps"
EXPERIMENT_NAME   = "/Users/tian.tan@databricks.com/ai-agent-telco-app"   # already linked to UC
SQL_WAREHOUSE_ID  = "11d6beea79229431"
# ─────────────────────────────────────────────────────────────────────────────

os.environ["MLFLOW_TRACING_SQL_WAREHOUSE_ID"] = SQL_WAREHOUSE_ID

# Create a fresh experiment (must have zero traces for linking to succeed)
existing = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if existing:
    experiment_id = existing.experiment_id
    print(f"Using existing experiment: {experiment_id}")
    print("⚠️  WARNING: If this experiment already has traces, linking will fail.")
    print("   Delete it or choose a different EXPERIMENT_NAME and re-run.")
else:
    experiment_id = mlflow.create_experiment(name=EXPERIMENT_NAME)
    print(f"Created new experiment: {experiment_id}")

# COMMAND ----------

# Link the UC schema to the experiment
result = set_experiment_trace_location(
    location=UCSchemaLocation(
        catalog_name=CATALOG_NAME,
        schema_name=SCHEMA_NAME,
    ),
    experiment_id=experiment_id,
)

print(f"\n✅ Linking successful!")
print(f"   Spans table : {result.full_otel_spans_table_name}")
print(f"\n👉 Update app.yaml  →  MLFLOW_EXPERIMENT_ID: \"{experiment_id}\"")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Grant permissions (run as workspace admin or catalog owner)
# MAGIC
# MAGIC Users / service principals that write or read traces need explicit grants.
# MAGIC `ALL_PRIVILEGES` is **not** sufficient — each permission must be granted individually.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Replace <principal> with the user email or service principal application ID
# MAGIC -- that runs the Databricks App (agent_server).

# MAGIC GRANT USE_CATALOG ON CATALOG ttan_demo_catalog_main TO `<principal>`;
# MAGIC GRANT USE_SCHEMA  ON SCHEMA  ttan_demo_catalog_main.ai_agent_apps TO `<principal>`;
# MAGIC GRANT MODIFY, SELECT ON TABLE ttan_demo_catalog_main.ai_agent_apps.mlflow_experiment_trace_otel_logs    TO `<principal>`;
# MAGIC GRANT MODIFY, SELECT ON TABLE ttan_demo_catalog_main.ai_agent_apps.mlflow_experiment_trace_otel_metrics TO `<principal>`;
# MAGIC GRANT MODIFY, SELECT ON TABLE ttan_demo_catalog_main.ai_agent_apps.mlflow_experiment_trace_otel_spans   TO `<principal>`;
