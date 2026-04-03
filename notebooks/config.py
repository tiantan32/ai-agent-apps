# Databricks notebook source
catalog = "ttan_demo_catalog_main"
schema = dbName = db = "ai_agent_apps"

volume_name = "raw_data"
VECTOR_SEARCH_ENDPOINT_NAME="dbdemos_vs_endpoint"

MODEL_NAME = "dbdemos_ai_agent_demo"
ENDPOINT_NAME = f'{MODEL_NAME}_{catalog}_{db}'[:60]

# This must be a tool-enabled model
LLM_ENDPOINT_NAME = 'databricks-claude-3-7-sonnet'
