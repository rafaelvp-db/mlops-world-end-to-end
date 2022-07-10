# Databricks notebook source
dbutils.widgets.text("run_name", "XGB Final Model")
dbutils.widgets.text("experiment_name", "telco_churn_mlops_experiment")
dbutils.widgets.text("model_name", "telco_churn_model")

run_name = dbutils.widgets.get("run_name")
experiment_name = dbutils.widgets.get("experiment_name")
model_name = dbutils.widgets.get("model_name")

# COMMAND ----------

from mlflow.tracking import MlflowClient

stage = "Production"
client = MlflowClient()
model_info = client.get_latest_versions(name = model_name, stages = [stage])[0]
model = mlflow.pyfunc.load_model(model_uri = f"{source}/)

# COMMAND ----------

model_info

# COMMAND ----------


