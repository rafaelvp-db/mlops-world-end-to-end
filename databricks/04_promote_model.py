# Databricks notebook source
dbutils.widgets.text("run_name", "XGB Final Model")
dbutils.widgets.text("experiment_name", "telco_churn_mlops_experiment")
dbutils.widgets.text("model_name", "telco_churn_model")

run_name = dbutils.widgets.get("run_name")
experiment_name = dbutils.widgets.get("experiment_name")
model_name = dbutils.widgets.get("model_name")

# COMMAND ----------

import mlflow

experiment_path = f"/Shared/{experiment_name}"
experiment = mlflow.get_experiment_by_name(experiment_path)

df = mlflow.search_runs(
  experiment_ids = [experiment.experiment_id],
  filter_string = "status = 'FINISHED'"
)

best_run_id = df.sort_values("metrics.test.log_loss")["run_id"].values[0]
best_run_id

# COMMAND ----------

model_uri = f"runs:/{best_run_id}/model"
model = mlflow.sklearn.load_model(model_uri = model_uri)

# COMMAND ----------

from mlflow.tracking import MlflowClient
client = MlflowClient()

model_version_info = client.get_latest_versions(name = model_name, stages = ["None"])[0]
target_version = None

if best_run_id == model_version_info.run_id:
  target_version = model_version_info.version
  client.transition_model_version_stage(
    name = model_name,
    version = target_version,
    stage = "Production",
    archive_existing_versions = True
  )

# COMMAND ----------


