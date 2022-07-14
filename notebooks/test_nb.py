# Databricks notebook source
import mlflow
from mlflow.tracking import MlflowClient
import telco_churn_mlops

client = MlflowClient()
mlflow.set_experiment("/Shared/telco_churn_mlops_experiment")
model_version_info = client.get_model_version(name = "telco_churn_model", version = 21)
model = mlflow.sklearn.load_model(model_uri = model_version_info.source)
model

# COMMAND ----------

# MAGIC %fs ls /local_disk0/.ephemeral_nfs/envs/pythonEnv-557eb19e-9e68-4fbf-a743-a6fbd01aa42e/lib/python3.9/site-packages/telco_churn_mlops/pipelines/

# COMMAND ----------


