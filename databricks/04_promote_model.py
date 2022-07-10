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

#TODO

#1. Activate serverless endpoint through the API
#2. Keep poking it until it finishes activating the endpoint
#3. Make sample requests to the API

# COMMAND ----------

# DBTITLE 1,Enable Endpoint
import requests

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
auth_header = {"Authorization" : "Bearer " + token}

endpoint_path = "/mlflow/endpoints-v2/enable"
payload = {
  "registered_model_name": model_name
}

host = "https://e2-demo-field-eng.cloud.databricks.com/api/2.0"
full_url = f"{host}{endpoint_path}"
response = requests.post(url = full_url, json = payload, headers = auth_header)

if response.status_code != 200:
  raise ValueError("Error making POST request to Mlflow API")

# COMMAND ----------

# DBTITLE 1,Check Endpoint Status
import time
import json

endpoint_enabled = False
attempt_number = 1

while not endpoint_enabled:
  time.sleep(2)
  print(f"Checking if Endpoint was enabled, attempt {attempt_number}...")
  endpoint_path = "/mlflow/endpoints-v2/get-status"
  full_url = f"{host}{endpoint_path}"
  response = requests.get(url = full_url, json = payload, headers = auth_header)
  json_response = json.loads(response.text)
  status = json_response["endpoint_status"]["state"]
  print(f"Current endpoint status: {status}")
  if status == "ENDPOINT_STATE_READY":
    endpoint_enabled = True
    print("Endpoint is enabled, exiting...")
  attempt_number += 1

# COMMAND ----------

# DBTITLE 1,Check Endpoint Version Status
version_endpoint_enabled = False
attempt_number = 1

max_attempts = 1000

while not version_endpoint_enabled:
  print(f"Checking if Endpoint for Version {target_version} was enabled, attempt {attempt_number}...")
  endpoint_path = "/mlflow/endpoints-v2/get-version-status"
  full_url = f"{host}{endpoint_path}"
  payload["endpoint_version_name"] = target_version
  response = requests.get(url = full_url, json = payload, headers = auth_header)
  json_response = json.loads(response.text)
  status = json_response["endpoint_status"]["service_status"]["state"]
  message = json_response["endpoint_status"]["service_status"]["message"]
  print(f"Current endpoint status: {status}, message: {message}")
  attempt_number += 1
  if attempt_number >= max_attempts:
    raise ValueError(f"Max attempts reached, last status: {status}")
  if status != "SERVICE_STATE_PENDING":
    print(f"Status: {status}, exiting...")
    pass
  time.sleep(10)

# COMMAND ----------


