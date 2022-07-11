# Databricks notebook source
!pip install scikit-learn==1.1.1
!pip install xgboost==1.5.0

# COMMAND ----------

dbutils.widgets.text("db_name", "telcochurndb")
dbutils.widgets.text("run_name", "XGB Final Model")
dbutils.widgets.text("experiment_name", "telco_churn_mlops_experiment")
dbutils.widgets.text("model_name", "telco_churn_model")

run_name = dbutils.widgets.get("run_name")
db_name = dbutils.widgets.get("db_name")
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

# DBTITLE 1,Setting tags to our model
from mlflow.tracking import MlflowClient
client = MlflowClient()

client.set_tag(run_id, key='demographic_vars', value='seniorCitizen,gender_Female')
client.set_tag(run_id, key='db_table', value=f'{db_name}.training')

# COMMAND ----------

# DBTITLE 1,Transition to Staging
model_version_info = client.get_latest_versions(name = model_name, stages = ["None"])[0]
target_version = None
target_stage = "Staging"

if best_run_id == model_version_info.run_id:
  target_version = model_version_info.version
  client.transition_model_version_stage(
    name = model_name,
    version = target_version,
    stage = target_stage
  )
  
  #Updating model version in Staging 
  #The main model description, typically done once.
  client.update_registered_model(
    name=model_version_info.name,
    description="This model predicts whether a customer will churn.  It is used to update the Telco Churn Dashboard in DB SQL."
  )

  #Gives more details on this specific model version
  client.update_model_version(
    name=model_version_info.name,
    version=model_version_info.version,
    description="This model version was built using XGBoost, with the best hyperparameters set identified with HyperOpt."
  )
  
print(f"Transitioned version {target_version} of model {model_name} to {target_stage}")

# COMMAND ----------

# DBTITLE 1,Test Predictions
from utils import export_df

current_stage = target_stage
model_version_info = client.get_latest_versions(name = model_name, stages = [current_stage])[0]

model = mlflow.sklearn.load_model(model_uri = model_version_info.source)
X_test, y_test = export_df(f"{db_name}.testing")
pred = model.predict(X_test.sample(10))
if pred is not None:
  print("Predictions OK")
  target_version = None
  target_stage = "Production"

# COMMAND ----------

# DBTITLE 1,Promote to Production
if best_run_id == model_version_info.run_id:
  target_version = model_version_info.version
  client.transition_model_version_stage(
    name = model_name,
    version = target_version,
    stage = target_stage,
    archive_existing_versions = True
  )
  
print(f"Transitioned version {target_version} of model {model_name} to {target_stage}")

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
    print(f"*** Status: {status}, exiting... ***")
    break
  time.sleep(10)

# COMMAND ----------


