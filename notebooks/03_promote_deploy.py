# Databricks notebook source
# !pip install scikit-learn==1.1.1
# !pip install xgboost==1.5.0

# COMMAND ----------

dbutils.widgets.text("db_name", "telcochurndb")
dbutils.widgets.text("run_name", "XGB Final Model")
dbutils.widgets.text("experiment_name", "telco_churn_mlops_experiment")
dbutils.widgets.text("user_name", "PLACE IT HERE")
dbutils.widgets.text("model_name", "telco_churn_model")

run_name = dbutils.widgets.get("run_name")
db_name = dbutils.widgets.get("db_name")
experiment_name = dbutils.widgets.get("experiment_name")
model_name = dbutils.widgets.get("model_name")
user_name = dbutils.widgets.get("user_name")


# COMMAND ----------

# MAGIC %md 
# MAGIC You can access experiments and query them with Spark by executing the following command
# MAGIC 
# MAGIC ```
# MAGIC display(spark.read.format("mlflow-experiment").load("3451728416871358").limit(3))
# MAGIC ```

# COMMAND ----------

import mlflow
archive_old_model = False

# here it depends if you modified your location where you store your artifact or not 
try:
  print(f"Your experiment will be placed under /Shared/{experiment_name}")
  experiment_path = f"/Shared/{experiment_name}"
  experiment = mlflow.get_experiment_by_name(experiment_path)
except:
  # if you used automl and kep MlFlow Artifact directory 
  # then you should follow this path 
  # /Users/user_name/databricks_automl/your_experiment_name 
  experiment_path = f"/Users/{user_name}/databricks_automl/{experiment_name}"

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

client.set_tag(best_run_id, key='demographic_vars', value='seniorCitizen, gender_Female')
client.set_tag(best_run_id, key='db_table', value=f'{db_name}.training')

# COMMAND ----------

# DBTITLE 1,Transition to Staging
target_version = None
target_stage = "Staging"
update_model_desc = None 
model_desc_str = """ This model predicts whether a customer will churn.  
It is used to update the Telco Churn Dashboard in DB SQL."""
model_version_desc = """This model version was built using XGBoost, with the best hyperparameters set identified with HyperOpt."""

print("Our target for the model is Staging")
try: 
  # Please keep in mind that if there is not a model in each of the stage this will give an error that the length of the list is not acceptable (because we are calling [0] the object)
  model_version_latest = client.get_latest_versions(name = model_name, stages = ["None"])[-1]
  print(f"Your latest model version in stage None is {model_version_latest.version}")
  print("")
  if best_run_id == model_version_latest.run_id:
    model_version_info = model_version_latest
    target_version = model_version_info.version
    
    client.transition_model_version_stage(
      name = model_name,
      version = target_version,
      stage = target_stage
    )

    #Updating model version in Staging 
    if update_model_desc:
      #The main model description, typically done once.
      client.update_registered_model(
        name=model_version_info.name,
        description=model_desc_str
      )

    #Gives more details on this specific model version
    client.update_model_version(
      name=model_version_info.name,
      version=model_version_info.version,
      description=model_version_desc
    )

    print(f"Transitioned version {target_version} of model {model_name} to {target_stage}")
  else:
    print("It seems that your best model is not in None Staging")
except:
  print( "There is not a model in a staging None, your best model was already transferred into a new Stage")
  

# COMMAND ----------

# DBTITLE 1,Test Predictions and Promote Target Stage into Production
try:  
  # Please keep in mind that if there is not a model in each of the stage this will give an error that the length of the list is not acceptable (because we are calling [0] the object)
  model_version_inStage = client.get_latest_versions(name = model_name, stages = ["Staging"], )[-1]
  if best_run_id == model_version_inStage.run_id: 
    model_version_info = model_version_inStage
    target_version = model_version_info.version
    print("Your model is already in Staging")
    from utils import export_df

    current_stage = target_stage
    model_version_info = client.get_latest_versions(name = model_name, stages = [current_stage])[-1]

    print("Loading our model to test predictions")
    model = mlflow.pyfunc.load_model(model_uri = model_version_info.source)
    X_test, y_test = export_df(f"{db_name}.testing")
    pred = model.predict(X_test.sample(10))
    
    if pred is not None:
      print("Predictions are OK")
      print(f"Moving target_stage to the Production, your model_version is {target_version}")
      target_stage = "Production"
    else:
      target_stage = "Production"
      print("You may need to go back an verify your model")
    
except:
  
  print("There is not model in a Staging")



# COMMAND ----------

# DBTITLE 1,Transitioning into Production
if target_stage == "Production":
  try:
    client.transition_model_version_stage(
      name = model_name,
      version = target_version,
      stage = target_stage,
      archive_existing_versions=False
    )
    print(f"Moving model version {target_version} of model {model_name} to {target_stage}")
  except:
    print(f"Please check that your model {model_name} passed necessary test before transitiining it into Production \nOR if it's not been transtioned into Production already ")

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



# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/><br/><a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>
