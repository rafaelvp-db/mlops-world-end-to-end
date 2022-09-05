# Databricks notebook source
dbutils.widgets.text("db_name", "churn_mlops_anastasia_prokaieva")
dbutils.widgets.text("model_name", "telco_churn_model")
dbutils.widgets.text("model_version", "43")


db_name = dbutils.widgets.get("db_name")
model_name = dbutils.widgets.get("model_name")

# COMMAND ----------

# Get latest model version in Production 
from mlflow.tracking import MlflowClient
client = MlflowClient()
model_version_inProd = client.get_latest_versions(name = model_name, stages = ["Production"], )[-1].version
model_version_inProd

# COMMAND ----------

import json
from utils import export_df

# Get our PAT Token
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
os.environ["DATABRICKS_TOKEN"] = token

# Get some data
X_test, y_test = export_df(f"{db_name}.testing")
X_test.head()

# COMMAND ----------

import os
import requests
import numpy as np
import pandas as pd
import json

def create_tf_serving_json(data):
  return {'inputs': {name: data[name].tolist() for name in data.keys()} if isinstance(data, dict) else data.tolist()}

def score_model(dataset):
  url = 'https://e2-demo-field-eng.cloud.databricks.com/model-endpoint/telco_churn_model/43/invocations'
  headers = {'Authorization': f'Bearer {os.environ.get("DATABRICKS_TOKEN")}', 'Content-Type': 'application/json'}
  ds_dict = {'dataframe_split': dataset.to_dict(orient='split')} if isinstance(dataset, pd.DataFrame) else create_tf_serving_json(dataset)
  data_json = json.dumps(ds_dict, allow_nan=True)
  response = requests.request(method='POST', headers=headers, url=url, data=data_json)
  if response.status_code != 200:
    raise Exception(f'Request failed with status {response.status_code}, {response.text}')
  return response.json()

# Make predictions
result = score_model(X_test)
result

# COMMAND ----------


