# Databricks notebook source
dbutils.widgets.text("db_name", "telcochurndb")
db_name = dbutils.widgets.get("db_name")

# COMMAND ----------

import os
import requests
import numpy as np
import pandas as pd

def create_tf_serving_json(data):
  return {'inputs': {name: data[name].tolist() for name in data.keys()} if isinstance(data, dict) else data.tolist()}

def score_model(dataset):
  url = 'https://e2-demo-field-eng.cloud.databricks.com/model-endpoint/telco_churn_model/18/invocations'
  headers = {'Authorization': f'Bearer {os.environ.get("DATABRICKS_TOKEN")}'}
  data_json = dataset.to_dict(orient='split') if isinstance(dataset, pd.DataFrame) else create_tf_serving_json(dataset)
  response = requests.request(method='POST', headers=headers, url=url, json=data_json)
  if response.status_code != 200:
    raise Exception(f'Request failed with status {response.status_code}, {response.text}')
  return response.json()

# COMMAND ----------

import json
from utils import export_df

# Get our PAT Token
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
os.environ["DATABRICKS_TOKEN"] = token

# Get some data
df_test = export_df(f"{db_name}.testing")

df_test.head()

# COMMAND ----------

# Make predictions
result = score_model(df_test[0])

# COMMAND ----------

result
