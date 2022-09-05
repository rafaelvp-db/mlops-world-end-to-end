# Databricks notebook source
dbutils.widgets.text("db_name", "churn_mlops_anastasia_prokaieva")
dbutils.widgets.text("model_name", "telco_churn_model")
dbutils.widgets.text("model_version", "43")


db_name = dbutils.widgets.get("db_name")
model_name = dbutils.widgets.get("model_name")

# COMMAND ----------

from utils import export_df
import mlflow
# Get latest model version in Production 
from mlflow.tracking import MlflowClient
client = MlflowClient()
# Getting the latest model version 
model_version_inProd = client.get_latest_versions(name = model_name, stages = ["Production"], )[-1]
model_version_inProd.version

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ##Deploying the model for batch inferences
# MAGIC 
# MAGIC <img style="float: right; margin-left: 20px" width="600" src="https://github.com/QuentinAmbard/databricks-demo/raw/main/retail/resources/images/churn_batch_inference.gif" />
# MAGIC 
# MAGIC Now that our model is available in the Registry, we can load it to compute our inferences and save them in a table to start building dashboards.
# MAGIC 
# MAGIC We will use MLFlow function to load a pyspark UDF and distribute our inference in the entire cluster. If the data is small, we can also load the model with plain python and use a pandas Dataframe.
# MAGIC 
# MAGIC If you don't know how to start, Databricks can generate a batch inference notebook in just one click from the model registry !

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Predictions using Pyfunc & Pandas

# COMMAND ----------

print("Loading our model to test predictions")
model = mlflow.pyfunc.load_model(model_uri = model_version_inProd.source)
X_test, y_test = export_df(f"{db_name}.testing")
pred = model.predict(X_test.sample(10))
pred

# COMMAND ----------

model_features = model.metadata.get_input_schema().input_names()
model_features

# COMMAND ----------

# MAGIC %md
# MAGIC ### Predictions using DeltaTable(FeatureTable) and SarkUDF

# COMMAND ----------

from databricks import feature_store
# Instantiate the feature store client
fs = feature_store.FeatureStoreClient()

## you can also load your model with a flavor of your model or with spark_udf 
model_spark = mlflow.pyfunc.spark_udf(spark, model_uri=f"models:/{model_name}/Production")
features = fs.read_table(f'{db_name}.telco_churn_features_ap')

predictions = features.withColumn('churnPredictions', model_spark(*model_features))
display(predictions.select("customerId", "churnPredictions"))

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ##Deploying the model for real-time inferences
# MAGIC 
# MAGIC <img style="float: right; margin-left: 20px" width="600" src="https://github.com/QuentinAmbard/databricks-demo/raw/main/retail/resources/images/churn_realtime_inference.gif" />
# MAGIC 
# MAGIC Our marketing team also needs to run inferences in real-time using REST api (send a customer ID and get back the inference).
# MAGIC 
# MAGIC While Feature store integration in real-time serving will come with Model Serving v2, you can deploy your Databricks Model in a single click.
# MAGIC 
# MAGIC Open the Model page and click on "Serving". It'll start your model behind a REST endpoint and you can start sending your HTTP requests!

# COMMAND ----------

import json

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


