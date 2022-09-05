# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Telco Churn Prediction Data Preparation

# COMMAND ----------

#!pip install bamboolib

# COMMAND ----------

# DBTITLE 1,Declaring our input widgets
dbutils.widgets.text("reinitialize", "True")
dbutils.widgets.text("db_name", "INSERT YOUR DATABASE NAME PLEASE") # insert you database into the widget
# EXAMPLE for mine 
# Try to keep Python and SQL widget with the same name, this helps when you sqitch around python and sql variables 
# dbutils.widgets.text("db_name", "churn_mlops_anastasia_prokaieva")

reinitialize = dbutils.widgets.get("reinitialize") # returns a str all the time, bool returns True 
db_name = dbutils.widgets.get("db_name")

# COMMAND ----------

# MAGIC %md 
# MAGIC ## End-to-End MLOps demo with MLFlow and Auto ML
# MAGIC 
# MAGIC ### Challenges moving ML project into production
# MAGIC 
# MAGIC 
# MAGIC Moving ML project from a standalone notebook to a production-grade data pipeline is complex and require multiple competencies. 
# MAGIC 
# MAGIC Having a model up and running in a notebook isn't enough. We need to cover the end to end ML Project life cycle and solve the following challenges:
# MAGIC 
# MAGIC * Update data over time (production-grade ingestion pipeline)
# MAGIC * How to save, share and re-use ML features in the organization
# MAGIC * How to ensure a new model version respect quality standard and won't break the pipeline
# MAGIC * Model governance: what is deployed, how is it trained, by who, which data?
# MAGIC * How to monitor and re-train the model...
# MAGIC 
# MAGIC In addition, these project typically invole multiple teams, creating friction and potential silos
# MAGIC 
# MAGIC * Data Engineers, in charge of ingesting, preparing and exposing the data
# MAGIC * Data Scientist, expert in data analysis, building ML model
# MAGIC * ML engineers, setuping the ML infrastructure pipelines (similar to devops)
# MAGIC 
# MAGIC This has a real impact on the business, slowing down projects and preventing them from being deployed in production and bringing ROI.
# MAGIC 
# MAGIC ## What's MLOps ?
# MAGIC 
# MAGIC MLOps is is a set of standards, tools, processes and methodology that aims to optimize time, efficiency and quality while ensuring governance in ML projects.
# MAGIC 
# MAGIC MLOps orchestrate a project life-cycle and adds the glue required between the component and teams to smoothly implement such ML pipelines.
# MAGIC 
# MAGIC Databricks is uniquely positioned to solve this challenge with the Lakehouse pattern. Not only we bring Data Engineers, Data Scientists and ML Engineers together in a unique platform, but we also provide tools to orchestrate ML project and accelerate the go to production.
# MAGIC 
# MAGIC ## MLOps pipeline we'll implement
# MAGIC 
# MAGIC In this demo, we'll implement a full MLOps pipeline, step by step:
# MAGIC 
# MAGIC <img src="https://github.com/QuentinAmbard/databricks-demo/raw/main/product_demos/mlops-end2end-flow-0.png" width="1200">

# COMMAND ----------



# COMMAND ----------

# MAGIC %md 
# MAGIC ### Exploring our data with Bamboolib 

# COMMAND ----------

# STEP 1: RUN THIS CELL TO INSTALL BAMBOOLIB

# You can also install bamboolib on the cluster. Just talk to your cluster admin for that
#%pip install bamboolib # I have it installed directly on my cluster 

# COMMAND ----------

# STEP 2: RUN THIS CELL TO IMPORT AND USE BAMBOOLIB

import bamboolib as bam

# This opens a UI from which you can import your data
bam  

# Already have a pandas data frame? Just display it!
# Here's an example
# import pandas as pd
# df_test = pd.DataFrame(dict(a=[1,2]))
# df_test  # <- You will see a green button above the data set if you display it

# COMMAND ----------

telco_df_raw = pd.read_csv("https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv", sep =',', header=0)
telco_df_raw

# COMMAND ----------

import plotly.express as px
fig = px.histogram(telco_df_raw, x='SeniorCitizen', color='Churn')
fig

# COMMAND ----------

import plotly.express as px
fig = px.box(telco_df_raw, x='MonthlyCharges', y='MultipleLines', facet_row='SeniorCitizen', color='Churn')
fig

# COMMAND ----------

import plotly.express as px
fig = px.density_heatmap(telco_df_raw, x='Churn', y='MonthlyCharges', facet_row='SeniorCitizen', facet_col='MultipleLines')
fig

# COMMAND ----------

import plotly.express as px
fig = px.treemap(telco_df_raw, path=['Churn', 'gender', 'SeniorCitizen', 'MultipleLines','Contract'])
fig

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Feature Prep ingesting Data into Delta Table or Feature Store 

# COMMAND ----------

from utils import *
telco_df_raw = pd.read_csv("https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv", sep =',', header=0)
telco_df_raw = spark.createDataFrame(telco_df_raw)
df_train, df_test = stratified_split_train_test(
df = telco_df_raw,
label="Churn",
join_on="customerID",
)

if reinitialize == "True":
  write_into_delta_table(df_train, f"{db_name}.full_set")
  write_into_delta_table(df_train, f"{db_name}.training")
  write_into_delta_table(df_test, f"{db_name}.testing")


# COMMAND ----------

# MAGIC %md 
# MAGIC #### explore data vis inside the Databricks Notebook Cell

# COMMAND ----------

display(telco_df_raw)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Example with Koalas 
# MAGIC 
# MAGIC Because our Data Scientist team is familiar with Pandas, we'll use `koalas` to scale `pandas` code. The Pandas instructions will be converted in the spark engine under the hood and distributed at scale.
# MAGIC 
# MAGIC *Note: Starting from `spark 3.2`, koalas is builtin and we can get an Pandas Dataframe using `to_pandas_on_spark`.*

# COMMAND ----------

import pyspark.pandas as ps

def compute_churn_features_koalas(data):
  
  # Convert to koalas
  data = data.to_pandas_on_spark()
  
  # OHE
  data = ps.get_dummies(data, 
                        columns=['gender', 'Partner', 'Dependents',
                                 'PhoneService', 'MultipleLines', 'InternetService',
                                 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                                 'TechSupport', 'StreamingTV', 'StreamingMovies',
                                 'Contract', 'PaperlessBilling', 'PaymentMethod'],dtype = 'int64')
  
  # Convert label to int and rename column
  data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})
  data = data.astype({'Churn': 'int32'})
  
  # Clean up column names
  data.columns = data.columns.str.replace(' ', '')
  data.columns = data.columns.str.replace('(', '-')
  data.columns = data.columns.str.replace(')', '')
  
  # Drop missing values
  data = data.dropna()
  
  return data

# COMMAND ----------

trying_koalas = compute_churn_features_koalas(telco_df_raw)
display(trying_koalas)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/><br/><a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>
