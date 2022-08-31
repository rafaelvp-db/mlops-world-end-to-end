# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Telco Churn Prediction
# MAGIC ## 1. Data Preparation

# COMMAND ----------

# DBTITLE 1,Declaring our input widgets
dbutils.widgets.text("reinitialize", "True")
dbutils.widgets.text("db_name", "telcochurndb")

reinitialize = dbutils.widgets.get("reinitialize") # returns a str all the time, bool returns True 
db_name = dbutils.widgets.get("db_name")

# COMMAND ----------

# MAGIC %md 
# MAGIC # End-to-End MLOps demo with MLFlow and Auto ML
# MAGIC 
# MAGIC ## Challenges moving ML project into production
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

# COMMAND ----------

telco_df_raw

# COMMAND ----------

import plotly.express as px
fig = px.histogram(telco_df_raw, x='SeniorCitizen', color='Churn')
fig

# COMMAND ----------

import plotly.express as px
fig = px.treemap(telco_df_raw, path=['Churn', 'gender', 'SeniorCitizen', 'MultipleLines','Contract'])
fig

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md 
# MAGIC ### Feature Prep ingesting Data into Delta Table or Feature Store 

# COMMAND ----------

# DBTITLE 1,Importing our library
from utils import *

# COMMAND ----------

telco_df_raw = spark.createDataFrame(telco_df_raw)
df_train, df_test = stratified_split_train_test(
df = telco_df_raw,
label="Churn",
join_on="customerID",
)

write_into_delta_table(df_train, f"{db_name}.training")
write_into_delta_table(df_test, f"{db_name}.testing")


# COMMAND ----------


