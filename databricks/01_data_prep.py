# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Telco Churn Prediction
# MAGIC ## 1. Data Preparation

# COMMAND ----------

# DBTITLE 1,Declaring our input widgets
dbutils.widgets.text("reinitialize", "True")
dbutils.widgets.text("db_name", "telcochurndb")
dbutils.widgets.text("input_data_path", "dbfs:/tmp/ibm_telco_churn.csv")

reinitialize = bool(dbutils.widgets.get("reinitialize"))
db_name = dbutils.widgets.get("db_name")
input_data_path = dbutils.widgets.get("input_data_path")


# COMMAND ----------

# DBTITLE 1,Importing our library
from utils import stratified_split_train_test, write_into_delta_table

# COMMAND ----------

if reinitialize: 
  spark.sql(f"drop database if exists {db_name}")
  spark.sql(f"create database {db_name}")

telco_df_raw = spark.read.option("header", True).csv(input_data_path)
df_train, df_test = stratified_split_train_test(
  df = telco_df_raw,
  label="Churn",
  join_on="customerID",
)

write_into_delta_table(df_train, f"{db_name}.training")
write_into_delta_table(df_test, f"{db_name}.testing")
