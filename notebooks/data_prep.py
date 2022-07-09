# Databricks notebook source
# MAGIC %run ./scripts

# COMMAND ----------



# COMMAND ----------

reInitialize = None
# this part can be removed 
if reInitialize: 
  # creating a database if does not exist
  _ = spark.sql(f'create database if not exists {db_name}') # should be add drop database ? 
  # drop a table if exists 
  _ = spark.sql(f"""DROP TABLE IF EXISTS {db_name}.trainDF""")
  _ = spark.sql(f"""DROP TABLE IF EXISTS {db_name}.testDF""")
  

telco_df_raw = spark.read.option("header", True).option("inferSchema", True).csv(DATA_LOAD)
# validation set can be added, but the dataset is already pretty small 
df_train, df_test = stratified_split_train_test(df=telco_df_raw, frac=FRACTION,
                                                label="Churn", join_on="customerID", seed=SEED)

#df_train, df_validation = stratified_split_train_test(df=df_train, frac=FRACTION,
#                                                label="Churn", join_on="customerID", seed=SEED)

# Save data into Delta 
_= writeIntoDeltaTable(df_train, f"{db_name}.trainDF")
_= writeIntoDeltaTable(df_test, f"{db_name}.testDF")


# COMMAND ----------


