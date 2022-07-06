# Databricks notebook source
# MAGIC %run ./scripts

# COMMAND ----------

# creating a database if does not exist
_ = spark.sql(f'create database if not exists {db_name}')
# drop a table if exists 
_ = spark.sql(f"""DROP TABLE IF EXISTS {db_name}.trainDF""")
_ = spark.sql(f"""DROP TABLE IF EXISTS {db_name}.testDF""")

# COMMAND ----------

telco_df_raw = spark.read.option("header", True).option("inferSchema", True).csv(DATA_LOAD)
df_train, df_test = stratified_split_train_test(df=telco_df_raw, frac=FRACTION,
                                                label="Churn", join_on="customerID", seed=SEED)

# Save data into Delta 
df_train.write.format('delta').mode('overwrite').option("overwriteSchema", "true").saveAsTable(f"{db_name}.trainDF")
df_test.write.format('delta').mode('overwrite').option("overwriteSchema", "true").saveAsTable(f"{db_name}.testDF")
