# Databricks notebook source
# MAGIC %run ./scripts

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

# MAGIC %md 
# MAGIC Saving Delta tables into parquet back to read them with Pandas 

# COMMAND ----------

trainDF = spark.read.format('delta').table("churn_mlops.trainDF")
trainDF = prepare_features(trainDF)
trainDF = compute_service_features(trainDF)

testDF = spark.read.format('delta').table("churn_mlops.testDF")
testDF = prepare_features(testDF)
testDF = compute_service_features(testDF)


# COMMAND ----------

display(trainDF)

# COMMAND ----------

def feature_prep():
  """
  :return object: return a sklearn Pipeline object 
  """
  transformers = []

  bool_pipeline = Pipeline(steps=[
      ("cast_type", FunctionTransformer(lambda df: df.astype(object))),
      ("imputer", SimpleImputer(missing_values=None, strategy="most_frequent")),
      ("onehot", OneHotEncoder(handle_unknown="ignore")),
  ])
  transformers.append(("boolean", bool_pipeline, 
                       ["Dependents", "PaperlessBilling", "Partner", "PhoneService", "SeniorCitizen"]))

  numerical_pipeline = Pipeline(steps=[
      ("converter", FunctionTransformer(lambda df: df.apply(pd.to_numeric, errors="coerce"))),
      ("imputer", SimpleImputer(strategy="mean"))
  ])
  transformers.append(("numerical", numerical_pipeline, 
                       ["AvgPriceIncrease", "Contract", "MonthlyCharges", "NumOptionalServices", "TotalCharges", "tenure"]))

  one_hot_pipeline = Pipeline(steps=[
      ("imputer", SimpleImputer(missing_values=None, strategy="constant", fill_value="")),
      ("onehot", OneHotEncoder(handle_unknown="ignore"))
  ])
  transformers.append(("onehot", one_hot_pipeline, 
                       ["DeviceProtection", "InternetService", "MultipleLines", "OnlineBackup", \
                        "OnlineSecurity", "PaymentMethod", "StreamingMovies", "StreamingTV", "TechSupport", "gender"]))

  pipeline = Pipeline([
      ("preprocessor", ColumnTransformer(transformers, remainder="passthrough", sparse_threshold=0)),
      ("standardizer", StandardScaler()),])

  return pipeline

# COMMAND ----------



# COMMAND ----------

trainDF.toPandas().to_parquet("/dbfs/Users/anastasia.prokaieva@databricks.com/demo/churn_mlops/train")
testDF.toPandas().to_parquet("/dbfs/Users/anastasia.prokaieva@databricks.com/demo/churn_mlops/test")

# COMMAND ----------


