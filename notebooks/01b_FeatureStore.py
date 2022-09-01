# Databricks notebook source
dbutils.widgets.text("reinitialize", "True")
# EXAMPLE for mine 
# Try to keep Python and SQL widget with the same name, this helps when you sqitch around python and sql variables 
dbutils.widgets.text("db_name", "churn_mlops_anastasia_prokaieva")

reinitialize = dbutils.widgets.get("reinitialize") # returns a str all the time, bool returns True 
db_name = dbutils.widgets.get("db_name")

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Creating a Feature Store Table

# COMMAND ----------

from databricks import feature_store
from utils import *
fs = feature_store.FeatureStoreClient()

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Creating Feature Store from an existing Delta Table 

# COMMAND ----------

fs.register_table(delta_table=f"{db_name}.full_set", primary_keys="customerID",
                  description=""" Raw Feature Table with minor preparation but cleaned, for future teams to work on and share it on churn dataset""",
                  tags={"useCase":"churn",
                        "team":"ds"
                       }
                 )

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC 
# MAGIC ## Write to Feature Store
# MAGIC 
# MAGIC <img src="https://github.com/QuentinAmbard/databricks-demo/raw/main/product_demos/mlops-end2end-flow-feature-store.png" style="float:right" width="500" />
# MAGIC 
# MAGIC Once our features are ready, we'll save them in Databricks Feature Store. Under the hood, features store are backed by a Delta Lake table.
# MAGIC 
# MAGIC This will allow discoverability and reusability of our feature accross our organization, increasing team efficiency.
# MAGIC 
# MAGIC Feature store will bring traceability and governance in our deployment, knowing which model is dependent of which set of features.
# MAGIC 
# MAGIC Make sure you're using the "Machine Learning" menu to have access to your feature store using the UI.

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Creating a feature store from freshly arriving data 

# COMMAND ----------

telco_df_raw = pd.read_csv("https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv", sep =',', header=0)
telco_df_raw = spark.createDataFrame(telco_df_raw)
# functions were predifined in utils 
telco_df_feat = compute_service_features(prepare_features(telco_df_raw))
display(telco_df_feat)

# COMMAND ----------

# Create the feature store based on df schema and write df to it
features_table = fs.create_table(
  name=f'{db_name}.telco_churn_features_ap',
  primary_keys=['customerID'],
  df=telco_df_feat, # you can create a table and write data into it later, keep in mind once created you should write into not recreate a table 
  description='Telco churn features preprocessed table, keep in mind a few categorical features were left asside only simple feature mapping was created. This set is ready to be consumed by the most of the models including AutoMl of Databricks.',
  tags={"useCase":"churn","team":"ds"}
)

# COMMAND ----------

# MAGIC %md 
# MAGIC Another way of creating a feature table example :
# MAGIC 
# MAGIC ```
# MAGIC from databricks.feature_store import FeatureStoreClient
# MAGIC 
# MAGIC fs = FeatureStoreClient()
# MAGIC 
# MAGIC # apply your prep function
# MAGIC churn_features_df = compute_churn_features(telcoDF)
# MAGIC 
# MAGIC churn_feature_table = fs.create_feature_table(
# MAGIC   name=f'{dbName}.churn_features',
# MAGIC   keys='customerID',
# MAGIC   schema=churn_features_df.spark.schema(),
# MAGIC   description='These features are derived from the u_ibm_telco_churn.bronze_customers table in the lakehouse.  We created dummy variables for the categorical columns, cleaned up their names, and added a boolean flag for whether the customer churned or not.  No aggregations were performed.'
# MAGIC )
# MAGIC 
# MAGIC fs.write_table(df=churn_features_df.to_spark(), name=f'{dbName}.churn_features', mode='overwrite')
# MAGIC ```

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Feature store in Production with MlFlow
# MAGIC 
# MAGIC ```
# MAGIC # mocking function example 
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC def train_model_fs():
# MAGIC   with mlflow.start_run() as run: 
# MAGIC     df = create_training_set_fs() # getting a delta table
# MAGIC     X, y = get_features(df) # create a split if necessary 
# MAGIC     model = pipeline_model(*params) # pipeline with the feature prep for this model
# MAGIC     model.fit(X,y) # fitting our pipeline object
# MAGIC     #logging the model into MlFlow and Feature Store
# MAGIC     fs.log_model(model, 
# MAGIC                   "artifact_name",
# MAGIC                   flavor, 
# MAGIC                   training_set,
# MAGIC                   registered_model_name, 
# MAGIC                   input_example, 
# MAGIC                   signature
# MAGIC                   )
# MAGIC                   
# MAGIC                   
# MAGIC train_model_fs(features_table_lookups)
# MAGIC ```

# COMMAND ----------



# COMMAND ----------

# MAGIC %md 
# MAGIC ## Examine FS tables and main functionalities 

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC 
# MAGIC ## Accelerating Churn model creation using Databricks Auto-ML
# MAGIC ### A glass-box solution that empowers data teams without taking away control
# MAGIC 
# MAGIC Databricks simplify model creation and MLOps. However, bootstraping new ML projects can still be long and inefficient. 
# MAGIC 
# MAGIC Instead of creating the same boilerplate for each new project, Databricks Auto-ML can automatically generate state of the art models for Classifications, regression, and forecast.
# MAGIC 
# MAGIC 
# MAGIC <img width="1000" src="https://github.com/QuentinAmbard/databricks-demo/raw/main/retail/resources/images/auto-ml-full.png"/>
# MAGIC 
# MAGIC <img style="float: right" width="600" src="https://github.com/QuentinAmbard/databricks-demo/raw/main/retail/resources/images/churn-auto-ml.png"/>
# MAGIC 
# MAGIC Models can be directly deployed, or instead leverage generated notebooks to boostrap projects with best-practices, saving you weeks of efforts.
# MAGIC 
# MAGIC ### Using Databricks Auto ML with our Churn dataset
# MAGIC 
# MAGIC Auto ML is available in the "Machine Learning" space. All we have to do is start a new Auto-ML experimentation and select the feature table we just created (`churn_features`)
# MAGIC 
# MAGIC Our prediction target is the `churn` column.
# MAGIC 
# MAGIC Click on Start, and Databricks will do the rest.
# MAGIC 
# MAGIC While this is done using the UI, you can also leverage the [python API](https://docs.databricks.com/applications/machine-learning/automl.html#automl-python-api-1)

# COMMAND ----------


