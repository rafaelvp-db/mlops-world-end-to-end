# Databricks notebook source
dbutils.widgets.text("reinitialize", "True")
# EXAMPLE for mine 
# Try to keep Python and SQL widget with the same name, this helps when you sqitch around python and sql variables 
dbutils.widgets.text("db_name", "churn_mlops_anastasia_prokaieva")
dbutils.widgets.text("feature_table_name", "telco_churn_features_ap")


reinitialize = dbutils.widgets.get("reinitialize") # returns a str all the time, bool returns True 
db_name = dbutils.widgets.get("db_name")
fs_table_name = dbutils.widgets.get("feature_table_name")

# importing all of our prepared scripts
from utils import *

from databricks import feature_store
# Instantiate the feature store client
fs = feature_store.FeatureStoreClient()

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Creating a Feature Store Table

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Creating Feature Store from an existing Delta Table 

# COMMAND ----------

try:
  fs.register_table(delta_table=f"{db_name}.full_set",
                    primary_keys="customerID",
                    description=""" 
                                Raw Feature Table with minor preparation but cleaned,
                                for future teams to work on and share it on churn dataset
                                """,
                    tags={"useCase":"churn",
                          "team":"ds"
                         }
                   )
except:
  print("Something went wrong, please check that your table exists")

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

try:  
  # Create the feature store based on df schema and write df to it
  features_table = fs.create_table(
    name=f'{db_name}.{}',
    primary_keys=['customerID'],
    df=telco_df_feat, # you can create a table and write data into it later, keep in mind once created you should write into not recreate a table 
    description="""
                Telco churn features preprocessed table, keep in mind
                a few categorical features were left asside only simple
                feature mapping was created. This set is ready to be 
                consumed by the most of the models including AutoMl 
                of Databricks.""",
    tags={"useCase":"churn","team":"ds"}
  )
except:
  print("Something went wrong, please check that your data exists")

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



# COMMAND ----------

# MAGIC %md 
# MAGIC ## Examine FS tables and main functionalities 

# COMMAND ----------

# Drop feature table if it exists
# fs.drop_table(name=f'{dbName}.telco_churn_features'),
help(fs.drop_table)

# COMMAND ----------

# Verify the feature store has been created and populated successfully
features_table_df = fs.read_table(f'{db_name}.{fs_table_name}')
display(features_table_df)

# COMMAND ----------

# Get metadata about the feature store
features_table = fs.get_table(f'{db_name}.{fs_table_name}')

print(f" Feature Table description {features_table.description}\n\n This table was create by {features_table.notebook_producers[0].creator_id} \n\n Our table has {len(features_table.features)} features")

# COMMAND ----------

help(fs)

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Creating FeatureLookup Tables as a prep for Training 

# COMMAND ----------

features_table.features

# COMMAND ----------

 from databricks.feature_store import FeatureLookup

# Remove some features that you dont need (or define features you would like to see )
features = features_table.features
features.remove("Churn")
features.remove("customerID")
# Define the feature lookup - you can specify the list of features as well to use
features_table_lookup = FeatureLookup(table_name = features_table.name, 
                                      lookup_key = 'customerID',
                                      feature_names= features
                                     ) 
             
features_table_lookups  = [features_table_lookup]

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Create a Training Dataset
# MAGIC 
# MAGIC When `fs.create_training_set(..)` is invoked below, the following steps will happen:
# MAGIC 
# MAGIC 1. A `TrainingSet` object will be created, which will select specific features from Feature Store to use in training your model. Each feature is specified by the `FeatureLookup'`s created above.
# MAGIC 
# MAGIC 2. Features are joined with the raw input data according to each `FeatureLookup's lookup_key`.
# MAGIC 
# MAGIC The TrainingSet is then transformed into a DataFrame to train on. This DataFrame includes the columns of taxi_data, as well as the features specified in the FeatureLookups.

# COMMAND ----------

import mlflow
import mlflow.shap
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

mlflow.sklearn.autolog()

# Create the training dataframe. We will feed this dataframe to our model to perform the feature lookup from the feature store and then train the model
train_data_df = telco_df.select("customerID", "Churn")

# Define a method for reuse later
def fit_model(model_feature_lookups):

  with mlflow.start_run():
    # Use a combination of Feature Store features and data residing outside Feature Store in the training set
    training_set = fs.create_training_set(train_data_df,
                                          feature_lookups=model_feature_lookups, #feature_lookups = list1_feature_lookups + list2_feature_lookups,
                                          label="Churn",
                                          exclude_columns="customerID")

    training_pd = training_set.load_df().toPandas()
    X = training_pd.drop("Churn", axis=1)
    y = training_pd["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Not attempting to tune the model at all for purposes here
    gb_classifier = GradientBoostingClassifier(n_iter_no_change=10)
    # Need to encode categorical cols
    encoders = ColumnTransformer(transformers=[('encoder', OneHotEncoder(handle_unknown='ignore'), X.columns[X.dtypes == 'object'])])
    pipeline = Pipeline([("encoder", encoders), ("gb_classifier", gb_classifier)])
    pipeline_model = pipeline.fit(X_train, y_train)
    
    mlflow.log_metric('test_accuracy', pipeline_model.score(X_test, y_test))

    fs.log_model(
      pipeline_model,
      "model",
      flavor=mlflow.sklearn,
      training_set=training_set,
      registered_model_name=model_name,
      input_example=X[:100],
      signature=infer_signature(X, y))
      
fit_model(features_table_lookups)

# COMMAND ----------



# COMMAND ----------



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

# MAGIC %md 
# MAGIC #### Example of Training a model with Feature Store and MLFlow 

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

# MAGIC %md 
# MAGIC #### Example how to run AutoML with the API instead of UI
# MAGIC 
# MAGIC ```
# MAGIC 
# MAGIC # read the data
# MAGIC spark_df = spark.read.format("delta").table(f'{db_name}.YOURTABLE')
# MAGIC train_df, test_df = spark_df.randomSplit([.8, .2], seed=42)
# MAGIC # launch AutoML
# MAGIC summary = automl.classify(train_df, target_col="target",
# MAGIC                           primary_metric="roc_auc",
# MAGIC                           timeout_minutes=5,
# MAGIC                           max_trials=20
# MAGIC                           )
# MAGIC # load the best model and make predictions                          
# MAGIC model_uri = f"runs:/{summary.best_trial.mlflow_run_id}/model"
# MAGIC predict = mlflow.pyfunc.spark_udf(spark, model_uri)
# MAGIC pred_df = test_df.withColumn("prediction", predict(*test_df.drop("target").columns))
# MAGIC 
# MAGIC ```

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Using the generated notebook to build our model
# MAGIC 
# MAGIC Next step: [Explore the generated Auto-ML notebook]($./02a_automl_code_generation)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md ## (Optional) Online Feature Store 
# MAGIC 
# MAGIC Example code how to publish your offline feature store online. 
# MAGIC Please consult our [the official documentation](https://docs.microsoft.com/en-us/azure/databricks/applications/machine-learning/feature-store/feature-tables#publish-features-to-an-online-feature-store) for the recent updates regarding the connection to the external SQL databases. 
# MAGIC 
# MAGIC ```
# MAGIC # 
# MAGIC # Step 1 Create database with the same name in the online store (Azure MySQL here)
# MAGIC # 
# MAGIC scope = "online_fs"
# MAGIC user = dbutils.secrets.get(scope, "ofs-user")
# MAGIC password = dbutils.secrets.get(scope, "ofs-password")
# MAGIC import mysql.connector
# MAGIC import pandas as pd
# MAGIC cnx = mysql.connector.connect(user=user,
# MAGIC                               password=password,
# MAGIC                               host=<hostname>)
# MAGIC cursor = cnx.cursor()
# MAGIC cursor.execute(f"CREATE DATABASE IF NOT EXISTS {dbName};")
# MAGIC 
# MAGIC #
# MAGIC # Step 2 Publish your data online
# MAGIC #
# MAGIC import datetime
# MAGIC from databricks.feature_store.online_store_spec import AzureMySqlSpec
# MAGIC  
# MAGIC online_store = AzureMySqlSpec(
# MAGIC   hostname=<hostname>,
# MAGIC   port=3306,
# MAGIC   read_secret_prefix='online_fs/ofs',
# MAGIC   write_secret_prefix='online_fs/ofs'
# MAGIC )
# MAGIC  
# MAGIC fs.publish_table(
# MAGIC   name=f'{dbName}.demographic_features',
# MAGIC   online_store=online_store,
# MAGIC   mode='overwrite'
# MAGIC )
# MAGIC  
# MAGIC fs.publish_table(
# MAGIC   name=f'{dbName}.service_features',
# MAGIC   online_store=online_store,
# MAGIC   mode='overwrite'
# MAGIC )
# MAGIC 
# MAGIC ```

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/><br/><a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>
