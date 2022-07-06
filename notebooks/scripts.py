# Databricks notebook source
# *****
# Importing all modules
# *****
import pyspark.sql.functions as F

from delta.tables import DeltaTable
import tempfile
import os
import numpy as np
import pandas as pd


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.compose import make_column_selector as selector
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score, average_precision_score

from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from xgboost import XGBClassifier


from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import RestException
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec

from hyperopt import hp, fmin, tpe, SparkTrials, STATUS_OK, space_eval


# COMMAND ----------

# *****
# Setting Parameters - can be passed into a file
# *****

db_name = "churn_mlops"
DATA_LOAD = "/mnt/databricks-datasets-private/ML/telco_churn/Telco-Customer-Churn.csv"

model_type = "xgb"
SEED = 454343
FRACTION = 0.8

model_registry_home = "/Shared/dbx/projects/"
model_name_registry = "churn_mlops"



# COMMAND ----------

# *****
# Functions
# *****

def stratified_split_train_test(df, frac, label, join_on, seed=42):
    """ 
    Stratfied split of a Spark DataDrame into a Train and Test sets
    """
    fractions = df.select(label).distinct().withColumn("fraction", F.lit(frac)).rdd.collectAsMap()
    df_frac = df.stat.sampleBy(label, fractions, seed)
    df_remaining = df.join(df_frac, on=join_on, how="left_anti")
    return df_frac, df_remaining

def prepare_features(sparkDF):
    # 0/1 -> boolean
    sparkDF = sparkDF.withColumn("SeniorCitizen", F.col("SeniorCitizen") == 1)
    # Yes/No -> boolean
    for yes_no_col in ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]:
      sparkDF = sparkDF.withColumn(yes_no_col, F.col(yes_no_col) == "Yes")
    sparkDF = sparkDF.withColumn("Churn", F.when(F.col("Churn") == "Yes", 1).otherwise(0))
    
    # Contract categorical -> duration in months
    sparkDF = sparkDF.withColumn("Contract",\
        F.when(F.col("Contract") == "Month-to-month", 1).\
        when(F.col("Contract") == "One year", 12).\
        when(F.col("Contract") == "Two year", 24))

    # Converting no Internet options into negative values 
    for icol in ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]:
      sparkDF = sparkDF.withColumn(str(icol),\
                             (F.when(F.col(icol) == "Yes", 1).when(F.col(icol) == "No internet service", -1).\
                             otherwise(0))
                                  )
      
    sparkDF = sparkDF.withColumn("MultipleLines",\
                             (F.when(F.col("MultipleLines") == "Yes", 1).when(F.col("MultipleLines") == "No phone service", -1).\
                             otherwise(0))
                                  )                              
    # Empty TotalCharges -> NaN
    sparkDF = sparkDF.withColumn("TotalCharges",\
        F.when(F.length(F.trim(F.col("TotalCharges"))) == 0, None).\
        otherwise(F.col("TotalCharges").cast('double')))

    return sparkDF
  
def compute_service_features(sparkDF):
    @F.pandas_udf('int')
    def num_optional_services(*cols):
      return sum(map(lambda s: (s == 1).astype('int'), cols))
    
    @F.pandas_udf('int')
    def num_no_services(*cols):
      return sum(map(lambda s: (s == -1).astype('int'), cols))

    # Below also add AvgPriceIncrease: current monthly charges compared to historical average
    sparkDF = sparkDF.fillna({"TotalCharges": 0.0}).\
      withColumn("NumOptionalServices",
          num_optional_services("OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies")).\
      withColumn("NumNoInternetServices",
          num_no_services("OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies")).\
      withColumn("AvgPriceIncrease",
          F.when(F.col("tenure") > 0, (F.col("MonthlyCharges") - (F.col("TotalCharges") / F.col("tenure")))).otherwise(0.0))
    
    return sparkDF  

# COMMAND ----------

def data4model(tableName):
  """
  Read Delta Table, compute features with prepare_features and compute_service_features
  then convert into Pandas DF select the main DF for train and validation
  
  :tableName str: Delta table Name 
  :return: X and y pandas DF
  """
  telco_df = spark.read.format("delta").table(f"{tableName}")

  telco_df = prepare_features(telco_df)
  telco_df  = compute_service_features(telco_df)

  dataset = telco_df.toPandas()
  X = dataset.drop(['customerID','Churn'], axis=1)
  y = dataset['Churn']
  
  return X, y

def weight_compute(y_train):
  """
  Define minimum positive class scale factor
  """
  weights = compute_class_weight(
    'balanced', 
    classes=np.unique(y_train), 
    y=y_train
    )
  scale = weights[1]/weights[0]
  return scale

# COMMAND ----------


