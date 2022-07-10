"""
    Module containing auxiliary functions for data preparation and model training.
    Author: Anastasia Prokaieva
"""


import pyspark.sql.functions as F
from pyspark.sql import SparkSession

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

import mlflow

from hyperopt import STATUS_OK


def stratified_split_train_test(df, label, join_on, seed=42, frac = 0.1):
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


def export_df(table_name):
    """
    Read Delta Table, compute features with prepare_features and compute_service_features
    then convert into Pandas DF select the main DF for train and validation

    :tableName str: Delta table Name 
    :return: X and y pandas DF
    """
    spark = SparkSession.builder.getOrCreate()
    telco_df = spark.read.format("delta").table(f"{table_name}")

    telco_df = prepare_features(telco_df)
    telco_df  = compute_service_features(telco_df)

    dataset = telco_df.toPandas()
    X = dataset.drop(['customerID','Churn'], axis=1)
    y = dataset['Churn']

    return X, y


def compute_weights(y_train):
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


def write_into_delta_table(
    df,
    path,
    schema_option="overwriteSchema",
    mode="overwrite",
    table_type="managed"
):
    if table_type == "managed":
        df.write.format('delta').mode(mode).option(schema_option, "true").saveAsTable(path)
    else:
        # you need to provide the full path 
        # example : /mnt/project/delta_ 
        df.write.format('delta').mode(mode).option(schema_option, "true").save(path)


def train_model(params, X_train, y_train):
    """ 
    Function that calls the pipeline to train a model and Tune it with HyperOpt 

    :params dict: all hyperparameters of the model
    :return dict: return a dictionary that contains a status and the loss of the model 
    """
    model = build_model(params)
    model.fit(X_train, y_train)
    loss = log_loss(y_train, model.predict_proba(X_train))
    mlflow.log_metrics(
        {
            'log_loss': loss,
            'accuracy': accuracy_score(y_train, model.predict(X_train))
        }
    )

    return { 'status': STATUS_OK, 'loss': loss }


def build_preprocessor() -> Pipeline:
    """
        Builds model.
        :params dict: all hyperparameters of the model
        :return Pipeline: returns an sklearn Pipeline object 
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

    

    return Pipeline([
        ("preprocessor", ColumnTransformer(transformers, remainder="passthrough", sparse_threshold=0)),
        ("standardizer", StandardScaler()),
    ])


def build_model(params):

    if 'max_depth' in params: 
        # hyperopt supplies values as float but must be int
        params['max_depth']=int(params['max_depth'])   
    if 'min_child_weight' in params: 
        # hyperopt supplies values as float but must be int
        params['min_child_weight']=int(params['min_child_weight']) 
    if 'max_delta_step' in params: 
        # hyperopt supplies values as float but must be int
        params['max_delta_step']=int(params['max_delta_step']) 
        
    # all other hyperparameters are taken as given by hyperopt
    xgb_classifier = XGBClassifier(**params)

    return xgb_classifier


