"""
    Module containing auxiliary functions for data preparation and model training.
    Authors: Anastasia Prokaieva & Rafael V. Pierre
"""


# *****
# Importing Modules
# *****


import tempfile
import os
import pickle

import numpy as np
import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from sklearn.utils.class_weight import compute_class_weight


# *****
# Data Ingestion / Read / Write Section
# *****


def stratified_split_train_test(df, label, join_on, seed=42, frac=0.1):
    """
    Stratfied split of a Spark DataDrame into a Train and Test sets
    """
    fractions = (
        df.select(label)
        .distinct()
        .withColumn("fraction", F.lit(frac))
        .rdd.collectAsMap()
    )
    df_frac = df.stat.sampleBy(label, fractions, seed)
    df_remaining = df.join(df_frac, on=join_on, how="left_anti")
    return df_frac, df_remaining


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
    telco_df = compute_service_features(telco_df)

    dataset = telco_df.toPandas()
    X = dataset.drop(["customerID", "Churn"], axis=1)
    y = dataset["Churn"]

    return X, y


def write_into_delta_table(
    df, path, schema_option="overwriteSchema", mode="overwrite", table_type="managed"
):
    if table_type == "managed":
        df.write.format("delta").mode(mode).option(schema_option, "true").saveAsTable(
            path
        )
    else:
        # you need to provide the full path
        # example : /mnt/project/delta_
        df.write.format("delta").mode(mode).option(schema_option, "true").save(path)

  
# *****
# Data Prep Section
# *****  

def prepare_features(sparkDF):
    # 0/1 -> boolean
    sparkDF = sparkDF.withColumn("SeniorCitizen", F.col("SeniorCitizen") == 1)
    # Yes/No -> boolean
    for yes_no_col in ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]:
        sparkDF = sparkDF.withColumn(yes_no_col, F.col(yes_no_col) == "Yes")
    sparkDF = sparkDF.withColumn(
        "Churn", F.when(F.col("Churn") == "Yes", 1).otherwise(0)
    )

    # Contract categorical -> duration in months
    sparkDF = sparkDF.withColumn(
        "Contract",
        F.when(F.col("Contract") == "Month-to-month", 1)
        .when(F.col("Contract") == "One year", 12)
        .when(F.col("Contract") == "Two year", 24),
    )

    # Converting no Internet options into negative values
    for icol in [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]:
        sparkDF = sparkDF.withColumn(
            str(icol),
            (
                F.when(F.col(icol) == "Yes", 1)
                .when(F.col(icol) == "No internet service", -1)
                .otherwise(0)
            ),
        )

    sparkDF = sparkDF.withColumn(
        "MultipleLines",
        (
            F.when(F.col("MultipleLines") == "Yes", 1)
            .when(F.col("MultipleLines") == "No phone service", -1)
            .otherwise(0)
        ),
    )
    # Empty TotalCharges -> NaN
    sparkDF = sparkDF.withColumn(
        "TotalCharges",
        F.when(F.length(F.trim(F.col("TotalCharges"))) == 0, None).otherwise(
            F.col("TotalCharges").cast("double")
        ),
    )

    return sparkDF


def compute_service_features(sparkDF):
    @F.pandas_udf("int")
    def num_optional_services(*cols):
        return sum(map(lambda s: (s == 1).astype("int"), cols))

    @F.pandas_udf("int")
    def num_no_services(*cols):
        return sum(map(lambda s: (s == -1).astype("int"), cols))

    # Below also add AvgPriceIncrease: current monthly charges compared to historical average
    sparkDF = (
        sparkDF.fillna({"TotalCharges": 0.0})
        .withColumn(
            "NumOptionalServices",
            num_optional_services(
                "OnlineSecurity",
                "OnlineBackup",
                "DeviceProtection",
                "TechSupport",
                "StreamingTV",
                "StreamingMovies",
            ),
        )
        .withColumn(
            "NumNoInternetServices",
            num_no_services(
                "OnlineSecurity",
                "OnlineBackup",
                "DeviceProtection",
                "TechSupport",
                "StreamingTV",
                "StreamingMovies",
            ),
        )
        .withColumn(
            "AvgPriceIncrease",
            F.when(
                F.col("tenure") > 0,
                (F.col("MonthlyCharges") - (F.col("TotalCharges") / F.col("tenure"))),
            ).otherwise(0.0),
        )
    )

    return sparkDF


def to_object(df):

    df = df.astype(object)
    return df


def to_numeric(df):

    df = df.apply(pd.to_numeric, errors="coerce")
    return df

  
def compute_weights(y_train):
    """
    Define minimum positive class scale factor
    """
    weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    scale = weights[1] / weights[0]
    return scale
  

# *****
# Model Section
# *****

  
def try_parse(str_value) -> float:
  
  result = str_value
  try:
    result = float(str_value)
  except ValueError:
    print(f"{str_value} can't be parsed to float, returning string...")
  return result

def calculate_metrics(target_metrics: dict, predicted, labels, stage = "train"):
  
  metric_results = {}
  for key in target_metrics.keys():
    if "accuracy" in key:
      metric_value = target_metrics[key](labels, np.round(predicted[:,1]))
    else:
      metric_value = target_metrics[key](labels, predicted[:,1])
    metric_results[f"{stage}.{key}"] = metric_value
  
  return metric_results

