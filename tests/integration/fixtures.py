from pyspark.sql import SparkSession
from pyspark.context import SparkContext
import pytest

import os

@pytest.fixture
def endpoint_url():
    return "https://e2-demo-field-eng.cloud.databricks.com/model-endpoint/telco_churn_model/{}/invocations"

@pytest.fixture
def token():
    return os.environ["DATABRICKS_TOKEN"]

@pytest.fixture
def version():
    #TODO: Do this dynamically
    return 22

@pytest.fixture
def spark_session():
    
    spark = SparkSession.builder.master("local[1]") \
                    .appName('telcochurn') \
                    .getOrCreate()

    return spark