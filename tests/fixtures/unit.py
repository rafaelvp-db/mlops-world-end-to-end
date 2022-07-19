import json
from pyspark.sql import SparkSession
from pyspark.context import SparkContext
import pytest


@pytest.fixture
def data_prep_init_conf():
    with open("conf/data_prep/data_prep_config_local.json", "r") as file:
        config = json.load(file)
        return config

@pytest.fixture
def train_init_conf():
    with open("conf/build_model/build_model_config_local.json", "r") as file:
        config = json.load(file)
        return config

@pytest.fixture
def db_name(data_prep_init_conf):
    return data_prep_init_conf["db_name"]

@pytest.fixture
def spark_session():
    
    spark = SparkSession.builder.master("local[1]") \
                    .appName('telcochurn') \
                    .getOrCreate()

    return spark
