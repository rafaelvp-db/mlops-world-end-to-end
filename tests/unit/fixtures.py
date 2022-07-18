from pyspark.sql import SparkSession
from pyspark.context import SparkContext
import pytest

@pytest.fixture
def db_name():
    return "telcochurnmlopsdb"

@pytest.fixture
def data_prep_init_conf():
    return {
        "db_name": "telcochurndb"
    }

@pytest.fixture
def spark_session():
    
    spark = SparkSession.builder.master("local[1]") \
                    .appName('telcochurn') \
                    .getOrCreate()

    return spark