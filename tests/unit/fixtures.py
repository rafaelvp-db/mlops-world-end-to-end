from pyspark.sql import SparkSession
from pyspark.context import SparkContext
import pytest

@pytest.fixture
def db_name():
    return "telcochurnmlopsdb"

@pytest.fixture
def spark_session():
    sc = SparkContext.getOrCreate()
    spark = SparkSession.builder.master(sc.master) \
                    .appName('telcochurn') \
                    .getOrCreate()

    return spark