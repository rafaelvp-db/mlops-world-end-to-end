import unittest
import tempfile
import os
import shutil

from telco_churn_mlops.jobs.train.entrypoint import TrainModelJob
from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip
from unittest.mock import MagicMock

class TrainModelUnitTest(unittest.TestCase):
    def setUp(self):
        builder = SparkSession.builder.master("local[1]")\
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        self.spark = configure_spark_with_delta_pip(builder).getOrCreate()
        self.test_config = {
            "dbname": "telcochurndb",
            "training_table": "training",
            "testing_table": "testing",
        }
        self.job = TrainModelJob(spark=self.spark, init_conf=self.test_config)

    def test_prepare_data(self):

        self.job.launch()


if __name__ == "__main__":
    unittest.main()
