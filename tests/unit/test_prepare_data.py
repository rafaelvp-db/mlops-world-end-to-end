import unittest
import tempfile
import os
import shutil

from telco_churn_mlops.jobs.data.entrypoint import PrepareDataJob
from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip
from unittest.mock import MagicMock

class PrepareDataUnitTest(unittest.TestCase):
    def setUp(self):
        builder = SparkSession.builder.master("local[1]")\
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        self.spark = configure_spark_with_delta_pip(builder).getOrCreate()
        self.test_config = {
            "dbname": "telcochurndb",
            "table_name": "training",
        }
        self.job = PrepareDataJob(spark=self.spark, init_conf=self.test_config)

    def test_prepare_data(self):

        self.job.launch()

        output_count = (
            self.spark.sql(f"select * from {self.test_config['dbname']}.{self.test_config['table_name']}")
            .count()
        )

        self.assertGreater(output_count, 0)


if __name__ == "__main__":
    unittest.main()
