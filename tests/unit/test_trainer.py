"""Unit tests for our Model Builder module"""

import pytest
from tests.fixtures.unit import *
from telco_churn_mlops.pipelines.trainer import ModelTrainingPipeline


@pytest.fixture
def pipeline(spark_session, db_name):

    pipeline = ModelTrainingPipeline(spark = spark_session, db_name = db_name)
    return pipeline


def test_train(caplog, pipeline):
    """Test model build."""

    pipeline.run(parallelism = 1)
    pass
    