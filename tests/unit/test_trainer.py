"""Unit tests for our Model Builder module"""

import pytest
from fixtures import *
from telco_churn_mlops.pipelines.trainer import ModelTrainingPipeline


@pytest.fixture
def pipeline(spark_session):

    pipeline = ModelTrainingPipeline(spark = spark_session, db_name = db_name)
    return pipeline


def test_train(caplog, pipeline):
    """Test model build."""

    pipeline.run()
    pass
    