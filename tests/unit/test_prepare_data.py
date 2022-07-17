"""Unit tests for our Model Builder module"""

from telco_churn_mlops.pipelines.data_preparation import DataPreparationPipeline
from fixtures import *
import pytest

@pytest.fixture
def pipeline(spark_session, db_name):

    spark = spark_session
    pipeline = DataPreparationPipeline(
        spark = spark,
        db_name = db_name
    )
    return pipeline


def test_prepare_data(caplog, pipeline):
    """Test prepare data."""

    pipeline.run()
    pass
    