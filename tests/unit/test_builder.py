"""Unit tests for our Model Builder module"""

import pytest
from telco_churn_mlops.pipelines.model_builder import ModelBuilder
from telco_churn_mlops.pipelines.trainer import ModelTrainingPipeline
from tests.fixtures.unit import *


@pytest.fixture
def pipeline(spark_session):
    builder = ModelBuilder()
    return builder.build_pipeline()

def test_build_model(caplog, pipeline):
    """Test model build."""

    result = True
    model = pipeline
    expected_steps = [
        "preprocessor",
        "standardizer",
        "XGBClassifier"
    ]
    step_descriptions = [step[0] for step in model.steps]
    for step in expected_steps:
        if step not in step_descriptions:
            result = False
            break

    assert result


def test_trainer(spark_session):
    """Test trainer."""

    pipeline = ModelTrainingPipeline(spark = spark_session)
    pass
    