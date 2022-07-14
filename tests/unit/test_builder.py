"""Unit tests for our Model Builder module"""

from telco_churn_mlops.pipelines.model_builder import ModelBuilder
from telco_churn_mlops.pipelines.trainer import ModelTrainingPipeline

def test_module():
    """Basic sanity checks."""

    builder = ModelBuilder()
    build_pipeline = builder.build_pipeline

    pass


def test_build_model(caplog):
    """Test model build."""

    result = True
    builder = ModelBuilder()
    model = builder.build_pipeline()
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


def test_trainer():
    """Test trainer."""

    pipeline = ModelTrainingPipeline()
    pass
    