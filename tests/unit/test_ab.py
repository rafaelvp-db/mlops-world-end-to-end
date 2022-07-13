"""Unit tests for our Model Builder module"""

from telco_churn_mlops.pipelines.ab_test import ABTestPipeline

def test_module():
    """Basic sanity checks."""

    pipeline = ABTestPipeline()

    pass


def test_build_model(caplog):
    """Test model build."""

    result = True
    model = model_builder.build_pipeline()
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
    