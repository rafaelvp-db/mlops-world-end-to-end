import pickle

from databricks.xgb_wrapper import SklearnModelWrapper
import mlflow


def save_model(
    model,
    run_id,
    preprocessor_pipeline,
    artifacts_folder = "artifacts",
    preprocessor_artifact_path = "/tmp/preprocessor.pkl",
    model_artifact_path = "/tmp/xgb.pkl",
    pip_requirements = ["sklearn", "pandas", "xgboost"]
):
    full_remote_path = f"runs://{run_id}/{artifacts_folder}"
    with open(model_artifact_path, "wb") as model_file:
      pickle.dump(model, model_file)
      
    with open(preprocessor_artifact_path, "wb") as preprocessor_file:
      pickle.dump(preprocessor_pipeline, preprocessor_file)
  
    artifacts = {
      "preprocessor": preprocessor_artifact_path,
      "model": model_artifact_path
    }
    
    model_info = mlflow.pyfunc.log_model(
        artifact_path = full_remote_path,
        python_model = SklearnModelWrapper(),
        code_path = ["./xgb_wrapper.py"],
        artifacts = artifacts,
        pip_requirements = pip_requirements

    )
        
    return model_info