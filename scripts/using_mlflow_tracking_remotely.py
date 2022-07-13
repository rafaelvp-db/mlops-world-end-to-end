import os
import mlflow

mlflow.set_tracking_uri("databricks")
os.environ["DATABRICKS_HOST"] = "https://e2-demo-field-eng.cloud.databricks.com/"
os.environ["DATABRICKS_TOKEN"] = ""

# set experiment
experiment_path = "/Users/databricksuser@email.com/experiments/leclub1"
mlflow.set_experiment(experiment_path)

# tracking
with mlflow.start_run(run_name="Training") as run:
    mlflow.set_tag("action", "test2222")
    mlflow.log_metric("acc", 0.99)
    mlflow.log_param("param", "my value")
