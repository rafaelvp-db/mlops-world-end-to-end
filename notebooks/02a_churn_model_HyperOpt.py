# Databricks notebook source
# !pip install scikit-learn==1.1.1 # install it on the cluster directly, would prefer to pass the init script rather then doing this 
# !pip install xgboost==1.5.0

# COMMAND ----------

dbutils.widgets.text("db_name", "telcochurndb")
dbutils.widgets.text("run_name", "XGB Final Model")
dbutils.widgets.text("experiment_name", "telco_churn_mlops_experiment")
dbutils.widgets.text("model_name", "telco_churn_model")
dbutils.widgets.text("spark_trials", "4")
dbutils.widgets.text("evaluations", "25")
dbutils.widgets.text("artifact_name", "model_churn")
dbutils.widgets.text("retrain", "False")

run_name = dbutils.widgets.get("run_name")
db_name = dbutils.widgets.get("db_name")
experiment_name = dbutils.widgets.get("experiment_name")
model_name = dbutils.widgets.get("model_name")
spark_trials = int(dbutils.widgets.get("spark_trials"))
evaluations = int(dbutils.widgets.get("evaluations"))
artifact_name = dbutils.widgets.get("artifact_name")
retrain = dbutils.widgets.get("retrain")

# COMMAND ----------

# MAGIC %md 
# MAGIC ## 2. Data Prep
# MAGIC 
# MAGIC Calling split function for Train and Test, the preprocessing part is only for the Train, our test will be kept for the final inference 
# MAGIC Saving everything into Delta before running the model 

# COMMAND ----------

from utils import export_df, compute_weights
from model_builder import *

# reading back the Delta table and calling a data4train -> can be a class then
print("Preparing X and y")
X_train, y_train = export_df(f"{db_name}.training")
X_test, y_test = export_df(f"{db_name}.testing")
scale = np.round(compute_weights(y_train), 3) # scale can be places also inside the parameters then 
print(f"Our target is imbalanced, computing the scale is {scale}")

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Model

# COMMAND ----------

import mlflow
experiment_path = f"/Shared/{experiment_name}"

try:
  print(f"Setting our existing experiment {experiment_path}")
  mlflow.set_experiment(experiment_path)
  experiment = mlflow.get_experiment_by_name(experiment_path)
except:
  print("Creating a new experiment and setting it")
  experiment = mlflow.create_experiment(name = experiment_path) # ?does this set it by default?
  mlflow.set_experiment(experiment_path)
  

# COMMAND ----------

# MAGIC %md 
# MAGIC ### HyperParameter Tuning 

# COMMAND ----------

if RETRAIN:
  # define hyperopt search space
  search_space = {
      'max_depth' : hp.quniform('max_depth', 5, 30, 1)                                  # depth of trees (preference is for shallow trees or even stumps (max_depth=1))
      ,'learning_rate' : hp.loguniform('learning_rate', np.log(0.01), np.log(0.10))     # learning rate for XGBoost
      ,'gamma': hp.quniform('gamma', 0.0, 1.0, 0.001)                                   # minimum loss reduction required to make a further partition on a leaf node
      ,'min_child_weight' : hp.quniform('min_child_weight', 4, 25, 1)                   # minimum number of instances per node
      ,'subsample' : hp.loguniform('subsample', np.log(0.1), np.log(1.0))               # random selection of rows for training,
      ,'colsample_bytree' : hp.loguniform('colsample_bytree', np.log(0.1), np.log(1.0)) # proportion of columns to use per tree
      ,'colsample_bylevel': hp.loguniform('colsample_bylevel', np.log(0.1), np.log(1.0))# proportion of columns to use per level
      ,'colsample_bynode' : hp.loguniform('colsample_bynode', np.log(0.1), np.log(1.0)) # proportion of columns to use per node
      ,'scale_pos_weight' : hp.loguniform('scale_pos_weight', np.log(1), np.log(scale * 10))   # weight to assign positive label to manage imbalance
  }

  def train_wrapper(params):
    model = train_model(params, X_train, y_train)
    prob = model.predict_proba(X_train)
    loss = log_loss(y_train, prob[:, 1])
    mlflow.log_metrics(
        {
            "train.log_loss": loss,
            "train.accuracy": accuracy_score(y_train, np.round(prob[:, 1])),
        }
    )
    return loss


  best_params = fmin(
    fn = train_wrapper,
    space = search_space,
    algo = tpe.suggest,
    max_evals = evaluations ,
    trials = SparkTrials(parallelism=spark_trials)
  )

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Finding the experiment with the best metrics

# COMMAND ----------

df = mlflow.search_runs(filter_string="status = 'FINISHED'")
best_run_id = df.sort_values("metrics.train.log_loss", ascending = True)["run_id"].values[0]
params_dict = mlflow.get_run(run_id = best_run_id).data.params
parsed_params = dict([(item[0], try_parse(item[1])) for item in params_dict.items()])
print(f"Best Run ID is {best_run_id}, params: \n {parsed_params}")

# COMMAND ----------

# configure params
#params = space_eval(search_space, parsed_params)

class SklearnModelWrapper(mlflow.pyfunc.PythonModel):
  def __init__(self, model):
    self.model = model
    
  def predict(self, context, model_input):
    return self.model.predict(model_input)

target_metrics = {
  "average_precision_score": average_precision_score,
  "accuracy_score": accuracy_score,
  "log_loss": log_loss
}
# train model with optimal settings 
with mlflow.start_run(experiment_id = experiment.experiment_id, run_name = run_name) as run:
  
  # preprocess features and train
  model = build_pipeline(parsed_params)
  model.fit(X_train, y_train)
  # predict
  pred_train = model.predict_proba(X_train)
  
  # ******
  # MlFlow Part Start 
  # ******
  
  # capture run info for later use
  run_id = run.info.run_id
  
  signature = infer_signature(X_train, model.predict(X_train))
  input_example = X_train.iloc[:5,:]
  
  wrappedModel = SklearnModelWrapper(model)
  mlflow.pyfunc.log_model( artifact_path=artifact_name,
                           python_model=wrappedModel,
                           signature=signature, 
                           input_example=input_example,
                           #pip_requirements = ["-r requirements.txt"],
                           #code_path = ["model_builder.py"]
                          )
  
  # score
  train_metrics = calculate_metrics(target_metrics, pred_train, y_train, "train")
  pred_test = model.predict_proba(X_test)
  test_metrics = calculate_metrics(target_metrics, pred_test, y_test, "test")
  
  mlflow.log_metrics(train_metrics)
  mlflow.log_metrics(test_metrics)
  mlflow.log_params(parsed_params)
  mlflow.set_tag("best_model", "true")

  print('Xgboost Trained with XGBClassifier')
  version_info = mlflow.register_model(model_uri = f"runs:/{run_id}/{artifact_name}", name = model_name)
  print(f"Model logged under run_id: {run_id}")
  print(f"Train metrics: {train_metrics}")
  print(f"Test metrics: {test_metrics}")

# COMMAND ----------



# COMMAND ----------


