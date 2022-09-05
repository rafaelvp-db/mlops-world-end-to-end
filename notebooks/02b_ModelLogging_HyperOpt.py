# Databricks notebook source
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
# MAGIC ## Data Init
# MAGIC 
# MAGIC Calling split function for Train and Test, the preprocessing part is only for the Train, our test will be kept for the final inference 
# MAGIC Saving everything into Delta before running the model 

# COMMAND ----------

from utils import *
import mlflow
from mlflow.models.signature import infer_signature
from hyperopt import hp, fmin, tpe, SparkTrials, space_eval, STATUS_OK

# reading back the Delta table and calling a data4train -> can be a class then
print(f"Preparing X and y \n using {db_name} database")
X_train, y_train = export_df(f"{db_name}.training")
X_test, y_test = export_df(f"{db_name}.testing")
scale = np.round(compute_weights(y_train), 3) # scale can be places also inside the parameters then 
print(f"Our target is imbalanced, computing the scale is {scale}")

# Setting our experiment to kep track of our model with MlFlow 
experiment_path = f"/Shared/{experiment_name}"

try:
  print(f"Setting our existing experiment {experiment_path}")
  mlflow.set_experiment(experiment_path)
  experiment = mlflow.get_experiment_by_name(experiment_path)
except:
  print("Creating a new experiment and setting it")
  experiment = mlflow.create_experiment(name = experiment_path)
  mlflow.set_experiment(experiment_path)
 

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Modelling 

# COMMAND ----------

# MAGIC %md 
# MAGIC ### HyperParameter Tuning for XgBoost
# MAGIC 
# MAGIC We are using HyperOpt here. 
# MAGIC For more information check here: 
# MAGIC - https://docs.databricks.com/applications/machine-learning/automl-hyperparam-tuning/hyperopt-best-practices.html
# MAGIC - https://docs.databricks.com/applications/machine-learning/automl-hyperparam-tuning/hyperopt-spark-mlflow-integration.html
# MAGIC - https://www.databricks.com/blog/2021/04/15/how-not-to-tune-your-model-with-hyperopt.html
# MAGIC 
# MAGIC Please keep in mind that HyperOpt often used with `SparkTrials`, to distribute independent models. Meanwhile, if you are using `SparkML` or anything that involved Spark Executors, HyperOpt should use `Trials()` instead. 

# COMMAND ----------

if retrain == "True":
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
# MAGIC ### Finding the experiment with the best metrics

# COMMAND ----------

try:
  runs_df = mlflow.search_runs(filter_string="status = 'FINISHED'")
  best_run_id = runs_df.sort_values("metrics.train.log_loss", ascending = True)["run_id"].values[0]
  params_dict = mlflow.get_run(run_id = best_run_id).data.params
  parsed_params = dict([(item[0], try_parse(item[1])) for item in params_dict.items()])
  print(f"Best Run ID is {best_run_id}, params: \n {parsed_params}")
except:
  print("You have no runs yet, please run your model first")

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Registering the best model

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Loggin model with a SkLearn flavor 

# COMMAND ----------

import class_builder # import your package (here we import only single script)
from class_builder import ModelBuilder  # import your model class
import inspect

# MlFlow requires to have a list of paths, to get the exact address of your package 
def _set_code_path():
    code_path = inspect.getfile(class_builder)
    root_path = code_path.replace(code_path.split("/")[-1], "")
    return root_path
  
code_path = _set_code_path()
print("Your code path is {code_path}")
# Set our Model Class 
model_builder = ModelBuilder()

target_metrics = {
  "average_precision_score": average_precision_score,
  "accuracy_score": accuracy_score,
  "log_loss": log_loss
}
# train model with optimal settings 
with mlflow.start_run(experiment_id = experiment.experiment_id, run_name = run_name) as run:

  # preprocess features and train
  model = model_builder.build_pipeline(parsed_params)
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
  
  #wrappedModel = SklearnModelWrapper(model)
  mlflow.sklearn.log_model(artifact_path=artifact_name,
                           sk_model=model,
                           signature=signature, 
                           input_example=input_example,
                           pip_requirements = ["-r requirements.txt"],
                           code_paths = ["./class_builder.py"] #OR [code_path+"class_builder.py"], you can also log the whole package if more then 1 script involved
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

# MAGIC %md 
# MAGIC ### Logging model as a pyfunc wrapper
# MAGIC 
# MAGIC If you have a flavor that does not relate to the classic flavors, you can use a pyfunc wrapper instead
# MAGIC ```
# MAGIC class SklearnModelWrapper(mlflow.pyfunc.PythonModel):
# MAGIC   def __init__(self, model):
# MAGIC     self.model = model
# MAGIC   def predict(self, context, model_input):
# MAGIC     return self.model.predict(model_input)
# MAGIC   
# MAGIC wrappedModel = SklearnModelWrapper(model)
# MAGIC mlflow.pyfunc.log_model(artifact_path=artifact_name,
# MAGIC                          python_model=wrappedModel,
# MAGIC                          signature=signature, 
# MAGIC                          input_example=input_example,
# MAGIC                          pip_requirements = ["-r requirements.txt"],
# MAGIC                          code_path = ["./class_builder.py"] #[code_path+"class_builder.py"]
# MAGIC                         )
# MAGIC 
# MAGIC 
# MAGIC ```

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/><br/><a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>
