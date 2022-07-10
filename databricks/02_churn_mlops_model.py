# Databricks notebook source
# MAGIC %md 
# MAGIC ## 2. Data Prep
# MAGIC 
# MAGIC Calling split function for Train and Test, the preprocessing part is only for the Train, our test will be kept for the final inference 
# MAGIC Saving everything into Delta before running the model 

# COMMAND ----------

dbutils.widgets.text("db_name", "telcochurndb")
db_name = dbutils.widgets.get("db_name")

# COMMAND ----------

from utils import export_df, compute_weights

# COMMAND ----------

import numpy as np

# reading back the Delta table and calling a data4train -> can be a class then
print("Preparing X and y")
X_train, y_train = export_df(f"{db_name}.training")
scale = np.round(compute_weights(y_train), 3) # scale can be places also inside the parameters then 
print(f"Our target is imbalanced, computing the scale is {scale}")

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Model

# COMMAND ----------

# MAGIC %md 
# MAGIC ### HyperParameter Tuning 

# COMMAND ----------

from hyperopt import hp, fmin, tpe, SparkTrials, space_eval
import mlflow
from utils import train_model

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
  return train_model(params, X_train, y_train)

experiment_name = "telco_churn_mlops_experiment"
experiment_path = f"/Shared/{experiment_name}"
experiment_id = mlflow.get_experiment_by_name(experiment_name)
#experiment_id = mlflow.create_experiment(experiment_path)

with mlflow.start_run(experiment_id = experiment_id) as run:

  best_params = fmin(
    fn = train_wrapper,
    space = search_space,
    algo = tpe.suggest,
    max_evals = 10,
    trials = SparkTrials(parallelism=10)
  )

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Finding the experiment with the best metrics

# COMMAND ----------

def try_parse(str_value) -> float:
  
  result = str_value
  try:
    result = float(str_value)
  except ValueError:
    print(f"{str_value} can't be parsed to float, returning string...")
  return result

df = mlflow.search_runs(filter_string="metrics.loss < 1")
best_run_id = df.sort_values("metrics.loss", ascending = True)["run_id"].values[0]
params_dict = mlflow.get_run(run_id = best_run_id).data.params
parsed_params = dict([(item[0], try_parse(item[1])) for item in params_dict.items()])
print(f"Best Run ID is {best_run_id}, params: \n {parsed_params}")

# COMMAND ----------

from xgb_wrapper import SklearnModelWrapper
import pickle

def save_model(
    model,
    preprocessor_pipeline,
    artifacts_path = "/artifacts/",
    preprocessor_artifact_path = "/dbfs/tmp/preprocessor.pkl",
    model_artifact_path = "/dbfs/tmp/xgb.pkl"
):
    with open(model_artifact_path, "wb") as model_file:
      pickle.dump(model, model_file)
      
    with open(preprocessor_artifact_path, "wb") as preprocessor_file:
      pickle.dump(preprocessor_pipeline, preprocessor_file)
  
    artifacts = {
      "preprocessor": f"{artifacts_path}/{preprocessor_artifact_path}",
      "model": f"{artifacts_path}/{model_artifact_path}"
    }
    
    model_info = mlflow.pyfunc.log_model(
        artifact_path = artifacts_path,
        python_model = SklearnModelWrapper(),
        code_path = ["./xgb_wrapper.py"],
        artifacts = artifacts,
    )
        
    return model_info

# COMMAND ----------

from mlflow.models.signature import infer_signature
from utils import build_model, build_preprocessor
from sklearn.metrics import average_precision_score, accuracy_score, log_loss
from hyperopt import space_eval


# configure params
params = space_eval(search_space, parsed_params)
# train model with optimal settings 
with mlflow.start_run(run_name='XGB Final Model') as run:
  
  # capture run info for later use
  run_id = run.info.run_id
  
  # preprocess features and train
  preprocessor_pipeline = build_preprocessor()
  xgb_model_best = build_model(parsed_params) 
  preprocessed_features = preprocessor.fit_transform(X_train)
  xgb_model_best.fit(preprocessed_features, y_train)
  # predict
  y_prob = xgb_model_best.predict_proba(preprocessed_features)
  # score
  model_ap = average_precision_score(y_train, y_prob[:,1])
  model_accuracy = accuracy_score(y_train, xgb_model_best.predict(preprocessed_features))
  loss = log_loss(y_train, y_prob)
  mlflow.log_metrics({'log_loss': loss,
                      'avg_precision': model_ap,
                      'accuracy': model_accuracy }
                    )
  mlflow.log_params(params)
  print('Xgboost Trained with XGBClassifier')
  save_model(
    model = xgb_model_best,
    preprocessor_pipeline = preprocessor_pipeline
  )
  model_name = "telco_churn_model"
  version_info = mlflow.register_model(model_uri = model_info.model_uri, name = model_name)
  
  print('Model logged under run_id "{0}" with AP score of {1:.5f}'.format(run_id, model_ap))

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Visualize the predictions from our best model 

# COMMAND ----------

model = mlflow.sklearn.load_model(model_uri = model_info.model_uri)

# COMMAND ----------

df = spark.sql("select * from telcochurndb.testing").toPandas().sample(1)

# COMMAND ----------

df

# COMMAND ----------

model.predict(df)

# COMMAND ----------


