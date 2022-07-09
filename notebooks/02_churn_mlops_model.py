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
experiment_id = mlflow.create_experiment(experiment_path)

with mlflow.start_run(experiment_id = experiment_id) as run:

  best_params = fmin(
    fn = train_wrapper,
    space = search_space,
    algo = tpe.suggest,
    max_evals = 36,
    trials = SparkTrials(parallelism=2)
  )

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Finding the experiment with the best metrics

# COMMAND ----------

#TODO

# COMMAND ----------

params_dict = {'colsample_bylevel': 0.7897107311909447,
 'colsample_bynode': 0.25823881509469926,
 'colsample_bytree': 0.7553513300708579,
 'gamma': 0.879,
 'learning_rate': 0.07393402778392641,
 'max_depth': 27.0,
 'min_child_weight': 1.0,
 'scale_pos_weight': 3.385912425949693,
 'subsample': 0.2582555157290116}

# COMMAND ----------

from mlflow.models.signature import infer_signature

# configure params
params = space_eval(search_space, best_params) 
# train model with optimal settings 
with mlflow.start_run(run_name='XGB Final Model') as run:
  
  # capture run info for later use
  run_id = run.info.run_id
  
  # preprocess features and train
  xgb_model_best = build_model(params) 
  xgb_model_best.fit(X_train, y_train)
  # predict
  y_prob = xgb_model_best.predict_proba(X_train)
  # score
  model_ap = average_precision_score(y_train, y_prob[:,1])
  model_accuracy = accuracy_score(y_train, xgb_model_best.predict(X_train))
  loss = log_loss(y_train, y_prob)
  mlflow.log_metrics({'log_loss': loss,
                      'avg_precision': model_ap,
                      'accuracy': model_accuracy }
                    )
  mlflow.log_params(params)
  
  #signature= infer_signature(X_train, y_train)
  input_example=X_train[:10],
  signature=infer_signature(X_train, y_train)
  print('Xgboost Trained with XGBClassifier')
  mlflow.sklearn.log_model(xgb_model_best, 'xgb_pipeline', 
                           registered_model_name = 'xbg_pipeline',
                           input_example=input_example, signature=signature)  # persist model with mlflow
  
  print('Model logged under run_id "{0}" with AP score of {1:.5f}'.format(run_id, model_ap))

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Vis the prediction from our best model 

# COMMAND ----------

# get the model 
# run_id "6af44d640dec4c429709d2afce745d00"
# vis the predictions + confusion matrix

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------


