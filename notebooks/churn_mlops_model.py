# Databricks notebook source
import mlflow

# COMMAND ----------

# MAGIC %run ./scripts

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Parameters 
# MAGIC Will be transeferd into .yaml or a .py or .json files as an input to read 

# COMMAND ----------

experiment_path = model_registry_home + model_name_registry

try:
  mlflow.set_experiment(experiment_path)
except:
  mlflow.create_experiment(name = f"{model_name_registry}", artifact_location=f"{model_registry_home}")
  mlflow.set_experiment(experiment_path)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Data Prep

# COMMAND ----------

# creating a database if does not exist
_ = spark.sql(f'create database if not exists {db_name}')
# drop a table if exists 
_ = spark.sql(f"""DROP TABLE IF EXISTS {db_name}.trainDF""")
_ = spark.sql(f"""DROP TABLE IF EXISTS {db_name}.testDF""")

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Feature Prep

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Train Test Split
# MAGIC 
# MAGIC I am going to use only Train data to train our model. Then the test should be separated from the dataframe. 
# MAGIC We are going to illustrate a real inference while data is arriving. 

# COMMAND ----------



# COMMAND ----------

# MAGIC %md 
# MAGIC ### Creating new features on the Train Table

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

print("Preparing X and y")
X_train, y_train = data4train(f"{db_name}.trainDF")
scale = np.round(weight_compute(y_train),3)
print(f"Our target is imbalanced, computing the scale is {scale}") 

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Model

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Defining the Pipeline

# COMMAND ----------

def train_model(params):
  model = build_model(params)
  model.fit(X_train, y_train)
  loss = log_loss(y_train, model.predict_proba(X_train))
  mlflow.log_metrics({'log_loss': loss,
                      'accuracy': accuracy_score(y_train, model.predict(X_train))}
                    )
  return { 'status': STATUS_OK, 'loss': loss }

def build_model(params):
  transformers = []

  bool_pipeline = Pipeline(steps=[
      ("cast_type", FunctionTransformer(lambda df: df.astype(object))),
      ("imputer", SimpleImputer(missing_values=None, strategy="most_frequent")),
      ("onehot", OneHotEncoder(handle_unknown="ignore")),
  ])
  transformers.append(("boolean", bool_pipeline, 
                       ["Dependents", "PaperlessBilling", "Partner", "PhoneService", "SeniorCitizen"]))

  numerical_pipeline = Pipeline(steps=[
      ("converter", FunctionTransformer(lambda df: df.apply(pd.to_numeric, errors="coerce"))),
      ("imputer", SimpleImputer(strategy="mean"))
  ])
  transformers.append(("numerical", numerical_pipeline, 
                       ["AvgPriceIncrease", "Contract", "MonthlyCharges", "NumOptionalServices", "TotalCharges", "tenure"]))

  one_hot_pipeline = Pipeline(steps=[
      ("imputer", SimpleImputer(missing_values=None, strategy="constant", fill_value="")),
      ("onehot", OneHotEncoder(handle_unknown="ignore"))
  ])
  transformers.append(("onehot", one_hot_pipeline, 
                       ["DeviceProtection", "InternetService", "MultipleLines", "OnlineBackup", \
                        "OnlineSecurity", "PaymentMethod", "StreamingMovies", "StreamingTV", "TechSupport", "gender"]))

  
  if 'max_depth' in params: 
      # hyperopt supplies values as float but must be int
      params['max_depth']=int(params['max_depth'])   
  if 'min_child_weight' in params: 
      # hyperopt supplies values as float but must be int
      params['min_child_weight']=int(params['min_child_weight']) 
  if 'max_delta_step' in params: 
      # hyperopt supplies values as float but must be int
      params['max_delta_step']=int(params['max_delta_step']) 
      
  # all other hyperparameters are taken as given by hyperopt
  xgb_classifier = XGBClassifier(**params)

  return Pipeline([
      ("preprocessor", ColumnTransformer(transformers, remainder="passthrough", sparse_threshold=0)),
      ("standardizer", StandardScaler()),
      ("classifier", xgb_classifier),
  ])

# COMMAND ----------

# MAGIC %md 
# MAGIC ### HyperParameter Tuning 

# COMMAND ----------

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

best_params = fmin(fn=train_model,
                   space=search_space,
                   algo=tpe.suggest,
                   max_evals=36,
                   trials=SparkTrials(parallelism=2)
                  )

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
  if 'max_depth' in params: params['max_depth']=int(params['max_depth'])       
  if 'min_child_weight' in params: params['min_child_weight']=int(params['min_child_weight'])
  if 'max_delta_step' in params: params['max_delta_step']=int(params['max_delta_step'])
  if 'scale_pos_weight' in params: params['scale_pos_weight']=int(params['scale_pos_weight'])    

  # train
  xgb_model_best = build_model(params)
  xgb_model_best.fit(X_train, y_train)
  # predict
  y_prob = xgb_model_best.predict_proba(X_train)
  # score
  model_ap = average_precision_score(y_train, y_prob[:,1])
  model_accuracy = accuracy_score(y_train, xgb_model_best.predict(X_train))
  loss = log_loss(y_train, y_prob)
  mlflow.log_metrics({'log_loss': loss,
                      'avg precision': model_ap,
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

# MAGIC %md
# MAGIC ### Wrapping the best model 

# COMMAND ----------



# COMMAND ----------

class SklearnModelWrapper(mlflow.pyfunc.PythonModel):
  
  def __init__(self, model):
    self.model = model
    
  def predict(self, context, model_input):
    return self.model.predict_proba(model_input)[:,1]
  
params = space_eval(search_space, best_params)

with mlflow.start_run(run_name='XGB Final Model') as run:
  
  run_id = run.info.run_id
  model_pipeline = build_model(params) # pipeline  
  # persist the model with the custom wrapper
  wrappedModel = SklearnModelWrapper(model_pipeline)
  
  mlflow.pyfunc.log_model(
    artifact_path='churn_model', 
    python_model=wrappedModel, 
    registered_model_name=model_name_registry)

print('Model logged under run_id "{0}" with log loss of {1:.5f}'.format(run_id, model_ap))



# COMMAND ----------



# COMMAND ----------

# MAGIC %md 
# MAGIC ## Inference 

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Getting information about our latest model version

# COMMAND ----------



# COMMAND ----------


