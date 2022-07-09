# Databricks notebook source
# MAGIC %run ./scripts 

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Simulating the config file read
# MAGIC 
# MAGIC with the dbx the config file is read from the deployment to adjust for it, we gonna parse the config file and include parameters into the inference part from it
# MAGIC 
# MAGIC this part goes into : 

# COMMAND ----------

import yaml
import mlflow

# parsing the yaml file 
def read_config(name, root):
    try:
        filename = root.replace('dbfs:', '/dbfs') + '/' + name
        with open(filename) as conf_file:
            conf = yaml.load(conf_file, Loader=yaml.FullLoader)
            return conf
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"{e}. Please include a config file!")
        

def setup_mlflow_conf(conf):
  """
  We are setting MlFlow experiment. 
  If the experiment did not exist it will create a new one and set it 
  """
  
  experiment_path = conf['experiment-path']
  model_name = conf['model-name']
  try:
    print(f"Setting experiment to {experiment_path}{model_name}")
    mlflow.set_experiment(experiment_path+model_name)
    experiment_id = mlflow.get_experiment_by_name(experiment_path+model_name).experiment_id
    return experiment_id
  except:
    print(f"Experiment was not found, creating one inside {experiment_path}")
    mlflow.create_experiment(name = f"{model_name}",
                             artifact_location=f"{experiment_path}")
    print(f"Setting a new created experiment")
    mlflow.set_experiment(experiment_path+model_name)
    experiment_id = mlflow.get_experiment_by_name(experiment_path+model_name).experiment_id
    return experiment_id

# COMMAND ----------

def clean_params(params):
    """
    XBGboost requires a particular format of inputs
    """
    if 'max_depth' in params: params['max_depth']=int(params['max_depth'])   
    if 'min_child_weight' in params: params['min_child_weight']=int(params['min_child_weight']) 
    if 'max_delta_step' in params: params['max_delta_step']=int(params['max_delta_step'])  
    if 'n_estimators' in params: params['n_estimators']=int(params['n_estimators'])
    return params

# COMMAND ----------

conf = read_config("project_config.yaml", "/Workspace/Repos/anastasia.prokaieva@databricks.com/mlops-world-end-to-end/notebooks/")
params = {ikey.split("-")[1]: ivalue for ikey, ivalue in conf.items() if ikey.split("-")[0] == 'params'}

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Wrapper if the model would be loaded and not retrained or retrained 

# COMMAND ----------

class SklearnModelWrapper(mlflow.pyfunc.PythonModel):
  """
  We are wrapping a SkLearn Pipeline and the predict to be able to serve it and log it into mlflow 
  The model will be retrained, if oyu wish to only predict, modify:
    - the data_provider function and build_model to remove training part
    - pass pre_trained model as an object into the init and load it with mlflow
  """
  mlflow.sklearn.autolog(disable=True)
  
  def __init__(self, conf=conf, params=params, retrain=None, data_provided=None):
    self.conf = conf
    self.params = params
    self.data_path = conf["scoring-input-path"]
    self.model_name = conf["model-name"]
    self.experimentID = setup_mlflow_conf(self.conf)
    self.stage = conf["model-stage"]
    self.retrain = retrain
    self.data_provided = data_provided
      
    
  def get_data(self, data_path):
    import pandas
    self.dataset = pandas.read_parquet(data_path)
    self.X = self.dataset.drop(['customerID','Churn'], axis=1)
    self.y = self.dataset['Churn']  
    return self.X, self.y
  
  def feature_prep(self):
    """
    :return object: return a sklearn Pipeline object 
    """
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

    self.pipeline = Pipeline([
        ("preprocessor", ColumnTransformer(transformers, remainder="passthrough", sparse_threshold=0)),
        ("standardizer", StandardScaler()),])
    
    return self.pipeline
  
  def model_train(self, params):   
    params = clean_params(params)
    self.model = XGBClassifier(**params)
    return self.model
  
  def model_call(self):
    logged_model = 'runs:/322e26961b4c4c649d5759a97ec6e202/retrained_clf_xgb'
    # Load model as a PyFuncModel.
    self.loaded_model = mlflow.pyfunc.load_model(logged_model)
    return self.loaded_model
  
  def predict(self, df=None):
    if self.data_provided: 
      X, y = self.get_data(self.data_path)
      trasformer = self.feature_prep().fit(X)
      self.X_prep = trasformer.transform(df)
    else:
      X, y = self.get_data(self.data_path)
      self.X_prep = self.feature_prep().fit_transform(X)
    
    if self.retrain: 
      with mlflow.start_run(experiment_id = self.experimentID) as run:
        model = self.model_train(self.params)
        evaluation = [(self.X_prep, y)]
        model.fit(self.X_prep, y,
                  eval_set=evaluation,
                  eval_metric=self.params['eval_metric'])
        mlflow.sklearn.log_model(model,"retrained_clf_xgb",
                                 input_example=self.X_prep[:3,:],
                                 registered_model_name = self.model_name
                                )
        predictions = model.predict_proba(self.X_prep)[:,1]
      return predictions 
        
    else: 
      model = self.model_call()
      predictions = model.predict(self.X_prep)
      return predictions 
      

# COMMAND ----------

wrappedModel = SklearnModelWrapper(conf, params, retrain=False, data_provided=True) 
X,y = wrappedModel.get_data(conf['input-path'])
wrappedModel.predict(df=X.iloc[:10,:])

# COMMAND ----------

X.iloc[:1,:].values

# COMMAND ----------



# COMMAND ----------

# MAGIC %md 
# MAGIC ### HERE THE REGISTERED MODEL DOES NOT evoke back..

# COMMAND ----------



# COMMAND ----------

wrappedModel = SklearnModelWrapper() 

mlflow.pyfunc.log_model(artifact_path='churn_model',python_model=wrappedModel,
                                registered_model_name='churn_mlops_ap')
mlflow.end_run()

# COMMAND ----------

import mlflow.pyfunc
 
model_version_uri = "models:/{model_name}/{model_version}".format(model_name="churn_mlops_ap", model_version='4')
 
print("Loading registered model version from URI: '{model_uri}'".format(model_uri=model_version_uri))
latest_model_version = mlflow.pyfunc.load_model(model_version_uri)

# COMMAND ----------

latest_model_version.predict(X.iloc[:1,:])

# COMMAND ----------


