import mlflow
from pyspark.context import SparkContext
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from xgboost import XGBClassifier
import pickle
import pandas as pd

from databricks.utils import export_df

class SklearnModelWrapper(mlflow.pyfunc.PythonModel):
  """
  We are wrapping a Sklearn Pipeline and the predict to be able to serve it and log it into mlflow 
  The model will be retrained, if oyu wish to only predict, modify:
    - the data_provider function and build_model to remove training part
    - pass pre_trained model as an object into the init and load it with mlflow
  """
  mlflow.sklearn.autolog(disable=True)
  
  def __init__(self, conf, params, retrain=None, spark = SparkContext.getOrCreate()):
    self.conf = conf
    self.params = params
    self.data_path = conf["scoring-input-path"]
    self.model_name = conf["model-name"]
    self.data_provider = export_df(f"{self.data_path}")
    #self.experimentID = setup_mlflow_conf(self.conf)
    #self.retrain = retrain
    #self.stage = conf["model-stage"]

  def load_context(self, context):
    """This method is called when loading an MLflow model with pyfunc.load_model(), as soon as the Python Model is constructed.
      Args:
          context: MLflow context where the model artifact is stored.
    """

    self._model = pickle.load(context["model-path"])
  
    
  def read_data(self, data_path):
    self.X, self.y = export_df(f"{data_path}")
    return self.X, self.y
  
  def get_feature_prep_pipeline(self):
    """
    :params dict: all hyperparameters of the model
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

    return Pipeline([
        ("preprocessor", ColumnTransformer(transformers, remainder="passthrough", sparse_threshold=0)),
        ("standardizer", StandardScaler()),])
  
  def model_train(self, params):   
    params = clean_params(params)
    self.model = XGBClassifier(**params)
    return self.model
  
  def model_call(self, model_name, stage):
    pass
  
  def predict(self, context, model_input):
    """This is an abstract function. We customized it into a method to fetch the Hugging Face model.
      Args:
          context ([type]): MLflow context where the model artifact is stored.
          model_input ([type]): the input data for inference.
      Returns:
          [type]: the prediction result.
    """

    feature_prep_pipeline = self.get_feature_prep_pipeline()
    input_df = pd.DataFrame.from_dict(model_input, orient="records")
    X_prep = feature_prep_pipeline.fit_transform(input_df)
    
    """if self.retrain: 
      
      with mlflow.start_run(experiment_id=self.experimentID, run_name='xgb_cl'):
        model = self.model_train(self.params)
        evaluation = [(X_prep, y)]
        model.fit(X_prep, y,
                  eval_set=evaluation,
                  eval_metric=self.params['eval_metric'])
        predictions = model.predict_proba(X_prep)[:,1]
        return predictions 
        
    else: """
    #model = self.model_call(self.model_name, self.model_stage)
    prediction = self._model.predict_proba(X_prep)
    return prediction
      
    """
    Place here the MLFlow registry part 
    """
    return self.model.predict_proba(X)[:,1]

def _load_pyfunc(data_path):
    """
    Load PyFunc implementation. Called by ``pyfunc.load_pyfunc``.
    """
    return SklearnModelWrapper(data_path)