import mlflow
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
  """

  def load_context(self, context):
    """This method is called when loading an MLflow model with pyfunc.load_model(), as soon as the Python Model is constructed.
      Args:
          context: MLflow context where the model artifact is stored.
    """

    self._model = pickle.load(context["model-path"])

  
  #def model_train(self, params):
    #TODO   
    #params = clean_params(params)
    #self.model = XGBClassifier(**params)
    #return self.model
  
  #def model_call(self, model_name, stage):
    #pass
  
  def predict(self, context, model_input):
    """This is an abstract function. We customized it into a method to fetch the Hugging Face model.
      Args:
          context ([type]): MLflow context where the model artifact is stored.
          model_input ([type]): the input data for inference.
      Returns:
          [type]: the prediction result.
    """

    #feature_prep_pipeline = self.get_feature_prep_pipeline()
    input_df = pd.DataFrame.from_dict(model_input, orient="records")
    result = self._model.fit_transform(input_df)
    
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