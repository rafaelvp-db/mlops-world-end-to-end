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

    self._preprocessor = pickle.load(context.artifacts["preprocessor"])
    self._model = pickle.load(context.artifacts["model"])

  
  def predict(self, context, model_input):
    """This is an abstract function to run the predictions.
      Args:mod
          context ([type]): MLflow context where the model artifact is stored.
          model_input ([type]): the input data for inference.
      Returns:
          [type]: the prediction result.
    """

    input_df = pd.DataFrame.from_dict(model_input, orient="records")
    preprocessed_input = self._preprocessor.transform(input_df)
    result = self._model.predict_proba(preprocessed_input)
  
    return result
      
    """
    Place here the MLFlow registry part 
    """
    return self.model.predict_proba(X)[:,1]

def _load_pyfunc(data_path):
    """
    Load PyFunc implementation. Called by ``pyfunc.load_pyfunc``.
    """
    return SklearnModelWrapper()