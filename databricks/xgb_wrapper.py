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

    with open(context.artifacts["preprocessor"], "rb") as preprocessor_file:
      self._preprocessor = pickle.load(preprocessor_file)

    with open(context.artifacts["model"], "rb") as model_file:
      self._model = pickle.load(model_file)

  
  def predict(self, context, model_input):
    """This is an abstract function to run the predictions.
      Args:mod
          context ([type]): MLflow context where the model artifact is stored.
          model_input ([type]): the input data for inference.
      Returns:
          [type]: the prediction result.
    """

    print(f"model input: {model_input}")
    input_df = pd.DataFrame.from_dict(model_input, orient="index")
    preprocessed_input = self._preprocessor.transform(input_df)
    result = self._model.predict_proba(preprocessed_input)
  

def _load_pyfunc(data_path):
    """
    Load PyFunc implementation. Called by ``pyfunc.load_pyfunc``.
    """
    return SklearnModelWrapper()