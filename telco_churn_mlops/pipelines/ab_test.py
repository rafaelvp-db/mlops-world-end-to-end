import inspect
import mlflow
import sys
import pickle
from mlflow.tracking import MlflowClient
from sklearn.metrics import log_loss
from pyspark.sql import DataFrame
from pyspark.sql.functions import lit, col

from telco_churn_mlops.pipelines.data_preparation import DataPreparationPipeline
from telco_churn_mlops.jobs.utils import export_df


class ABTestPipeline:
    def __init__(
        self,
        spark,
        model_name,
        db_name,
        prod_version,
        test_version,
        limit=None,
    ):
        self.spark = spark
        self.model_name = model_name
        self.db_name = db_name
        self.prod_version = prod_version
        self.test_version = test_version
        self.limit = limit

    def run(self):
        test_df = export_df(table_name=f"{self.db_name}.testing")
        versions = [
            {"name": "prod", "number": self.prod_version},
            {"name": "test", "number": self.test_version},
        ]

        for version in versions:
            with mlflow.start_run(run_name=f"{version['name']}_model") as run:
                mlflow.set_tags(version)
                metric = self._score_model(df=test_df, version=version["number"])
                mlflow.log_metric(metric)

    def _score_model(self, df: DataFrame, version: str) -> DataFrame:
        
        from telco_churn_mlops.pipelines import model_builder as builder
        module_path = inspect.getfile(builder)
        index = module_path.rindex("/")
        dst_path = module_path[:index]
        sys.path.insert(1, dst_path)
        #import model_builder

        client = MlflowClient()
        from mlflow.artifacts import download_artifacts
        model_version_info = client.get_model_version(name = self.model_name, version = version)
        #model = mlflow.sklearn.load_model(model_uri = model_version_info.source, dst_path = dst_path)
        download_artifacts(artifact_uri = model_version_info.source, dst_path = dst_path)
        model = None
        with open(f"{dst_path}/model/model.pkl", "rb") as file:
            model = pickle.load(file)
        pred = model.predict_proba(df.drop("Churn", axis=1))
        return {"loss": pred}
