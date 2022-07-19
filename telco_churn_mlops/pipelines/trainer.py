from hyperopt import (
    hp,
    fmin,
    tpe,
    Trials,
    SparkTrials,
    space_eval,
    STATUS_OK
)
import inspect
import os
import glob
import mlflow
from mlflow.models.signature import infer_signature
import numpy as np
from sklearn.metrics import log_loss, accuracy_score, average_precision_score
from sklearn.utils.class_weight import compute_class_weight
import telco_churn_mlops
from telco_churn_mlops.pipelines.model_builder import ModelBuilder
from typing import List
from telco_churn_mlops.pipelines.data_preparation import DataPreparationPipeline


class ModelTrainingPipeline:
    def __init__(
        self,
        spark,
        db_name: str = "telcochurndb",
        training_table: str = "training",
        testing_table: str = "testing",
        experiment_name: str = "telco_churn_xgb",
        model_name: str = "xgboost",
        run_name: str = "XGB Final Model",
        pip_requirements: List[str] = [
            "scikit-learn==1.1.1",
            "xgboost==1.5.0"
        ],
    ):
        self._db_name = db_name
        self._run_name = run_name
        self._spark = spark
        self._training_table = training_table
        self._testing_table = testing_table
        self._experiment_name = experiment_name
        self._pip_requirements = pip_requirements
        self._model_name = model_name
        self._set_code_path()
        self._model_builder = ModelBuilder()
        self._data_preparation_pipeline = DataPreparationPipeline(
            spark = self._spark,
            db_name = self._db_name
        )

    def _set_code_path(self):
        code_path = inspect.getfile(telco_churn_mlops)
        root_path = code_path.replace(code_path.split("/")[-1], "")
        self._code_path = root_path


    def _calculate_scale(self, decimal_places=3):
        self.scale = np.round(self._compute_weights(self.y_train), decimal_places)

    def _get_splits(self):

        self.X_train, self.y_train = self._data_preparation_pipeline.export_df(
            table_name=self._training_table
        )

        self.X_test, self.y_test = self._data_preparation_pipeline.export_df(
            table_name=self._testing_table
        )

    def _train_wrapper(self, params):
        model = self._model_builder.build_pipeline(params)
        model.fit(self.X_train, self.y_train)
        # model = self._model_builder.train_model(params, self.X_train, self.y_train)
        prob = model.predict_proba(self.X_train)
        loss = log_loss(self.y_train, prob[:, 1])
        mlflow.log_metrics(
            {
                "train.log_loss": loss,
                "train.accuracy": accuracy_score(self.y_train, np.round(prob[:, 1])),
            }
        )
        return loss

    def _initialize_search_space(self):

        self._calculate_scale()
        self._search_space = {
            "max_depth": hp.quniform("max_depth", 5, 30, 1),
            "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.10)),
            "gamma": hp.quniform("gamma", 0.0, 1.0, 0.001),
            "min_child_weight": hp.quniform("min_child_weight", 4, 25, 1),
            "subsample": hp.loguniform("subsample", np.log(0.1), np.log(1.0)),
            "colsample_bytree": hp.loguniform(
                "colsample_bytree", np.log(0.1), np.log(1.0)
            ),
            "colsample_bylevel": hp.loguniform(
                "colsample_bylevel", np.log(0.1), np.log(1.0)
            ),
            "colsample_bynode": hp.loguniform(
                "colsample_bynode", np.log(0.1), np.log(1.0)
            ),
            "scale_pos_weight": hp.loguniform(
                "scale_pos_weight", np.log(1), np.log(self.scale * 10)
            ),
        }

    def _get_best_params(self, parallelism=5, max_evals=10):

        best_params = None

        mlflow.set_experiment(f"/Shared/{self._experiment_name}")
        trials = SparkTrials(spark_session = self._spark, parallelism = parallelism)
        if parallelism == 1:
            trials = Trials()
        with mlflow.start_run(run_name = self._run_name) as run:
            best_params = fmin(
                fn=self._train_wrapper,
                space=self._search_space,
                algo=tpe.suggest,
                max_evals=max_evals,
                trials=Trials()
            )

        self._best_params = best_params

    def _compute_weights(self, y_train):
        """
        Define minimum positive class scale factor
        """
        weights = compute_class_weight(
            "balanced", classes=np.unique(y_train), y=y_train
        )
        scale = weights[1] / weights[0]
        return scale

    def run(self, parallelism = 5):

        self._get_splits()
        self._initialize_search_space()
        self._get_best_params(parallelism = parallelism)
        # configure params
        params = space_eval(self._search_space, self._best_params)
        # train model with optimal settings
        with mlflow.start_run(run_name = self._run_name) as run:

            # capture run info for later use
            run_id = run.info.run_id

            # preprocess features and train
            xgb_model_best = self._model_builder.build_pipeline(self._best_params)
            xgb_model_best.fit(self.X_train, self.y_train)
            # predict
            pred_train = xgb_model_best.predict_proba(self.X_train)
            # score
            target_metrics = {
                "average_precision_score": average_precision_score,
                "accuracy_score": accuracy_score,
                "log_loss": log_loss,
            }

            train_metrics = self._calculate_metrics(
                target_metrics, pred_train, self.y_train, "train"
            )
            pred_test = xgb_model_best.predict_proba(self.X_test)
            test_metrics = self._calculate_metrics(
                target_metrics, pred_test, self.y_test, "test"
            )

            mlflow.log_metrics(train_metrics)
            mlflow.log_metrics(test_metrics)
            mlflow.log_params(params)

            model_info = mlflow.sklearn.log_model(
                sk_model=xgb_model_best,
                artifact_path="model",
                pip_requirements=self._pip_requirements,
                code_paths = [self._code_path]
            )

            self.model_info = model_info
            self._register_model(self._model_name)

    def get_best_run(
        self,
        filter_string: str = "status = 'FINISHED'",
        sort_by: str = "metrics.train.log_loss",
        ascending = True
    ):

        df = mlflow.search_runs(filter_string=filter_string)
        best_run_id = df.sort_values(by=sort_by, ascending = ascending)["run_id"].values[0]
        best_run = mlflow.get_run(run_id=best_run_id)

        return best_run

    def _calculate_metrics(
        self, target_metrics: dict, predicted, labels, stage="train"
    ):

        metric_results = {}
        for key in target_metrics.keys():
            if "accuracy" in key:
                metric_value = target_metrics[key](labels, np.round(predicted[:, 1]))
            else:
                metric_value = target_metrics[key](labels, predicted[:, 1])
            metric_results[f"{stage}.{key}"] = metric_value

        return metric_results

    def _register_model(self, model_name):

        version_info = mlflow.register_model(
            model_uri=self.model_info.model_uri, name=model_name
        )

        return version_info
