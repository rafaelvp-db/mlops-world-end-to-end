import mlflow
from mlflow.tracking import MlflowClient
import requests
from pipelines.utils import export_df


class DeployModelPipeline:
    def __init__(
        self,
        spark,
        model_name,
        experiment_name,
        token,
        host="https://e2-demo-field-eng.cloud.databricks.com/api/2.0",
    ):
        self._model_name = model_name
        self._spark = spark
        self._experiment_path = f"/Shared/{experiment_name}"
        self._client = MlflowClient()
        self._token = token
        self._host = host

    def _get_candidate(self, status="FINISHED", sort_by="metrics.test.log_loss"):

        experiment = mlflow.get_experiment_by_name(self._experiment_path)

        df = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"status = '{status}'",
        )

        best_run_id = df.sort_values(sort_by)["run_id"].values[0]
        return best_run_id

    def _set_tags(
        self,
        run_id,
        tags: dict = {
            "demographic_vars": "seniorCitizen,gender_Female",
            "db_table": "telcochurnmlopsdb.training",
        },
    ):

        for key, value in tags.items():
            self._client.set_tag(run_id, key=key, value=value)

    def _promote_to_staging(
        self,
        best_run_id,
        model_description="This model predicts whether a customer will churn.  \
            It is used to update the Telco Churn Dashboard in DB SQL.",
        version_description="This model version was built using XGBoost, \
            with the best hyperparameters set identified with HyperOpt.",
    ):

        model_version_info = self._client.get_latest_versions(
            name=self._model_name, stages=["None"]
        )[0]
        target_version = None
        target_stage = "Staging"

        if best_run_id == model_version_info.run_id:
            target_version = model_version_info.version
            self._client.transition_model_version_stage(
                name=self._model_name, version=target_version, stage=target_stage
            )

        # More details about the model
        self._client.update_registered_model(
            name=model_version_info.name, description=model_description
        )

        # Gives more details on this specific model version
        self._client.update_model_version(
            name=model_version_info.name,
            version=model_version_info.version,
            description=version_description,
        )

        print(
            f"Transitioned version {target_version} of model {self._model_name} \
            to {target_stage}"
        )

    def _test_predictions(self, model_version_info):

        model = mlflow.pyfunc.load_model(model_uri=model_version_info.source)
        X_test, y_test = export_df(self.db_name, self.table_name)
        pred = model.predict(X_test.sample(10))

        if pred is not None:
            return True

        return False

    def _promote_to_production(
        self, best_run_id, from_stage="Staging", to_stage="Production"
    ):

        model_version_info = self._client.get_latest_versions(
            name=self._model_name, stages=[from_stage]
        )[0]

        if self._test_prediction(model_version_info=model_version_info):
            if best_run_id == model_version_info.run_id:
                target_version = model_version_info.version
                self._client.transition_model_version_stage(
                    name=self._model_name,
                    version=target_version,
                    stage=to_stage,
                    archive_existing_versions=True,
                )

        print(
            f"Transitioned version {target_version} of model \
            {self.model_name} to {to_stage}"
        )

    def _enable_endpoint(self):

        auth_header = {"Authorization": "Bearer " + self._token}
        endpoint_path = "/mlflow/endpoints-v2/enable"
        payload = {"registered_model_name": self._model_name}
        full_url = f"{self._host}{endpoint_path}"
        response = requests.post(url=full_url, json=payload, headers=auth_header)

        if response.status_code != 200:
            raise ValueError("Error making POST request to Mlflow API")

    def run(self):

        best_run_id = self._get_candidate()
        self._promote_to_staging(best_run_id=best_run_id)
        self._promote_to_production(best_run_id=best_run_id)
        self._enable_endpoint()
