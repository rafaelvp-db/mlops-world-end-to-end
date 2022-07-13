import mlflow
from telco_churn_mlops.common import Job
from telco_churn_mlops.pipelines.ab_test import ABTestPipeline


class ABTestJob(Job):
    def launch(self):

        self.logger.info("Launching AB Test job")
        db_name = self.conf["db_name"]
        model_name = self.conf["model_name"]
        prod_version = self.conf["prod_version"]
        test_version = self.conf["test_version"]
        experiment_name = self.conf["experiment_name"]

        pipeline = ABTestPipeline(
            spark=self.spark,
            db_name=db_name,
            model_name=model_name,
            prod_version=prod_version,
            test_version=test_version
        )
        mlflow.set_experiment(f"/Shared/{self.conf['experiment_name']}")
        pipeline.run()
        self.logger.info("AB Test job finished!")


if __name__ == "__main__":
    job = ABTestJob()
    job.launch()
