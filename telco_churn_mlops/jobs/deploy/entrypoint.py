from telco_churn_mlops.common import Job
from telco_churn_mlops.pipelines.deploy import ModelDeploymentPipeline


class ModelDeploymentJob(Job):
    def launch(self):
        self.logger.info("Launching deploy model job")
        db_name = self.conf["db_name"]
        model_name = self.conf["model_name"]
        experiment_name = self.conf["experiment_name"]
        run_name = self.conf["run_name"]
        pipeline = ModelDeploymentPipeline(
            spark = self.spark,
            db_name = db_name,
            run_name = run_name,
            model_name = model_name,
            experiment_name = experiment_name
        )
        pipeline.run()
        self.logger.info("deploy model job finished!")


if __name__ == "__main__":
    job = ModelDeploymentJob()
    job.launch()
