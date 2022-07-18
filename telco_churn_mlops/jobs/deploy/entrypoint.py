import os
from telco_churn_mlops.common import Job
from telco_churn_mlops.pipelines.deploy import DeployModelPipeline


class DeployModelJob(Job):
    def launch(self):
        self.logger.info("Launching deploy model job")
        db_name = self.conf["db_name"]
        model_name = self.conf["model_name"]
        experiment_name = self.conf["experiment_name"]
        pipeline = DeployModelPipeline(
            model_name=model_name,
            experiment_name=experiment_name,
            token=os.environ["DATABRICKS_TOKEN"],
        )
        pipeline.run()
        self.logger.info("deploy model job finished!")


if __name__ == "__main__":
    job = DeployModelJob()
    job.launch()
