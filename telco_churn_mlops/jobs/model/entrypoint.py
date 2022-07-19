import mlflow
from telco_churn_mlops.common import Job
from telco_churn_mlops.pipelines.trainer import ModelTrainingPipeline


class TrainModelJob(Job):
    def launch(self):
        self.logger.info("Launching model training job")

        pipeline = ModelTrainingPipeline(
            db_name=self.conf["db_name"],
            spark=self.spark,
            training_table=self.conf["training_table"],
            testing_table=self.conf["testing_table"],
            model_name=self.conf["model_name"],
            experiment_name=self.conf["experiment_name"]
        )
        mlflow.set_experiment(f"/Shared/{self.conf['experiment_name']}")
        pipeline.run()

        self.logger.info("training job finished!")


if __name__ == "__main__":
    job = TrainModelJob()
    job.launch()
