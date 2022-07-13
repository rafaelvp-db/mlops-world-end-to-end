import mlflow
from telco_churn_mlops.common import Job
from telco_churn_mlops.pipelines.trainer import ModelTrainingPipeline


class TrainModelJob(Job):
    def launch(self):
        self.logger.info("Launching model training job")

        trainer = ModelTrainingPipeline(
            db_name=self.conf["db_name"],
            training_table=self.conf["training_table"],
            testing_table=self.conf["testing_table"],
            experiment_name=self.conf["experiment_name"],
            model_name=self.conf["model_name"],
        )
        mlflow.set_experiment()
        trainer.run()

        self.logger.info("training job finished!")


if __name__ == "__main__":
    job = TrainModelJob()
    job.launch()
