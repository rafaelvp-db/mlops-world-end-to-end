from telco_churn_mlops.common import Job
from telco_churn_mlops.jobs.trainer import Trainer


class TrainModelJob(Job):

    def launch(self):
        self.logger.info("Launching model training job")
        
        trainer = Trainer(
            db_name = self.init_config["db_name"],
            training_table = self.init_config["training_table"],
            testing_table = self.init_config["testing_table"]
        )
        trainer.train()

        self.logger.info("training job finished!")


if __name__ == "__main__":
    job = TrainModelJob()
    job.launch()
