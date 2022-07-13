from telco_churn_mlops.common import Job
from telco_churn_mlops.pipelines.data_preparation import DataPreparationPipeline


class PrepareDataJob(Job):

    def launch(self):
        self.logger.info("Launching data prep job")
        db_name = self.conf["db_name"]
        pipeline = DataPreparationPipeline(
            spark = self.spark,
            db_name = db_name
        )
        pipeline.write_delta_tables()
        self.logger.info("Data prep job finished!")


if __name__ == "__main__":
    job = PrepareDataJob()
    job.launch()
