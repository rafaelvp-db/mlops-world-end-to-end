from telco_churn_mlops.common import Job
from telco_churn_mlops.jobs.prepare_data import write_delta_tables


class PrepareDataJob(Job):

    def launch(self):
        self.logger.info("Launching sample job")
        write_delta_tables(spark = self.spark)
        self.logger.info("Sample job finished!")


if __name__ == "__main__":
    job = PrepareDataJob()
    job.launch()
