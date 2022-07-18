from telco_churn_mlops.jobs.data.entrypoint import PrepareDataJob
from fixtures import *


def test_prep_data_job(spark_session, data_prep_init_conf):
    job = PrepareDataJob(
        spark = spark_session,
        init_conf = data_prep_init_conf
    )

    job.launch()