from telco_churn_mlops.jobs.data.entrypoint import PrepareDataJob
from telco_churn_mlops.jobs.model.entrypoint import TrainModelJob
from tests.fixtures.unit import *


def test_prep_data_job(spark_session, data_prep_init_conf):
    job = PrepareDataJob(
        spark = spark_session,
        init_conf = data_prep_init_conf
    )
    job.launch()

def test_train_model(spark_session, train_init_conf):
    job = TrainModelJob(spark = spark_session, init_conf = train_init_conf)
    job.launch()