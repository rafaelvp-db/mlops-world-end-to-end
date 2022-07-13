from telco_churn_mlops.jobs.utils import (
    stratified_split_train_test,
    write_into_delta_table,
)


class DataPreparationPipeline:
    def __init__(
        self,
        spark,
        db_name,
    ):

        self.spark = spark
        self.db_name = db_name

    def run(
        self,
        input_data_path: str = "/tmp/ibm_telco_churn.csv",
        label: str = "Churn",
        join_on: str = "customerID",
    ):

        telco_df_raw = self.spark.read.option("header", True).csv(input_data_path)
        df_train, df_test = stratified_split_train_test(
            df=telco_df_raw,
            label=label,
            join_on=join_on,
        )

        for item in [(df_train, "training"), (df_test, "testing")]:
            write_into_delta_table(df=item[0], db_name=self.db_name, table_name=item[1])
