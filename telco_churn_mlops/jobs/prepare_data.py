from pyspark.context import SparkContext
from utils import stratified_split_train_test, write_into_delta_table

spark = SparkContext.getOrCreate()


def write_delta_tables(
    db_name: str,
    input_data_path: str,
    label: str = "Churn",
    join_on: str = "customerID"
):

    spark.sql(f"CREATE DATABASE IF NOT EXISTS {db_name}")

    telco_df_raw = spark.read.option("header", True).csv(input_data_path)
    df_train, df_test = stratified_split_train_test(
        df = telco_df_raw,
        label = label,
        join_on = join_on,
    )

    write_into_delta_table(df_train, f"{db_name}.training")
    write_into_delta_table(df_test, f"{db_name}.testing")
