from telco_churn_mlops.jobs.utils import stratified_split_train_test, write_into_delta_table


def write_delta_tables(
    spark,
    db_name: str = "telcochurndb",
    input_data_path: str = "/tmp/churn.csv",
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
