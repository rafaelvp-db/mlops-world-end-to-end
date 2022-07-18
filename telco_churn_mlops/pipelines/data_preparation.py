import pandas as pd
from pyspark.sql import functions as F


class DataPreparationPipeline:
    def __init__(
        self,
        spark,
        db_name,
    ):

        self.spark = spark
        self.db_name = db_name

    def stratified_split_train_test(self, df, label, join_on, seed=42, frac=0.1):
        """
        Stratfied split of a Spark DataDrame into a Train and Test sets
        """
        fractions = (
            df.select(label)
            .distinct()
            .withColumn("fraction", F.lit(frac))
            .rdd.collectAsMap()
        )
        df_frac = df.stat.sampleBy(label, fractions, seed)
        df_remaining = df.join(df_frac, on=join_on, how="left_anti")
        return df_frac, df_remaining

    def export_df(self, table_name):

        telco_df = self.spark.read.format("delta").table(f"{self.db_name}.{table_name}")

        telco_df = self.prepare_features(telco_df)
        telco_df = self.compute_service_features(telco_df)

        dataset = telco_df.toPandas()
        X = dataset.drop(["customerID", "Churn"], axis=1)
        y = dataset["Churn"]

        return X, y

    def prepare_features(self, sparkDF):
        # 0/1 -> boolean
        sparkDF = sparkDF.withColumn("SeniorCitizen", F.col("SeniorCitizen") == 1)
        # Yes/No -> boolean
        for yes_no_col in ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]:
            sparkDF = sparkDF.withColumn(yes_no_col, F.col(yes_no_col) == "Yes")
        sparkDF = sparkDF.withColumn(
            "Churn", F.when(F.col("Churn") == "Yes", 1).otherwise(0)
        )

        # Contract categorical -> duration in months
        sparkDF = sparkDF.withColumn(
            "Contract",
            F.when(F.col("Contract") == "Month-to-month", 1)
            .when(F.col("Contract") == "One year", 12)
            .when(F.col("Contract") == "Two year", 24),
        )

        # Converting no Internet options into negative values
        for icol in [
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
        ]:
            sparkDF = sparkDF.withColumn(
                str(icol),
                (
                    F.when(F.col(icol) == "Yes", 1)
                    .when(F.col(icol) == "No internet service", -1)
                    .otherwise(0)
                ),
            )

        sparkDF = sparkDF.withColumn(
            "MultipleLines",
            (
                F.when(F.col("MultipleLines") == "Yes", 1)
                .when(F.col("MultipleLines") == "No phone service", -1)
                .otherwise(0)
            ),
        )
        # Empty TotalCharges -> NaN
        sparkDF = sparkDF.withColumn(
            "TotalCharges",
            F.when(F.length(F.trim(F.col("TotalCharges"))) == 0, None).otherwise(
                F.col("TotalCharges").cast("double")
            ),
        )

        return sparkDF

    def compute_service_features(self, sparkDF):
        @F.pandas_udf("int")
        def num_optional_services(*cols):
            return sum(map(lambda s: (s == 1).astype("int"), cols))

        @F.pandas_udf("int")
        def num_no_services(*cols):
            return sum(map(lambda s: (s == -1).astype("int"), cols))

        # Below also add AvgPriceIncrease: current monthly charges compared to historical average
        sparkDF = (
            sparkDF.fillna({"TotalCharges": 0.0})
            .withColumn(
                "NumOptionalServices",
                num_optional_services(
                    "OnlineSecurity",
                    "OnlineBackup",
                    "DeviceProtection",
                    "TechSupport",
                    "StreamingTV",
                    "StreamingMovies",
                ),
            )
            .withColumn(
                "NumNoInternetServices",
                num_no_services(
                    "OnlineSecurity",
                    "OnlineBackup",
                    "DeviceProtection",
                    "TechSupport",
                    "StreamingTV",
                    "StreamingMovies",
                ),
            )
            .withColumn(
                "AvgPriceIncrease",
                F.when(
                    F.col("tenure") > 0,
                    (
                        F.col("MonthlyCharges")
                        - (F.col("TotalCharges") / F.col("tenure"))
                    ),
                ).otherwise(0.0),
            )
        )

        return sparkDF


    def write_into_delta_table(
        self,
        df,
        table_name,
        db_name="telcochurndb",
        schema_option="overwriteSchema",
        mode="overwrite",
        table_type="managed",
    ):
        if table_type == "managed":

            table = f"{db_name}.{table_name}"
            self.spark.sql(f"CREATE DATABASE IF NOT EXISTS {db_name}")
            self.spark.sql(f"DROP TABLE IF EXISTS {db_name}.{table_name}")
            df.write.saveAsTable(f"{db_name}.{table_name}")

    def to_object(df):

        df = df.astype(object)
        return df

    def to_numeric(df):

        df = df.apply(pd.to_numeric, errors="coerce")
        return df

    def run(
        self,
        input_data_path: str = "./data/ibm_telco_churn.csv",
        label: str = "Churn",
        join_on: str = "customerID",
    ):

        telco_df_raw = self.spark.read.option("header", True).csv(input_data_path)
        df_train, df_test = self.stratified_split_train_test(
            df=telco_df_raw,
            label=label,
            join_on=join_on,
        )

        print(f"df_train: {df_train}")

        for item in [(df_train, "training"), (df_test, "testing")]:
            self.write_into_delta_table(
                df=item[0], db_name=self.db_name, table_name=item[1]
            )
