

def export_df(spark, db_name, table_name):
    df = spark.sql(f"select * from {db_name}.{table_name}").toPandas()
    X = df.drop("Churn")
    y = df["Churn"]

    return X, y