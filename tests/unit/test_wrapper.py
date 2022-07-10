"""Unit tests for our Pyfunc XGB Wrapper"""

from os.path import abspath

from databricks.xgb_wrapper import SklearnModelWrapper
from pyspark.context import SparkContext
from pyspark.sql import SparkSession

# warehouse_location points to the default location for managed databases and tables
warehouse_location = abspath('/tmp/spark-warehouse')

sc = SparkContext.getOrCreate()
sc.stop()

spark = SparkSession.builder \
    .appName("test") \
    .config("spark.sql.warehouse.dir", warehouse_location) \
    .enableHiveSupport() \
    .getOrCreate()

create_table_sql = """
    CREATE TABLE telco_churn
    (
        customerID INT,
        Dependents BOOLEAN,
        PaperlessBilling BOOLEAN,
        Partner BOOLEAN,
        PhoneService BOOLEAN,
        SeniorCitizen BOOLEAN,
        AvgPriceIncrease FLOAT,
        Contract INT,
        MonthlyCharges FLOAT,
        NumOptionalServices INT,
        TotalCharges FLOAT,
        tenure FLOAT,
        DeviceProtection STRING,
        InternetService STRING,
        MultipleLines STRING,
        OnlineBackup STRING,
        OnlineSecurity STRING,
        PaymentMethod STRING,
        StreamingMovies STRING,
        StreamingTV STRING,
        TechSupport STRING,
        gender STRING,
        Churn BOOLEAN
    ) USING hive
"""

spark.sql("DROP TABLE IF EXISTS telco_churn")
spark.sql(create_table_sql)


def test_module():
    """Basic sanity checks."""
    
    conf = {
        "scoring-input-path": "telco_churn",
        "model-name": "mymodel"
    }

    params = {"test_param": "test_param"}

    wrapper = SklearnModelWrapper(conf = conf, params = params, spark = spark)

    pass