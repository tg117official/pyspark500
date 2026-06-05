from pyspark.sql import SparkSession
from pyspark.sql.functions import trim

# Initialize SparkSession
spark = SparkSession.builder \
    .appName("EmployeeDataCleaning") \
    .getOrCreate()

# Read raw data
input_path = "data/input_data/employee_new.csv"
raw_data = spark.read.csv(input_path, header=True, inferSchema=True)

# Data cleaning
cleaned_data = raw_data.dropDuplicates() \
    .withColumn("first_name", trim(raw_data["first_name"])) \
    .withColumn("last_name", trim(raw_data["last_name"])) \
    .withColumn("email", trim(raw_data["email"])) \
    .withColumn("dept", trim(raw_data["dept"]))

# Write cleaned data
clean_data_path = "data/clean_data/"
cleaned_data.coalesce(1).write.mode("overwrite").parquet(clean_data_path)

input("Press Enter to End APP : ")

spark.stop()
