from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as _sum, count

# Initialize SparkSession
spark = SparkSession.builder \
    .appName("EmployeeDataAggregation") \
    .getOrCreate()

# Read transformed data
final_output_path = "data/final_op/"
transformed_data = spark.read.parquet(final_output_path)

# Aggregation
aggregated_data = transformed_data.groupBy("dept").agg(
    _sum("salary").alias("total_salary_expense"),
    count("id").alias("employee_count")
)

# Write aggregated data
aggregated_output_path = "data/aggregated_output/"
aggregated_data.coalesce(1).write.mode("overwrite").parquet(aggregated_output_path)

spark.stop()
