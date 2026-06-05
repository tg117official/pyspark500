from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, when

# Initialize SparkSession
spark = SparkSession.builder \
    .appName("EmployeeDataTransformation") \
    .getOrCreate()

# Read cleaned data
clean_data_path = "data/clean_data/"
cleaned_data = spark.read.parquet(clean_data_path)

# Transformation
avg_salary_per_dept = cleaned_data.groupBy("dept").agg(avg("salary").alias("avg_salary"))
transformed_data = cleaned_data.join(avg_salary_per_dept, on="dept", how="inner") \
    .withColumn("salary_category", when(col("salary") > col("avg_salary"), "High").otherwise("Low"))

# Write transformed data
final_output_path = "data/final_op/"
transformed_data.coalesce(1).write.mode("overwrite").parquet(final_output_path)

spark.stop()
