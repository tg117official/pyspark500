# Create a DataFrame from employee.txt (csv) file by implicitly providing Schema
# Importing the necessary libraries
from pyspark.sql import SparkSession

# Creating a Spark session
spark = SparkSession.builder \
    .appName("Employee Data Analysis") \
    .getOrCreate()

# Read the employee data from a CSV file into a DataFrame
# Replace '<path to employee.csv>' with the actual path to your CSV file
employeeDF = spark.read.option("inferSchema", "true").\
    option("header", "true").csv(r"data/employee.txt")

# Print the inferred schema of the DataFrame to show the data types and structure
employeeDF.printSchema()

# Display the contents of the DataFrame to provide a snapshot of the data
employeeDF.show()

# Stop the Spark session
spark.stop()