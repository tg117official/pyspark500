# Each use case provides an example of how to apply sorting to your data, covering
# scenarios from 01_basic_broadcast to more complex sorting conditions.

# Both sort and orderBy can be used interchangeably in most cases, though orderBy is
# typically used more explicitly for clarity, especially in complex queries.

# SQL equivalents are provided to demonstrate how these operations can be mimicked in
# SQL queries within Spark SQL.

# Ensure the path to the CSV file is correctly set in the script and the CSV has headers
# that match the column references used in the commands.

# Adjust schema specifics and data types as necessary based on the actual content of your
# CSV file.

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, asc, desc, rand, expr, length

# Initialize a Spark session
spark = SparkSession.builder.appName("SortingDemonstration").getOrCreate()

# Load the data into a DataFrame
df = spark.read.csv("data\employee.csv", header=True, inferSchema=True)

# Register the DataFrame as a temporary view to use SQL
df.createOrReplaceTempView("employee")

# Use Case 1: Sort by One Column in Ascending Order
df.sort("salary").show()
# SQL Equivalent
spark.sql("SELECT * FROM employee ORDER BY salary").show()

# Use Case 2: Sort by One Column in Descending Order
df.sort(df.salary.desc()).show()
df.sort(desc(col("salary"))).show()
# SQL Equivalent
spark.sql("SELECT * FROM employee ORDER BY salary DESC").show()

# Use Case 3: OrderBy with Ascending and Descending Together
df.orderBy(col("dept").asc(), col("salary").desc()).show()
# SQL Equivalent
spark.sql("SELECT * FROM employee ORDER BY dept ASC, salary DESC").show()

# Use Case 4: Sort by Multiple Columns
df.sort("dept", "salary").show()
# SQL Equivalent
spark.sql("SELECT * FROM employee ORDER BY dept, salary").show()

# Use Case 5: Using asc() and desc() in sort
df.sort(col("dept").asc(), col("ename").desc()).show()
# SQL Equivalent
spark.sql("SELECT * FROM employee ORDER BY dept ASC, ename DESC").show()

# Use Case 6: OrderBy Using Expression
df.orderBy(length(col("ename")).desc()).show()
# SQL Equivalent
spark.sql("SELECT * FROM employee ORDER BY LENGTH(ename) DESC").show()

df = spark.read.csv("data\employee_with_nulls.csv", header=True, inferSchema=True)

# Register the DataFrame as a temporary view to use SQL
df.createOrReplaceTempView("employee_nulls")


# Use Case 7: Case-Insensitive Sorting
df.orderBy(col("salary").asc_nulls_last()).show()
# SQL Equivalent
spark.sql("SELECT * FROM employee ORDER BY ename ASC NULLS LAST").show()

df.orderBy(col("salary").desc_nulls_first()).show()

# Use Case 8: Sorting with Null Values at the End
df.sort(col("salary").asc_nulls_last()).show()
# SQL Equivalent
spark.sql("SELECT * FROM employee ORDER BY date_of_joining ASC NULLS LAST").show()

# Use Case 9: Sort Numerically on String Type Column by Casting
df.withColumn("eid_numeric", col("eid").cast("integer")).sort("eid_numeric").show()
# Note: SQL Equivalent would require a similar CAST operation

# Use Case 10: Random Sorting (for example purposes, not often useful)
df.orderBy(rand()).show()
# Note: SQL Equivalent would use ORDER BY RAND()

# Stop the Spark session
spark.stop()


# Questions:

# A. Basic Sorting
#   1. Sort by One Column in Ascending Order: How can you sort a DataFrame based on the salary column in ascending order?
#   2. Sort by One Column in Descending Order: What is the method to sort a DataFrame by the salary column in descending order?

# B. Complex Sorting
#   3. OrderBy with Ascending and Descending Together: How do you perform a sort operation using multiple columns with different sort orders (e.g., ascending for one column and descending for another)?
#   4. Sort by Multiple Columns: Demonstrate how to sort a DataFrame by multiple columns (e.g., department followed by salary).
#   5. Using asc() and desc() in sort: How do you explicitly specify the sort order of columns using asc() and desc() functions?

# C. Advanced Sorting Techniques
#   6. OrderBy Using Expression: How can you sort a DataFrame by the length of names in descending order?
#   7. Case-Insensitive Sorting: Show how to sort a DataFrame by employee names in a case-insensitive manner, considering null values.
#   8. Sorting with Null Values at the End: How can you sort a DataFrame by a date column where rows with null values appear at the end?

# D. Specific Use Cases
#   9. Sort Numerically on String Type Column by Casting: How to sort a DataFrame by casting a string type employee ID to an integer?
#   10. Random Sorting: What method would you use to randomly sort the entries in a DataFrame?

# Each use case demonstrates the flexibility of PySpark in handling different types of sorting scenarios, from straightforward to more nuanced conditions. These exercises also illustrate how similar operations can be achieved using SQL within a Spark environment.
