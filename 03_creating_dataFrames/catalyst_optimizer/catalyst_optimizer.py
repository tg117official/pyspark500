# ============================================================
# Catalyst Optimizer Practical Demo in PySpark
# ============================================================
# Objective:
# This script helps you understand how Spark Catalyst Optimizer works.
#
# You will learn:
# 1. What happens before Spark executes SQL/DataFrame code
# 2. Difference between parsing and analysis
# 3. How to read explain(True) output
# 4. Projection Pruning
# 5. Predicate Pushdown
# 6. Constant Folding
# 7. Partition Pruning
# 8. Join Strategy Selection
# 9. Broadcast Join
# 10. Difference between Catalyst Optimizer and AQE
# ============================================================

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, broadcast


# ------------------------------------------------------------
# Step 1: Create Spark Session
# ------------------------------------------------------------

spark = (
    SparkSession.builder
    .appName("CatalystOptimizerDemo")
    .master("local[*]")
    # AQE is disabled initially so that we can first observe
    # normal Catalyst planning clearly.
    .config("spark.sql.adaptive.enabled", "false")
    # Reducing shuffle partitions makes the plan easier to understand
    # in local/demo environments.
    .config("spark.sql.shuffle.partitions", "4")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

print("\nSpark Session Created Successfully")
print("AQE Enabled:", spark.conf.get("spark.sql.adaptive.enabled"))
print("Shuffle Partitions:", spark.conf.get("spark.sql.shuffle.partitions"))


# ------------------------------------------------------------
# Step 2: Create Sample DataFrames
# ------------------------------------------------------------
# In real projects, data may come from files, Hive tables,
# Delta tables, JDBC sources, or catalog tables.
#
# For this demo, we are creating small DataFrames manually.

employees_data = [
    (1, "Sandeep", "Engineering", 90000, "India"),
    (2, "Rahul", "Engineering", 70000, "India"),
    (3, "Priya", "HR", 60000, "India"),
    (4, "Amit", "Finance", 80000, "USA"),
    (5, "Sneha", "Finance", 95000, "USA"),
    (6, "John", "Engineering", 120000, "USA"),
    (7, "Neha", "HR", 65000, "India"),
    (8, "Ravi", "Sales", 50000, "India")
]

employees_columns = ["emp_id", "emp_name", "dept_name", "salary", "country"]

employees_df = spark.createDataFrame(employees_data, employees_columns)


departments_data = [
    ("Engineering", "Bangalore"),
    ("HR", "Mumbai"),
    ("Finance", "New York"),
    ("Sales", "Delhi")
]

departments_columns = ["dept_name", "dept_location"]

departments_df = spark.createDataFrame(departments_data, departments_columns)


print("\nEmployees Data")
employees_df.show()

print("\nDepartments Data")
departments_df.show()

print("\nEmployees Schema")
employees_df.printSchema()


# ------------------------------------------------------------
# Step 3: Register Temporary Views
# ------------------------------------------------------------
# Spark SQL queries can be executed on temporary views.
# Catalyst Optimizer also works for SQL queries written on temp views.

employees_df.createOrReplaceTempView("employees")
departments_df.createOrReplaceTempView("departments")

print("\nTemporary views created: employees, departments")


# ============================================================
# DEMO 1: Parsing vs Analysis
# ============================================================
# Important idea:
#
# Parsing:
# Spark checks the basic SQL structure and creates an unresolved logical plan.
#
# Analysis:
# Spark checks whether table names, column names, and data types are valid.
#
# In the query below, the SQL structure is correct,
# but the column 'wrong_column' does not exist.
# So the query fails during the analysis stage.

print("\n" + "=" * 80)
print("DEMO 1: Parsing vs Analysis")
print("=" * 80)

try:
    wrong_query_df = spark.sql("""
        SELECT emp_name, wrong_column
        FROM employees
    """)

    wrong_query_df.show()

except Exception as e:
    print("\nExpected Error")
    print("The query structure is valid, but the column does not exist.")
    print("This means Spark fails while resolving the column during analysis.")
    print("\nError Message:")
    print(str(e)[:700])


# ============================================================
# DEMO 2: Viewing Catalyst Plans using explain(True)
# ============================================================
# explain(True) shows the internal plans created by Spark.
#
# Important sections:
#
# 1. Parsed Logical Plan
#    Spark has understood the SQL structure.
#
# 2. Analyzed Logical Plan
#    Spark has verified table names, column names, and data types.
#
# 3. Optimized Logical Plan
#    Catalyst has applied optimization rules.
#
# 4. Physical Plan
#    Spark has selected actual execution operators.

print("\n" + "=" * 80)
print("DEMO 2: Viewing Catalyst Plans using explain(True)")
print("=" * 80)

simple_query_df = spark.sql("""
    SELECT emp_name, salary
    FROM employees
    WHERE salary > 70000
""")

print("\nQuery Output")
simple_query_df.show()

print("\nFull Execution Plan")
simple_query_df.explain(True)


# ============================================================
# DEMO 3: Projection Pruning
# ============================================================
# Projection means selecting columns.
#
# Projection pruning means Spark tries to remove unnecessary columns
# from the plan as early as possible.
#
# Example:
# If we only need emp_name, Spark does not need to carry emp_id,
# dept_name, salary, and country in later stages of the plan.
#
# This optimization becomes more powerful with columnar formats
# like Parquet and Delta because Spark can physically read only
# required columns.

print("\n" + "=" * 80)
print("DEMO 3: Projection Pruning")
print("=" * 80)

projection_df = employees_df.select("emp_name")

print("\nOnly emp_name column selected")
projection_df.show()

print("\nExecution Plan for Projection Pruning")
projection_df.explain(True)


# ============================================================
# DEMO 4: Predicate Pushdown
# ============================================================
# Predicate means filter condition.
#
# Predicate pushdown means Spark tries to push the filter condition
# closer to the data source.
#
# This helps Spark avoid unnecessary data reading.
#
# Parquet supports predicate pushdown, so we will first write data
# in Parquet format and then read it again.

print("\n" + "=" * 80)
print("DEMO 4: Predicate Pushdown with Parquet")
print("=" * 80)

parquet_path = "/tmp/catalyst_employees_parquet"

employees_df.write.mode("overwrite").parquet(parquet_path)

parquet_employees_df = spark.read.parquet(parquet_path)

predicate_df = parquet_employees_df.select("emp_id", "salary").\
    filter(col("salary") > 70000)

print("\nEmployees with salary > 70000")
predicate_df.show()

print("\nExecution Plan for Predicate Pushdown")
predicate_df.explain(True)

print("""
Observation:
In the Physical Plan, look for a term like:

PushedFilters

If PushedFilters is visible, it means Spark has pushed the filter
condition closer to the Parquet reader.
""")


# ============================================================
# DEMO 5: Constant Folding
# ============================================================
# Constant folding means Spark simplifies constant expressions
# during planning itself.
#
# Example:
# salary > 50000 + 20000
#
# Spark can simplify this internally as:
# salary > 70000
#
# This avoids unnecessary calculation during execution.

print("\n" + "=" * 80)
print("DEMO 5: Constant Folding")
print("=" * 80)

constant_folding_df = employees_df.filter(col("salary") > (50000 + 20000))

print("\nEmployees with salary > 50000 + 20000")
constant_folding_df.show()

print("\nExecution Plan for Constant Folding")
constant_folding_df.explain(True)


# ============================================================
# DEMO 6: Partition Pruning
# ============================================================
# Partition pruning means Spark reads only the required partitions.
#
# Example:
# If data is partitioned by country and we filter country = 'India',
# Spark should avoid reading partitions that are not required.
#
# First, we write data partitioned by country.
# Then, we read the data and apply a filter on country.

print("\n" + "=" * 80)
print("DEMO 6: Partition Pruning")
print("=" * 80)

partitioned_path = "/tmp/catalyst_employees_partitioned"

employees_df.write.mode("overwrite").partitionBy("country").parquet(partitioned_path)

partitioned_df = spark.read.parquet(partitioned_path)

india_df = partitioned_df.filter(col("country") == "India")

print("\nEmployees from India")
india_df.show()

print("\nExecution Plan for Partition Pruning")
india_df.explain(True)

print("""
Observation:
In the Physical Plan, look for a term like:

PartitionFilters

If PartitionFilters is visible, it means Spark is using the partition
column to reduce unnecessary data scanning.
""")


# ============================================================
# DEMO 7: Join Strategy Selection
# ============================================================
# Logical plan says what operation needs to happen.
# Physical plan decides how that operation will happen.
#
# For joins, Spark can choose different physical strategies:
#
# 1. SortMergeJoin
# 2. BroadcastHashJoin
# 3. ShuffleHashJoin
#
# The selected strategy depends on data size, statistics,
# configuration, and hints.

print("\n" + "=" * 80)
print("DEMO 7: Join Strategy Selection")
print("=" * 80)

normal_join_df = employees_df.join(
    departments_df,
    on="dept_name",
    how="inner"
)

print("\nNormal Join Output")
normal_join_df.show()

print("\nPhysical Plan for Normal Join")
normal_join_df.explain(True)

print("""
Observation:
Check the Physical Plan and identify which join strategy Spark selected.

Possible join strategies:
- SortMergeJoin
- BroadcastHashJoin
- ShuffleHashJoin

The important point is:
Logical plan says JOIN.
Physical plan decides HOW to perform that join.
""")


# ============================================================
# DEMO 8: Broadcast Join
# ============================================================
# Broadcast join is useful when one DataFrame is small.
#
# Instead of shuffling both DataFrames, Spark can send the small
# DataFrame to all executors.
#
# Here, departments_df is small, so we will broadcast it explicitly.

print("\n" + "=" * 80)
print("DEMO 8: Broadcast Join")
print("=" * 80)

broadcast_join_df = employees_df.join(
    broadcast(departments_df),
    on="dept_name",
    how="inner"
)

print("\nBroadcast Join Output")
broadcast_join_df.show()

print("\nPhysical Plan for Broadcast Join")
broadcast_join_df.explain(True)

print("""
Observation:
In the Physical Plan, look for:

BroadcastHashJoin

This means Spark is broadcasting the smaller DataFrame.
Broadcast join can reduce shuffle when one side of the join is small.
""")


# ============================================================
# DEMO 9: Complete SQL Query Optimization
# ============================================================
# Real-world queries usually contain multiple operations together:
#
# 1. Select
# 2. Filter
# 3. Join
# 4. Projection
#
# Catalyst optimizes the complete plan, not only one operation.

print("\n" + "=" * 80)
print("DEMO 9: Complete SQL Query Optimization")
print("=" * 80)

complex_query_df = spark.sql("""
    SELECT 
        e.emp_name,
        e.salary,
        d.dept_location
    FROM employees e
    INNER JOIN departments d
        ON e.dept_name = d.dept_name
    WHERE e.salary > 70000
""")

print("\nComplex Query Output")
complex_query_df.show()

print("\nFull Plan for Complex Query")
complex_query_df.explain(True)

print("""
Observation:
This query contains:
1. Join
2. Filter
3. Projection

Check the optimized logical plan and physical plan.

Important question:
Does Spark reduce unnecessary data before the join?

Usually, filtering before join is better because it reduces the amount
of data participating in the join.
""")


# ============================================================
# DEMO 10: Catalyst Optimizer vs AQE
# ============================================================
# Catalyst Optimizer:
# - Works mainly before execution
# - Creates and optimizes logical/physical plans
#
# AQE:
# - AQE stands for Adaptive Query Execution
# - Works during execution
# - Uses runtime statistics
#
# AQE can:
# 1. Change join strategy at runtime
# 2. Coalesce shuffle partitions
# 3. Handle skew joins

print("\n" + "=" * 80)
print("DEMO 10: Catalyst Optimizer vs AQE")
print("=" * 80)

print("\nCurrent AQE Setting")
print(spark.conf.get("spark.sql.adaptive.enabled"))

print("""
At the beginning of this script, AQE was disabled.

Now we will enable AQE and observe the physical plan again.
""")

spark.conf.set("spark.sql.adaptive.enabled", "true")

print("\nAQE Enabled Now")
print(spark.conf.get("spark.sql.adaptive.enabled"))

aqe_join_df = employees_df.join(
    departments_df,
    on="dept_name",
    how="inner"
)

print("\nJoin Output with AQE Enabled")
aqe_join_df.show()

print("\nPhysical Plan with AQE Enabled")
aqe_join_df.explain(True)

print("""
Observation:
When AQE is enabled, Spark may show:

AdaptiveSparkPlan

in the physical plan.

Simple difference:

Catalyst Optimizer:
Planning-time optimization

AQE:
Runtime optimization
""")


# ============================================================
# Final Summary
# ============================================================

print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

print("""
Catalyst Optimizer Summary:

1. Spark does not execute SQL/DataFrame code directly.
2. Spark first creates a logical plan.
3. During parsing, Spark understands the structure of the query.
4. During analysis, Spark verifies table names, column names, and data types.
5. Catalyst applies optimization rules.
6. Optimized logical plan is created.
7. Spark generates one or more physical plans.
8. Physical plan decides HOW the query will be executed.
9. Finally, Spark executes the selected physical plan using distributed tasks.
10. AQE can further optimize the plan during runtime.

Important Optimizations Covered:

1. Projection Pruning
2. Predicate Pushdown
3. Constant Folding
4. Partition Pruning
5. Join Strategy Selection
6. Broadcast Join

Very Important Line:

Logical Plan = What needs to be done
Physical Plan = How Spark will do it
""")

# ============================================================
# One Logical Plan, Multiple Possible Physical Plans
# ============================================================


# ------------------------------------------------------------
# Create sample DataFrames
# ------------------------------------------------------------

employees_data = [
    (1, "Sandeep", 10, 90000),
    (2, "Rahul", 10, 70000),
    (3, "Priya", 20, 60000),
    (4, "Amit", 30, 80000),
    (5, "Sneha", 30, 95000),
    (6, "John", 10, 120000),
]

departments_data = [
    (10, "Engineering"),
    (20, "HR"),
    (30, "Finance"),
]

employees_df = spark.createDataFrame(
    employees_data,
    ["emp_id", "emp_name", "dept_id", "salary"]
)

departments_df = spark.createDataFrame(
    departments_data,
    ["dept_id", "dept_name"]
)

# ============================================================
# CASE 1: Normal Join
# ============================================================
# Logical operation:
# employees JOIN departments ON dept_id
#
# Spark will choose a physical join strategy.
# Depending on configuration/statistics, it may choose SortMergeJoin
# or BroadcastHashJoin.

print("\n" + "=" * 80)
print("CASE 1: Normal Join")
print("=" * 80)

normal_join_df = employees_df.join(
    departments_df,
    on="dept_id",
    how="inner"
)

normal_join_df.show()

print("\nPlan for Normal Join")
normal_join_df.explain(True)


# ============================================================
# CASE 2: Force Broadcast Join
# ============================================================
# Logical operation is still the same:
# employees JOIN departments ON dept_id
#
# But now we are giving Spark a broadcast hint.
# Spark will most likely choose BroadcastHashJoin.

print("\n" + "=" * 80)
print("CASE 2: Broadcast Join using Hint")
print("=" * 80)

broadcast_join_df = employees_df.join(
    broadcast(departments_df),
    on="dept_id",
    how="inner"
)

broadcast_join_df.show()

print("\nPlan for Broadcast Join")
broadcast_join_df.explain(True)


# ============================================================
# CASE 3: Disable Broadcast Join
# ============================================================
# Logical operation is still the same:
# employees JOIN departments ON dept_id
#
# Here we disable automatic broadcast joins by setting threshold to -1.
# Now Spark cannot use automatic broadcast join.
# It will usually choose SortMergeJoin for equi-join.

print("\n" + "=" * 80)
print("CASE 3: Broadcast Disabled")
print("=" * 80)

spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")

no_broadcast_join_df = employees_df.join(
    departments_df,
    on="dept_id",
    how="inner"
)

no_broadcast_join_df.show()

print("\nPlan when Broadcast Join is Disabled")
no_broadcast_join_df.explain(True)

spark.stop()