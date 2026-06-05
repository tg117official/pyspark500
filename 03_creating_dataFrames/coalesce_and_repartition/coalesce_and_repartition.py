# ============================================================
# PySpark Shell Demo: Parallelism, coalesce(), and repartition()
# ============================================================
#
# Objective:
# This script demonstrates how to handle parallelism using:
#
# 1. Partition count inspection
# 2. coalesce()
# 3. repartition()
# 4. repartition by column
# 5. Reducing small output files
# 6. Increasing parallelism
# 7. Rebalancing skewed data
# 8. Trade-offs between coalesce and repartition
#
# Important:
# This script is designed for PySpark shell.
# In PySpark shell, spark and sc are already available.
#
# Spark UI:
# http://localhost:4040
#
# Check:
# - Jobs tab
# - Stages tab
# - Number of tasks
# - Task duration
# - Shuffle read/write
# - SQL tab
# ============================================================

from pyspark.sql.functions import col, rand, spark_partition_id, count, sum as spark_sum, when, lit
import os
import shutil


# ============================================================
# Helper Function 1: Print Clean Heading
# ============================================================

def print_heading(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


# ============================================================
# Helper Function 2: Inspect DataFrame Partitions
# ============================================================
# Purpose:
# Shows:
# 1. Number of partitions
# 2. Number of records in each partition
#
# This helps us understand whether data is balanced or imbalanced.

def inspect_partition_counts(df, title):
    print_heading(title)

    print("\nTotal partitions:")
    print(df.rdd.getNumPartitions())

    partition_counts = (
        df
        .withColumn("partition_id", spark_partition_id())
        .groupBy("partition_id")
        .agg(count("*").alias("record_count"))
        .orderBy("partition_id")
    )

    print("\nRecords per partition:")
    partition_counts.show(200, truncate=False)


# ============================================================
# Helper Function 3: Inspect Partition Content
# ============================================================
# Purpose:
# Shows actual data present inside each partition.
#
# Use only for small DataFrames.

def inspect_partition_content(df, title):
    print_heading(title)

    partition_data = (
        df.rdd
        .mapPartitionsWithIndex(
            lambda pid, iterator: [(pid, [row.asDict() for row in iterator])]
        )
        .collect()
    )

    for pid, records in partition_data:
        if records:
            print(f"Partition {pid}: {records}")
        else:
            print(f"Partition {pid}: EMPTY")


# ============================================================
# Helper Function 4: Count Output Data Files
# ============================================================
# Purpose:
# Spark usually writes one part file per output partition.
#
# This function counts part files in an output directory.
#
# Note:
# Works for local file paths like /tmp/path.

def count_part_files(path):
    if not os.path.exists(path):
        print("Path does not exist:", path)
        return

    part_files = []
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.startswith("part-"):
                part_files.append(os.path.join(root, f))

    print("Part file count:", len(part_files))
    for f in part_files:
        print(f)


# ============================================================
# Initial Spark Information
# ============================================================

print_heading("INITIAL SPARK INFORMATION")

print("Spark UI:", spark.sparkContext.uiWebUrl)
print("Spark Version:", spark.version)
print("Default Parallelism:", sc.defaultParallelism)

spark.conf.set("spark.sql.shuffle.partitions", "8")

print("spark.sql.shuffle.partitions:", spark.conf.get("spark.sql.shuffle.partitions"))

print("""
Important Idea:

1 partition usually creates 1 task in a Spark stage.

More partitions:
- More parallel tasks
- But more scheduling overhead

Fewer partitions:
- Fewer tasks
- But less parallelism and possibly large partitions
""")


# ============================================================
# EXERCISE 1: Understand Current Partitions
# ============================================================
# Purpose:
# Before using coalesce() or repartition(), always check current partitions.
#
# This exercise creates a DataFrame with 8 partitions and inspects
# records per partition.

print_heading("EXERCISE 1: Understand Current Partitions")

ex1_df = (
    spark.range(0, 80, 1, 8)
    .withColumn("amount", col("id") * 10)
)

inspect_partition_counts(
    ex1_df,
    "Exercise 1: DataFrame with 8 Partitions"
)

print("""
Learning:

The DataFrame has 8 partitions.
Each partition can become one task during execution.
""")


# ============================================================
# EXERCISE 2: coalesce() to Reduce Partitions
# ============================================================
# Purpose:
# coalesce() is mainly used to reduce the number of partitions
# without full shuffle.
#
# Real use case:
# After filtering data, the output becomes small.
# We reduce partitions before writing to avoid too many small files.

print_heading("EXERCISE 2: coalesce() to Reduce Partitions")

ex2_df = (
    spark.range(0, 100, 1, 10)
    .withColumn("amount", col("id") * 10)
)

inspect_partition_counts(
    ex2_df,
    "Before coalesce(): 10 Partitions"
)

ex2_coalesced_df = ex2_df.coalesce(3)

inspect_partition_counts(
    ex2_coalesced_df,
    "After coalesce(3): Reduced to 3 Partitions"
)

print("\nPhysical Plan for coalesce(3):")
ex2_coalesced_df.explain(True)

print("""
Learning:

coalesce(3) reduces partitions from 10 to 3.

Important:
coalesce() usually avoids full shuffle.
It is useful when we only want to reduce partitions.
""")


# ============================================================
# EXERCISE 3: coalesce() Cannot Increase Partitions Effectively
# ============================================================
# Purpose:
# coalesce() is not used to increase partitions.
#
# If current partitions = 3 and we call coalesce(10),
# Spark will not increase partitions to 10.

print_heading("EXERCISE 3: coalesce() Cannot Increase Partitions Effectively")

ex3_df = spark.range(0, 30, 1, 3)

inspect_partition_counts(
    ex3_df,
    "Before coalesce(10): 3 Partitions"
)

ex3_coalesced_df = ex3_df.coalesce(10)

inspect_partition_counts(
    ex3_coalesced_df,
    "After coalesce(10): Partition Count Does Not Increase"
)

print("""
Learning:

coalesce() is mainly for reducing partitions.
To increase partitions, use repartition().
""")


# ============================================================
# EXERCISE 4: repartition() to Increase Parallelism
# ============================================================
# Purpose:
# repartition() can increase partitions because it performs shuffle.
#
# Real use case:
# If a DataFrame has very few partitions, only a few tasks run.
# Increasing partitions can improve parallelism.

print_heading("EXERCISE 4: repartition() to Increase Parallelism")

ex4_df = spark.range(0, 100, 1, 2)

inspect_partition_counts(
    ex4_df,
    "Before repartition(): 2 Partitions"
)

ex4_repartitioned_df = ex4_df.repartition(8)

inspect_partition_counts(
    ex4_repartitioned_df,
    "After repartition(8): Increased to 8 Partitions"
)

print("\nPhysical Plan for repartition(8):")
ex4_repartitioned_df.explain(True)

print("""
Learning:

repartition(8) increases partitions from 2 to 8.

Important:
repartition() causes shuffle.
Use it when you need redistribution or more parallelism.
""")


# ============================================================
# EXERCISE 5: repartition() to Rebalance Data
# ============================================================
# Purpose:
# repartition() can rebalance data across partitions.
#
# Real use case:
# If data is unevenly distributed, some tasks become slow.
# repartition() can redistribute the data more evenly.

print_heading("EXERCISE 5: repartition() to Rebalance Data")

# Create intentionally imbalanced data by coalescing to 1 first.
ex5_df = (
    spark.range(0, 1000, 1, 10)
    .coalesce(1)
    .withColumn("amount", col("id") * 10)
)

inspect_partition_counts(
    ex5_df,
    "Before repartition(): Imbalanced Data with 1 Partition"
)

ex5_rebalanced_df = ex5_df.repartition(5)

inspect_partition_counts(
    ex5_rebalanced_df,
    "After repartition(5): Data Redistributed"
)

print("""
Learning:

repartition() performs shuffle and redistributes data.
This helps when data is concentrated in very few partitions.
""")


# ============================================================
# EXERCISE 6: repartition by Column
# ============================================================
# Purpose:
# repartition(numPartitions, column) redistributes data using the column.
#
# Real use case:
# Before groupBy/join on a key, we may repartition by that key
# to control data distribution.
#
# Important:
# Same key values are sent to the same target partition.

print_heading("EXERCISE 6: repartition by Column")

ex6_data = [
    ("C101", 100),
    ("C101", 200),
    ("C101", 300),
    ("C102", 400),
    ("C102", 500),
    ("C103", 600),
    ("C104", 700),
    ("C105", 800),
]

ex6_df = spark.createDataFrame(ex6_data, ["customer_id", "amount"])

ex6_repartitioned_df = ex6_df.repartition(4, "customer_id")

print("\nData after repartition(4, 'customer_id') with partition ID:")
ex6_repartitioned_df.withColumn("partition_id", spark_partition_id()).show(truncate=False)

inspect_partition_content(
    ex6_repartitioned_df,
    "Exercise 6: Partition Content after repartition by customer_id"
)

print("\nPhysical Plan for repartition(4, 'customer_id'):")
ex6_repartitioned_df.explain(True)

print("""
Learning:

repartition(4, 'customer_id') hash partitions the data by customer_id.

Same customer_id values go to the same partition.
This can be useful before key-based operations.
""")


# ============================================================
# EXERCISE 7: coalesce() After Filtering
# ============================================================
# Purpose:
# After filtering, data size may become very small but partition count
# may remain high.
#
# Real use case:
# Reduce small output files before writing filtered data.

print_heading("EXERCISE 7: coalesce() After Filtering")

ex7_df = (
    spark.range(0, 10000, 1, 20)
    .withColumn("amount", col("id") * 10)
)

ex7_filtered_df = ex7_df.filter(col("id") < 100)

inspect_partition_counts(
    ex7_filtered_df,
    "After Filter: Small Data but Still Many Partitions"
)

ex7_output_df = ex7_filtered_df.coalesce(2)

inspect_partition_counts(
    ex7_output_df,
    "After coalesce(2): Better for Small Output"
)

print("""
Learning:

After filtering, data became small.
Instead of writing many small files, use coalesce() to reduce partitions.
""")


# ============================================================
# EXERCISE 8: Output File Count without coalesce()
# ============================================================
# Purpose:
# Spark usually writes one part file per output partition.
#
# Real issue:
# Too many partitions can create too many small output files.

print_heading("EXERCISE 8: Output File Count without coalesce()")

ex8_path = "/tmp/parallelism_demo_without_coalesce"

if os.path.exists(ex8_path):
    shutil.rmtree(ex8_path)

ex8_df = (
    spark.range(0, 100, 1, 10)
    .withColumn("amount", col("id") * 10)
)

print("\nPartitions before write:")
print(ex8_df.rdd.getNumPartitions())

ex8_df.write.mode("overwrite").parquet(ex8_path)

print("\nOutput files without coalesce():")
count_part_files(ex8_path)

print("""
Learning:

If DataFrame has 10 partitions, write may create around 10 part files.
This can create small-file problems when data is small.
""")


# ============================================================
# EXERCISE 9: Output File Count with coalesce()
# ============================================================
# Purpose:
# Demonstrate how coalesce() can reduce output file count.

print_heading("EXERCISE 9: Output File Count with coalesce()")

ex9_path = "/tmp/parallelism_demo_with_coalesce"

if os.path.exists(ex9_path):
    shutil.rmtree(ex9_path)

ex9_df = (
    spark.range(0, 100, 1, 10)
    .withColumn("amount", col("id") * 10)
)

ex9_output_df = ex9_df.coalesce(2)

print("\nPartitions before write after coalesce(2):")
print(ex9_output_df.rdd.getNumPartitions())

ex9_output_df.write.mode("overwrite").parquet(ex9_path)

print("\nOutput files with coalesce(2):")
count_part_files(ex9_path)

print("""
Learning:

coalesce(2) reduces output partitions.
This usually reduces the number of output part files.
""")


# ============================================================
# EXERCISE 10: repartition() for Balanced Output Files
# ============================================================
# Purpose:
# repartition() is useful when we want balanced output files.
#
# Real use case:
# If data is uneven, coalesce may reduce files but may not balance them.
# repartition can redistribute data before writing.

print_heading("EXERCISE 10: repartition() for Balanced Output Files")

ex10_path = "/tmp/parallelism_demo_repartition_balanced_output"

if os.path.exists(ex10_path):
    shutil.rmtree(ex10_path)

ex10_df = (
    spark.range(0, 10000, 1, 4)
    .withColumn("category", when(col("id") < 9000, lit("A")).otherwise(lit("B")))
    .withColumn("amount", rand() * 1000)
)

ex10_repartitioned_df = ex10_df.repartition(4)

inspect_partition_counts(
    ex10_repartitioned_df,
    "After repartition(4): Balanced Output Partitions"
)

ex10_repartitioned_df.write.mode("overwrite").parquet(ex10_path)

print("\nOutput files after repartition(4):")
count_part_files(ex10_path)

print("""
Learning:

repartition() shuffles and redistributes data.
This can produce more balanced output files compared to reducing partitions blindly.
""")


# ============================================================
# EXERCISE 11: repartition by Column Before Aggregation
# ============================================================
# Purpose:
# Demonstrate a key-based partitioning pattern.
#
# Real use case:
# If we are going to aggregate by customer_id, repartitioning by customer_id
# can place same customer records in the same partition.
#
# Note:
# groupBy itself will cause shuffle.
# This exercise is for understanding key distribution.

print_heading("EXERCISE 11: repartition by Column Before Aggregation")

ex11_df = (
    spark.range(0, 10000, 1, 8)
    .withColumn("customer_id", col("id") % 100)
    .withColumn("amount", rand() * 1000)
)

ex11_by_customer_df = ex11_df.repartition(8, "customer_id")

print("\nData after repartition(8, 'customer_id'):")
ex11_by_customer_df.withColumn("partition_id", spark_partition_id()).show(20, truncate=False)

ex11_agg_df = (
    ex11_by_customer_df
    .groupBy("customer_id")
    .agg(spark_sum("amount").alias("total_amount"))
)

print("\nAggregation Result:")
ex11_agg_df.show(20, truncate=False)

print("\nPhysical Plan:")
ex11_agg_df.explain(True)

print("""
Learning:

repartition by key is useful when you want to control key distribution.
However, groupBy may still introduce its own shuffle depending on the plan.
Always confirm using explain(True).
""")


# ============================================================
# EXERCISE 12: coalesce(1) Warning
# ============================================================
# Purpose:
# coalesce(1) creates a single partition.
#
# Real issue:
# This can force one task to process all data.
# It reduces parallelism and can create bottlenecks.
#
# Use coalesce(1) only when:
# - Data is very small
# - Single output file is strictly required
# - It is not a large production workload

print_heading("EXERCISE 12: coalesce(1) Warning")

ex12_df = spark.range(0, 100000, 1, 10)

inspect_partition_counts(
    ex12_df,
    "Before coalesce(1): 10 Partitions"
)

ex12_single_partition_df = ex12_df.coalesce(1)

inspect_partition_counts(
    ex12_single_partition_df,
    "After coalesce(1): Single Partition"
)

print("""
Learning:

coalesce(1) reduces everything to one partition.

This may be okay for small data.
But for large production data, it can create a serious bottleneck.
""")


# ============================================================
# EXERCISE 13: repartition by Range
# ============================================================
# Purpose:
# repartitionByRange() distributes data based on ordered ranges.
#
# Real use case:
# Useful before range-based processing or sorted/range-style writes.
#
# Example:
# Data can be distributed by salary/id/date ranges.

print_heading("EXERCISE 13: repartitionByRange()")

ex13_df = (
    spark.range(0, 100, 1, 4)
    .withColumn("amount", col("id") * 10)
)

ex13_range_df = ex13_df.repartitionByRange(4, "amount")

print("\nData after repartitionByRange(4, 'amount'):")
ex13_range_df.withColumn("partition_id", spark_partition_id()).show(100, truncate=False)

inspect_partition_counts(
    ex13_range_df,
    "Partition Counts after repartitionByRange"
)

print("""
Learning:

repartitionByRange() distributes data by ranges of a column.
This is different from hash-based repartition by column.
""")


# ============================================================
# EXERCISE 14: SQL Hints for Coalesce and Repartition
# ============================================================
# Purpose:
# Spark SQL also supports partitioning hints.
#
# Useful hints:
# COALESCE
# REPARTITION
# REPARTITION_BY_RANGE
#
# These can be used in SQL queries to control output partitions.

print_heading("EXERCISE 14: SQL Hints for Coalesce and Repartition")

ex14_df = (
    spark.range(0, 1000, 1, 8)
    .withColumn("customer_id", col("id") % 100)
    .withColumn("amount", rand() * 1000)
)

ex14_df.createOrReplaceTempView("transactions_parallelism_demo")

sql_coalesce_df = spark.sql("""
    SELECT /*+ COALESCE(2) */
        customer_id,
        amount
    FROM transactions_parallelism_demo
""")

inspect_partition_counts(
    sql_coalesce_df,
    "SQL Hint COALESCE(2)"
)

sql_repartition_df = spark.sql("""
    SELECT /*+ REPARTITION(6, customer_id) */
        customer_id,
        amount
    FROM transactions_parallelism_demo
""")

inspect_partition_counts(
    sql_repartition_df,
    "SQL Hint REPARTITION(6, customer_id)"
)

print("""
Learning:

SQL hints can control partition count and repartitioning behavior.
This is useful when working with Spark SQL instead of DataFrame API.
""")


# ============================================================
# FINAL SUMMARY
# ============================================================

print_heading("FINAL SUMMARY")

print("""
Parallelism and Partition Handling Summary:

1. Partition count controls the number of tasks in many stages.

2. Use df.rdd.getNumPartitions() to check partition count.

3. Use spark_partition_id() to inspect record distribution.

4. coalesce() is mainly used to reduce partitions.

5. coalesce() usually avoids full shuffle.

6. coalesce() is useful after filtering or before writing small output.

7. coalesce(1) should be avoided for large production data.

8. repartition() can increase or decrease partitions.

9. repartition() causes shuffle.

10. repartition() is useful when data needs redistribution or balancing.

11. repartition(numPartitions, column) hash partitions data by column.

12. repartitionByRange() partitions data by value ranges.

13. Use coalesce() when:
    - You only want to reduce partitions
    - Data is already reasonably balanced
    - You want fewer output files without shuffle

14. Use repartition() when:
    - You want to increase parallelism
    - You want to rebalance data
    - You want to distribute by a key
    - You can afford shuffle

15. Industry practice:
    - Check Spark UI before tuning.
    - Avoid too many small files.
    - Avoid very large partitions.
    - Tune partition count based on data size and cluster resources.
    - Use AQE where possible to automatically coalesce shuffle partitions.
""")