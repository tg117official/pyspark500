# ============================================================
# PySpark Demo: Shuffle Partitions Challenge and AQE Coalescing
# ============================================================
#
# Objective:
# This script demonstrates the internal behavior of shuffle partitions.
#
# You will observe:
#
# 1. How many shuffle partitions are created when AQE is disabled
# 2. Which partition contains which records
# 3. How empty shuffle partitions are created
# 4. How AQE coalesces small/empty post-shuffle partitions
# 5. How to inspect partition-level content using mapPartitionsWithIndex()
# 6. How to read physical plans using explain(True)
#
# Important:
# This script is designed for PySpark shell.
# In PySpark shell, spark and sc are already available.
#
# Open Spark UI:
# http://localhost:4040
#
# Check:
# - SQL tab
# - Jobs tab
# - Stages tab
# - Number of tasks
# - Shuffle Read
# - Shuffle Write
# - Physical plan
# ============================================================


from pyspark.sql.functions import col, sum as spark_sum, spark_partition_id


# ============================================================
# Helper Function 1: Print Section Heading
# ============================================================
# Purpose:
# This function prints a clean heading for every exercise.

def print_heading(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


# ============================================================
# Helper Function 2: Inspect Partitions
# ============================================================
# Purpose:
# This function helps us see:
#
# 1. Total number of partitions in a DataFrame
# 2. Which partition contains which records
# 3. Which partitions are empty
#
# Important:
# df.rdd.getNumPartitions() shows number of partitions.
#
# mapPartitionsWithIndex() allows us to inspect data partition by partition.
#
# This is very useful for understanding shuffle internals.

def inspect_partitions(df, title):
    print_heading(title)
    total_partitions = df.rdd.getNumPartitions()
    print("\nTotal partitions in this DataFrame:")
    print(total_partitions)
    partition_data = (
        df.rdd
        .mapPartitionsWithIndex(
            lambda partition_id, iterator: [
                (
                    partition_id,
                    [row.asDict() for row in iterator]
                )
            ]
        )
        .collect()
    )
    empty_partitions = []
    non_empty_partitions = []
    print("\nPartition-wise content:")
    for partition_id, records in partition_data:
        if len(records) == 0:
            empty_partitions.append(partition_id)
            print(f"Partition {partition_id}: EMPTY")
        else:
            non_empty_partitions.append(partition_id)
            print(f"Partition {partition_id}: {records}")
    print("\nSummary:")
    print("Total partitions:", total_partitions)
    print("Non-empty partition count:", len(non_empty_partitions))
    print("Non-empty partition IDs:", non_empty_partitions)
    print("Empty partition count:", len(empty_partitions))
    print("Empty partition IDs:", empty_partitions)


# ============================================================
# Helper Function 3: Create Small Key-Based DataFrame
# ============================================================
# Purpose:
# We intentionally create data with only 3 unique keys: A, B, C.
#
# Later, we will set spark.sql.shuffle.partitions = 10.
#
# Since there are only 3 keys but 10 shuffle partitions,
# many shuffle partitions can become empty.

def create_small_key_df():
    data = [
        ("A", 10),
        ("A", 20),
        ("A", 30),
        ("B", 100),
        ("B", 200),
        ("C", 1000),
        ("C", 2000),
    ]
    columns = ["key", "amount"]
    return spark.createDataFrame(data, columns)

# ============================================================
# Helper Function 4: Create GroupBy Result
# ============================================================
# Purpose:
# groupBy is a wide transformation.
#
# It requires Spark to bring the same keys together.
# Therefore, groupBy causes shuffle.
#
# Example:
# All A records should come together.
# All B records should come together.
# All C records should come together.

def create_grouped_df(df):
    return (
        df
        .groupBy("key")
        .agg(spark_sum("amount").alias("total_amount"))
    )


# ============================================================
# Initial Information
# ============================================================

print_heading("INITIAL SPARK INFORMATION")
print("Spark UI:", spark.sparkContext.uiWebUrl)
print("Spark Version:", spark.version)

print("""
Important Concept:

Shuffle happens when Spark needs to redistribute data across partitions.

Example:
df.groupBy("key").sum("amount")

For this operation, Spark must bring all same keys together.

If we configure:
spark.sql.shuffle.partitions = 10

Spark initially plans 10 shuffle partitions.

Without AQE:
Spark generally keeps these 10 post-shuffle partitions.

With AQE:
Spark can observe runtime shuffle statistics and coalesce small/empty
post-shuffle partitions into fewer partitions.
""")


# ============================================================
# EXERCISE 1: AQE Disabled - Empty Shuffle Partitions
# ============================================================
# Purpose:
# This exercise demonstrates a challenge when AQE is disabled.
#
# We have:
# - Only 3 unique keys: A, B, C
# - But shuffle partitions are set to 10
#
# Because Spark uses hash partitioning, only a few partitions may receive data.
# Remaining shuffle partitions may be empty.
#
# Without AQE, Spark still works with the configured number of shuffle partitions.
#
# Expected learning:
# Too many shuffle partitions can create empty/tiny partitions and unnecessary tasks.

print_heading("EXERCISE 1: AQE Disabled - Empty Shuffle Partitions")

# Disable AQE
spark.conf.set("spark.sql.adaptive.enabled", "false")

# Set shuffle partitions higher than required for this small dataset
spark.conf.set("spark.sql.shuffle.partitions", "10")

print("AQE Enabled:", spark.conf.get("spark.sql.adaptive.enabled"))
print("Shuffle Partitions:", spark.conf.get("spark.sql.shuffle.partitions"))

df_no_aqe = create_small_key_df()

print("\nInput Data:")
df_no_aqe.show()

print("\nInput DataFrame partition count:")
print(df_no_aqe.rdd.getNumPartitions())

grouped_no_aqe = create_grouped_df(df_no_aqe)

print("\nPhysical Plan with AQE Disabled:")
grouped_no_aqe.explain(True)

# This action triggers execution.
# The helper will collect partition-wise content.
inspect_partitions(
    grouped_no_aqe,
    "AQE Disabled Result: Partition-wise Content After groupBy"
)

print("""
Key Observation:

1. We configured spark.sql.shuffle.partitions = 10.
2. The result DataFrame may have 10 partitions.
3. Since we have only 3 keys, only a few partitions contain data.
4. Many partitions can be empty.
5. Without AQE, Spark does not automatically reduce these partitions.

Challenge:
Empty/tiny shuffle partitions can create unnecessary task scheduling overhead.
""")


# ============================================================
# EXERCISE 2: Manual Repartition by Key
# ============================================================
# Purpose:
# This exercise makes hash partitioning easier to see.
#
# We manually call:
# df.repartition(10, "key")
#
# This tells Spark:
# "Redistribute the data into 10 partitions based on key."
#
# Since only A, B, C keys exist:
# - A records will go to one partition
# - B records will go to one partition
# - C records will go to one partition
# - Many other partitions may be empty
#
# This helps students understand why empty shuffle partitions can exist.

print_heading("EXERCISE 2: Manual Repartition by Key - Which Key Goes Where?")

spark.conf.set("spark.sql.adaptive.enabled", "false")
spark.conf.set("spark.sql.shuffle.partitions", "10")

print("AQE Enabled:", spark.conf.get("spark.sql.adaptive.enabled"))
print("Shuffle Partitions:", spark.conf.get("spark.sql.shuffle.partitions"))

df_repartition_demo = create_small_key_df()

repartitioned_df = df_repartition_demo.repartition(10, "key")

print("\nData after repartition(10, 'key') with visible partition IDs:")
repartitioned_df.withColumn("partition_id", spark_partition_id()).show()

print("\nPhysical Plan for repartition(10, 'key'):")
repartitioned_df.explain(True)

inspect_partitions(
    repartitioned_df,
    "Manual Repartition Result: Partition-wise Content"
)

print("""
Key Observation:

Spark uses hash-based distribution for repartition by key.

Conceptually:
partition_id = hash(key) % number_of_partitions

Since we have only 3 keys and 10 target partitions:
- A goes to one partition
- B goes to one partition
- C goes to one partition
- Remaining partitions may remain empty

This is why too many shuffle partitions can lead to empty partitions.
""")


# ============================================================
# EXERCISE 3: AQE Enabled - Coalescing Shuffle Partitions
# ============================================================
# Purpose:
# This exercise demonstrates how AQE helps.
#
# We again use:
# - Only 3 keys: A, B, C
# - shuffle partitions = 10
#
# But this time AQE is enabled.
#
# AQE can observe actual shuffle output sizes at runtime.
# Then it can coalesce small/empty post-shuffle partitions.
#
# Expected learning:
# AQE can reduce unnecessary post-shuffle partitions/tasks.

print_heading("EXERCISE 3: AQE Enabled - Coalescing Small/Empty Shuffle Partitions")

# Enable AQE
spark.conf.set("spark.sql.adaptive.enabled", "true")

# Enable post-shuffle partition coalescing
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")

# Keep the initially planned shuffle partitions high
spark.conf.set("spark.sql.shuffle.partitions", "10")

# These settings make coalescing easier to observe in a small local demo.
# parallelismFirst=false tells Spark to respect advisory partition size more.
# advisoryPartitionSizeInBytes=1MB means small shuffle partitions can be combined.
spark.conf.set("spark.sql.adaptive.coalescePartitions.parallelismFirst", "false")
spark.conf.set("spark.sql.adaptive.advisoryPartitionSizeInBytes", "1MB")

print("AQE Enabled:", spark.conf.get("spark.sql.adaptive.enabled"))
print("Coalesce Partitions Enabled:", spark.conf.get("spark.sql.adaptive.coalescePartitions.enabled"))
print("Shuffle Partitions:", spark.conf.get("spark.sql.shuffle.partitions"))
print("AQE parallelismFirst:", spark.conf.get("spark.sql.adaptive.coalescePartitions.parallelismFirst"))
print("AQE advisory partition size:", spark.conf.get("spark.sql.adaptive.advisoryPartitionSizeInBytes"))

df_aqe = create_small_key_df()

grouped_aqe = create_grouped_df(df_aqe)

print("\nPhysical Plan Before Action with AQE Enabled:")
grouped_aqe.explain(True)

# This action triggers execution and AQE can adapt the post-shuffle partitions.
inspect_partitions(
    grouped_aqe,
    "AQE Enabled Result: Partition-wise Content After Coalescing"
)

print("\nPhysical Plan After Action with AQE Enabled:")
grouped_aqe.explain(True)

print("""
Key Observation:

With AQE enabled:
1. Spark initially plans based on spark.sql.shuffle.partitions = 10.
2. During execution, Spark observes actual shuffle output size.
3. Since data is very small, many shuffle partitions are tiny/empty.
4. AQE can coalesce these post-shuffle partitions.
5. The resulting DataFrame may have fewer partitions than 10.

Possible physical plan keywords:
- AdaptiveSparkPlan
- AQEShuffleRead
- coalesced

Important:
AQE does not change the final query result.
It changes the physical execution strategy.
""")


# ============================================================
# EXERCISE 4: Compare AQE Disabled vs AQE Enabled Side by Side
# ============================================================
# Purpose:
# This exercise gives a clean comparison summary.
#
# We run the same groupBy twice:
# 1. AQE disabled
# 2. AQE enabled
#
# Then we print:
# - Partition count
# - Partition-wise data
#
# This makes the difference easier to explain in class.

print_heading("EXERCISE 4: Side-by-Side Comparison")

# -------------------------------
# AQE Disabled Version
# -------------------------------

spark.conf.set("spark.sql.adaptive.enabled", "false")
spark.conf.set("spark.sql.shuffle.partitions", "10")

df_compare_no_aqe = create_small_key_df()
result_no_aqe = create_grouped_df(df_compare_no_aqe)

# Trigger execution and collect partition data
no_aqe_partition_data = (
    result_no_aqe.rdd
    .mapPartitionsWithIndex(
        lambda partition_id, iterator: [
            (
                partition_id,
                [row.asDict() for row in iterator]
            )
        ]
    )
    .collect()
)

# -------------------------------
# AQE Enabled Version
# -------------------------------

spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
spark.conf.set("spark.sql.shuffle.partitions", "10")
spark.conf.set("spark.sql.adaptive.coalescePartitions.parallelismFirst", "false")
spark.conf.set("spark.sql.adaptive.advisoryPartitionSizeInBytes", "1MB")

df_compare_aqe = create_small_key_df()
result_aqe = create_grouped_df(df_compare_aqe)

# Trigger execution and collect partition data
aqe_partition_data = (
    result_aqe.rdd
    .mapPartitionsWithIndex(
        lambda partition_id, iterator: [
            (
                partition_id,
                [row.asDict() for row in iterator]
            )
        ]
    )
    .collect()
)

print("\nAQE Disabled Partition Count:")
print(result_no_aqe.rdd.getNumPartitions())

print("\nAQE Disabled Partition Content:")
for partition_id, records in no_aqe_partition_data:
    print(f"Partition {partition_id}: {records if records else 'EMPTY'}")

print("\nAQE Enabled Partition Count:")
print(result_aqe.rdd.getNumPartitions())

print("\nAQE Enabled Partition Content:")
for partition_id, records in aqe_partition_data:
    print(f"Partition {partition_id}: {records if records else 'EMPTY'}")

print("""
Comparison Summary:

Without AQE:
- Spark uses the configured shuffle partition count.
- Empty/tiny shuffle partitions may remain.
- More post-shuffle tasks can be created.

With AQE:
- Spark starts with the configured shuffle partition count.
- Runtime shuffle statistics are collected.
- Small/empty post-shuffle partitions can be coalesced.
- Fewer post-shuffle tasks may be created.
""")


# ============================================================
# FINAL SUMMARY
# ============================================================

print_heading("FINAL SUMMARY")

print("""
Important Points:

1. Shuffle happens when Spark needs to redistribute data.
   Example:
   groupBy, join, distinct, repartition, orderBy.

2. spark.sql.shuffle.partitions controls the planned number of shuffle partitions
   for Spark SQL/DataFrame shuffle operations.

3. If the number of unique keys is small and shuffle partitions are high,
   many shuffle partitions can be empty.

4. Without AQE:
   Spark generally follows the configured shuffle partition count.

5. With AQE:
   Spark can use runtime shuffle statistics to coalesce small/empty
   post-shuffle partitions.

6. AQE does not change the query result.
   It changes the physical execution plan.

7. Useful functions for internal inspection:
   df.rdd.getNumPartitions()
   df.withColumn("partition_id", spark_partition_id()).show()
   df.rdd.mapPartitionsWithIndex(...)
   df.explain(True)

8. Useful Spark UI areas:
   SQL tab
   Jobs tab
   Stages tab
   Number of tasks
   Shuffle Read
   Shuffle Write

9. Useful physical plan keywords:
   Exchange
   hashpartitioning
   AdaptiveSparkPlan
   AQEShuffleRead
   coalesced
""")