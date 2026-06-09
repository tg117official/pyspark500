# ============================================================
# PySpark Shell Lab: Data Skew in DataFrame Operations
# ============================================================
#
# Objective:
# Practice skew diagnosis and mitigation for common PySpark DataFrame operations.
#
# Designed for PySpark shell:
# - spark and sc are already available.
#
# Spark UI:
# http://localhost:4040
#
# What to observe:
# - Spark UI > SQL tab
# - Spark UI > Stages tab
# - Task duration imbalance
# - Shuffle read/write
# - Spill memory/disk
# - Number of records per partition
#
# Important:
# These examples use small data for teaching.
# In production, the same patterns appear with much larger datasets.
# ============================================================

from pyspark.sql.functions import (
    col, lit, when, rand, floor, concat_ws, pmod,
    spark_partition_id, count, sum as spark_sum,
    row_number, desc, asc, expr
)
from pyspark.sql.window import Window
import time


# ============================================================
# Helper functions
# ============================================================

def print_heading(title):
    print("\n" + "=" * 110)
    print(title)
    print("=" * 110)


def run_action(label, action_func):
    start = time.time()
    result = action_func()
    end = time.time()
    print("\n" + "-" * 90)
    print(label)
    print("Result:", result)
    print("Time Taken:", round(end - start, 2), "seconds")
    print("-" * 90)
    return result


def inspect_partition_counts(df, title, limit_rows=200):
    """
    Diagnosis window:
    Shows how many records are present in each current partition.
    """
    print_heading(title)
    print("Total partitions:", df.rdd.getNumPartitions())
    (
        df.withColumn("partition_id", spark_partition_id())
          .groupBy("partition_id")
          .agg(count("*").alias("record_count"))
          .orderBy("partition_id")
          .show(limit_rows, truncate=False)
    )


def inspect_key_distribution(df, key_col, title, top_n=20):
    """
    Diagnosis window:
    Shows most frequent keys.
    Useful for identifying hot/skewed keys.
    """
    print_heading(title)
    (
        df.groupBy(key_col)
          .agg(count("*").alias("record_count"))
          .orderBy(desc("record_count"))
          .show(top_n, truncate=False)
    )


def inspect_partition_content_small(df, title):
    """
    Diagnosis window:
    Shows actual content partition-wise.
    Use only for small DataFrames.
    """
    print_heading(title)
    rows = (
        df.rdd
          .mapPartitionsWithIndex(
              lambda pid, itr: [(pid, [r.asDict() for r in itr])]
          )
          .collect()
    )
    for pid, records in rows:
        print(f"Partition {pid}: {records if records else 'EMPTY'}")


# ============================================================
# Base configs for local
# ============================================================

print_heading("BASE CONFIGURATION")

spark.conf.set("spark.sql.shuffle.partitions", "8")
spark.conf.set("spark.sql.adaptive.enabled", "false")
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")

print("Spark UI:", spark.sparkContext.uiWebUrl)
print("Spark Version:", spark.version)
print("Shuffle Partitions:", spark.conf.get("spark.sql.shuffle.partitions"))
print("AQE Enabled:", spark.conf.get("spark.sql.adaptive.enabled"))
print("Auto Broadcast Threshold:", spark.conf.get("spark.sql.autoBroadcastJoinThreshold"))

print("""
note:
Skew hurts when one key/range/group receives much more data than others.
The usual symptom is:
- one/few tasks run much longer than others
- one/few shuffle partitions contain much more data
- high spill or GC in specific tasks
""")


# ============================================================
# EXERCISE 1: join() with skewed key
# ============================================================
# Problem:
# One hot join key sends a huge amount of data to one shuffle partition.
#
# Regular way:
# Large table joins dimension table using customer_id.
# customer_id = 0 is extremely frequent.
#
# Skew-aware solutions shown:
# A) Broadcast small dimension table
# B) Enable AQE skew join handling
# ============================================================

print_heading("EXERCISE 1: join() with skewed key")

# ----------------------------
# Data setup
# ----------------------------

orders_e1 = (
    spark.range(0, 100000)
    .withColumn(
        "customer_id",
        when(col("id") < 80000, lit(0)).otherwise((col("id") % 1000).cast("int"))
    )
    .withColumn("amount", rand() * 1000)
    .repartition(8, "customer_id")
)

customers_e1 = (
    spark.range(0, 1000)
    .withColumnRenamed("id", "customer_id")
    .withColumn("customer_type", when(col("customer_id") == 0, lit("HOT")).otherwise(lit("NORMAL")))
)

inspect_key_distribution(orders_e1, "customer_id", "Exercise 1 Diagnosis: Hot join keys in orders")
inspect_partition_counts(orders_e1, "Exercise 1 Diagnosis: orders after repartition by customer_id")

# ----------------------------
# Problem version
# ----------------------------

spark.conf.set("spark.sql.adaptive.enabled", "false")
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")
spark.conf.set("spark.sql.shuffle.partitions", "8")

join_problem_e1 = orders_e1.join(customers_e1, "customer_id", "inner")

print("\nProblem version plan: shuffle join likely")
join_problem_e1.explain(True)

run_action(
    "Exercise 1 Problem: Join count with skewed key",
    lambda: join_problem_e1.count()
)

# ----------------------------
# Solution A: Broadcast small side
# ----------------------------
# If one side is small, broadcast avoids shuffling the large skewed side
# for the join. This is often the best solution.

from pyspark.sql.functions import broadcast

join_broadcast_e1 = orders_e1.join(broadcast(customers_e1), "customer_id", "inner")

print("\nSolution A plan: Broadcast small dimension table")
join_broadcast_e1.explain(True)

run_action(
    "Exercise 1 Solution A: Broadcast join count",
    lambda: join_broadcast_e1.count()
)

# ----------------------------
# Solution B: AQE skew join
# ----------------------------
# AQE can split skewed shuffle partitions in sort-merge join scenarios.
# In local demos, we lower thresholds to make the behavior easier to observe.

spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")
spark.conf.set("spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes", "1")
spark.conf.set("spark.sql.adaptive.skewJoin.skewedPartitionFactor", "2")

join_aqe_e1 = orders_e1.join(customers_e1, "customer_id", "inner")

print("\nSolution B plan before/after action: AQE skew join enabled")
join_aqe_e1.explain(True)

run_action(
    "Exercise 1 Solution B: AQE skew join count",
    lambda: join_aqe_e1.count()
)

join_aqe_e1.explain(True)


# ============================================================
# EXERCISE 2: groupBy().agg() with skewed group key
# ============================================================
# Problem:
# DataFrame groupBy performs partial aggregation, but final aggregation
# still sends the same hot key to one reducer.
#
# Regular way:
# groupBy("customer_id").sum("amount")
#
# Skew-aware solution:
# Two-phase salted aggregation:
# 1. Add salt to distribute hot key into multiple groups
# 2. Aggregate by customer_id + salt
# 3. Aggregate again by customer_id
# ============================================================

print_heading("EXERCISE 2: groupBy().agg() with skewed key")

spark.conf.set("spark.sql.adaptive.enabled", "false")
spark.conf.set("spark.sql.shuffle.partitions", "8")

sales_e2 = (
    spark.range(0, 120000)
    .withColumn(
        "customer_id",
        when(col("id") < 90000, lit(0)).otherwise((col("id") % 1000).cast("int"))
    )
    .withColumn("amount", rand() * 100)
)

inspect_key_distribution(sales_e2, "customer_id", "Exercise 2 Diagnosis: Hot groupBy key")

# Problem version
group_problem_e2 = (
    sales_e2
    .groupBy("customer_id")
    .agg(spark_sum("amount").alias("total_amount"))
)

print("\nProblem version plan: normal groupBy")
group_problem_e2.explain(True)

run_action(
    "Exercise 2 Problem: groupBy sum count",
    lambda: group_problem_e2.count()
)

# Solution: salted two-phase aggregation
salt_count = 10

sales_salted_e2 = (
    sales_e2
    .withColumn(
        "salt",
        when(col("customer_id") == 0, pmod(col("id"), lit(salt_count))).otherwise(lit(0))
    )
)

partial_agg_e2 = (
    sales_salted_e2
    .groupBy("customer_id", "salt")
    .agg(spark_sum("amount").alias("partial_amount"))
)

final_agg_e2 = (
    partial_agg_e2
    .groupBy("customer_id")
    .agg(spark_sum("partial_amount").alias("total_amount"))
)

print("\nSolution plan: two-phase salted aggregation")
final_agg_e2.explain(True)

inspect_key_distribution(
    sales_salted_e2.filter(col("customer_id") == 0),
    "salt",
    "Exercise 2 Diagnosis: Hot key distributed across salt values"
)

run_action(
    "Exercise 2 Solution: salted two-phase aggregation count",
    lambda: final_agg_e2.count()
)


# ============================================================
# EXERCISE 3: distinct() with duplicate-heavy values
# ============================================================
# Problem:
# distinct() brings same rows/values together.
# A very frequent value can create skew.
#
# Regular way:
# df.select("event_type").distinct()
#
# Skew-aware solution:
# Two-phase distinct using salt:
# 1. distinct on value + salt
# 2. final distinct on value
#
# Note:
# This is useful only when duplicate-heavy values are creating pressure.
# ============================================================

print_heading("EXERCISE 3: distinct() with duplicate-heavy values")

events_e3 = (
    spark.range(0, 100000)
    .withColumn(
        "event_type",
        when(col("id") < 85000, lit("click")).otherwise(concat_ws("_", lit("event"), (col("id") % 1000)))
    )
    .select("event_type", "id")
)

inspect_key_distribution(events_e3, "event_type", "Exercise 3 Diagnosis: Duplicate-heavy values")

# Problem version
distinct_problem_e3 = events_e3.select("event_type").distinct()

print("\nProblem version plan: normal distinct")
distinct_problem_e3.explain(True)

run_action(
    "Exercise 3 Problem: normal distinct count",
    lambda: distinct_problem_e3.count()
)

# Solution: two-phase salted distinct
events_salted_e3 = (
    events_e3
    .withColumn(
        "salt",
        when(col("event_type") == "click", pmod(col("id"), lit(10))).otherwise(lit(0))
    )
)

distinct_phase1_e3 = events_salted_e3.select("event_type", "salt").distinct()
distinct_solution_e3 = distinct_phase1_e3.select("event_type").distinct()

print("\nSolution plan: salted two-phase distinct")
distinct_solution_e3.explain(True)

inspect_key_distribution(
    events_salted_e3.filter(col("event_type") == "click"),
    "salt",
    "Exercise 3 Diagnosis: click distributed across salt values"
)

run_action(
    "Exercise 3 Solution: salted two-phase distinct count",
    lambda: distinct_solution_e3.count()
)


# ============================================================
# EXERCISE 4: dropDuplicates() with skewed duplicate keys
# ============================================================
# Problem:
# dropDuplicates(["device_id"]) groups duplicate keys together.
# One extremely frequent device_id can overload one partition.
#
# Regular way:
# df.dropDuplicates(["device_id"])
#
# Skew-aware solution:
# Split hot key and normal keys:
# - normal keys use dropDuplicates normally
# - hot key is handled separately with limit/first depending business rule
#
# Important:
# Dedup logic must match business requirement.
# ============================================================

print_heading("EXERCISE 4: dropDuplicates() with skewed duplicate key")

devices_e4 = (
    spark.range(0, 100000)
    .withColumn(
        "device_id",
        when(col("id") < 90000, lit("UNKNOWN_DEVICE")).otherwise(concat_ws("_", lit("D"), (col("id") % 1000)))
    )
    .withColumn("event_ts", col("id"))
)

inspect_key_distribution(devices_e4, "device_id", "Exercise 4 Diagnosis: Duplicate-heavy device_id")

# Problem version
dedup_problem_e4 = devices_e4.dropDuplicates(["device_id"])

print("\nProblem version plan: normal dropDuplicates")
dedup_problem_e4.explain(True)

run_action(
    "Exercise 4 Problem: normal dropDuplicates count",
    lambda: dedup_problem_e4.count()
)

# Solution: isolate hot/default key
normal_devices_e4 = devices_e4.filter(col("device_id") != "UNKNOWN_DEVICE")
hot_devices_e4 = devices_e4.filter(col("device_id") == "UNKNOWN_DEVICE")

normal_dedup_e4 = normal_devices_e4.dropDuplicates(["device_id"])

# Business rule: keep one row for UNKNOWN_DEVICE.
# In real projects, you may drop unknowns, keep latest, or process separately.
hot_dedup_e4 = hot_devices_e4.orderBy("event_ts").limit(1)

dedup_solution_e4 = normal_dedup_e4.unionByName(hot_dedup_e4)

print("\nSolution plan: isolate hot key and deduplicate separately")
dedup_solution_e4.explain(True)

run_action(
    "Exercise 4 Solution: hot-key isolated dropDuplicates count",
    lambda: dedup_solution_e4.count()
)


# ============================================================
# EXERCISE 5: orderBy() / sort() with uneven ranges
# ============================================================
# Problem:
# Sorting/range partitioning can become uneven if most data has the same
# or very close sort values.
#
# Regular way:
# df.orderBy("score")
#
# Skew-aware solution:
# If exact global order by one column is not required for downstream storage,
# use a secondary column to break ties or use sortWithinPartitions after
# better repartitioning.
#
# Note:
# Global orderBy is expensive. For file writes, global sorting is often
# unnecessary; partition-level sorting may be enough.
# ============================================================

print_heading("EXERCISE 5: orderBy() / sort() with uneven ranges")

scores_e5 = (
    spark.range(0, 100000)
    .withColumn("score", when(col("id") < 90000, lit(100)).otherwise((col("id") % 1000).cast("int")))
    .withColumn("user_id", col("id"))
)

inspect_key_distribution(scores_e5, "score", "Exercise 5 Diagnosis: Skewed sort/range values")

# Problem version: global sort
sort_problem_e5 = scores_e5.orderBy("score")

print("\nProblem version plan: global orderBy on skewed score")
sort_problem_e5.explain(True)

run_action(
    "Exercise 5 Problem: orderBy count",
    lambda: sort_problem_e5.count()
)

# Solution: add secondary key to break large equal ranges when acceptable
# If exact ordering by score only is required, the secondary key is still valid
# as deterministic tie-breaking: ORDER BY score, user_id.
sort_solution_e5 = scores_e5.orderBy("score", "user_id")

print("\nSolution plan: orderBy score, user_id to break tie ranges")
sort_solution_e5.explain(True)

run_action(
    "Exercise 5 Solution: orderBy with tie-breaker count",
    lambda: sort_solution_e5.count()
)

# Alternative for file layout, not exact global order:
layout_solution_e5 = scores_e5.repartitionByRange(8, "score", "user_id").sortWithinPartitions("score", "user_id")
inspect_partition_counts(layout_solution_e5, "Exercise 5 Diagnosis: range repartition with secondary key")


# ============================================================
# EXERCISE 6: Window.partitionBy() with hot partition key
# ============================================================
# Problem:
# Window.partitionBy("customer_id") requires all rows of one customer to be
# processed as one logical window partition.
# A hot customer can create one huge task.
#
# Regular way:
# row_number over Window.partitionBy("customer_id").orderBy("event_ts")
#
# Skew-aware solution:
# If business logic allows, use a finer partition key:
# customer_id + event_date.
#
# If exact window over all records of hot customer is required, salting may
# not be semantically correct.
# ============================================================

print_heading("EXERCISE 6: Window.partitionBy() with hot key")

events_e6 = (
    spark.range(0, 100000)
    .withColumn(
        "customer_id",
        when(col("id") < 85000, lit(0)).otherwise((col("id") % 1000).cast("int"))
    )
    .withColumn(
        "event_date",
        when(col("id") % 3 == 0, lit("2026-06-01"))
        .when(col("id") % 3 == 1, lit("2026-06-02"))
        .otherwise(lit("2026-06-03"))
    )
    .withColumn("event_ts", col("id"))
)

inspect_key_distribution(events_e6, "customer_id", "Exercise 6 Diagnosis: Hot window partition key")

# Problem version
w_problem = Window.partitionBy("customer_id").orderBy("event_ts")

window_problem_e6 = events_e6.withColumn("rn", row_number().over(w_problem))

print("\nProblem version plan: window partitioned only by customer_id")
window_problem_e6.explain(True)

run_action(
    "Exercise 6 Problem: window count",
    lambda: window_problem_e6.count()
)

# Solution: finer partitioning if business logic allows
# Example: ranking per customer per date instead of ranking per customer globally.
w_solution = Window.partitionBy("customer_id", "event_date").orderBy("event_ts")

window_solution_e6 = events_e6.withColumn("rn", row_number().over(w_solution))

print("\nSolution plan: window partitioned by customer_id and event_date")
window_solution_e6.explain(True)

run_action(
    "Exercise 6 Solution: finer window partition count",
    lambda: window_solution_e6.count()
)

print("""
Exercise 6 Important Note:
If the business requirement is truly 'rank all events per customer globally',
then salting will break correctness.
In that case, handle hot customer separately, increase resources/partitions,
or redesign the logic.
""")


# ============================================================
# EXERCISE 7: repartition("skewed_column")
# ============================================================
# Problem:
# Repartitioning by a skewed column sends the hot key to one partition.
#
# Regular way:
# df.repartition(8, "status")
#
# Skew-aware solution:
# Add salt for hot value if the goal is only to distribute workload.
# Do not use salted key if exact key grouping is required without final merge.
# ============================================================

print_heading("EXERCISE 7: repartition('skewed_column')")

status_e7 = (
    spark.range(0, 100000)
    .withColumn("status", when(col("id") < 95000, lit("ACTIVE")).otherwise(lit("INACTIVE")))
    .withColumn("amount", rand() * 100)
)

inspect_key_distribution(status_e7, "status", "Exercise 7 Diagnosis: Skewed repartition column")

# Problem version
repart_problem_e7 = status_e7.repartition(8, "status")
inspect_partition_counts(repart_problem_e7, "Exercise 7 Problem: repartition(8, 'status')")

# Solution: distribute hot value using salt
repart_solution_e7 = (
    status_e7
    .withColumn(
        "salt",
        when(col("status") == "ACTIVE", pmod(col("id"), lit(8))).otherwise(lit(0))
    )
    .repartition(8, "status", "salt")
)

inspect_partition_counts(repart_solution_e7, "Exercise 7 Solution: repartition by status + salt")

print("""
Exercise 7 Important Note:
Salting is good for distributing workload.
If the next operation requires final result by status, aggregate/merge back by status later.
""")


# ============================================================
# EXERCISE 8: pivot() with skewed group/pivot values
# ============================================================
# Problem:
# pivot involves grouping and aggregation.
# Dominant group or pivot values can create uneven workload.
#
# Regular way:
# df.groupBy("customer_id").pivot("category").sum("amount")
#
# Skew-aware solution:
# Pre-aggregate with salt for hot customer, then final pivot/aggregation.
# Also limit pivot values explicitly.
# ============================================================

print_heading("EXERCISE 8: pivot() with skewed values")

pivot_e8 = (
    spark.range(0, 100000)
    .withColumn("customer_id", when(col("id") < 85000, lit(0)).otherwise((col("id") % 1000).cast("int")))
    .withColumn(
        "category",
        when(col("id") % 4 == 0, lit("mobile"))
        .when(col("id") % 4 == 1, lit("laptop"))
        .when(col("id") % 4 == 2, lit("book"))
        .otherwise(lit("other"))
    )
    .withColumn("amount", rand() * 100)
)

inspect_key_distribution(pivot_e8, "customer_id", "Exercise 8 Diagnosis: Hot pivot group key")

# Problem version
pivot_problem_e8 = (
    pivot_e8
    .groupBy("customer_id")
    .pivot("category")
    .sum("amount")
)

print("\nProblem version plan: normal pivot")
pivot_problem_e8.explain(True)

run_action(
    "Exercise 8 Problem: pivot count",
    lambda: pivot_problem_e8.count()
)

# Solution: salted pre-aggregation + explicit pivot values
pivot_salted_e8 = (
    pivot_e8
    .withColumn("salt", when(col("customer_id") == 0, pmod(col("id"), lit(10))).otherwise(lit(0)))
)

pivot_partial_e8 = (
    pivot_salted_e8
    .groupBy("customer_id", "salt", "category")
    .agg(spark_sum("amount").alias("partial_amount"))
)

pivot_solution_e8 = (
    pivot_partial_e8
    .groupBy("customer_id")
    .pivot("category", ["mobile", "laptop", "book", "other"])
    .agg(spark_sum("partial_amount"))
)

print("\nSolution plan: salted pre-aggregation + explicit pivot values")
pivot_solution_e8.explain(True)

run_action(
    "Exercise 8 Solution: skew-aware pivot count",
    lambda: pivot_solution_e8.count()
)


# ============================================================
# EXERCISE 9: cube() / rollup() with skewed grouping columns
# ============================================================
# Problem:
# cube/rollup create multiple grouping levels.
# Skew in base grouping columns can be amplified.
#
# Regular way:
# df.rollup("region", "status").sum("amount")
#
# Skew-aware solution:
# Pre-aggregate at a finer level, isolate hot values, or use salted
# two-phase aggregation for hot combinations.
# ============================================================

print_heading("EXERCISE 9: cube() / rollup() with skewed values")

rollup_e9 = (
    spark.range(0, 100000)
    .withColumn("region", when(col("id") < 90000, lit("APAC")).otherwise(lit("EU")))
    .withColumn("status", when(col("id") < 95000, lit("ACTIVE")).otherwise(lit("INACTIVE")))
    .withColumn("amount", rand() * 100)
)

inspect_key_distribution(rollup_e9, "region", "Exercise 9 Diagnosis: Skewed region")
inspect_key_distribution(rollup_e9, "status", "Exercise 9 Diagnosis: Skewed status")

# Problem version
rollup_problem_e9 = (
    rollup_e9
    .rollup("region", "status")
    .agg(spark_sum("amount").alias("total_amount"))
)

print("\nProblem version plan: normal rollup")
rollup_problem_e9.explain(True)

run_action(
    "Exercise 9 Problem: rollup count",
    lambda: rollup_problem_e9.count()
)

# Solution: salted pre-aggregation for hot combination
rollup_salted_e9 = (
    rollup_e9
    .withColumn(
        "salt",
        when((col("region") == "APAC") & (col("status") == "ACTIVE"), pmod(col("id"), lit(10))).otherwise(lit(0))
    )
)

rollup_partial_e9 = (
    rollup_salted_e9
    .groupBy("region", "status", "salt")
    .agg(spark_sum("amount").alias("partial_amount"))
)

rollup_solution_e9 = (
    rollup_partial_e9
    .rollup("region", "status")
    .agg(spark_sum("partial_amount").alias("total_amount"))
)

print("\nSolution plan: pre-aggregate hot combination before rollup")
rollup_solution_e9.explain(True)

run_action(
    "Exercise 9 Solution: skew-aware rollup count",
    lambda: rollup_solution_e9.count()
)


# ============================================================
# EXERCISE 10: groupBy on low-cardinality columns
# ============================================================
# Problem:
# Low-cardinality columns like status/gender/is_active can create few huge groups.
# Partial aggregation helps, but final reducers may still be imbalanced.
#
# Regular way:
# df.groupBy("status").count()
#
# Skew-aware solution:
# Two-phase salted aggregation.
# ============================================================

print_heading("EXERCISE 10: groupBy on low-cardinality skewed column")

low_card_e10 = (
    spark.range(0, 150000)
    .withColumn("status", when(col("id") < 140000, lit("ACTIVE")).otherwise(lit("INACTIVE")))
    .withColumn("amount", rand() * 50)
)

inspect_key_distribution(low_card_e10, "status", "Exercise 10 Diagnosis: Low-cardinality skew")

# Problem version
low_card_problem_e10 = (
    low_card_e10
    .groupBy("status")
    .agg(spark_sum("amount").alias("total_amount"))
)

print("\nProblem version plan: direct groupBy on low-cardinality column")
low_card_problem_e10.explain(True)

run_action(
    "Exercise 10 Problem: low-cardinality groupBy count",
    lambda: low_card_problem_e10.count()
)

# Solution: salted two-phase aggregation
low_card_salted_e10 = (
    low_card_e10
    .withColumn("salt", when(col("status") == "ACTIVE", pmod(col("id"), lit(10))).otherwise(lit(0)))
)

low_card_partial_e10 = (
    low_card_salted_e10
    .groupBy("status", "salt")
    .agg(spark_sum("amount").alias("partial_amount"))
)

low_card_solution_e10 = (
    low_card_partial_e10
    .groupBy("status")
    .agg(spark_sum("partial_amount").alias("total_amount"))
)

print("\nSolution plan: salted two-phase low-cardinality aggregation")
low_card_solution_e10.explain(True)

run_action(
    "Exercise 10 Solution: salted groupBy count",
    lambda: low_card_solution_e10.count()
)


# ============================================================
# EXERCISE 11: join() with null/default keys
# ============================================================
# Problem:
# Null/default keys like 0, UNKNOWN, blank can create one huge join partition.
#
# Regular way:
# df.join(dim, "customer_id")
#
# Skew-aware solution:
# Separate abnormal keys:
# - join valid keys normally
# - handle null/default keys separately based on business rule
# ============================================================

print_heading("EXERCISE 11: join() with null/default keys")

orders_e11 = (
    spark.range(0, 100000)
    .withColumn(
        "customer_id",
        when(col("id") < 80000, lit(0)).otherwise((col("id") % 1000).cast("int"))
    )
    .withColumn("amount", rand() * 100)
)

dim_e11 = (
    spark.range(1, 1000)
    .withColumnRenamed("id", "customer_id")
    .withColumn("segment", lit("KNOWN"))
)

inspect_key_distribution(orders_e11, "customer_id", "Exercise 11 Diagnosis: Default key skew")

# Problem version
join_default_problem_e11 = orders_e11.join(dim_e11, "customer_id", "left")

print("\nProblem version plan: join with default key included")
join_default_problem_e11.explain(True)

run_action(
    "Exercise 11 Problem: join with default key count",
    lambda: join_default_problem_e11.count()
)

# Solution: isolate default key
valid_orders_e11 = orders_e11.filter(col("customer_id") != 0)
default_orders_e11 = orders_e11.filter(col("customer_id") == 0)

valid_join_e11 = valid_orders_e11.join(dim_e11, "customer_id", "left")

# Business rule: mark default customers separately without joining.
default_handled_e11 = default_orders_e11.withColumn("segment", lit("UNKNOWN_OR_DEFAULT"))

join_default_solution_e11 = valid_join_e11.unionByName(default_handled_e11.select(valid_join_e11.columns))

print("\nSolution plan: isolate default key before join")
join_default_solution_e11.explain(True)

run_action(
    "Exercise 11 Solution: default-key isolated join count",
    lambda: join_default_solution_e11.count()
)


# ============================================================
# EXERCISE 12: repartitionByRange() on uneven ranges
# ============================================================
# Problem:
# If most records have the same/range-close value, range partitioning can
# create imbalanced partitions.
#
# Regular way:
# df.repartitionByRange(8, "score")
#
# Skew-aware solution:
# Add secondary column to split ties/ranges:
# df.repartitionByRange(8, "score", "id")
# ============================================================

print_heading("EXERCISE 12: repartitionByRange() on uneven ranges")

range_e12 = (
    spark.range(0, 100000)
    .withColumn("score", when(col("id") < 90000, lit(100)).otherwise((col("id") % 1000).cast("int")))
    .withColumn("amount", rand() * 100)
)

inspect_key_distribution(range_e12, "score", "Exercise 12 Diagnosis: Uneven range values")

# Problem version
range_problem_e12 = range_e12.repartitionByRange(8, "score")
inspect_partition_counts(range_problem_e12, "Exercise 12 Problem: repartitionByRange(8, 'score')")

# Solution: range partition by score + id to split huge equal score group
range_solution_e12 = range_e12.repartitionByRange(8, "score", "id")
inspect_partition_counts(range_solution_e12, "Exercise 12 Solution: repartitionByRange(8, 'score', 'id')")

print("""
Exercise 12 Important Note:
If most rows have the same score, score alone cannot distribute them well.
Adding a secondary column like id can split tie-heavy ranges.
""")


# ============================================================
# FINAL SUMMARY
# ============================================================

print_heading("FINAL SUMMARY")

print("""
Skew Diagnosis Tools Used:

1. groupBy(key).count().orderBy(desc("record_count"))
   -> identifies hot keys.

2. spark_partition_id()
   -> shows how records are distributed across partitions.

3. df.rdd.getNumPartitions()
   -> shows partition count.

4. df.explain(True)
   -> shows Exchange, join strategy, sort, aggregate, AQE plan.

5. Spark UI
   -> shows task duration imbalance, shuffle read/write, spill, GC.

General Skew Handling Patterns:

1. Broadcast small dimension table for skewed joins.
2. Enable AQE skew join handling.
3. Use salting for hot keys.
4. Use selective salting only for hot values.
5. Split hot/default/null keys and process them separately.
6. Use two-phase aggregation for skewed groupBy/pivot/rollup.
7. Avoid repartitioning by a skewed column directly.
8. Add secondary keys for tie-heavy sort/range partitioning.
9. Do not salt window functions unless business semantics allow it.
10. Always verify correctness after skew handling.
""")