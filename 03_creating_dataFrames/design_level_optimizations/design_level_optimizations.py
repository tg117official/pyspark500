# ============================================================
# PySpark Shell Demo:
# Design-Level Optimization - Partitioning, Bucketing,
# and Dynamic Partition Pruning
# ============================================================
#
# This script is written as student-facing notes + hands-on lab.
#
# Designed for PySpark shell:
# spark and sc are already available.
#
# Recommended PySpark shell start:
#
# pyspark --conf spark.sql.warehouse.dir=/tmp/spark_warehouse \
#         --conf spark.sql.shuffle.partitions=4
#
# Spark UI:
# http://localhost:4040
#
# Why this topic is important:
#
# Partitioning and bucketing are design-level optimizations.
# They are not just one-time runtime tricks.
# We design data layout based on query patterns.
#
# Partitioning mainly helps with filtering/pruning.
# Bucketing mainly helps with repeated joins on the same key.
# Dynamic Partition Pruning helps Spark skip partitions during joins.
# ============================================================


from pyspark.sql.functions import (
    col,
    lit,
    when,
    rand,
    sum as spark_sum,
    count,
    spark_partition_id
)

import os
import shutil


# ============================================================
# Helper Functions
# ============================================================

def print_heading(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def remove_path(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        print("Removed path:", path)


def list_dirs(path, max_items=100):
    print("\nListing:", path)

    if not os.path.exists(path):
        print("Path does not exist:", path)
        return

    items = sorted(os.listdir(path))

    for item in items[:max_items]:
        print(item)

    if len(items) > max_items:
        print("... more items:", len(items) - max_items)


def count_part_files(path):
    part_files = []

    if not os.path.exists(path):
        print("Path does not exist:", path)
        return []

    for root, dirs, files in os.walk(path):
        for f in files:
            if f.startswith("part-"):
                part_files.append(os.path.join(root, f))

    print("Part file count:", len(part_files))
    return part_files


def inspect_plan(df, title):
    print_heading(title)
    df.explain(True)


def inspect_partition_counts(df, title):
    print_heading(title)

    print("Total partitions:", df.rdd.getNumPartitions())

    (
        df
        .withColumn("partition_id", spark_partition_id())
        .groupBy("partition_id")
        .agg(count("*").alias("record_count"))
        .orderBy("partition_id")
        .show(100, truncate=False)
    )


# ============================================================
# Initial Configuration
# ============================================================

print_heading("INITIAL CONFIGURATION")

spark.conf.set("spark.sql.shuffle.partitions", "4")
spark.conf.set("spark.sql.sources.bucketing.enabled", "true")

# Dynamic Partition Pruning config.
# DPP is enabled by default in many Spark 3.x+ environments,
# but we explicitly set it for this demo.
spark.conf.set("spark.sql.optimizer.dynamicPartitionPruning.enabled", "true")

# Broadcast is useful for many DPP examples because Spark can collect
# filtering values from the small side of a join.
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "10MB")

print("Spark UI:", spark.sparkContext.uiWebUrl)
print("Spark Version:", spark.version)
print("Warehouse Directory:", spark.conf.get("spark.sql.warehouse.dir"))
print("Shuffle Partitions:", spark.conf.get("spark.sql.shuffle.partitions"))
print("Bucketing Enabled:", spark.conf.get("spark.sql.sources.bucketing.enabled"))
print("Dynamic Partition Pruning Enabled:",
      spark.conf.get("spark.sql.optimizer.dynamicPartitionPruning.enabled"))
print("Auto Broadcast Join Threshold:",
      spark.conf.get("spark.sql.autoBroadcastJoinThreshold"))

print("""
NOTES:

1. Partitioning:
   Partitioning creates folder-level data layout.
   Example:
   country=India/
   country=USA/

2. Bucketing:
   Bucketing creates a fixed number of bucket files based on hash of column.
   Bucketing metadata must be stored in a table/catalog.
   Therefore, we use saveAsTable() for bucketing.

3. Dynamic Partition Pruning:
   DPP is useful when a large fact table is partitioned and joined with
   a smaller filtered dimension table.
   Spark can use the dimension-side filter to avoid scanning irrelevant
   partitions of the fact table.
""")


# ============================================================
# EXERCISE 1: Create Sample Fact and Dimension Data
# ============================================================
# Purpose:
# We create data similar to a simple e-commerce model.
#
# Fact table:
# orders
#
# Dimension table:
# countries
#
# orders has:
# - order_id
# - customer_id
# - country_id
# - country
# - order_date
# - amount
#
# countries has:
# - country_id
# - country
# - region
#
# This will help demonstrate:
# - partitioning by country
# - partition pruning
# - dynamic partition pruning
# - bucketing by customer_id

print_heading("EXERCISE 1: Create Sample Fact and Dimension Data")

orders_df = (
    spark.range(0, 20000)
    .withColumnRenamed("id", "order_id")
    .withColumn("customer_id", (col("order_id") % 2000).cast("int"))
    .withColumn("country_id", (col("order_id") % 4).cast("int"))
    .withColumn(
        "country",
        when(col("country_id") == 0, lit("India"))
        .when(col("country_id") == 1, lit("USA"))
        .when(col("country_id") == 2, lit("UK"))
        .otherwise(lit("Germany"))
    )
    .withColumn(
        "order_date",
        when(col("order_id") % 3 == 0, lit("2026-06-01"))
        .when(col("order_id") % 3 == 1, lit("2026-06-02"))
        .otherwise(lit("2026-06-03"))
    )
    .withColumn("amount", (rand() * 1000).cast("double"))
    .repartition(4)
)

countries_df = spark.createDataFrame(
    [
        (0, "India", "APAC"),
        (1, "USA", "NA"),
        (2, "UK", "EU"),
        (3, "Germany", "EU"),
    ],
    ["country_id", "country", "region"]
)

orders_df.show(10, truncate=False)
countries_df.show(truncate=False)

print("orders_df partitions:", orders_df.rdd.getNumPartitions())


# ============================================================
# EXERCISE 2: Partitioning by country
# ============================================================
# Purpose:
# Demonstrate folder-level partitioning.
#
# Good partition columns:
# - Frequently used in filters
# - Low/medium cardinality
# - Business meaningful
#
# country is a good demo column because it has only 4 values.
#
# Output folder layout:
# country=India/
# country=USA/
# country=UK/
# country=Germany/

print_heading("EXERCISE 2: Write Orders Partitioned by country")

partition_path = "/tmp/design_opt_orders_partitioned_country"

remove_path(partition_path)

(
    orders_df
    .write
    .mode("overwrite")
    .partitionBy("country")
    .parquet(partition_path)
)

print("Partitioned path:", partition_path)
list_dirs(partition_path)

print("\nTotal part files:")
count_part_files(partition_path)

print("""
NOTES:

partitionBy("country") physically creates folders based on country values.
This helps when queries frequently filter by country.

Good:
filter country = 'India'

Not directly useful:
filter amount > 500
because amount is not a partition column.
""")


# ============================================================
# EXERCISE 3: Static Partition Pruning
# ============================================================
# Purpose:
# Demonstrate normal/static partition pruning.
#
# Query:
# Read only India orders.
#
# Since the data is partitioned by country, Spark can skip other folders.
#
# In physical plan, look for:
# PartitionFilters
#
# This is the most important proof.

print_heading("EXERCISE 3: Static Partition Pruning")

orders_partitioned_read_df = spark.read.parquet(partition_path)

india_orders_df = orders_partitioned_read_df.filter(col("country") == "India")

india_orders_df.show(10, truncate=False)

inspect_plan(
    india_orders_df,
    "Physical Plan: Static Partition Pruning country = India"
)

print("""
NOTES:

Static partition pruning happens when the filter is directly available
on the partition column.

Example:
country = 'India'

Spark can skip:
country=USA/
country=UK/
country=Germany/

and read only:
country=India/

Look for:
PartitionFilters
""")


# ============================================================
# EXERCISE 4: No Partition Pruning for Non-Partition Column
# ============================================================
# Purpose:
# Show that partitioning helps only when the filter uses partition column.
#
# Data is partitioned by country.
# But this query filters by amount.
#
# Spark cannot use country folder pruning for amount filter.
#
# In physical plan:
# - DataFilters may contain amount
# - PartitionFilters will not contain amount

print_heading("EXERCISE 4: No Partition Pruning for Non-Partition Column")

high_amount_df = orders_partitioned_read_df.filter(col("amount") > 900)

high_amount_df.show(10, truncate=False)

inspect_plan(
    high_amount_df,
    "Physical Plan: Filter on Non-Partition Column amount"
)

print("""
NOTES:

Partitioning by country does not help amount-only filters.

This is why partition column selection must be based on query patterns.
""")


# ============================================================
# EXERCISE 5: Bad Partitioning Choice - High Cardinality
# ============================================================
# Purpose:
# Demonstrate why high-cardinality partition columns are risky.
#
# customer_id has many values.
# If we partition by customer_id, Spark creates many folders.
#
# Problems:
# - Too many directories
# - Too many small files
# - Metadata overhead
# - Slow file listing
#
# We write a smaller dataset to avoid creating too many files.

print_heading("EXERCISE 5: Bad Partitioning Choice - customer_id")

bad_partition_path = "/tmp/design_opt_orders_partitioned_customer_id"

remove_path(bad_partition_path)

small_orders_df = orders_df.limit(1000)

(
    small_orders_df
    .write
    .mode("overwrite")
    .partitionBy("customer_id")
    .parquet(bad_partition_path)
)

customer_partition_dirs = [
    d for d in os.listdir(bad_partition_path)
    if d.startswith("customer_id=")
]

print("Number of customer_id folders:", len(customer_partition_dirs))
print("First 25 customer_id folders:")
for d in sorted(customer_partition_dirs)[:25]:
    print(d)

print("""
NOTES:

Avoid high-cardinality partition columns like:
customer_id, transaction_id, email, uuid.

They create too many folders and small files.
""")


# ============================================================
# EXERCISE 6: Multi-column Partitioning
# ============================================================
# Purpose:
# Demonstrate partitioning by more than one column.
#
# Example:
# partitionBy("country", "order_date")
#
# Folder layout:
# country=India/order_date=2026-06-01/
#
# Benefit:
# Good when queries filter by both country and order_date.
#
# Trade-off:
# More partition columns can multiply folder count.

print_heading("EXERCISE 6: Multi-column Partitioning")

multi_partition_path = "/tmp/design_opt_orders_partitioned_country_date"

remove_path(multi_partition_path)

(
    orders_df
    .write
    .mode("overwrite")
    .partitionBy("country", "order_date")
    .parquet(multi_partition_path)
)

print("Top-level partition folders:")
list_dirs(multi_partition_path)

print("\nTotal part files:")
count_part_files(multi_partition_path)

multi_read_df = spark.read.parquet(multi_partition_path)

india_date_df = (
    multi_read_df
    .filter(
        (col("country") == "India") &
        (col("order_date") == "2026-06-01")
    )
)

india_date_df.show(10, truncate=False)

inspect_plan(
    india_date_df,
    "Physical Plan: Partition Pruning on country and order_date"
)

print("""
NOTES:

Multi-column partitioning can help when filters include multiple
partition columns.

But be careful:
country has 4 values and order_date has 3 values.
Possible folder combinations = 4 * 3 = 12.

With many values, folder count can explode.
""")


# ============================================================
# EXERCISE 7: Bucketing Setup
# ============================================================
# Purpose:
# Bucketing needs table/catalog metadata.
#
# We use:
# saveAsTable()
#
# This stores table metadata and bucket information.
#
# If this fails, restart PySpark shell with:
#
# pyspark --conf spark.sql.warehouse.dir=/tmp/spark_warehouse
#
# In production environments:
# Hive metastore, Unity Catalog, Glue Catalog, or similar catalog
# usually stores table metadata.

print_heading("EXERCISE 7: Bucketing Setup")

spark.sql("CREATE DATABASE IF NOT EXISTS design_opt_demo")
spark.sql("USE design_opt_demo")

print("Current database:")
spark.sql("SELECT current_database()").show()

print("Existing tables:")
spark.sql("SHOW TABLES").show(truncate=False)

print("""
NOTES:

Partitioning can work with simple Parquet paths.

Bucketing should be demonstrated with tables because Spark needs bucket
metadata to understand bucket layout during query planning.
""")


# ============================================================
# EXERCISE 8: Create Bucketed Tables
# ============================================================
# Purpose:
# Demonstrate bucketing.
#
# We create:
# - orders_bucketed
# - customers_bucketed
#
# Both bucketed by customer_id into 8 buckets.
#
# Why same bucket key and bucket count?
# For bucketed join benefits, both tables should be bucketed on the join key
# with compatible bucket count.
#
# Concept:
# bucket_id = hash(customer_id) % number_of_buckets
#
# Use case:
# Repeated large joins on customer_id.

print_heading("EXERCISE 8: Create Bucketed Tables")

spark.sql("DROP TABLE IF EXISTS orders_bucketed")
spark.sql("DROP TABLE IF EXISTS customers_bucketed")

customers_df = (
    spark.range(0, 2000)
    .withColumnRenamed("id", "customer_id")
    .withColumn(
        "customer_type",
        when(col("customer_id") % 2 == 0, lit("Premium"))
        .otherwise(lit("Regular"))
    )
)

(
    orders_df
    .write
    .mode("overwrite")
    .format("parquet")
    .bucketBy(8, "customer_id")
    .sortBy("customer_id")
    .saveAsTable("orders_bucketed")
)

(
    customers_df
    .write
    .mode("overwrite")
    .format("parquet")
    .bucketBy(8, "customer_id")
    .sortBy("customer_id")
    .saveAsTable("customers_bucketed")
)

print("Created bucketed tables.")

spark.sql("SHOW TABLES").show(truncate=False)

print("DESCRIBE EXTENDED orders_bucketed")
spark.sql("DESCRIBE EXTENDED orders_bucketed").show(100, truncate=False)

print("""
NOTES:

Bucketing does not create folders like country=India.
It creates bucketed files and stores bucket metadata in table definition.

Use bucketed tables when large tables are frequently joined on the same key.
""")


# ============================================================
# EXERCISE 9: Bucketed Join
# ============================================================
# Purpose:
# Demonstrate join between bucketed tables.
#
# Both tables:
# - bucketed by customer_id
# - same number of buckets
#
# Expected benefit:
# Spark may reduce or avoid shuffle because data is already organized
# by the join key.
#
# To make shuffle behavior visible:
# - Disable broadcast
# - Disable AQE
# - Enable bucketing

print_heading("EXERCISE 9: Bucketed Join")

spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")
spark.conf.set("spark.sql.adaptive.enabled", "false")
spark.conf.set("spark.sql.sources.bucketing.enabled", "true")

bucketed_orders = spark.table("orders_bucketed")
bucketed_customers = spark.table("customers_bucketed")

bucketed_join_df = (
    bucketed_orders
    .join(bucketed_customers, on="customer_id", how="inner")
    .select("order_id", "customer_id", "customer_type", "amount")
)

bucketed_join_df.show(10, truncate=False)

inspect_plan(
    bucketed_join_df,
    "Physical Plan: Bucketed Join"
)

print("""
NOTES:

In the plan, compare Exchange operators.

If Spark uses bucket metadata effectively, it may avoid shuffle on one
or both sides of the join.

This depends on Spark version, table metadata, configuration, and query plan.

Always verify using explain(True).
""")


# ============================================================
# EXERCISE 10: Non-Bucketed Join Comparison
# ============================================================
# Purpose:
# Compare bucketed join with normal non-bucketed DataFrame join.
#
# Normal DataFrames are not saved with bucket metadata.
# Spark usually needs Exchange hashpartitioning before sort merge join.

print_heading("EXERCISE 10: Non-Bucketed Join Comparison")

spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")
spark.conf.set("spark.sql.adaptive.enabled", "false")
spark.conf.set("spark.sql.shuffle.partitions", "8")

normal_join_df = (
    orders_df
    .join(customers_df, on="customer_id", how="inner")
    .select("order_id", "customer_id", "customer_type", "amount")
)

normal_join_df.show(10, truncate=False)

inspect_plan(
    normal_join_df,
    "Physical Plan: Non-Bucketed Join"
)

print("""
NOTES:

Normal non-bucketed joins usually require Exchange hashpartitioning.

Compare this plan with the bucketed join plan.
""")


# ============================================================
# EXERCISE 11: Dynamic Partition Pruning Setup
# ============================================================
# Purpose:
# Create a partitioned fact table and a dimension table.
#
# Fact table:
# orders_dpp_fact
# Partitioned by country_id
#
# Dimension table:
# countries_dpp_dim
#
# Query:
# Join fact with dimension.
# Filter dimension on region = 'EU'.
#
# Since region = 'EU' maps to country_id 2 and 3,
# Spark can use the dimension-side filter to prune fact partitions:
#
# country_id=2
# country_id=3
#
# This is Dynamic Partition Pruning.
#
# Why dynamic?
# The filter is not directly written on fact.country_id.
# It is discovered through the join with filtered dimension table.

print_heading("EXERCISE 11: Dynamic Partition Pruning Setup")

dpp_fact_path = "/tmp/design_opt_dpp_orders_fact"

remove_path(dpp_fact_path)

# Write fact table partitioned by country_id.
# Note:
# We keep country_id as partition column.
(
    orders_df
    .write
    .mode("overwrite")
    .partitionBy("country_id")
    .parquet(dpp_fact_path)
)

countries_dpp_df = countries_df

orders_dpp_fact = spark.read.parquet(dpp_fact_path)
countries_dpp_df.createOrReplaceTempView("countries_dpp_dim")
orders_dpp_fact.createOrReplaceTempView("orders_dpp_fact")

print("Fact table partition folders:")
list_dirs(dpp_fact_path)

orders_dpp_fact.show(10, truncate=False)
countries_dpp_df.show(truncate=False)

print("""
NOTES:

Fact table is partitioned by country_id.

Dimension table has:
country_id, country, region

We will filter dimension by region = 'EU'.

Spark can infer:
EU countries are country_id 2 and 3.

Then Spark can dynamically prune fact partitions:
country_id=0 and country_id=1 can be skipped.
""")


# ============================================================
# EXERCISE 12: Dynamic Partition Pruning Enabled
# ============================================================
# Purpose:
# Demonstrate DPP.
#
# DPP is enabled.
#
# Query:
# SELECT orders joined with countries
# WHERE countries.region = 'EU'
#
# Notice:
# We are NOT directly filtering orders.country_id.
#
# Still, Spark can use dimension filter to prune partitions of orders.
#
# In physical plan, look for:
# dynamicpruning
# dynamicpruningexpression
# SubqueryBroadcast
# PartitionFilters containing dynamic pruning expression
#
# Exact wording can vary by Spark version.

print_heading("EXERCISE 12: Dynamic Partition Pruning Enabled")

spark.conf.set("spark.sql.optimizer.dynamicPartitionPruning.enabled", "true")
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "10MB")
spark.conf.set("spark.sql.adaptive.enabled", "true")

print("DPP Enabled:",
      spark.conf.get("spark.sql.optimizer.dynamicPartitionPruning.enabled"))
print("AQE Enabled:", spark.conf.get("spark.sql.adaptive.enabled"))
print("Broadcast Threshold:",
      spark.conf.get("spark.sql.autoBroadcastJoinThreshold"))

dpp_enabled_df = spark.sql("""
    SELECT
        o.order_id,
        o.country_id,
        o.country,
        c.region,
        o.amount
    FROM orders_dpp_fact o
    INNER JOIN countries_dpp_dim c
        ON o.country_id = c.country_id
    WHERE c.region = 'EU'
""")

dpp_enabled_df.show(10, truncate=False)

inspect_plan(
    dpp_enabled_df,
    "Physical Plan: Dynamic Partition Pruning Enabled"
)

print("""
NOTES:

This is the key idea of DPP:

The fact table is partitioned by country_id.
The filter is on the dimension table:
c.region = 'EU'

Spark can use the filtered dimension table to discover relevant country_ids.
Then it can prune fact partitions dynamically.

Look in the physical plan for:
- dynamicpruning
- dynamicpruningexpression
- SubqueryBroadcast
- PartitionFilters

This proves Spark is trying to avoid unnecessary fact partition scans.
""")


# ============================================================
# EXERCISE 13: Dynamic Partition Pruning Disabled
# ============================================================
# Purpose:
# Run the same query with DPP disabled.
#
# Expected:
# Spark should not add dynamic partition pruning expression.
#
# Compare physical plan with Exercise 12.

print_heading("EXERCISE 13: Dynamic Partition Pruning Disabled")

spark.conf.set("spark.sql.optimizer.dynamicPartitionPruning.enabled", "false")

print("DPP Enabled:",
      spark.conf.get("spark.sql.optimizer.dynamicPartitionPruning.enabled"))

dpp_disabled_df = spark.sql("""
    SELECT
        o.order_id,
        o.country_id,
        o.country,
        c.region,
        o.amount
    FROM orders_dpp_fact o
    INNER JOIN countries_dpp_dim c
        ON o.country_id = c.country_id
    WHERE c.region = 'EU'
""")

dpp_disabled_df.show(10, truncate=False)

inspect_plan(
    dpp_disabled_df,
    "Physical Plan: Dynamic Partition Pruning Disabled"
)

print("""
NOTES:

Compare this plan with the DPP-enabled plan.

With DPP disabled:
Spark should not add the dynamic pruning predicate.

This means Spark may scan more fact partitions.
""")


# ============================================================
# EXERCISE 14: Static Pruning vs Dynamic Pruning
# ============================================================
# Purpose:
# Understand the difference.
#
# Static pruning:
# Filter is directly on partition column.
#
# Example:
# WHERE o.country_id IN (2, 3)
#
# Dynamic pruning:
# Filter comes indirectly through joined dimension table.
#
# Example:
# WHERE c.region = 'EU'
# and join condition o.country_id = c.country_id

print_heading("EXERCISE 14: Static Pruning vs Dynamic Pruning")

spark.conf.set("spark.sql.optimizer.dynamicPartitionPruning.enabled", "true")

static_pruning_df = spark.sql("""
    SELECT
        o.order_id,
        o.country_id,
        o.country,
        o.amount
    FROM orders_dpp_fact o
    WHERE o.country_id IN (2, 3)
""")

static_pruning_df.show(10, truncate=False)

inspect_plan(
    static_pruning_df,
    "Physical Plan: Static Partition Pruning"
)

print("""
NOTES:

Static pruning:
The filter is directly on the partition column:
o.country_id IN (2, 3)

Dynamic pruning:
The filter comes from another joined table:
c.region = 'EU'

Both reduce partition scanning, but dynamic pruning is more intelligent
because Spark derives the partition filter through the join.
""")


# ============================================================
# EXERCISE 15: Case Where DPP May Not Help Much
# ============================================================
# Purpose:
# DPP is most useful when dimension-side filter is selective.
#
# If filter selects almost all dimension rows, then most fact partitions
# are still needed.
#
# Example:
# region IN ('EU', 'APAC', 'NA')
#
# This selects all countries in our tiny dimension table.
#
# DPP may exist in the plan, but benefit is low because almost all
# partitions are required.

print_heading("EXERCISE 15: Case Where DPP May Not Help Much")

dpp_less_helpful_df = spark.sql("""
    SELECT
        o.order_id,
        o.country_id,
        o.country,
        c.region,
        o.amount
    FROM orders_dpp_fact o
    INNER JOIN countries_dpp_dim c
        ON o.country_id = c.country_id
    WHERE c.region IN ('EU', 'APAC', 'NA')
""")

dpp_less_helpful_df.show(10, truncate=False)

inspect_plan(
    dpp_less_helpful_df,
    "Physical Plan: DPP Less Helpful When Filter Is Not Selective"
)

print("""
NOTES:

DPP helps most when the dimension-side filter is selective.

Good case:
region = 'EU'
Only country_id 2 and 3 are needed.

Less useful case:
region IN ('EU', 'APAC', 'NA')
All country_ids are needed.

DPP may still appear, but the pruning benefit is low.
""")


# ============================================================
# EXERCISE 16: Case Where DPP Cannot Help
# ============================================================
# Purpose:
# DPP requires that the large fact side be partitioned by the join key
# or a key that can be dynamically filtered.
#
# If fact data is not partitioned by the join key, Spark cannot skip
# folders using that key.
#
# Here we create a non-partitioned fact path and run the same join.
#
# Expected:
# No useful partition pruning on country_id folders because no such
# folders exist.

print_heading("EXERCISE 16: Case Where DPP Cannot Help - Non-Partitioned Fact")

non_partition_fact_path = "/tmp/design_opt_dpp_orders_fact_non_partitioned"

remove_path(non_partition_fact_path)

(
    orders_df
    .write
    .mode("overwrite")
    .parquet(non_partition_fact_path)
)

orders_non_partitioned_fact = spark.read.parquet(non_partition_fact_path)
orders_non_partitioned_fact.createOrReplaceTempView("orders_non_partitioned_fact")

dpp_cannot_help_df = spark.sql("""
    SELECT
        o.order_id,
        o.country_id,
        o.country,
        c.region,
        o.amount
    FROM orders_non_partitioned_fact o
    INNER JOIN countries_dpp_dim c
        ON o.country_id = c.country_id
    WHERE c.region = 'EU'
""")

dpp_cannot_help_df.show(10, truncate=False)

inspect_plan(
    dpp_cannot_help_df,
    "Physical Plan: DPP Cannot Prune Non-Partitioned Fact"
)

print("""
NOTES:

DPP is mainly useful when Spark can skip partitions/folders.

If the fact table is not partitioned by country_id,
there are no country_id folders to skip.

So DPP cannot provide partition-pruning benefit.
""")


# ============================================================
# EXERCISE 17: Combined Partitioning and Bucketing
# ============================================================
# Purpose:
# Demonstrate that partitioning and bucketing can be combined.
#
# Example:
# Partition by country
# Bucket by customer_id
#
# Use case:
# - Queries filter by country
# - Joins frequently happen on customer_id
#
# Warning:
# This is more complex and should be used only when query patterns justify it.

print_heading("EXERCISE 17: Combined Partitioning + Bucketing")

spark.sql("DROP TABLE IF EXISTS orders_partitioned_bucketed")

(
    orders_df
    .write
    .mode("overwrite")
    .format("parquet")
    .partitionBy("country")
    .bucketBy(8, "customer_id")
    .sortBy("customer_id")
    .saveAsTable("orders_partitioned_bucketed")
)

combined_df = spark.table("orders_partitioned_bucketed").filter(col("country") == "India")

combined_df.show(10, truncate=False)

inspect_plan(
    combined_df,
    "Physical Plan: Partitioned + Bucketed Table Filter"
)

print("DESCRIBE EXTENDED orders_partitioned_bucketed")
spark.sql("DESCRIBE EXTENDED orders_partitioned_bucketed").show(100, truncate=False)

print("""
NOTES:

Partitioning + Bucketing together:

Partitioning:
country folders help filter/pruning.

Bucketing:
customer_id buckets can help repeated joins on customer_id.

Use this only when both access patterns are important.
""")


# ============================================================
# FINAL SUMMARY
# ============================================================

print_heading("FINAL SUMMARY")

print("""
DESIGN-LEVEL OPTIMIZATION SUMMARY

Partitioning:

1. Partitioning creates folder-level layout.
2. Best for frequently filtered low/medium-cardinality columns.
3. Example good columns:
   date, country, region, status, order_date.
4. Avoid high-cardinality partition columns:
   customer_id, email, transaction_id, uuid.
5. Main benefit:
   partition pruning.
6. Proof in physical plan:
   PartitionFilters.

Bucketing:

1. Bucketing creates a fixed number of bucket files using hash of column.
2. Best for repeated large joins on the same key.
3. Requires table metadata, so use saveAsTable().
4. Both large join tables should ideally use same bucket key and compatible
   bucket count.
5. Main benefit:
   reduced shuffle during joins.
6. Proof:
   compare Exchange operators in explain(True).

Dynamic Partition Pruning:

1. DPP is used in joins.
2. Large fact table is partitioned by join key.
3. Small dimension table has selective filter.
4. Spark uses filtered dimension values to prune fact partitions.
5. Example:
   Fact partitioned by country_id.
   Dimension filtered by region = 'EU'.
   Spark scans only required country_id partitions.
6. Proof in plan:
   dynamicpruning / dynamicpruningexpression / SubqueryBroadcast /
   PartitionFilters.
7. DPP helps most when dimension filter is selective.
8. DPP cannot help much if:
   - fact table is not partitioned by the join key
   - dimension filter selects almost all values
   - join/filter pattern does not allow partition pruning.

One-line understanding:

Partitioning reduces scan during filters.
Bucketing reduces shuffle during repeated joins.
Dynamic Partition Pruning reduces fact-table scan during joins using
runtime filters from the dimension side.
""")