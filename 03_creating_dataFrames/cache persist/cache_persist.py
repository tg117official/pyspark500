# ============================================================
# PySpark Independent Exercises: Cache, Persist, Storage Levels
# ============================================================
#
# Objective:
# This script contains independent hands-on exercises for:
#
# 1. Lazy nature of cache()
# 2. cache() vs repeated computation
# 3. persist() with MEMORY_AND_DISK
# 4. persist() with DISK_ONLY
# 5. persist() with MEMORY_ONLY
# 6. Changing storage level correctly
# 7. unpersist()
# 8. Proving cache usage using explain(True)
# 9. Serialized storage levels
# 10. Fractional caching concept
# 11. When not to cache
#
# Important:
# Each exercise has its own setup.
# Each exercise creates its own DataFrame.
# Each exercise performs its own cleanup using unpersist().
#
# This makes every exercise independent.
#
# Keep Spark UI open while running this script.
#
# In local mode, Spark UI usually opens at:
# http://localhost:4040
#
# Go to the "Storage" tab to observe cached/persisted data.
# ============================================================


from pyspark.sql import SparkSession
from pyspark.sql.functions import col, rand, sum as spark_sum, avg
from pyspark.storagelevel import StorageLevel
import time


# ------------------------------------------------------------
# Create SparkSession
# ------------------------------------------------------------
# SparkSession is created once for the complete script.
# Individual exercises are independent at DataFrame/cache level.
# We do not stop SparkSession after every exercise because starting
# Spark again and again will make the demo slow.

spark = (
    SparkSession.builder
    .appName("IndependentCachePersistExercises")
    .master("local[*]")
    .config("spark.sql.shuffle.partitions", "4")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

print("\nSparkSession created successfully")
print("Spark Version:", spark.version)
print("Spark UI:", spark.sparkContext.uiWebUrl)


# ------------------------------------------------------------
# Utility function to measure execution time
# ------------------------------------------------------------

def measure_time(label, action_func):
    start_time = time.time()
    result = action_func()
    end_time = time.time()
    print(f"\n{label}")
    print(f"Result: {result}")
    print(f"Time Taken: {round(end_time - start_time, 2)} seconds")
    return result


# ------------------------------------------------------------
# Utility function to print section heading
# ------------------------------------------------------------

def print_heading(title):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


# ============================================================
# EXERCISE 1: cache() is Lazy
# ============================================================
# Purpose:
# This exercise proves that cache() does not immediately store data.
#
# Important idea:
# cache() only marks a DataFrame for caching.
# Actual caching happens when the first action runs.
#
# Steps:
# 1. Create a DataFrame.
# 2. Call cache().
# 3. Check storage level.
# 4. Run count() to materialize the cache.
# 5. Check Spark UI Storage tab.

print_heading("EXERCISE 1: cache() is Lazy")

ex1_df = (
    spark.range(0, 2_000_000)
    .withColumn("random_value", rand())
    .withColumn("amount", col("random_value") * 1000)
    .filter(col("id") % 2 == 0)
)

print("\nDataFrame created.")
print("No action has been executed yet.")

print("\nStorage level before cache():")
print(ex1_df.storageLevel)

ex1_cached_df = ex1_df.cache()

print("\ncache() called.")
print("Data is marked for caching, but it is not materialized yet.")

print("\nStorage level after cache():")
print(ex1_cached_df.storageLevel)

measure_time(
    "First action after cache(): count()",
    lambda: ex1_cached_df.count()
)

print("""
Observation:
After count(), the cache is materialized.
Now check Spark UI Storage tab.
You should see cached data there.
""")

ex1_cached_df.unpersist()

print("\nExercise 1 cleanup completed using unpersist().")


# ============================================================
# EXERCISE 2: Without Cache vs With Cache
# ============================================================
# Purpose:
# This exercise compares repeated actions without cache and with cache.
#
# Important idea:
# Without cache, Spark may recompute the lineage for each action.
# With cache, Spark can reuse the cached result after the first action.

print_heading("EXERCISE 2: Without Cache vs With Cache")

ex2_df = (
    spark.range(0, 5_000_000)
    .withColumn("random_value", rand())
    .withColumn("amount", col("random_value") * 1000)
    .filter(col("id") % 3 == 0)
    .withColumn("final_amount", col("amount") * 0.9)
    .repartition(4)
)

print("\nRunning repeated actions WITHOUT cache")

measure_time(
    "Without cache - first count()",
    lambda: ex2_df.count()
)

measure_time(
    "Without cache - second count()",
    lambda: ex2_df.count()
)

print("\nNow applying cache()")

ex2_cached_df = ex2_df.cache()

measure_time(
    "With cache - first count() materializes cache",
    lambda: ex2_cached_df.count()
)

measure_time(
    "With cache - second count() reuses cache",
    lambda: ex2_cached_df.count()
)

print("""
Observation:
The first action after cache() builds the cache.
The second action can reuse cached data.
""")

ex2_cached_df.unpersist()

print("\nExercise 2 cleanup completed using unpersist().")


# ============================================================
# EXERCISE 3: cache() Default Storage Level
# ============================================================
# Purpose:
# This exercise shows the default storage level used by cache().
#
# Important idea:
# For PySpark DataFrame, cache() uses the default DataFrame storage level.
# Commonly, it appears as:
#
# Disk Memory Deserialized 1x Replicated
#
# This means Spark can store cached data in memory and disk,
# in deserialized form.

print_heading("EXERCISE 3: cache() Default Storage Level")

ex3_df = (
    spark.range(0, 1_000_000)
    .withColumn("amount", col("id") * 10)
    .filter(col("amount") > 1000)
)

print("\nStorage level before cache():")
print(ex3_df.storageLevel)

ex3_cached_df = ex3_df.cache()

print("\nStorage level after cache():")
print(ex3_cached_df.storageLevel)

measure_time(
    "Materializing cache using count()",
    lambda: ex3_cached_df.count()
)

print("""
Observation:
Check the storage level printed above.
Also check Spark UI Storage tab.
""")

ex3_cached_df.unpersist()

print("\nExercise 3 cleanup completed using unpersist().")


# ============================================================
# EXERCISE 4: persist() with MEMORY_AND_DISK
# ============================================================
# Purpose:
# This exercise demonstrates persist(StorageLevel.MEMORY_AND_DISK).
#
# Important idea:
# MEMORY_AND_DISK means:
# 1. Try to store data in memory.
# 2. If memory is not enough, store remaining partitions on disk.
#
# This is safer than MEMORY_ONLY for large DataFrames.

print_heading("EXERCISE 4: persist() with MEMORY_AND_DISK")

ex4_df = (
    spark.range(0, 3_000_000)
    .withColumn("random_value", rand())
    .withColumn("amount", col("random_value") * 5000)
    .filter(col("id") % 5 == 0)
    .repartition(4)
)

ex4_persisted_df = ex4_df.persist(StorageLevel.MEMORY_AND_DISK)

print("\nStorage level after persist(StorageLevel.MEMORY_AND_DISK):")
print(ex4_persisted_df.storageLevel)

measure_time(
    "MEMORY_AND_DISK - first count() materializes persisted data",
    lambda: ex4_persisted_df.count()
)

measure_time(
    "MEMORY_AND_DISK - second count() reuses persisted data",
    lambda: ex4_persisted_df.count()
)

print("""
Observation:
Open Spark UI Storage tab.
Depending on available memory, data may be visible in memory and/or disk.
""")

ex4_persisted_df.unpersist()

print("\nExercise 4 cleanup completed using unpersist().")


# ============================================================
# EXERCISE 5: persist() with DISK_ONLY
# ============================================================
# Purpose:
# This exercise demonstrates persist(StorageLevel.DISK_ONLY).
#
# Important idea:
# DISK_ONLY stores persisted partitions only on disk.
#
# Benefit:
# Saves executor memory.
#
# Cost:
# Reading from disk is slower than reading from memory.
#
# Use case:
# Data is too large for memory, but recomputing lineage is more expensive.

print_heading("EXERCISE 5: persist() with DISK_ONLY")

ex5_df = (
    spark.range(0, 3_000_000)
    .withColumn("random_value", rand())
    .withColumn("amount", col("random_value") * 1000)
    .filter(col("id") % 7 == 0)
    .repartition(4)
)

ex5_persisted_df = ex5_df.persist(StorageLevel.DISK_ONLY)

print("\nStorage level after persist(StorageLevel.DISK_ONLY):")
print(ex5_persisted_df.storageLevel)

measure_time(
    "DISK_ONLY - first count() materializes persisted data",
    lambda: ex5_persisted_df.count()
)

measure_time(
    "DISK_ONLY - second count() reuses disk persisted data",
    lambda: ex5_persisted_df.count()
)

print("""
Observation:
Open Spark UI Storage tab.
The data should be stored on disk.
""")

ex5_persisted_df.unpersist()

print("\nExercise 5 cleanup completed using unpersist().")


# ============================================================
# EXERCISE 6: persist() with MEMORY_ONLY
# ============================================================
# Purpose:
# This exercise demonstrates persist(StorageLevel.MEMORY_ONLY).
#
# Important idea:
# MEMORY_ONLY stores partitions only in memory.
#
# If all partitions do not fit in memory:
# - Some partitions may not be cached.
# - Missing partitions can be recomputed from lineage when needed.
#
# Use case:
# Data fits comfortably in memory and fast reuse is needed.

print_heading("EXERCISE 6: persist() with MEMORY_ONLY")

ex6_df = (
    spark.range(0, 3_000_000)
    .withColumn("random_value", rand())
    .withColumn("amount", col("random_value") * 1000)
    .filter(col("id") % 4 == 0)
    .repartition(4)
)

ex6_persisted_df = ex6_df.persist(StorageLevel.MEMORY_ONLY)

print("\nStorage level after persist(StorageLevel.MEMORY_ONLY):")
print(ex6_persisted_df.storageLevel)

measure_time(
    "MEMORY_ONLY - first count() materializes persisted data",
    lambda: ex6_persisted_df.count()
)

measure_time(
    "MEMORY_ONLY - second count() reuses memory persisted data",
    lambda: ex6_persisted_df.count()
)

print("""
Observation:
Open Spark UI Storage tab.

If memory is enough:
- All partitions may be stored in memory.

If memory is not enough:
- Some partitions may not be cached.
- Missing partitions may be recomputed.
""")

ex6_persisted_df.unpersist()

print("\nExercise 6 cleanup completed using unpersist().")


# ============================================================
# EXERCISE 7: Correct Way to Change Storage Level
# ============================================================
# Purpose:
# This exercise shows that storage level should not be changed directly
# on an already persisted DataFrame.
#
# Correct process:
# 1. Persist DataFrame with one storage level.
# 2. Materialize it using an action.
# 3. unpersist().
# 4. Persist again with the new storage level.
# 5. Materialize again using an action.

print_heading("EXERCISE 7: Correct Way to Change Storage Level")

ex7_df = (
    spark.range(0, 1_000_000)
    .withColumn("amount", col("id") * 5)
    .filter(col("id") % 2 == 0)
)

ex7_persisted_df = ex7_df.persist(StorageLevel.MEMORY_AND_DISK)
ex7_persisted_df.count()

print("\nInitial storage level:")
print(ex7_persisted_df.storageLevel)

print("\nTrying to persist the same DataFrame with DISK_ONLY without unpersisting:")

ex7_persisted_df.persist(StorageLevel.DISK_ONLY)

print("\nStorage level after trying direct change:")
print(ex7_persisted_df.storageLevel)

print("""
Observation:
The storage level does not change directly because the DataFrame
is already persisted.

Now we will unpersist and persist again with DISK_ONLY.
""")

ex7_persisted_df.unpersist()

ex7_persisted_df = ex7_df.persist(StorageLevel.DISK_ONLY)
ex7_persisted_df.count()

print("\nStorage level after unpersist() and persist(StorageLevel.DISK_ONLY):")
print(ex7_persisted_df.storageLevel)

ex7_persisted_df.unpersist()

print("\nExercise 7 cleanup completed using unpersist().")


# ============================================================
# EXERCISE 8: unpersist()
# ============================================================
# Purpose:
# This exercise focuses only on unpersist().
#
# Important idea:
# Cached/persisted data uses executor memory and/or disk.
# If the data is no longer needed, remove it using unpersist().
#
# unpersist() removes cached blocks from memory/disk.

print_heading("EXERCISE 8: unpersist()")

ex8_df = (
    spark.range(0, 2_000_000)
    .withColumn("amount", col("id") * 2)
    .filter(col("amount") > 100)
)

ex8_cached_df = ex8_df.cache()

print("\nStorage level after cache():")
print(ex8_cached_df.storageLevel)

measure_time(
    "Materializing cache using count()",
    lambda: ex8_cached_df.count()
)

print("""
Now check Spark UI Storage tab.
The DataFrame should be visible there.
""")

ex8_cached_df.unpersist()

print("\nunpersist() called.")

print("""
Now check Spark UI Storage tab again.
The cached DataFrame should be removed.
""")

print("\nStorage level after unpersist():")
print(ex8_cached_df.storageLevel)

print("\nExercise 8 cleanup completed.")


# ============================================================
# EXERCISE 9: Prove Cache Usage using explain(True)
# ============================================================
# Purpose:
# This exercise shows how to prove that Spark is reading from cache.
#
# Important idea:
# After cache is materialized, explain(True) may show:
#
# InMemoryTableScan
# InMemoryRelation
#
# These indicate that Spark is using cached data.

print_heading("EXERCISE 9: Prove Cache Usage using explain(True)")

ex9_df = (
    spark.range(0, 2_000_000)
    .withColumn("amount", col("id") * 10)
    .withColumn("discounted_amount", col("amount") * 0.9)
    .filter(col("id") % 3 == 0)
)

ex9_cached_df = ex9_df.cache()

measure_time(
    "Materializing cache using count()",
    lambda: ex9_cached_df.count()
)

ex9_result_df = (
    ex9_cached_df
    .filter(col("amount") > 10000)
    .select("id", "amount", "discounted_amount")
)

print("\nExecution plan after cache materialization:")
ex9_result_df.explain(True)

print("""
Observation:
Look for:

InMemoryTableScan
InMemoryRelation

These terms show that Spark is reading from cached data.
""")

ex9_cached_df.unpersist()

print("\nExercise 9 cleanup completed using unpersist().")


# ============================================================
# EXERCISE 10: Serialized Storage Level
# ============================================================
# Purpose:
# This exercise demonstrates serialized storage.
#
# Important idea:
# Serialized storage stores data in compact byte format.
#
# Benefit:
# - Lower memory/storage usage.
#
# Cost:
# - Extra CPU is required for serialization/deserialization.
#
# Important:
# Serialization is not encryption.
# It is mainly for compact storage and transfer efficiency.

print_heading("EXERCISE 10: Serialized Storage Level")

ex10_df = (
    spark.range(0, 3_000_000)
    .withColumn("random_value", rand())
    .withColumn("amount", col("random_value") * 1000)
    .filter(col("id") % 2 == 0)
    .repartition(4)
)

ex10_ser_df = ex10_df.persist(StorageLevel.MEMORY_AND_DISK_SER)

print("\nStorage level after persist(StorageLevel.MEMORY_AND_DISK_SER):")
print(ex10_ser_df.storageLevel)

measure_time(
    "MEMORY_AND_DISK_SER - first count() materializes persisted data",
    lambda: ex10_ser_df.count()
)

measure_time(
    "MEMORY_AND_DISK_SER - second count() reuses persisted data",
    lambda: ex10_ser_df.count()
)

print("""
Observation:
Open Spark UI Storage tab and compare memory/storage usage.

Serialized storage may use less memory/storage but may require extra CPU
because Spark has to deserialize data before processing.
""")

ex10_ser_df.unpersist()

print("\nExercise 10 cleanup completed using unpersist().")


# ============================================================
# EXERCISE 11: Fractional Caching Concept
# ============================================================
# Purpose:
# This exercise explains partition-level caching.
#
# Important idea:
# Spark caches data at partition level.
#
# A partition is either cached or not cached.
# Spark does not cache half of a single partition.
#
# If memory is limited:
# - Some partitions may be cached.
# - Some partitions may not be cached.
# - Missing partitions can be recomputed from lineage.

print_heading("EXERCISE 11: Fractional Caching Concept")

ex11_df = (
    spark.range(0, 4_000_000)
    .withColumn("random_value", rand())
    .withColumn("amount", col("random_value") * 1000)
    .filter(col("id") % 2 == 0)
    .repartition(8)
)

print("\nNumber of partitions:")
print(ex11_df.rdd.getNumPartitions())

ex11_cached_df = ex11_df.persist(StorageLevel.MEMORY_ONLY)

measure_time(
    "Materializing MEMORY_ONLY cache using count()",
    lambda: ex11_cached_df.count()
)

print("""
Observation:
Open Spark UI Storage tab.

Check:
- Number of cached partitions
- Memory used
- Disk used

If all partitions fit in memory, all may be cached.
If memory is not enough, only some partitions may be cached.
""")

ex11_cached_df.unpersist()

print("\nExercise 11 cleanup completed using unpersist().")


# ============================================================
# EXERCISE 12: Cache with Aggregation Reuse
# ============================================================
# Purpose:
# This exercise shows a realistic case where cache is useful.
#
# Scenario:
# We create a cleaned DataFrame once.
# Then we use it for multiple aggregations.
#
# Without cache:
# Spark may recompute the cleaned DataFrame for each aggregation.
#
# With cache:
# Spark can reuse the cleaned DataFrame.

print_heading("EXERCISE 12: Cache with Aggregation Reuse")

ex12_raw_df = (
    spark.range(0, 5_000_000)
    .withColumn("customer_id", (col("id") % 1000))
    .withColumn("amount", rand() * 1000)
    .withColumn("valid_flag", col("id") % 2)
)

ex12_clean_df = (
    ex12_raw_df
    .filter(col("valid_flag") == 1)
    .withColumn("final_amount", col("amount") * 0.9)
    .select("customer_id", "final_amount")
    .repartition(4)
)

ex12_cached_df = ex12_clean_df.cache()

measure_time(
    "Materializing cleaned DataFrame cache",
    lambda: ex12_cached_df.count()
)

measure_time(
    "Aggregation 1: Total final amount",
    lambda: ex12_cached_df.agg(spark_sum("final_amount")).collect()[0][0]
)

measure_time(
    "Aggregation 2: Average final amount",
    lambda: ex12_cached_df.agg(avg("final_amount")).collect()[0][0]
)

measure_time(
    "Aggregation 3: Customer count",
    lambda: ex12_cached_df.select("customer_id").distinct().count()
)

print("""
Observation:
This is a common production-style use case.

A cleaned/reusable DataFrame is cached once and used for multiple downstream
actions or aggregations.
""")

ex12_cached_df.unpersist()

print("\nExercise 12 cleanup completed using unpersist().")


# ============================================================
# EXERCISE 13: When Not to Cache
# ============================================================
# Purpose:
# This exercise explains that caching is not always beneficial.
#
# Important idea:
# If a DataFrame is used only once, caching can add unnecessary overhead.
#
# Cache should usually be considered when:
# - The DataFrame is reused multiple times.
# - The DataFrame is expensive to recompute.
# - The source is slow.
# - The DataFrame is created after heavy transformations or joins.

print_heading("EXERCISE 13: When Not to Cache")

ex13_df = (
    spark.range(0, 2_000_000)
    .withColumn("amount", col("id") * 10)
    .filter(col("amount") > 1000)
)

measure_time(
    "Single-use DataFrame action without cache",
    lambda: ex13_df.count()
)

print("""
Observation:
This DataFrame is used only once.

If we cache it, Spark will spend extra effort storing the data,
but we will not reuse it later.

So caching is not useful for every DataFrame.
""")

print("\nExercise 13 completed.")


# ============================================================
# EXERCISE 14: cache() on DataFrame Created from File
# ============================================================
# Purpose:
# This exercise simulates a file-source use case.
#
# Important idea:
# If a DataFrame is read from a file and reused multiple times,
# caching can avoid repeatedly scanning the file source.
#
# Steps:
# 1. Create sample data.
# 2. Write it to Parquet.
# 3. Read it back.
# 4. Cache the read DataFrame.
# 5. Reuse it multiple times.

print_heading("EXERCISE 14: cache() on DataFrame Created from File")

ex14_path = "/tmp/cache_persist_ex14_parquet"

ex14_source_df = (
    spark.range(0, 2_000_000)
    .withColumn("amount", rand() * 1000)
    .withColumn("category_id", col("id") % 10)
)

ex14_source_df.write.mode("overwrite").parquet(ex14_path)

ex14_file_df = spark.read.parquet(ex14_path)

ex14_cached_file_df = ex14_file_df.cache()

measure_time(
    "Materializing cache for file DataFrame",
    lambda: ex14_cached_file_df.count()
)

measure_time(
    "Action 1 on cached file DataFrame",
    lambda: ex14_cached_file_df.filter(col("category_id") == 5).count()
)

measure_time(
    "Action 2 on cached file DataFrame",
    lambda: ex14_cached_file_df.agg(avg("amount")).collect()[0][0]
)

print("""
Observation:
If the same file-based DataFrame is reused multiple times,
cache can reduce repeated file scanning.
""")

ex14_cached_file_df.unpersist()

print("\nExercise 14 cleanup completed using unpersist().")


# ============================================================
# FINAL SUMMARY
# ============================================================

print_heading("FINAL SUMMARY")

print("""
Cache and Persist Summary:

1. Spark transformations are lazy.
2. cache() and persist() are also lazy.
3. cache() only marks a DataFrame for caching.
4. Actual cache materialization happens when the first action runs.
5. cache() uses the default storage level.
6. persist() allows us to choose storage level.
7. MEMORY_ONLY stores data only in memory.
8. MEMORY_AND_DISK stores data in memory first and disk if required.
9. DISK_ONLY stores data only on disk.
10. Serialized storage can save memory/storage but may add CPU cost.
11. Spark caches data at partition level.
12. A partition is either cached or not cached.
13. unpersist() removes cached data from memory/disk.
14. Cache is temporary, not permanent storage.
15. Do not cache every DataFrame.
16. Cache only when the DataFrame is reused or expensive to recompute.
17. Spark UI Storage tab is the best place to observe cached data.
18. explain(True) can show InMemoryTableScan when cached data is used.

Useful Commands:

df.cache()

df.persist(StorageLevel.MEMORY_ONLY)

df.persist(StorageLevel.MEMORY_AND_DISK)

df.persist(StorageLevel.DISK_ONLY)

df.persist(StorageLevel.MEMORY_AND_DISK_SER)

df.unpersist()

df.storageLevel

df.explain(True)
""")


# ------------------------------------------------------------
# Stop SparkSession
# ------------------------------------------------------------

spark.stop()

print("\nSparkSession stopped successfully")