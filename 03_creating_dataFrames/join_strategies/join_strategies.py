# ============================================================
# PySpark Shell Demo: Spark Join Strategies
# ============================================================
#
# Objective:
# This script demonstrates common Spark SQL join strategies:
#
# 1. Broadcast Hash Join
# 2. Sort Merge Join
# 3. Shuffle Hash Join
# 4. Broadcast Nested Loop Join
# 5. Shuffle Replicate Nested Loop Join
# 6. Left Semi Join / Existential-style join
# 7. Left Anti Join / NOT EXISTS-style join
#
# Important:
# This script is designed for PySpark shell.
# In PySpark shell, spark and sc are already available.
#
# To check actual join strategy:
# Use:
# df.explain(True)
#
# Important physical plan keywords:
#
# BroadcastHashJoin
# SortMergeJoin
# ShuffledHashJoin
# BroadcastNestedLoopJoin
# CartesianProduct
# BroadcastExchange
# Exchange hashpartitioning
# LeftSemi
# LeftAnti
#
# Spark UI:
# http://localhost:4040
#
# Check:
# - SQL tab
# - Stages tab
# - Shuffle read/write
# - Physical plan
# ============================================================


from pyspark.sql.functions import col, broadcast, expr, rand, when, lit


# ============================================================
# Helper Function: Print Clean Headings
# ============================================================

def print_heading(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


# ============================================================
# Helper Function: Show Result and Physical Plan
# ============================================================
# Purpose:
# This function executes the DataFrame and prints the physical plan.
#
# show() triggers execution.
# explain(True) shows parsed, analyzed, optimized, and physical plans.

def show_and_explain(df, title, n=20):
    print_heading(title)

    print("\nResult:")
    df.show(n, truncate=False)

    print("\nDetailed Execution Plan:")
    df.explain(True)


# ============================================================
# Base Configuration for Demo
# ============================================================
# AQE is disabled initially so students can observe the static join plan.
# Later, AQE can be enabled separately if required.
#
# autoBroadcastJoinThreshold is set to -1 initially to avoid automatic
# broadcast joins unless we explicitly use broadcast() or BROADCAST hint.

print_heading("INITIAL CONFIGURATION")

spark.conf.set("spark.sql.adaptive.enabled", "false")
spark.conf.set("spark.sql.shuffle.partitions", "4")
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")

print("AQE Enabled:", spark.conf.get("spark.sql.adaptive.enabled"))
print("Shuffle Partitions:", spark.conf.get("spark.sql.shuffle.partitions"))
print("Auto Broadcast Join Threshold:", spark.conf.get("spark.sql.autoBroadcastJoinThreshold"))
print("Spark UI:", spark.sparkContext.uiWebUrl)


# ============================================================
# Create Small Base DataFrames
# ============================================================
# These DataFrames will be used in multiple exercises.
#
# orders_df:
# Transaction/order data.
#
# customers_df:
# Customer dimension data.
#
# products_df:
# Product dimension data.

print_heading("CREATING SAMPLE DATAFRAMES")

orders_data = [
    (1, 101, 1001, 500),
    (2, 102, 1002, 700),
    (3, 103, 1001, 300),
    (4, 101, 1003, 900),
    (5, 104, 1002, 200),
    (6, 105, 1004, 1000),
]

orders_columns = ["order_id", "customer_id", "product_id", "amount"]

customers_data = [
    (101, "Sandeep", "Pune"),
    (102, "Rahul", "Mumbai"),
    (103, "Priya", "Delhi"),
    (104, "Amit", "Pune"),
]

customers_columns = ["customer_id", "customer_name", "city"]

products_data = [
    (1001, "Laptop"),
    (1002, "Mobile"),
    (1003, "Keyboard"),
    (1004, "Mouse"),
]

products_columns = ["product_id", "product_name"]

orders_df = spark.createDataFrame(orders_data, orders_columns)
customers_df = spark.createDataFrame(customers_data, customers_columns)
products_df = spark.createDataFrame(products_data, products_columns)

orders_df.createOrReplaceTempView("orders")
customers_df.createOrReplaceTempView("customers")
products_df.createOrReplaceTempView("products")

print("\nOrders:")
orders_df.show()

print("\nCustomers:")
customers_df.show()

print("\nProducts:")
products_df.show()


# ============================================================
# EXERCISE 1: Broadcast Hash Join
# ============================================================
# Purpose:
# Broadcast Hash Join is used when one side of the join is small.
#
# Spark broadcasts the small DataFrame to all executors.
# Then each executor joins its local partition of the large DataFrame
# with the broadcasted small DataFrame.
#
# Benefit:
# Avoids shuffling the large DataFrame.
#
# Physical plan keyword:
# BroadcastHashJoin
#
# We explicitly use broadcast(customers_df).

broadcast_join_df = (
    orders_df
    .join(
        broadcast(customers_df),
        on="customer_id",
        how="inner"
    )
)

show_and_explain(
    broadcast_join_df,
    "EXERCISE 1: Broadcast Hash Join"
)


# ============================================================
# EXERCISE 2: Broadcast Hash Join using SQL Hint
# ============================================================
# Purpose:
# Same strategy as Exercise 1, but using SQL hint.
#
# Spark SQL hint:
# /*+ BROADCAST(table_name) */
#
# Physical plan keyword:
# BroadcastHashJoin
# BroadcastExchange

broadcast_sql_df = spark.sql("""
    SELECT /*+ BROADCAST(c) */
        o.order_id,
        o.customer_id,
        c.customer_name,
        o.amount
    FROM orders o
    INNER JOIN customers c
        ON o.customer_id = c.customer_id
""")

show_and_explain(
    broadcast_sql_df,
    "EXERCISE 2: Broadcast Hash Join using SQL Hint"
)


# ============================================================
# EXERCISE 3: Sort Merge Join
# ============================================================
# Purpose:
# Sort Merge Join is commonly used for large equi-joins.
#
# Both sides are shuffled by join key.
# Then both sides are sorted by join key.
# Then Spark merges matching rows.
#
# Conditions:
# - Equi-join
# - Broadcast disabled
# - MERGE hint can be used
#
# Physical plan keyword:
# SortMergeJoin
#
# We disable broadcast and use MERGE hint.

spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")

sort_merge_join_df = spark.sql("""
    SELECT /*+ MERGE(o, c) */
        o.order_id,
        o.customer_id,
        c.customer_name,
        o.amount
    FROM orders o
    INNER JOIN customers c
        ON o.customer_id = c.customer_id
""")

show_and_explain(
    sort_merge_join_df,
    "EXERCISE 3: Sort Merge Join"
)


# ============================================================
# EXERCISE 4: Shuffle Hash Join
# ============================================================
# Purpose:
# Shuffle Hash Join is also used for equi-joins.
#
# Both sides are shuffled by join key.
# On each shuffle partition, Spark builds a hash table for one side
# and probes it with the other side.
#
# Conditions:
# - Equi-join
# - Broadcast disabled
# - SHUFFLE_HASH hint used
#
# Physical plan keyword:
# ShuffledHashJoin
#
# Note:
# Spark may not always select Shuffle Hash Join automatically.
# Using SHUFFLE_HASH hint makes it easier to demonstrate.

spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")

shuffle_hash_join_df = spark.sql("""
    SELECT /*+ SHUFFLE_HASH(c) */
        o.order_id,
        o.customer_id,
        c.customer_name,
        o.amount
    FROM orders o
    INNER JOIN customers c
        ON o.customer_id = c.customer_id
""")

show_and_explain(
    shuffle_hash_join_df,
    "EXERCISE 4: Shuffle Hash Join"
)


# ============================================================
# EXERCISE 5: Broadcast Nested Loop Join
# ============================================================
# Purpose:
# Broadcast Nested Loop Join is used when:
#
# - Join condition is non-equi condition, or
# - Cross join-like condition is used, and
# - One side can be broadcast
#
# Unlike hash join, it does not require equality condition.
#
# Example condition:
# orders.amount > discount_rules.min_amount
#
# Physical plan keyword:
# BroadcastNestedLoopJoin
#
# This is usually expensive if data is large.
# It is okay when one side is very small.

discount_rules_data = [
    (1, 0, 100, "LOW"),
    (2, 101, 800, "MEDIUM"),
    (3, 801, 2000, "HIGH"),
]

discount_rules_columns = ["rule_id", "min_amount", "max_amount", "amount_band"]

discount_rules_df = spark.createDataFrame(
    discount_rules_data,
    discount_rules_columns
)

discount_rules_df.createOrReplaceTempView("discount_rules")

broadcast_nested_loop_df = (
    orders_df
    .join(
        broadcast(discount_rules_df),
        (orders_df.amount >= discount_rules_df.min_amount) &
        (orders_df.amount <= discount_rules_df.max_amount),
        "inner"
    )
    .select(
        "order_id",
        "amount",
        "amount_band"
    )
)

show_and_explain(
    broadcast_nested_loop_df,
    "EXERCISE 5: Broadcast Nested Loop Join for Non-Equi Join"
)


# ============================================================
# EXERCISE 6: Shuffle Replicate Nested Loop Join
# ============================================================
# Purpose:
# Shuffle Replicate Nested Loop Join is generally used for joins where
# Spark cannot use normal equi-join strategies and no broadcast side is used.
#
# It can be very expensive because data may be replicated across partitions.
#
# Spark SQL hint:
# SHUFFLE_REPLICATE_NL
#
# Physical plan may show:
# CartesianProduct
# or nested-loop style behavior depending on Spark version and plan.
#
# This strategy is usually not preferred for large data.

spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")

shuffle_replicate_nl_df = spark.sql("""
    SELECT /*+ SHUFFLE_REPLICATE_NL(r) */
        o.order_id,
        o.amount,
        r.amount_band
    FROM orders o
    INNER JOIN discount_rules r
        ON o.amount BETWEEN r.min_amount AND r.max_amount
""")

show_and_explain(
    shuffle_replicate_nl_df,
    "EXERCISE 6: Shuffle Replicate Nested Loop Join / Cartesian-style Join"
)


# ============================================================
# EXERCISE 7: Cross Join / Cartesian Product
# ============================================================
# Purpose:
# Cross join joins every row from left side with every row from right side.
#
# If left has 6 rows and right has 4 rows:
# output = 6 * 4 = 24 rows.
#
# This is very expensive for large data.
#
# Physical plan keyword:
# CartesianProduct
# or BroadcastNestedLoopJoin if one side is broadcast.

cross_join_df = orders_df.crossJoin(products_df)

show_and_explain(
    cross_join_df,
    "EXERCISE 7: Cross Join / Cartesian Product"
)


# ============================================================
# EXERCISE 8: Left Semi Join
# ============================================================
# Purpose:
# Left Semi Join returns rows from the left DataFrame only if a match
# exists in the right DataFrame.
#
# It is similar to SQL EXISTS logic.
#
# Important:
# It returns only columns from the left side.
#
# Example:
# Return only those orders whose customer exists in customers table.
#
# Physical plan keyword:
# LeftSemi
#
# Spark DataFrame API supports left_semi join type.

left_semi_df = orders_df.join(
    customers_df,
    on="customer_id",
    how="left_semi"
)

show_and_explain(
    left_semi_df,
    "EXERCISE 8: Left Semi Join / EXISTS-style Join"
)


# ============================================================
# EXERCISE 9: Left Anti Join
# ============================================================
# Purpose:
# Left Anti Join returns rows from the left DataFrame where no matching row
# exists in the right DataFrame.
#
# It is similar to SQL NOT EXISTS logic.
#
# Example:
# Return orders where customer is not present in customers table.
#
# Physical plan keyword:
# LeftAnti
#
# In our data, customer_id = 105 is present in orders but not in customers.

left_anti_df = orders_df.join(
    customers_df,
    on="customer_id",
    how="left_anti"
)

show_and_explain(
    left_anti_df,
    "EXERCISE 9: Left Anti Join / NOT EXISTS-style Join"
)


# ============================================================
# EXERCISE 10: EXISTS Subquery in SQL
# ============================================================
# Purpose:
# SQL EXISTS can often be optimized internally as a semi join.
#
# Query:
# Return orders where matching customer exists.
#
# Physical plan may show:
# LeftSemi
# BroadcastHashJoin LeftSemi
# SortMergeJoin LeftSemi
#
# depending on configuration and Spark version.

exists_df = spark.sql("""
    SELECT
        o.order_id,
        o.customer_id,
        o.product_id,
        o.amount
    FROM orders o
    WHERE EXISTS (
        SELECT 1
        FROM customers c
        WHERE c.customer_id = o.customer_id
    )
""")

show_and_explain(
    exists_df,
    "EXERCISE 10: SQL EXISTS Subquery / Existential-style Join"
)


# ============================================================
# EXERCISE 11: NOT EXISTS Subquery in SQL
# ============================================================
# Purpose:
# SQL NOT EXISTS can often be optimized internally as an anti join.
#
# Query:
# Return orders where matching customer does not exist.
#
# Physical plan may show:
# LeftAnti
# BroadcastHashJoin LeftAnti
# SortMergeJoin LeftAnti
#
# depending on configuration and Spark version.

not_exists_df = spark.sql("""
    SELECT
        o.order_id,
        o.customer_id,
        o.product_id,
        o.amount
    FROM orders o
    WHERE NOT EXISTS (
        SELECT 1
        FROM customers c
        WHERE c.customer_id = o.customer_id
    )
""")

show_and_explain(
    not_exists_df,
    "EXERCISE 11: SQL NOT EXISTS Subquery / Anti Join"
)


# ============================================================
# EXERCISE 12: Join Strategy Hint Priority
# ============================================================
# Purpose:
# Spark has multiple join hints:
#
# BROADCAST
# MERGE
# SHUFFLE_HASH
# SHUFFLE_REPLICATE_NL
#
# When conflicting hints are provided, Spark has priority rules.
#
# Spark prioritizes:
# BROADCAST > MERGE > SHUFFLE_HASH > SHUFFLE_REPLICATE_NL
#
# This exercise gives conflicting hints to observe which one wins.

hint_priority_df = spark.sql("""
    SELECT /*+ MERGE(o), BROADCAST(c) */
        o.order_id,
        o.customer_id,
        c.customer_name,
        o.amount
    FROM orders o
    INNER JOIN customers c
        ON o.customer_id = c.customer_id
""")

show_and_explain(
    hint_priority_df,
    "EXERCISE 12: Join Hint Priority - BROADCAST vs MERGE"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print_heading("FINAL SUMMARY")

print("""
Join Strategy Summary:

1. Broadcast Hash Join
   - Used when one side is small.
   - Small side is broadcast to executors.
   - Physical plan keyword: BroadcastHashJoin.

2. Sort Merge Join
   - Common for large equi-joins.
   - Both sides are shuffled and sorted by join key.
   - Physical plan keyword: SortMergeJoin.

3. Shuffle Hash Join
   - Both sides are shuffled by join key.
   - Spark builds hash table per partition for one side.
   - Physical plan keyword: ShuffledHashJoin.

4. Broadcast Nested Loop Join
   - Used for non-equi joins when one side can be broadcast.
   - Physical plan keyword: BroadcastNestedLoopJoin.

5. Shuffle Replicate Nested Loop Join
   - Used for nested-loop style joins without broadcast.
   - Can be expensive.
   - May appear as CartesianProduct or nested-loop style plan.

6. Cross Join
   - Every row from left joins with every row from right.
   - Very expensive for large data.
   - Physical plan keyword: CartesianProduct or BroadcastNestedLoopJoin.

7. Left Semi Join
   - Returns rows from left side where match exists in right side.
   - Similar to EXISTS.
   - Physical plan keyword: LeftSemi.

8. Left Anti Join
   - Returns rows from left side where match does not exist in right side.
   - Similar to NOT EXISTS.
   - Physical plan keyword: LeftAnti.

How to check actual strategy:
df.explain(True)

Important physical plan keywords:
BroadcastHashJoin
SortMergeJoin
ShuffledHashJoin
BroadcastNestedLoopJoin
CartesianProduct
BroadcastExchange
Exchange hashpartitioning
LeftSemi
LeftAnti
""")