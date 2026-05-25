from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, expr, year, current_date, month, expr, lit

from pyspark import StorageLevel

# Initialize a Spark session
spark = SparkSession.builder.appName("AdvancedDataManipulation").enableHiveSupport().getOrCreate()

sc = spark.sparkContext

df = spark.read.csv("data/employee.csv", header=True, inferSchema=True)

# Register DataFrame as SQL temporary view
df.createOrReplaceTempView("employees")

# 1. salary_range
df = df.withColumn("salary_range",
                    when(col("salary") < 50000, "Low")
                   .when(col("salary") <= 100000, "Medium")
                   .otherwise("High"))

df = df.withColumn("salary_range", expr("""CASE
    WHEN salary < 50000 THEN 'Low'
    WHEN salary <= 100000 THEN 'Medium'
    ELSE 'High'
END as salary_range"""))

spark.sql("""
SELECT *, CASE 
    WHEN salary < 50000 THEN 'Low'
    WHEN salary <= 100000 THEN 'Medium'
    ELSE 'High'
END as salary_range FROM employees
""").show()

# 2. experience_level
df = df.withColumn("experience_level",
                    when(year("date_of_joining") < 2010, "Veteran")
                   .when(year("date_of_joining") <= 2015, "Experienced")
                   .otherwise("Novice"))
spark.sql("""
SELECT *, CASE 
    WHEN year(date_of_joining) < 2010 THEN 'Veteran'
    WHEN year(date_of_joining) <= 2015 THEN 'Experienced'
    ELSE 'Novice'
END as experience_level FROM employees
""").show()

# 3. dept_alias
df = df.withColumn("dept_alias",
                    when(col("dept") == "HR", "Human Resources")
                   .when(col("dept") == "Tech", "Technology")
                   .when(col("dept") == "Finance", "Financial Affairs")
                   .otherwise("Marketing and Sales"))
spark.sql("""
SELECT *, CASE 
    WHEN dept = 'HR' THEN 'Human Resources'
    WHEN dept = 'Tech' THEN 'Technology'
    WHEN dept = 'Finance' THEN 'Financial Affairs'
    ELSE 'Marketing and Sales'
END as dept_alias FROM employees
""").show()

# 4. is_high_earner
average_salary = df.selectExpr("avg(salary)").first()[0]
df = df.withColumn("is_high_earner", col("salary") > lit(average_salary))
spark.sql(f"""
SELECT *, salary > (SELECT avg(salary) FROM employees) as is_high_earner FROM employees
""").show()

# 5. adjusted_salary
df = df.withColumn("adjusted_salary",
                   when(col("dept") == "Tech", col("salary") * 1.1)
                   .otherwise(col("salary") * 1.05))
spark.sql("""
SELECT *, CASE
    WHEN dept = 'Tech' THEN salary * 1.1
    ELSE salary * 1.05
END as adjusted_salary FROM employees
""").show()

# 6. tenure_years
df = df.withColumn("tenure_years", year(current_date()) - year("date_of_joining"))
spark.sql("""
SELECT *, YEAR(current_date()) - YEAR(date_of_joining) as tenure_years FROM employees
""").show()

# 7. role_type
df = df.withColumn("role_type",
                   when((col("dept") == "Tech") & (col("salary") > 100000), "Senior Tech")
                   .when((col("dept") == "HR") & (col("salary") > 50000), "Senior HR")
                   .otherwise("General Staff"))
spark.sql("""
SELECT *, CASE
    WHEN dept = 'Tech' AND salary > 100000 THEN 'Senior Tech'
    WHEN dept = 'HR' AND salary > 50000 THEN 'Senior HR'
    ELSE 'General Staff'
END as role_type FROM employees
""").show()

# 8. hire_date_season
df = df.withColumn("hire_date_season",
                   when(month("date_of_joining").isin(1, 2, 3), "Winter")
                   .when(month("date_of_joining").isin(4, 5, 6), "Spring")
                   .when(month("date_of_joining").isin(7, 8, 9), "Summer")
                   .otherwise("Fall"))
spark.sql("""
SELECT *, CASE
    WHEN month(date_of_joining) IN (1, 2, 3) THEN 'Winter'
    WHEN month(date_of_joining) IN (4, 5, 6) THEN 'Spring'
    WHEN month(date_of_joining) IN (7, 8, 9) THEN 'Summer'
    ELSE 'Fall'
END as hire_date_season FROM employees
""").show()

# 9. salary_normalization (Requires more complex SQL)
min_salary, max_salary = df.selectExpr("min(salary)", "max(salary)").first()
df = df.withColumn("salary_normalization", (col("salary") - lit(min_salary)) / (lit(max_salary) - lit(min_salary)))
spark.sql(f"""
SELECT *, (salary - (SELECT min(salary) FROM employees)) / ((SELECT max(salary) FROM employees) - (SELECT min(salary) FROM employees)) as salary_normalization FROM employees
""").show()

# 10. initials
df = df.withColumn("initials", expr("concat(substring(ename, 1, 1), substring(ename, locate(' ', ename)+1, 1))"))
spark.sql("""
SELECT *, CONCAT(SUBSTRING(ename, 1, 1), SUBSTRING(ename, LOCATE(' ', ename) + 1, 1)) as initials FROM employees
""").show()

# Stop the Spark session
spark.stop()


# Questions :
# Add a new column "salary_range" based on the salary:
#       Less than 50,000: 'Low'
#       Between 50,000 and 100,000: 'Medium'
#       More than 100,000: 'High'

# Add a new column "experience_level" based on the year of joining:
#       Before 2010: 'Veteran'
#       Between 2010 and 2015: 'Experienced'
#       After 2015: 'Novice'

# Add a new column "dept_alias" based on the department:
#       HR: 'Human Resources'
#       Tech: 'Technology'
#       Finance: 'Financial Affairs'
#       Marketing: 'Marketing and Sales'

# Add a new column "is_high_earner" to indicate if the salary is above the average salary of the dataset.

# Create a column "adjusted_salary" that multiplies the salary by 1.1 if the department is Tech, otherwise by 1.05.

# Add a column "tenure_years" showing the number of years with the company based on the date of joining.

# Classify employees into "role_type" based on their department and salary:
#       Tech and salary > 100,000: 'Senior Tech'
#       HR and salary > 50,000: 'Senior HR'
#       Otherwise: 'General Staff'

# Add "hire_date_season" based on the month of joining:
#       Q1 (Jan-Mar): 'Winter'
#       Q2 (Apr-Jun): 'Spring'
#       Q3 (Jul-Sep): 'Summer'
#       Q4 (Oct-Dec): 'Fall'

# Add a column "salary_normalization" that normalizes the salary between 0 and 1 across the dataset.

# Add "initials" column that contains the initials of the employee's name.

