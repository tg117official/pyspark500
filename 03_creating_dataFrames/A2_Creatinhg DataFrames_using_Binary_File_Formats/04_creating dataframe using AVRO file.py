from pyspark.sql import SparkSession
from pyspark.sql.avro.functions import from_avro, to_avro
from pyspark.sql.functions import col

# Initialize a SparkSession
# Include the Avro package if necessary. This line can be modified depending on your Spark version and setup.
# For example, in Spark 3.x and above, you might not need to add the package explicitly if already included.
# Following package is for spark 3.5.1, if you have a different version, change it accourding to you version
spark = SparkSession.builder \
    .appName("ReadAvroFile") \
    .config("spark.jars.packages", "org.apache.spark:spark-avro_2.12:3.5.1") \
    .getOrCreate()

# Load the Avro schema file
# It's good practice to load the schema from a file to ensure compatibility and avoid manual coding errors.
schema_path = r"data\twitter.avsc"
with open(schema_path, 'r') as f:
    avro_schema = f.read()
#avro_schema = """
# {
#   "type" : "record",
#   "name" : "twitter_schema",
#   "namespace" : "com.miguno.avro",
#   "fields" : [ {
#     "name" : "username",
#     "type" : "string",
#     "doc" : "Name of the user account on Twitter.com"
#   }, {
#     "name" : "tweet",
#     "type" : "string",
#     "doc" : "The content of the user's Twitter message"
#   }, {
#     "name" : "timestamp",
#     "type" : "long",
#     "doc" : "Unix epoch time in seconds"
#   } ],
#   "doc:" : "A 01_basic_broadcast schema for storing Twitter messages"
# } """


# Read the Avro file into a DataFrame using the schema
# This ensures that the DataFrame conforms to the expected structure and types as defined in the Avro schema.
df = spark.read.format("avro") \
    .option("avroSchema", avro_schema) \
    .load(r"data\twitter.avro")

# Show the DataFrame to verify the contents
df.show()

# Stop the Spark session
spark.stop()

# To run this code in pyspark shell, start pyspark shell with following command
# pyspark --packages org.apache.spark:spark-avro_2.12:3.4.3
