import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, window, count, sum as spark_sum, when
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, TimestampType

# -------------------------------------------------------------
# Configuration & Logging 🛠️
# -------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - ⚡ %(message)s')

KAFKA_BROKER = "localhost:9092" # Use "kafka:29092" if running inside docker
KAFKA_TOPIC = "user_events"
POSTGRES_URL = "jdbc:postgresql://localhost:5432/feature_store"
POSTGRES_PROPERTIES = {
    "user": "algofriend",
    "password": "algopassword",
    "driver": "org.postgresql.Driver"
}

# -------------------------------------------------------------
# Define Event Schema 📄
# -------------------------------------------------------------
schema = StructType([
    StructField("event_id", StringType(), True),         # Advanced: Added unique ID
    StructField("user_id", IntegerType(), True),
    StructField("item_id", IntegerType(), True),
    StructField("event_type", StringType(), True),
    StructField("timestamp", TimestampType(), True),
    StructField("session_id", StringType(), True),
    StructField("device_type", StringType(), True)
])

def create_spark_session() -> SparkSession:
    """Initializes and configures the SparkSession. ✨"""
    logging.info("Initializing SparkSession... 🚀")
    return SparkSession.builder \
        .appName("RecommendationFeaturePipeline") \
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.1,org.postgresql:postgresql:42.6.0") \
        .config("spark.sql.shuffle.partitions", "200") \
        .getOrCreate()

def process_stream():
    """Main streaming pipeline to process Kafka events and sink to PostgreSQL. 🌊"""
    spark = create_spark_session()
    
    # 1. Read from Kafka 📥
    logging.info(f"Connecting to Kafka topic '{KAFKA_TOPIC}'... 🎧")
    raw_df = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", KAFKA_BROKER) \
        .option("subscribe", KAFKA_TOPIC) \
        .option("startingOffsets", "latest") \
        .option("failOnDataLoss", "false") \
        .load()

    # 2. Parse JSON 🧩
    parsed_df = raw_df.selectExpr("CAST(value AS STRING) as json_str") \
        .select(from_json(col("json_str"), schema).alias("data")) \
        .select("data.*")

    # 3. Add derived columns (One-hot encoding for event types) 🏷️
    enriched_df = parsed_df \
        .withColumn("is_view", when(col("event_type") == "view", 1).otherwise(0)) \
        .withColumn("is_purchase", when(col("event_type") == "purchase", 1).otherwise(0)) \
        .withColumn("is_like", when(col("event_type") == "like", 1).otherwise(0))
        
    # 4. Aggregate User Features (Tumbling Window) 👤
    user_features_df = enriched_df \
        .withWatermark("timestamp", "10 minutes") \
        .groupBy(
            col("user_id"),
            window(col("timestamp"), "5 minutes")
        ) \
        .agg(
            count("*").alias("total_events_5m"),
            spark_sum("is_view").alias("total_views_5m"),
            spark_sum("is_purchase").alias("total_purchases_5m"),
            spark_sum("is_like").alias("total_likes_5m")
        )

    # 5. Aggregate Item Features 📦
    item_features_df = enriched_df \
        .withWatermark("timestamp", "10 minutes") \
        .groupBy(
            col("item_id"),
            window(col("timestamp"), "5 minutes")
        ) \
        .agg(
            count("*").alias("item_total_events_5m"),
            spark_sum("is_view").alias("item_total_views_5m"),
            spark_sum("is_purchase").alias("item_total_purchases_5m"),
            spark_sum("is_like").alias("item_total_likes_5m")
        )

    # 6. Write stream to PostgreSQL (Using ForeachBatch for JDBC sink) 💾
    def write_user_to_postgres(df, epoch_id):
        db_df = df.select(
            col("user_id"), 
            col("window.start").alias("event_timestamp"),
            "total_events_5m", "total_views_5m", "total_purchases_5m", "total_likes_5m"
        )
        db_df.write.jdbc(url=POSTGRES_URL, table="user_features", mode="append", properties=POSTGRES_PROPERTIES)
        
    def write_item_to_postgres(df, epoch_id):
        db_df = df.select(
            col("item_id"), 
            col("window.start").alias("event_timestamp"),
            "item_total_events_5m", "item_total_views_5m", "item_total_purchases_5m", "item_total_likes_5m"
        )
        db_df.write.jdbc(url=POSTGRES_URL, table="item_features", mode="append", properties=POSTGRES_PROPERTIES)

    logging.info("Starting streaming queries... ⏳")
    query_users = user_features_df.writeStream \
        .outputMode("append") \
        .foreachBatch(write_user_to_postgres) \
        .option("checkpointLocation", "/tmp/spark_checkpoints/users") \
        .start()
        
    query_items = item_features_df.writeStream \
        .outputMode("append") \
        .foreachBatch(write_item_to_postgres) \
        .option("checkpointLocation", "/tmp/spark_checkpoints/items") \
        .start()

    query_users.awaitTermination()
    query_items.awaitTermination()

if __name__ == "__main__":
    process_stream()
