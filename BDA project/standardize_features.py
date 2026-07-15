import os
from pyspark.sql import SparkSession
from pyspark.ml.feature import StandardScaler
from pyspark.ml.linalg import VectorUDT

# ==================== CONFIGURATION ====================
HDFS_HOST = "namenode"
HDFS_PORT = "8020"
SPARK_MASTER = "spark://spark-master:7077"

INPUT_BASE = f"hdfs://{HDFS_HOST}:{HDFS_PORT}/data/facebook_hateful_memes/concatenated_features"
OUTPUT_BASE = f"hdfs://{HDFS_HOST}:{HDFS_PORT}/data/facebook_hateful_memes/standardized_features"
MODEL_PATH = f"hdfs://{HDFS_HOST}:{HDFS_PORT}/data/facebook_hateful_memes/scaler_model"

# ==================== SPARK SESSION ====================
spark = SparkSession.builder \
    .appName("Standardize Multimodal Features") \
    .master(SPARK_MASTER) \
    .config("spark.executor.memory", "6g") \
    .config("spark.driver.memory", "8g") \
    .config("spark.sql.parquet.compression.codec", "snappy") \
    .config("spark.hadoop.fs.defaultFS", f"hdfs://{HDFS_HOST}:{HDFS_PORT}") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# ==================== LOAD DATA ====================
print("\n📥 Loading concatenated features from HDFS...")
train_df = spark.read.parquet(f"{INPUT_BASE}_train")
dev_df   = spark.read.parquet(f"{INPUT_BASE}_dev")
test_df  = spark.read.parquet(f"{INPUT_BASE}_test")

print(f"Train count: {train_df.count()}")
print(f"Dev count:   {dev_df.count()}")
print(f"Test count:  {test_df.count()}")

# ==================== STANDARDIZATION (FIT ON TRAIN ONLY) ====================
print("\n⚙️ Fitting StandardScaler on training data...")
scaler = StandardScaler(
    inputCol="features_vector",
    outputCol="scaled_features",
    withStd=True,
    withMean=True   # as per paper equation (1)
)

scaler_model = scaler.fit(train_df)
print("✅ Scaler fitted.")

# Save the scaler model to HDFS
print(f"\n💾 Saving scaler model to {MODEL_PATH}")
scaler_model.write().overwrite().save(MODEL_PATH)

# ==================== TRANSFORM ALL SPLITS ====================
print("\n🔄 Transforming training data...")
train_scaled = scaler_model.transform(train_df)

print("🔄 Transforming dev data...")
dev_scaled = scaler_model.transform(dev_df)

print("🔄 Transforming test data...")
test_scaled = scaler_model.transform(test_df)

# ==================== SAVE SCALED FEATURES TO HDFS ====================
print("\n💾 Saving standardized training features...")
train_scaled.select("filename", "label", "scaled_features") \
    .write.mode("overwrite").parquet(f"{OUTPUT_BASE}_train")

print("💾 Saving standardized dev features...")
dev_scaled.select("filename", "label", "scaled_features") \
    .write.mode("overwrite").parquet(f"{OUTPUT_BASE}_dev")

print("💾 Saving standardized test features...")
test_scaled.select("filename", "label", "scaled_features") \
    .write.mode("overwrite").parquet(f"{OUTPUT_BASE}_test")

print("\n✅ Standardization completed.")
print(f"📁 Output locations:")
print(f"   {OUTPUT_BASE}_train")
print(f"   {OUTPUT_BASE}_dev")
print(f"   {OUTPUT_BASE}_test")
print(f"📁 Scaler model: {MODEL_PATH}")

spark.stop()