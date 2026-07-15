import numpy as np
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.sql import Row
from imblearn.over_sampling import SMOTE

# ==================== CONFIGURATION ====================
HDFS_HOST = "namenode"
HDFS_PORT = "8020"
BASE_PATH = f"hdfs://{HDFS_HOST}:{HDFS_PORT}/data/facebook_hateful_memes"

TRAIN_PCA = f"{BASE_PATH}/pca_features_train"
DEV_PCA   = f"{BASE_PATH}/pca_features_dev"
TEST_PCA  = f"{BASE_PATH}/pca_features_test"

OUTPUT_BALANCED = f"{BASE_PATH}/balanced_train_pca"

# ==================== SPARK SESSION ====================
spark = SparkSession.builder \
    .appName("SMOTE on PCA Features") \
    .master("spark://spark-master:7077") \
    .config("spark.executor.memory", "6g") \
    .config("spark.driver.memory", "8g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# ==================== LOAD TRAINING PCA FEATURES ====================
print("\n📥 Loading training PCA features...")
train_df = spark.read.parquet(TRAIN_PCA)
print(f"Training samples: {train_df.count()}")

# Convert to NumPy arrays (fits in driver memory: 8500 x 512 = ~17 MB)
train_features = np.array(train_df.select("pca_features").rdd.map(lambda r: r[0].toArray()).collect())
train_labels   = np.array(train_df.select("label").rdd.map(lambda r: r[0]).collect())

print(f"Original class distribution: {np.bincount(train_labels)}")

# ==================== APPLY SMOTE ====================
print("\n🔄 Applying SMOTE (k_neighbors=3)...")
smote = SMOTE(k_neighbors=3, random_state=42)
X_res, y_res = smote.fit_resample(train_features, train_labels)

print(f"Resampled class distribution: {np.bincount(y_res)}")
print(f"New training size: {len(X_res)}")

# ==================== SAVE BALANCED TRAINING SET TO HDFS ====================
print("\n💾 Saving balanced training set to HDFS...")

# Convert back to Spark DataFrame
def vector_from_array(arr):
    return Vectors.dense(arr)

rows = [Row(features=vector_from_array(x), label=int(y)) for x, y in zip(X_res, y_res)]
balanced_df = spark.createDataFrame(rows)

balanced_df.write.mode("overwrite").parquet(OUTPUT_BALANCED)
print(f"✅ Balanced training set saved to {OUTPUT_BALANCED}")

spark.stop()