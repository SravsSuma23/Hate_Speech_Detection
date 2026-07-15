import os
import sys
import numpy as np
import io
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, ArrayType, FloatType
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql import functions as F

# ==================== CONFIGURATION ====================
HDFS_HOST = "namenode"
HDFS_PORT = "8020"
SPARK_MASTER = "spark://spark-master:7077"

# ResNet features base path
RESNET_BASE = f"hdfs://{HDFS_HOST}:{HDFS_PORT}/data/facebook_hateful_memes/features/resnet_features1"

# LSTM features base path
LSTM_BASE = f"hdfs://{HDFS_HOST}:{HDFS_PORT}/data/facebook_hateful_memes/lstm_features"

# Output path for concatenated features
OUTPUT_BASE = f"hdfs://{HDFS_HOST}:{HDFS_PORT}/data/facebook_hateful_memes/concatenated_features_lstm"

# ==================== SPARK SESSION ====================
def create_spark_session():
    spark = SparkSession.builder \
        .appName("Concatenate ResNet and LSTM Features") \
        .master(SPARK_MASTER) \
        .config("spark.executor.memory", "6g") \
        .config("spark.driver.memory", "8g") \
        .config("spark.driver.maxResultSize", "4g") \
        .config("spark.sql.parquet.compression.codec", "snappy") \
        .config("spark.hadoop.fs.defaultFS", f"hdfs://{HDFS_HOST}:{HDFS_PORT}") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark

# ==================== READ RESNET BATCHES ====================
def read_resnet_split(spark, split_name):
    print(f"\n🔍 Loading ResNet features for split: {split_name}")
    hadoop_conf = spark.sparkContext._jsc.hadoopConfiguration()
    fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(hadoop_conf)

    split_path = spark._jvm.org.apache.hadoop.fs.Path(f"{RESNET_BASE}/{split_name}")
    if not fs.exists(split_path):
        print(f"❌ ResNet split directory not found: {split_path}")
        return None

    batch_dirs = [status.getPath() for status in fs.listStatus(split_path) if fs.isDirectory(status.getPath())]
    batch_dirs.sort(key=lambda p: str(p))

    if not batch_dirs:
        print(f"⚠️ No batch directories found in {split_path}")
        return None

    all_rows = []
    for batch_dir in batch_dirs:
        batch_name = batch_dir.getName()
        print(f"  📦 Processing {batch_name}...")
        feat_path = spark._jvm.org.apache.hadoop.fs.Path(batch_dir, "resnet_features.npy")
        label_path = spark._jvm.org.apache.hadoop.fs.Path(batch_dir, "labels.npy")
        fname_path = spark._jvm.org.apache.hadoop.fs.Path(batch_dir, "filenames.npy")

        try:
            stream = fs.open(feat_path)
            feat_bytes = bytearray(stream.readAllBytes())
            stream.close()
            feat_arr = np.load(io.BytesIO(feat_bytes))

            stream = fs.open(label_path)
            label_bytes = bytearray(stream.readAllBytes())
            stream.close()
            label_arr = np.load(io.BytesIO(label_bytes))

            stream = fs.open(fname_path)
            fname_bytes = bytearray(stream.readAllBytes())
            stream.close()
            fname_arr = np.load(io.BytesIO(fname_bytes), allow_pickle=True)

            assert len(feat_arr) == len(label_arr) == len(fname_arr), \
                f"Array length mismatch in {batch_name}"

            for i in range(len(fname_arr)):
                filename = str(fname_arr[i])
                label = int(label_arr[i])
                features = feat_arr[i].tolist()
                all_rows.append((filename, label, features))
        except Exception as e:
            print(f"    ⚠️ Failed to read/parse {batch_name}: {e}")
            continue

    if not all_rows:
        print(f"❌ No ResNet data loaded for {split_name}")
        return None

    schema = StructType([
        StructField("filename", StringType(), True),
        StructField("label", IntegerType(), True),
        StructField("resnet_features", ArrayType(FloatType()), True)
    ])
    df = spark.createDataFrame(all_rows, schema)
    print(f"✅ Loaded {df.count()} ResNet samples for {split_name}")
    return df

# ==================== READ LSTM BATCHES ====================
def read_lstm_split(spark, split_name):
    print(f"\n🔍 Loading LSTM features for split: {split_name}")
    hadoop_conf = spark.sparkContext._jsc.hadoopConfiguration()
    fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(hadoop_conf)

    split_path = spark._jvm.org.apache.hadoop.fs.Path(f"{LSTM_BASE}/{split_name}")
    if not fs.exists(split_path):
        print(f"❌ LSTM split directory not found: {split_path}")
        return None

    batch_dirs = [status.getPath() for status in fs.listStatus(split_path) if fs.isDirectory(status.getPath())]
    batch_dirs.sort(key=lambda p: str(p))

    if not batch_dirs:
        print(f"⚠️ No batch directories found in {split_path}")
        return None

    all_rows = []
    for batch_dir in batch_dirs:
        batch_name = batch_dir.getName()
        print(f"  📦 Processing {batch_name}...")
        feat_path = spark._jvm.org.apache.hadoop.fs.Path(batch_dir, "lstm_features.npy")
        label_path = spark._jvm.org.apache.hadoop.fs.Path(batch_dir, "labels.npy")
        fname_path = spark._jvm.org.apache.hadoop.fs.Path(batch_dir, "filenames.npy")

        try:
            stream = fs.open(feat_path)
            feat_bytes = bytearray(stream.readAllBytes())
            stream.close()
            feat_arr = np.load(io.BytesIO(feat_bytes))

            stream = fs.open(label_path)
            label_bytes = bytearray(stream.readAllBytes())
            stream.close()
            label_arr = np.load(io.BytesIO(label_bytes))

            stream = fs.open(fname_path)
            fname_bytes = bytearray(stream.readAllBytes())
            stream.close()
            fname_arr = np.load(io.BytesIO(fname_bytes), allow_pickle=True)

            assert len(feat_arr) == len(label_arr) == len(fname_arr), \
                f"Array length mismatch in {batch_name}"

            for i in range(len(fname_arr)):
                full_path = str(fname_arr[i])
                filename = os.path.basename(full_path)
                label = int(label_arr[i])
                features = feat_arr[i].tolist()
                all_rows.append((filename, label, features))
        except Exception as e:
            print(f"    ⚠️ Failed to read/parse {batch_name}: {e}")
            continue

    if not all_rows:
        print(f"❌ No LSTM data loaded for {split_name}")
        return None

    schema = StructType([
        StructField("filename", StringType(), True),
        StructField("label", IntegerType(), True),
        StructField("lstm_features", ArrayType(FloatType()), True)
    ])
    df = spark.createDataFrame(all_rows, schema)
    print(f"✅ Loaded {df.count()} LSTM samples for {split_name}")
    return df

# ==================== CONCATENATE FOR ONE SPLIT ====================
def process_split(spark, split_name):
    print(f"\n{'='*60}\nProcessing split: {split_name}\n{'='*60}")

    # 1. Load ResNet features
    resnet_df = read_resnet_split(spark, split_name)
    if resnet_df is None:
        return

    # 2. Load LSTM features
    lstm_df = read_lstm_split(spark, split_name)
    if lstm_df is None:
        return

    # 3. Drop the duplicate label column from LSTM DataFrame
    lstm_df = lstm_df.drop("label")

    # 4. Join on filename (inner join)
    joined_df = resnet_df.join(lstm_df, on="filename", how="inner")
    count = joined_df.count()
    print(f"✅ Joined {count} samples")

    if count == 0:
        print("⚠️ No matches – stopping.")
        return

    # 5. Concatenate feature vectors
    def concat_vectors(arr1, arr2):
        return arr1 + arr2
    concat_udf = F.udf(concat_vectors, ArrayType(FloatType()))

    joined_df = joined_df.withColumn(
        "multimodal_features",
        concat_udf("resnet_features", "lstm_features")
    )

    # 6. Convert to ML Vector for PCA/SMOTE
    def to_vector(arr):
        return Vectors.dense(arr)
    to_vector_udf = F.udf(to_vector, VectorUDT())
    joined_df = joined_df.withColumn("features_vector", to_vector_udf("multimodal_features"))

    # 7. Repartition to avoid OOM during write
    joined_df = joined_df.repartition(4)

    # 8. Save to HDFS
    output_path = f"{OUTPUT_BASE}_{split_name}"
    joined_df.select("filename", "label", "multimodal_features", "features_vector") \
        .write.mode("overwrite").parquet(output_path)
    print(f"✅ Saved concatenated features to {output_path}")

    return joined_df

# ==================== MAIN ====================
def main():
    print("=" * 60)
    print("CONCATENATE RESNET AND LSTM FEATURES")
    print("=" * 60)

    spark = create_spark_session()

    try:
        splits = ['train', 'dev', 'test']
        for split in splits:
            process_split(spark, split)

        print("\n✅ All splits processed successfully.")
        print("\n📁 Output locations (HDFS):")
        for split in splits:
            print(f"   {OUTPUT_BASE}_{split}")
    except Exception as e:
        print(f"\n❌ Pipeline error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        spark.stop()

if __name__ == "__main__":
    main()