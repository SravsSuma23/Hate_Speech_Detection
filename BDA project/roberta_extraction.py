import os
import sys
import warnings
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, ArrayType, FloatType
import torch

# ==================== FORCE CACHE DIRECTORY ====================
os.environ["HF_HOME"] = "/tmp/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/hf_cache"
os.makedirs("/tmp/hf_cache", exist_ok=True)
print(f"Cache directory set to: {os.environ['TRANSFORMERS_CACHE']}")

# Import transformers AFTER setting cache
from transformers import RobertaTokenizer, RobertaModel

# ==================== CONFIGURATION ====================
HDFS_HOST = "namenode"
HDFS_PORT = "8020"
SPARK_MASTER = "spark://spark-master:7077"

INPUT_BASE = f"hdfs://{HDFS_HOST}:{HDFS_PORT}/data/facebook_hateful_memes/processed_text"
OUTPUT_BASE = f"hdfs://{HDFS_HOST}:{HDFS_PORT}/data/facebook_hateful_memes/features/roberta_features"

MAX_LENGTH = 128                     # max tokens per text (as in paper)
BATCH_SIZE = 32                       # inference batch size per partition

# ==================== SPARK SESSION ====================
def create_spark_session():
    spark = SparkSession.builder \
        .appName("RoBERTa Feature Extraction (mapPartitions)") \
        .master(SPARK_MASTER) \
        .config("spark.executor.memory", "8g") \
        .config("spark.driver.memory", "8g") \
        .config("spark.sql.parquet.compression.codec", "snappy") \
        .config("spark.hadoop.fs.defaultFS", f"hdfs://{HDFS_HOST}:{HDFS_PORT}") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.executorEnv.HF_HOME", "/tmp/hf_cache") \
        .config("spark.executorEnv.TRANSFORMERS_CACHE", "/tmp/hf_cache") \
        .config("spark.pyspark.python", "/usr/bin/python3") \
        .config("spark.executorEnv.PYTHONPATH", ":".join([p for p in sys.path if p])) \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark

# ==================== PARTITION PROCESSOR (RDD) ====================
def process_partition(iterator):
    """
    Process a partition of rows (each as a tuple) and yield (filename, label, embedding list)
    """
    import traceback
    import sys
    try:
        # Ensure cache directory exists
        os.makedirs("/tmp/hf_cache", exist_ok=True)

        # Load model once per partition
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        model = RobertaModel.from_pretrained("roberta-base")
        model.eval()
        torch.set_num_threads(1)

        # Collect rows into batches for efficiency
        batch_texts = []
        batch_filenames = []
        batch_labels = []

        for row in iterator:
            # row is a Row object from DataFrame; convert to dict or access by index
            # We'll assume the DataFrame has columns: id, img, text, label
            # But we need filename and text and label
            filename = row.img
            text = row.text
            label = row.label
            batch_texts.append(text)
            batch_filenames.append(filename)
            batch_labels.append(label)

            if len(batch_texts) >= BATCH_SIZE:
                # Process batch
                inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH)
                with torch.no_grad():
                    outputs = model(**inputs)
                cls_embeddings = outputs.last_hidden_state[:, 0, :].numpy().astype(np.float32)

                # Yield results
                for i in range(len(batch_texts)):
                    yield (batch_filenames[i], batch_labels[i], cls_embeddings[i].tolist())

                # Clear batch
                batch_texts = []
                batch_filenames = []
                batch_labels = []

        # Process remaining
        if batch_texts:
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH)
            with torch.no_grad():
                outputs = model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].numpy().astype(np.float32)
            for i in range(len(batch_texts)):
                yield (batch_filenames[i], batch_labels[i], cls_embeddings[i].tolist())

    except Exception as e:
        print("=" * 60, file=sys.stderr)
        print("ERROR IN PARTITION PROCESSOR:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        raise  # Re-raise to fail the task

# ==================== EXTRACT FOR ONE SPLIT ====================
def extract_split(split_name, spark):
    print(f"\n📁 Processing {split_name}...")
    input_path = f"{INPUT_BASE}/{split_name}"
    output_path = f"{OUTPUT_BASE}_{split_name}"

    # Read the preprocessed Parquet (should have columns: id, img, text, label)
    df = spark.read.parquet(input_path)

    # Repartition to control parallelism
    df = df.repartition(4)

    # Convert to RDD of rows and apply mapPartitions
    rdd = df.rdd.mapPartitions(process_partition)

    # Define schema for the result
    schema = StructType([
        StructField("img", StringType(), True),
        StructField("label", IntegerType(), True),
        StructField("text_embedding", ArrayType(FloatType()), True)
    ])

    # Convert back to DataFrame
    result_df = spark.createDataFrame(rdd, schema)

    # Write to HDFS
    result_df.write.mode("overwrite").parquet(output_path)
    count = result_df.count()
    print(f"✅ Saved {count} samples to {output_path}")
    return count

# ==================== MAIN ====================
def main():
    print("=" * 60)
    print("ROBERTA FEATURE EXTRACTION (mapPartitions)")
    print("=" * 60)

    warnings.filterwarnings('ignore')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    spark = create_spark_session()

    try:
        splits = ['train', 'dev', 'test']
        total_processed = 0
        for split in splits:
            count = extract_split(split, spark)
            total_processed += count

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