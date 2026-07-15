import tensorflow as tf
import numpy as np
from pyspark.sql import SparkSession
from pyspark.storagelevel import StorageLevel
import os
import warnings
import shutil

# ==================== CONFIGURATION ====================
MAX_TOKENS = 80
EMBEDDING_DIM = 300
LSTM_HIDDEN = 150
LEARNING_RATE = 1e-4
TRAIN_BATCH_SIZE = 64
EPOCHS = 20
PATIENCE = 3
SHUFFLE_BUFFER_SIZE = 50
FEATURE_BATCH_SIZE = 256   # safe and efficient

HDFS_HOST = "namenode"
HDFS_PORT = "8020"
SPARK_MASTER = "spark://spark-master:7077"

# HDFS base path for features (writable by spark user)
HDFS_FEATURES_BASE = f"hdfs://{HDFS_HOST}:{HDFS_PORT}/data/facebook_hateful_memes/lstm_features"

# ==================== INITIALIZE SPARK ====================
def create_spark_session():
    spark = SparkSession.builder \
        .appName("LSTM Hate Speech Training (Streaming)") \
        .master(SPARK_MASTER) \
        .config("spark.executor.memory", "8g") \
        .config("spark.driver.memory", "8g") \
        .config("spark.driver.maxResultSize", "4g") \
        .config("spark.hadoop.fs.defaultFS", f"hdfs://{HDFS_HOST}:{HDFS_PORT}") \
        .config("spark.sql.parquet.compression.codec", "snappy") \
        .config("spark.sql.parquet.enableVectorizedReader", "false") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark

# ==================== PERSIST DATAFRAME (MEMORY+DISK) ====================
def load_and_persist_df(spark, split_name):
    """Load parquet DataFrame, repartition, and persist with MEMORY_AND_DISK."""
    path = f"hdfs://{HDFS_HOST}:{HDFS_PORT}/data/facebook_hateful_memes/fasttext_lstm_embeddings/{split_name}"
    print(f"📥 Reading {split_name} from {path} and persisting (MEMORY_AND_DISK)...")
    df = spark.read.parquet(path)
    
    # Repartition to make each partition smaller
    if split_name == "train":
        df = df.repartition(50)   # 8500 rows → ~170 per partition
    else:
        df = df.repartition(8)    # dev 500, test 1000 → small partitions
    
    df.persist(StorageLevel.MEMORY_AND_DISK)
    df.count()  # force persistence
    print(f"✅ {split_name} persisted. Row count: {df.count()}")
    return df

# ==================== GENERATOR FOR TF.DATA ====================
def df_to_numpy_generator(df, batch_size):
    iterator = df.toLocalIterator()
    batch_X = []
    batch_y = []
    for row in iterator:
        seq = np.array(row.text_embedding_seq, dtype=np.float32)  # (80,300)
        label = row.label
        batch_X.append(seq)
        batch_y.append(label)

        if len(batch_X) == batch_size:
            yield np.array(batch_X, dtype=np.float32), np.array(batch_y, dtype=np.int32)
            batch_X = []
            batch_y = []

    if batch_X:
        yield np.array(batch_X, dtype=np.float32), np.array(batch_y, dtype=np.int32)

def create_tf_dataset(df, batch_size, shuffle=False):
    output_signature = (
        tf.TensorSpec(shape=(None, MAX_TOKENS, EMBEDDING_DIM), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.int32)
    )
    ds = tf.data.Dataset.from_generator(
        lambda: df_to_numpy_generator(df, batch_size),
        output_signature=output_signature
    )
    if shuffle:
        ds = ds.shuffle(SHUFFLE_BUFFER_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

# ==================== COMPUTE CLASS WEIGHTS (SPARK) ====================
def compute_class_weights_spark(df):
    label_counts = df.groupBy("label").count().collect()
    count_dict = {row.label: row['count'] for row in label_counts}
    valid_counts = {k: v for k, v in count_dict.items() if k in (0, 1)}
    if len(valid_counts) < 2:
        print("⚠️ Warning: Not both labels 0 and 1 found. Using equal weights.")
        return {0: 1.0, 1: 1.0}
    total = sum(valid_counts.values())
    n_classes = len(valid_counts)
    weights = {}
    for cls, cnt in valid_counts.items():
        weights[cls] = total / (n_classes * cnt)
    print(f"⚖️ Class weights: {weights}")
    return weights

# ==================== VALIDATE LABELS ====================
def validate_labels(df, split_name):
    invalid = df.filter((df.label < 0) | (df.label > 1)).count()
    if invalid > 0:
        print(f"⚠️ {split_name} contains {invalid} rows with invalid labels (not 0 or 1).")
        return False
    return True

# ==================== BUILD LSTM MODEL ====================
def build_lstm_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(MAX_TOKENS, EMBEDDING_DIM)),
        tf.keras.layers.LSTM(LSTM_HIDDEN, return_sequences=False, name='lstm_layer'),
        tf.keras.layers.Dense(2, activation='softmax', name='output')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# ==================== SAVE MODEL LOCALLY (HDFS SKIPPED) ====================
def save_model_locally(model, model_name="lstm_hate_speech.h5"):
    print(f"\n💾 Saving model locally...")
    local_path = f'/tmp/{model_name}'
    model.save(local_path)
    print(f"✅ Model saved locally: {local_path}")

# ==================== EXTRACT LSTM FEATURES (BATCHED) ====================
def extract_lstm_features(model, df, split_name, spark):
    print(f"\n🔧 Extracting LSTM features for {split_name} (batched)...")

    feature_extractor = tf.keras.Model(
        inputs=model.input,
        outputs=model.get_layer('lstm_layer').output
    )

    # Accumulate rows in a list, then predict in batches
    batch_seqs = []
    batch_labels = []
    batch_ids = []
    batch_filenames = []

    # Create base directory for this split locally
    local_base_dir = f"/tmp/lstm_features/{split_name}"
    os.makedirs(local_base_dir, exist_ok=True)

    # Determine next batch index by checking existing batch_* directories
    existing = [d for d in os.listdir(local_base_dir) if d.startswith('batch_')]
    batch_idx = len(existing)

    # Ensure HDFS base directory exists (try to create)
    try:
        hadoop_conf = spark.sparkContext._jsc.hadoopConfiguration()
        fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(hadoop_conf)
        hdfs_base = spark._jvm.org.apache.hadoop.fs.Path(f"{HDFS_FEATURES_BASE}/{split_name}")
        if not fs.exists(hdfs_base):
            fs.mkdirs(hdfs_base)
            print(f"  ☁️  Created HDFS directory: {hdfs_base}")
    except Exception as e:
        print(f"  ⚠️ Could not create HDFS base directory (will skip HDFS upload): {e}")

    # Iterate over rows, collect until we have FEATURE_BATCH_SIZE rows
    for row in df.toLocalIterator():
        seq = np.array(row.text_embedding_seq, dtype=np.float32)  # (80,300)
        batch_seqs.append(seq)
        batch_labels.append(row.label)
        batch_ids.append(row.id)
        batch_filenames.append(row.img)

        if len(batch_seqs) >= FEATURE_BATCH_SIZE:
            # Convert to numpy array (batch, 80, 300)
            X_batch = np.array(batch_seqs, dtype=np.float32)
            # Predict
            feats = feature_extractor.predict(X_batch, verbose=0)  # (batch, 150)
            # Write this batch to disk and HDFS
            _write_feature_batch(
                feats, batch_labels, batch_ids, batch_filenames,
                split_name, batch_idx, spark
            )
            # Reset accumulators
            batch_seqs = []
            batch_labels = []
            batch_ids = []
            batch_filenames = []
            batch_idx += 1

    # Final batch (if any)
    if batch_seqs:
        X_batch = np.array(batch_seqs, dtype=np.float32)
        feats = feature_extractor.predict(X_batch, verbose=0)
        _write_feature_batch(
            feats, batch_labels, batch_ids, batch_filenames,
            split_name, batch_idx, spark
        )

    print(f"✅ LSTM feature extraction completed for {split_name}")

def _write_feature_batch(feats, labs, ids, fnames, split_name, batch_idx, spark):
    """
    feats: numpy array of shape (batch, 150)
    labs: list of labels (ints)
    ids: list of ids
    fnames: list of filenames
    """
    feats_arr = np.array(feats, dtype=np.float32)   # already numpy
    labs_arr = np.array(labs, dtype=np.int32)
    ids_arr = np.array(ids, dtype=np.int32)
    fnames_arr = np.array(fnames, dtype=str)

    # Local save
    local_dir = f"/tmp/lstm_features/{split_name}/batch_{batch_idx}"
    os.makedirs(local_dir, exist_ok=True)

    np.save(os.path.join(local_dir, "lstm_features.npy"), feats_arr)
    np.save(os.path.join(local_dir, "labels.npy"), labs_arr)
    np.save(os.path.join(local_dir, "ids.npy"), ids_arr)
    np.save(os.path.join(local_dir, "filenames.npy"), fnames_arr)
    print(f"  💾 Saved batch {batch_idx} with {len(feats)} rows to {local_dir}")

    # Upload to HDFS
    try:
        hadoop_conf = spark.sparkContext._jsc.hadoopConfiguration()
        fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(hadoop_conf)
        hdfs_batch_dir = spark._jvm.org.apache.hadoop.fs.Path(f"{HDFS_FEATURES_BASE}/{split_name}/batch_{batch_idx}")
        fs.mkdirs(hdfs_batch_dir)

        for fname in ["lstm_features.npy", "labels.npy", "ids.npy", "filenames.npy"]:
            local_file = os.path.join(local_dir, fname)
            local_path_obj = spark._jvm.org.apache.hadoop.fs.Path(local_file)
            hdfs_file_obj = spark._jvm.org.apache.hadoop.fs.Path(f"{HDFS_FEATURES_BASE}/{split_name}/batch_{batch_idx}/{fname}")
            fs.copyFromLocalFile(local_path_obj, hdfs_file_obj)
        print(f"  ☁️  Uploaded batch {batch_idx} to HDFS: {HDFS_FEATURES_BASE}/{split_name}/batch_{batch_idx}")
    except Exception as e:
        print(f"  ⚠️ Could not upload batch {batch_idx} to HDFS: {e}")

# ==================== MAIN PIPELINE ====================
def main():
    print("=" * 60)
    print("LSTM HATE SPEECH TRAINING (STREAMING, MEMORY+DISK PERSIST)")
    print("=" * 60)

    warnings.filterwarnings('ignore')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.get_logger().setLevel('ERROR')
    tf.debugging.set_log_device_placement(False)
    tf.config.set_visible_devices([], 'GPU')

    spark = create_spark_session()

    try:
        print("\n[1/4] Persisting datasets (MEMORY_AND_DISK)...")
        df_train = load_and_persist_df(spark, "train")
        df_dev   = load_and_persist_df(spark, "dev")
        df_test  = load_and_persist_df(spark, "test")

        print("\n[2/4] Computing class weights...")
        class_weight_dict = compute_class_weights_spark(df_train)

        # Validate dev labels only (we'll skip test)
        dev_valid = validate_labels(df_dev, "dev")
        if not dev_valid:
            print("❌ Dev set has invalid labels. Exiting.")
            return

        # Test set validation (just for info)
        test_valid = validate_labels(df_test, "test")
        if not test_valid:
            print("⚠️ Test set has invalid labels; will skip test evaluation.")

        # Create datasets
        train_ds = create_tf_dataset(df_train, TRAIN_BATCH_SIZE, shuffle=True)
        dev_ds   = create_tf_dataset(df_dev,   TRAIN_BATCH_SIZE, shuffle=False)
        # No test_ds needed for evaluation

        print("\n[3/4] Building LSTM model...")
        model = build_lstm_model()

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=PATIENCE, restore_best_weights=True, verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath='/tmp/lstm_best.h5',
                monitor='val_accuracy', save_best_only=True, verbose=1
            )
        ]

        print("\n🚀 Starting training...")
        history = model.fit(
            train_ds,
            validation_data=dev_ds,
            epochs=EPOCHS,
            callbacks=callbacks,
            class_weight=class_weight_dict,
            verbose=1
        )

        # -------------------- SAVE MODEL --------------------
        save_model_locally(model, "lstm_hate_speech.h5")

        # -------------------- EXTRACT LSTM FEATURES (BATCHED) --------------------
        print("\n🎯 Extracting LSTM features (batched)...")
        shutil.rmtree("/tmp/lstm_features", ignore_errors=True)
        os.makedirs("/tmp/lstm_features/train", exist_ok=True)
        os.makedirs("/tmp/lstm_features/dev", exist_ok=True)
        os.makedirs("/tmp/lstm_features/test", exist_ok=True)

        # Extract features (batched) and upload to HDFS
        extract_lstm_features(model, df_train, "train", spark)
        extract_lstm_features(model, df_dev,   "dev",   spark)
        extract_lstm_features(model, df_test,  "test",  spark)

        print("\n" + "=" * 60)
        print("✅ LSTM PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'df_train' in locals(): df_train.unpersist()
        if 'df_dev' in locals(): df_dev.unpersist()
        if 'df_test' in locals(): df_test.unpersist()
        spark.stop()
        print("\n👋 Spark stopped.")

if __name__ == "__main__":
    main()