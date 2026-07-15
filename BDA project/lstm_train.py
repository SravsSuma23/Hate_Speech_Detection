import tensorflow as tf
import numpy as np
from pyspark.sql import SparkSession
from pyspark.storagelevel import StorageLevel
import os
import warnings
import math

# ==================== CONFIGURATION ====================
MAX_TOKENS = 80
EMBEDDING_DIM = 300
LSTM_HIDDEN = 150
LEARNING_RATE = 1e-4
TRAIN_BATCH_SIZE = 64
EPOCHS = 20
PATIENCE = 3
SHUFFLE_BUFFER_SIZE = 50

HDFS_HOST = "namenode"
HDFS_PORT = "8020"
SPARK_MASTER = "spark://spark-master:7077"

# ==================== INITIALIZE SPARK ====================
def create_spark_session():
    spark = SparkSession.builder \
        .appName("LSTM Training") \
        .master(SPARK_MASTER) \
        .config("spark.executor.memory", "6g") \
        .config("spark.driver.memory", "8g") \
        .config("spark.driver.maxResultSize", "4g") \
        .config("spark.hadoop.fs.defaultFS", f"hdfs://{HDFS_HOST}:{HDFS_PORT}") \
        .config("spark.sql.parquet.compression.codec", "snappy") \
        .config("spark.sql.parquet.enableVectorizedReader", "false") \
        .config("spark.sql.adaptive.enabled", "true") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark

# ==================== PERSIST DATAFRAME ====================
def load_and_persist_df(spark, split_name):
    path = f"hdfs://{HDFS_HOST}:{HDFS_PORT}/data/facebook_hateful_memes/fasttext_lstm_embeddings/{split_name}"
    print(f"📥 Reading {split_name}...")
    df = spark.read.parquet(path)
    if split_name == "train":
        df = df.repartition(50)
    else:
        df = df.repartition(8)
    df.persist(StorageLevel.MEMORY_AND_DISK)
    cnt = df.count()
    print(f"✅ {split_name}: {cnt} rows")
    return df, cnt

# ==================== GENERATOR ====================
def df_to_numpy_generator(df, batch_size):
    iterator = df.toLocalIterator()
    batch_X, batch_y = [], []
    for row in iterator:
        seq = np.array(row.text_embedding_seq, dtype=np.float32)
        batch_X.append(seq)
        batch_y.append(row.label)
        if len(batch_X) == batch_size:
            yield np.array(batch_X, dtype=np.float32), np.array(batch_y, dtype=np.int32)
            batch_X, batch_y = [], []
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

# ==================== CLASS WEIGHTS ====================
def compute_class_weights_spark(df):
    counts = df.groupBy("label").count().collect()
    d = {row.label: row['count'] for row in counts}
    valid = {k: v for k, v in d.items() if k in (0, 1)}
    if len(valid) < 2:
        return {0: 1.0, 1: 1.0}
    total = sum(valid.values())
    n = len(valid)
    return {cls: total / (n * cnt) for cls, cnt in valid.items()}

# ==================== BUILD MODEL ====================
def build_lstm_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(MAX_TOKENS, EMBEDDING_DIM)),
        tf.keras.layers.LSTM(LSTM_HIDDEN, return_sequences=False, name='lstm_layer'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# ==================== MAIN ====================
def main():
    print("=" * 60)
    print("LSTM TRAINING ONLY")
    print("=" * 60)

    warnings.filterwarnings('ignore')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.get_logger().setLevel('ERROR')
    tf.config.set_visible_devices([], 'GPU')

    spark = create_spark_session()

    try:
        df_train, train_cnt = load_and_persist_df(spark, "train")
        df_dev, dev_cnt     = load_and_persist_df(spark, "dev")
        df_test, test_cnt   = load_and_persist_df(spark, "test")  # not used for eval

        class_weight = compute_class_weights_spark(df_train)
        print(f"⚖️ Class weights: {class_weight}")

        train_steps = math.ceil(train_cnt / TRAIN_BATCH_SIZE)
        dev_steps   = math.ceil(dev_cnt / TRAIN_BATCH_SIZE)

        train_ds = create_tf_dataset(df_train, TRAIN_BATCH_SIZE, shuffle=True)
        dev_ds   = create_tf_dataset(df_dev,   TRAIN_BATCH_SIZE, shuffle=False)

        model = build_lstm_model()

        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE,
                                             restore_best_weights=True, verbose=1),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                                 patience=2, min_lr=1e-6, verbose=1),
            tf.keras.callbacks.ModelCheckpoint('/tmp/lstm_best.h5',
                                               monitor='val_accuracy', save_best_only=True, verbose=1)
        ]

        print(f"\n🚀 Training with {train_steps} steps per epoch...")
        model.fit(train_ds,
                  steps_per_epoch=train_steps,
                  validation_data=dev_ds,
                  validation_steps=dev_steps,
                  epochs=EPOCHS,
                  callbacks=callbacks,
                  class_weight=class_weight,
                  verbose=1)

        # Save final model
        model.save('/tmp/lstm_hate_speech.h5')
        print("✅ Model saved to /tmp/lstm_hate_speech.h5")

    finally:
        df_train.unpersist()
        df_dev.unpersist()
        df_test.unpersist()
        spark.stop()
        print("👋 Spark stopped.")

if __name__ == "__main__":
    main()