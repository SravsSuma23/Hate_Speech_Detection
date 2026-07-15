import tensorflow as tf
import numpy as np
import io
from PIL import Image
from pyspark.sql import SparkSession
import os
import time
import warnings
import gc
from sklearn.utils.class_weight import compute_class_weight

# ==================== CONFIGURATION ====================
IMG_SIZE = 224
RESNET_FEATURE_DIM = 2048
LEARNING_RATE = 1e-5               
WEIGHT_DECAY = 0.1                  
FINE_TUNE_EPOCHS = 20           

# Batch streaming configuration
TRAIN_BATCH_SIZE = 12               

# HDFS CONFIGURATION
HDFS_HOST = "namenode"
HDFS_PORT = "8020"
SPARK_MASTER = "spark://spark-master:7077"

# ==================== INITIALIZE SPARK ====================
def create_spark_session():
    spark = SparkSession.builder \
        .appName("ResNet50 Streaming Training") \
        .master(SPARK_MASTER) \
        .config("spark.executor.memory", "6g") \
        .config("spark.driver.memory", "6g") \
        .config("spark.driver.maxResultSize", "2g") \
        .config("spark.hadoop.fs.defaultFS", f"hdfs://{HDFS_HOST}:{HDFS_PORT}") \
        .config("spark.network.timeout", "600s") \
        .config("spark.sql.parquet.compression.codec", "snappy") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark

# ==================== SAVE MODEL TO HDFS ====================
def save_model_to_hdfs(spark, model, model_name="resnet50_fine_tuned.h5"):
    print(f"\n💾 Saving model to HDFS...")
    try:
        local_path = f'/tmp/{model_name}'
        model.save(local_path)
        print(f"✅ Model saved locally: {local_path}")

        hadoop_conf = spark.sparkContext._jsc.hadoopConfiguration()
        fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(hadoop_conf)

        hdfs_dir = f"/data/models"
        hdfs_path = f"{hdfs_dir}/{model_name}"
        hdfs_dir_obj = spark._jvm.org.apache.hadoop.fs.Path(hdfs_dir)
        hdfs_path_obj = spark._jvm.org.apache.hadoop.fs.Path(hdfs_path)

        if not fs.exists(hdfs_dir_obj):
            fs.mkdirs(hdfs_dir_obj)
            print(f"📁 Created HDFS directory: {hdfs_dir}")

        if fs.exists(hdfs_path_obj):
            fs.delete(hdfs_path_obj, True)
            print(f"🗑️ Removed existing HDFS file")

        local_path_obj = spark._jvm.org.apache.hadoop.fs.Path(local_path)
        fs.copyFromLocalFile(local_path_obj, hdfs_path_obj)

        if fs.exists(hdfs_path_obj):
            file_status = fs.getFileStatus(hdfs_path_obj)
            file_size = file_status.getLen() / (1024 * 1024)
            print(f"✅ Model saved to HDFS: hdfs://{HDFS_HOST}:{HDFS_PORT}{hdfs_path}")
            print(f"📏 File size: {file_size:.2f} MB")
            return hdfs_path
        else:
            print("⚠️ Warning: Could not verify HDFS save")
            return None
    except Exception as e:
        print(f"❌ Error saving to HDFS: {e}")
        return None

# ==================== RESNET50 MODEL DEFINITIONS ====================
def create_resnet50_base():
    print("\n🧠 Creating ResNet50 base model...")
    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    # Freeze all but the last 10 layers
    for layer in base_model.layers[:-10]:
        layer.trainable = False
    trainable_count = sum([layer.trainable for layer in base_model.layers])
    print(f"✅ Trainable layers: {trainable_count}/{len(base_model.layers)}")
    return base_model

def create_fine_tuning_model(base_model):
    """Add classification head. Use AdamW with weight decay as in the paper."""
    print("\n🔧 Creating fine‑tuning model...")
    x = tf.keras.layers.GlobalAveragePooling2D(name='global_avg_pool')(base_model.output)
    x = tf.keras.layers.Dense(256, activation='relu', name='fc1')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(128, activation='relu', name='fc2')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(2, activation='softmax', name='predictions')(x)

    model = tf.keras.Model(inputs=base_model.input, outputs=outputs)

    # Use AdamW to incorporate weight decay (requires TF 2.11+)
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# ==================== STREAMING DATA GENERATOR ====================
class HDFSDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, split_name, batch_size=32, shuffle=True):
        self.split_name = split_name
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.base_path = f"hdfs://{HDFS_HOST}:{HDFS_PORT}/data/facebook_hateful_memes"
        self.split_path = f"{self.base_path}/image_{split_name}"
        self.file_paths = []
        self.labels = []
        self._load_file_list()
        self.on_epoch_end()   # initial shuffle

    def _load_file_list(self):
        print(f"📥 Indexing {self.split_name} files...")
        hadoop_conf = spark.sparkContext._jsc.hadoopConfiguration()
        fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(hadoop_conf)
        path_obj = spark._jvm.org.apache.hadoop.fs.Path(self.split_path)

        self.file_paths = []
        self.labels = []

        for status in fs.listStatus(path_obj):
            if fs.isDirectory(status.getPath()):
                folder_name = status.getPath().getName().lower()
                label = 1 if 'hateful' in folder_name and 'non_hateful' not in folder_name else 0
                for file_status in fs.listStatus(status.getPath()):
                    if file_status.getPath().getName().endswith('.png'):
                        file_path = str(file_status.getPath())
                        self.file_paths.append(file_path)
                        self.labels.append(label)

        print(f"✅ Indexed {len(self.file_paths)} images for {self.split_name}")
        print(f"   Class 0 (non-hateful): {self.labels.count(0)}")
        print(f"   Class 1 (hateful): {self.labels.count(1)}")

    def __len__(self):
        # One epoch = all images exactly once
        return int(np.ceil(len(self.file_paths) / self.batch_size))

    def __getitem__(self, index):
        batch_start = index * self.batch_size
        batch_end = min(batch_start + self.batch_size, len(self.file_paths))
        batch_indices = self.indices[batch_start:batch_end]

        batch_images = []
        batch_labels = []

        for idx in batch_indices:
            file_path = self.file_paths[idx]
            label = self.labels[idx]
            try:
                hadoop_conf = spark.sparkContext._jsc.hadoopConfiguration()
                fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(hadoop_conf)
                path_obj = spark._jvm.org.apache.hadoop.fs.Path(file_path)
                stream = fs.open(path_obj)
                content = bytearray(stream.readAllBytes())
                stream.close()

                with Image.open(io.BytesIO(content)) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img = img.resize((IMG_SIZE, IMG_SIZE))
                    img_array = np.array(img, dtype=np.float32)
                    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)

                batch_images.append(img_array)
                batch_labels.append(label)
            except Exception as e:
                print(f"⚠️ Skipping {os.path.basename(file_path)}: {str(e)[:100]}")
                continue

        # If we skipped some images, we might have fewer than batch_size.
        # Pad with duplicates of the first image (any image) to keep batch size constant.
        # This is a simple fix; you could also drop the last incomplete batch.
        while len(batch_images) < self.batch_size:
            if len(batch_images) == 0:
                # No valid images in this batch – return zeros
                batch_images.append(np.zeros((IMG_SIZE, IMG_SIZE, 3)))
                batch_labels.append(0)
            else:
                batch_images.append(batch_images[0])
                batch_labels.append(batch_labels[0])

        return np.array(batch_images), np.array(batch_labels)

    def on_epoch_end(self):
        self.indices = np.arange(len(self.file_paths))
        if self.shuffle:
            np.random.shuffle(self.indices)

# ==================== STREAMING FINE‑TUNING ====================
def fine_tune_resnet50_streaming():
    print("\n" + "=" * 60)
    print("STREAMING RESNET50 FINE‑TUNING FROM HDFS")
    print("=" * 60)

    try:
        # Create streaming generators
        print("\n📥 Creating streaming data generators...")
        train_gen = HDFSDataGenerator(
            split_name='train',
            batch_size=TRAIN_BATCH_SIZE,
            shuffle=True
        )
        val_gen = HDFSDataGenerator(
            split_name='dev',
            batch_size=TRAIN_BATCH_SIZE,
            shuffle=False
        )

        # Compute steps per epoch from generator lengths
        train_steps = len(train_gen)
        val_steps = len(val_gen)

        print(f"\n📊 Training configuration (paper parameters):")
        print(f"   Batch size: {TRAIN_BATCH_SIZE}")
        print(f"   Steps per epoch (train): {train_steps} (full dataset)")
        print(f"   Steps per epoch (val): {val_steps}")
        print(f"   Epochs: {FINE_TUNE_EPOCHS}")
        print(f"   Learning rate: {LEARNING_RATE}")
        print(f"   Weight decay: {WEIGHT_DECAY}")

        # Create model
        base_model = create_resnet50_base()
        model = create_fine_tuning_model(base_model)

        # Class weights
        print("\n📊 Computing class weights from sample...")
        sample_images, sample_labels = train_gen[0]
        class_weights = compute_class_weight('balanced', classes=[0, 1], y=sample_labels)
        class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
        print(f"⚖️ Class weights: {class_weight_dict}")

        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=7, restore_best_weights=True, verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath='/tmp/resnet50_fine_tuned_best.h5',
                monitor='val_accuracy', save_best_only=True, verbose=1
            )
        ]

        # Train
        print("\n🚀 Starting streaming fine‑tuning...")
        start_time = time.time()

        history = model.fit(
            train_gen,
            steps_per_epoch=train_steps,
            validation_data=val_gen,
            validation_steps=val_steps,
            epochs=FINE_TUNE_EPOCHS,
            callbacks=callbacks,
            class_weight=class_weight_dict,
            verbose=1
        )

        training_time = time.time() - start_time
        print(f"\n⏱️ Streaming training completed in {training_time:.2f}s")

        # Save locally and to HDFS
        model.save('/tmp/resnet50_fine_tuned.h5')
        print("✅ Model saved locally to /tmp/resnet50_fine_tuned.h5")
        hdfs_path = save_model_to_hdfs(spark, model, "resnet50_fine_tuned.h5")

        # Final evaluation
        print("\n📊 Final evaluation on validation set...")
        val_loss, val_acc = model.evaluate(val_gen, steps=val_steps, verbose=0)
        print(f"   Validation Loss: {val_loss:.4f}")
        print(f"   Validation Accuracy: {val_acc:.4f}")

        tf.keras.backend.clear_session()
        gc.collect()

        return model, hdfs_path

    except Exception as e:
        print(f"❌ Streaming fine‑tuning failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# ==================== MAIN ====================
def main():
    print("=" * 60)
    print("RESNET50 STREAMING FINE‑TUNING (PAPER PARAMETERS)")
    print("=" * 60)

    warnings.filterwarnings('ignore')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.get_logger().setLevel('ERROR')
    tf.config.set_visible_devices([], 'GPU')
    tf.keras.backend.clear_session()

    global spark
    spark = create_spark_session()

    try:
        fine_tuned_model, hdfs_model_path = fine_tune_resnet50_streaming()

        if fine_tuned_model is None:
            print("\n⚠️ Streaming training failed!")
        else:
            print("\n✅ Streaming training completed successfully")
            print("\n" + "=" * 60)
            print("🎯 TRAINING SUMMARY:")
            print("=" * 60)
            print(f"📁 Local model: /tmp/resnet50_fine_tuned.h5")
            if hdfs_model_path:
                print(f"📁 HDFS model: hdfs://{HDFS_HOST}:{HDFS_PORT}{hdfs_model_path}")
            print(f"📁 Best model: /tmp/resnet50_fine_tuned_best.h5")
            print("=" * 60)

        print("\n💡 NEXT: Run feature extraction separately:")
        print("   python3 resnet50_feature_extraction.py")

    except Exception as e:
        print(f"\n❌ Pipeline error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n🛑 Stopping Spark...")
        spark.stop()
        print("👋 Training completed.")

if __name__ == "__main__":
    main()