import tensorflow as tf
import numpy as np
import io
from PIL import Image
from pyspark.sql import SparkSession
import os
import time
import warnings
import shutil
import gc

# ==================== CONFIGURATION ====================
IMG_SIZE = 224
RESNET_FEATURE_DIM = 2048               # after global average pooling

HDFS_HOST = "namenode"
HDFS_PORT = "8020"
SPARK_MASTER = "spark://spark-master:7077"

# HDFS base path for extracted features (batches)
HDFS_RESNET_BASE = f"hdfs://{HDFS_HOST}:{HDFS_PORT}/data/facebook_hateful_memes/features/resnet_features1"

# Batch sizes
PRED_BATCH_SIZE = 32        # images fed to model at once (inference batch)
SAVE_BATCH_SIZE = 500       # number of images after which we write to disk

# ==================== INITIALIZE SPARK ====================
def create_spark_session():
    spark = SparkSession.builder \
        .appName("ResNet50 Feature Extraction (Disk Spilling)") \
        .master(SPARK_MASTER) \
        .config("spark.executor.memory", "6g") \
        .config("spark.driver.memory", "8g") \
        .config("spark.driver.maxResultSize", "4g") \
        .config("spark.hadoop.fs.defaultFS", f"hdfs://{HDFS_HOST}:{HDFS_PORT}") \
        .config("spark.sql.parquet.compression.codec", "snappy") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark

# ==================== LOAD TRAINED RESNET50 MODEL ====================
def load_trained_resnet50():
    """Load the fine‑tuned ResNet50 model and create a feature extractor (global_avg_pool)."""
    print("\n🔍 Loading trained ResNet50 model...")

    model_paths = ['/tmp/resnet50_fine_tuned_best.h5', '/tmp/resnet50_fine_tuned.h5']
    base_model = None

    for path in model_paths:
        if os.path.exists(path):
            print(f"  Loading {path}")
            full_model = tf.keras.models.load_model(path)
            # Try to extract the global_avg_pool layer
            try:
                base_model = tf.keras.Model(
                    inputs=full_model.input,
                    outputs=full_model.get_layer('global_avg_pool').output
                )
                print("✅ Feature extractor created from fine‑tuned model (global_avg_pool).")
                break
            except Exception as e:
                print(f"  Could not extract global_avg_pool: {e}. Using whole model.")
                base_model = full_model
                break

    if base_model is None:
        print("⚠️ No fine‑tuned model found, using pre‑trained ResNet50.")
        base_model = tf.keras.applications.ResNet50(
            include_top=False,
            weights='imagenet',
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
            pooling='avg'      # directly gives 2048‑dim features
        )

    return base_model

# ==================== PROCESS BATCH OF IMAGES ====================
def process_image_batch(image_data_list, feature_extractor):
    """
    image_data_list: list of tuples (file_path, content, label)
    Returns lists of features, labels, filenames for successfully processed images.
    """
    batch_images = []
    batch_labels = []
    batch_filenames = []

    for file_path, content, label in image_data_list:
        try:
            with Image.open(io.BytesIO(bytearray(content))) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img = img.resize((IMG_SIZE, IMG_SIZE))
                img_array = np.array(img, dtype=np.float32)
                img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
                batch_images.append(img_array)
                batch_labels.append(label)
                batch_filenames.append(os.path.basename(file_path))
        except Exception as e:
            print(f"⚠️ Error processing {file_path}: {e}")

    if not batch_images:
        return [], [], []

    # Stack into a single batch for prediction
    X_batch = np.stack(batch_images, axis=0)  # (batch, 224, 224, 3)
    features = feature_extractor.predict(X_batch, verbose=0)  # (batch, 2048)

    # Ensure correct dimension (should be 2048 already, but pad/trim just in case)
    features_list = []
    for i in range(features.shape[0]):
        f = features[i].flatten()
        if len(f) != RESNET_FEATURE_DIM:
            if len(f) > RESNET_FEATURE_DIM:
                f = f[:RESNET_FEATURE_DIM]
            else:
                f = np.pad(f, (0, RESNET_FEATURE_DIM - len(f)))
        features_list.append(f)

    return features_list, batch_labels, batch_filenames

# ==================== EXTRACT FEATURES WITH DISK SPILLING ====================
def extract_features_split(split_name, feature_extractor, spark):
    print(f"\n📁 Processing {split_name}...")
    base_path = f"hdfs://{HDFS_HOST}:{HDFS_PORT}/data/facebook_hateful_memes"
    split_path = f"{base_path}/image_{split_name}"

    hadoop_conf = spark.sparkContext._jsc.hadoopConfiguration()
    fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(hadoop_conf)

    # Collect all image paths and labels
    image_paths = []
    split_path_obj = spark._jvm.org.apache.hadoop.fs.Path(split_path)
    try:
        for folder_status in fs.listStatus(split_path_obj):
            if fs.isDirectory(folder_status.getPath()):
                folder_name = folder_status.getPath().getName().lower()
                label = 1 if 'hateful' in folder_name and 'non_hateful' not in folder_name else 0
                for file_status in fs.listStatus(folder_status.getPath()):
                    if file_status.getPath().getName().endswith('.png'):
                        image_paths.append((str(file_status.getPath()), label))
    except Exception as e:
        print(f"⚠️ Could not list directory {split_path}: {e}")
        return 0

    total = len(image_paths)
    print(f"📊 Found {total} images")

    if total == 0:
        print(f"⚠️ No images found for {split_name}. Skipping.")
        return 0

    # Prepare local base directory for batches
    local_base = f"/tmp/resnet_features_batches/{split_name}"
    if os.path.exists(local_base):
        shutil.rmtree(local_base)
    os.makedirs(local_base)

    # Prepare HDFS base directory
    try:
        hdfs_split_dir = spark._jvm.org.apache.hadoop.fs.Path(f"{HDFS_RESNET_BASE}/{split_name}")
        if not fs.exists(hdfs_split_dir):
            fs.mkdirs(hdfs_split_dir)
            print(f"  ☁️  Created HDFS directory: {hdfs_split_dir}")
    except Exception as e:
        print(f"  ⚠️ Could not create HDFS base directory: {e}")

    # Accumulate features in memory until we reach SAVE_BATCH_SIZE
    all_features = []
    all_labels = []
    all_filenames = []
    batch_idx = 0
    processed = 0

    # Process images in chunks of PRED_BATCH_SIZE
    for i in range(0, total, PRED_BATCH_SIZE):
        chunk_paths = image_paths[i:i+PRED_BATCH_SIZE]

        # Read image contents for this chunk
        chunk_data = []
        for file_path, label in chunk_paths:
            try:
                path_obj = spark._jvm.org.apache.hadoop.fs.Path(file_path)
                stream = fs.open(path_obj)
                content = bytearray(stream.readAllBytes())
                stream.close()
                chunk_data.append((file_path, content, label))
            except Exception as e:
                print(f"⚠️ Failed to read {file_path}: {e}")

        # Process batch
        feats, labs, fnames = process_image_batch(chunk_data, feature_extractor)
        if feats:
            all_features.extend(feats)
            all_labels.extend(labs)
            all_filenames.extend(fnames)
            processed += len(feats)

        # If we have accumulated enough, write to disk
        while len(all_features) >= SAVE_BATCH_SIZE:
            # Take first SAVE_BATCH_SIZE
            feats_slice = all_features[:SAVE_BATCH_SIZE]
            labs_slice = all_labels[:SAVE_BATCH_SIZE]
            fnames_slice = all_filenames[:SAVE_BATCH_SIZE]

            _write_batch(feats_slice, labs_slice, fnames_slice,
                         split_name, batch_idx, local_base, fs, spark)

            # Keep the rest
            all_features = all_features[SAVE_BATCH_SIZE:]
            all_labels = all_labels[SAVE_BATCH_SIZE:]
            all_filenames = all_filenames[SAVE_BATCH_SIZE:]
            batch_idx += 1

        # Print progress every 500 images
        if processed % 500 == 0 or processed == total:
            print(f"  Processed {processed}/{total} images...")

    # Write remaining features
    if all_features:
        _write_batch(all_features, all_labels, all_filenames,
                     split_name, batch_idx, local_base, fs, spark)
        batch_idx += 1

    print(f"✅ Completed {split_name}: {processed} images processed in {batch_idx} batches.")
    return processed

def _write_batch(feats, labs, fnames, split_name, batch_idx, local_base, fs, spark):
    """Save a batch of features to local disk and upload to HDFS."""
    feats_arr = np.array(feats, dtype=np.float32)          # (batch, 2048)
    labs_arr = np.array(labs, dtype=np.int32)
    fnames_arr = np.array(fnames, dtype=str)

    # Local save
    batch_dir = os.path.join(local_base, f"batch_{batch_idx:04d}")
    os.makedirs(batch_dir, exist_ok=True)

    np.save(os.path.join(batch_dir, "resnet_features.npy"), feats_arr)
    np.save(os.path.join(batch_dir, "labels.npy"), labs_arr)
    np.save(os.path.join(batch_dir, "filenames.npy"), fnames_arr)
    print(f"  💾 Saved batch {batch_idx} locally ({len(feats)} images)")

    # Upload to HDFS
    try:
        hdfs_batch_dir = spark._jvm.org.apache.hadoop.fs.Path(
            f"{HDFS_RESNET_BASE}/{split_name}/batch_{batch_idx:04d}"
        )
        fs.mkdirs(hdfs_batch_dir)

        for fname in ["resnet_features.npy", "labels.npy", "filenames.npy"]:
            local_file = os.path.join(batch_dir, fname)
            local_path_obj = spark._jvm.org.apache.hadoop.fs.Path(local_file)
            hdfs_file_obj = spark._jvm.org.apache.hadoop.fs.Path(
                f"{HDFS_RESNET_BASE}/{split_name}/batch_{batch_idx:04d}/{fname}"
            )
            fs.copyFromLocalFile(local_path_obj, hdfs_file_obj)

        print(f"  ☁️  Uploaded batch {batch_idx} to HDFS")
    except Exception as e:
        print(f"  ⚠️ HDFS upload failed for batch {batch_idx}: {e}")

# ==================== MAIN ====================
def main():
    print("=" * 60)
    print("RESNET50 FEATURE EXTRACTION (DISK SPILLING)")
    print("=" * 60)

    warnings.filterwarnings('ignore')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.get_logger().setLevel('ERROR')
    tf.config.set_visible_devices([], 'GPU')

    # Check for trained model
    model_paths = ['/tmp/resnet50_fine_tuned_best.h5', '/tmp/resnet50_fine_tuned.h5']
    found = any(os.path.exists(p) for p in model_paths)
    if not found:
        print("\n⚠️  WARNING: No fine‑tuned ResNet50 model found in /tmp/")
        print("   Will use pre‑trained ResNet50 (ImageNet weights) instead.\n")
    else:
        print("\n✅ Fine‑tuned ResNet50 model found.")

    spark = create_spark_session()
    feature_extractor = load_trained_resnet50()

    try:
        total = 0
        for split in ['train', 'dev', 'test']:
            count = extract_features_split(split, feature_extractor, spark)
            if count is not None:
                total += count

        print("\n" + "=" * 60)
        print(f"✅ EXTRACTION COMPLETED: {total} total images")
        print("📁 Local batches: /tmp/resnet_features_batches/[train|dev|test]/batch_*")
        print(f"📁 HDFS batches:  {HDFS_RESNET_BASE}/[train|dev|test]/batch_*")
        print("\nEach batch contains:")
        print("  - resnet_features.npy : (batch, 2048)")
        print("  - labels.npy          : (batch,)")
        print("  - filenames.npy       : (batch,)")
    finally:
        spark.stop()

if __name__ == "__main__":
    main()