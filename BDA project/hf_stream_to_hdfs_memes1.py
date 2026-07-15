import os
import json
import requests
from hdfs import InsecureClient
from datasets import load_dataset

# ==================== CONFIGURATION ====================
HDFS_HOST = "namenode"
HDFS_PORT = "8020"
HDFS_BASE_URL = f"hdfs://{HDFS_HOST}:{HDFS_PORT}/data/facebook_hateful_memes1"  # for Spark later
HDFS_BASE_PATH = "/data/facebook_hateful_memes1"                                 # for client operations

# Use RAM disk for Hugging Face cache (optional, but avoids disk writes)
CACHE_DIR = "/dev/shm/hf_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["HF_DATASETS_CACHE"] = CACHE_DIR
os.environ["HF_HOME"] = CACHE_DIR

# Hugging Face dataset identifier
DATASET_NAME = "neuralcatcher/hateful_memes"

# Base URL for image files on Hugging Face
HF_BASE_URL = "https://huggingface.co/datasets/neuralcatcher/hateful_memes/resolve/main"

# HDFS directories (absolute paths for client)
HDFS_RAW = f"{HDFS_BASE_PATH}/raw"
HDFS_IMG = f"{HDFS_BASE_PATH}/img"

# ==================== HDFS CLIENT ====================
client = InsecureClient(f"http://{HDFS_HOST}:9870", user="spark")

def ensure_hdfs_dir(path):
    """Create HDFS directory if it doesn't exist."""
    try:
        client.status(path)
    except:
        client.makedirs(path)

ensure_hdfs_dir(HDFS_RAW)
ensure_hdfs_dir(HDFS_IMG)

# ==================== STREAM DATASET ====================
print("📥 Loading dataset in streaming mode...")
ds = load_dataset(DATASET_NAME, streaming=True)

# We'll collect JSONL lines per split in memory (small)
jsonl_buffers = {"train": [], "validation": [], "test": []}

for split_name, split_ds in ds.items():
    print(f"\n🔹 Processing split: {split_name}")
    for idx, example in enumerate(split_ds):
        # Example fields: 'id', 'img', 'label', 'text'
        img_rel_path = example['img']          # e.g., "img/42953.png"
        img_url = f"{HF_BASE_URL}/{img_rel_path}"
        filename = os.path.basename(img_rel_path)

        # Upload image to HDFS by streaming from Hugging Face
        print(f"  ☁️  Uploading image: {filename}")
        try:
            # Stream the image from HF
            r = requests.get(img_url, stream=True)
            r.raise_for_status()

            # Write directly to HDFS in chunks
            hdfs_img_path = f"{HDFS_IMG}/{filename}"
            with client.write(hdfs_img_path, overwrite=True, encoding=None) as writer:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        writer.write(chunk)
        except Exception as e:
            print(f"    ⚠️ Failed to upload {filename}: {e}")
            continue

        # Prepare JSONL line
        json_line = json.dumps({
            "id": example['id'],
            "img": f"img/{filename}",
            "label": example['label'],
            "text": example['text']
        }) + "\n"
        jsonl_buffers[split_name].append(json_line)

        # Optional: progress indicator
        if idx % 100 == 0:
            print(f"    Processed {idx} examples...")

# ==================== WRITE JSONL FILES ====================
print("\n📝 Writing JSONL files to HDFS...")
for split_name, lines in jsonl_buffers.items():
    hdfs_jsonl_path = f"{HDFS_RAW}/{split_name}.jsonl"
    all_lines = "".join(lines).encode('utf-8')
    client.write(hdfs_jsonl_path, data=all_lines, overwrite=True)
    print(f"  ✅ Uploaded {split_name}.jsonl ({len(lines)} lines)")

# ==================== CLEANUP ====================
print(f"\n🧹 Cleaning up cache directory: {CACHE_DIR}")
import shutil
shutil.rmtree(CACHE_DIR, ignore_errors=True)
print("👋 Done.")