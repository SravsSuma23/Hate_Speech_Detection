import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pyspark.sql import SparkSession
from pyspark.sql import Row
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
HDFS_HOST = "namenode"
HDFS_PORT = "8020"
BASE_PATH = f"hdfs://{HDFS_HOST}:{HDFS_PORT}/data/facebook_hateful_memes"

TRAIN_BALANCED = f"{BASE_PATH}/balanced_train_pca"
DEV_PATH       = f"{BASE_PATH}/pca_features_dev"
TEST_PATH      = f"{BASE_PATH}/pca_features_test"
PREDICTIONS_OUTPUT = f"{BASE_PATH}/test_predictions"

# Training hyperparameters (as per paper)
BATCH_SIZE = 12
EPOCHS = 90
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.1
HIDDEN1 = 256
HIDDEN2 = 128
DROPOUT1 = 0.5
DROPOUT2 = 0.3
INPUT_DIM = 512
NUM_CLASSES = 2

# ==================== SPARK SESSION ====================
spark = SparkSession.builder \
    .appName("Normal Training (no adversarial)") \
    .master("spark://spark-master:7077") \
    .config("spark.executor.memory", "6g") \
    .config("spark.driver.memory", "8g") \
    .getOrCreate()
spark.sparkContext.setLogLevel("WARN")

# ==================== LOAD DATA ====================
print("\n📥 Loading balanced training data...")
train_df = spark.read.parquet(TRAIN_BALANCED)
train_features = np.array(train_df.select("features").rdd.map(lambda r: r[0].toArray()).collect())
train_labels = np.array(train_df.select("label").rdd.map(lambda r: r[0]).collect())

print("📥 Loading dev data...")
dev_df = spark.read.parquet(DEV_PATH)
dev_features = np.array(dev_df.select("pca_features").rdd.map(lambda r: r[0].toArray()).collect())
dev_labels = np.array(dev_df.select("label").rdd.map(lambda r: r[0]).collect())

print("📥 Loading test data...")
test_df = spark.read.parquet(TEST_PATH)
test_features = np.array(test_df.select("pca_features").rdd.map(lambda r: r[0].toArray()).collect())
test_filenames = test_df.select("filename").rdd.map(lambda r: r[0]).collect()

print(f"Train size: {len(train_features)}")
print(f"Dev size:   {len(dev_features)}")
print(f"Test size:  {len(test_features)}")

# ==================== PYTORCH DATASETS ====================
train_dataset = TensorDataset(torch.FloatTensor(train_features), torch.LongTensor(train_labels))
dev_dataset   = TensorDataset(torch.FloatTensor(dev_features),   torch.LongTensor(dev_labels))
test_dataset  = TensorDataset(torch.FloatTensor(test_features))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
dev_loader   = DataLoader(dev_dataset,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

# ==================== MODEL DEFINITION ====================
class MLP(nn.Module):
    def __init__(self, input_dim, hidden1, hidden2, num_classes, dropout1, dropout2):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.dropout1 = nn.Dropout(dropout1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.dropout2 = nn.Dropout(dropout2)
        self.fc3 = nn.Linear(hidden2, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

model = MLP(INPUT_DIM, HIDDEN1, HIDDEN2, NUM_CLASSES, DROPOUT1, DROPOUT2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Model on {device}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# ==================== NORMAL TRAINING LOOP (NO ADVERSARIAL) ====================
best_dev_acc = 0.0
best_model_path = "/tmp/best_model.pt"

for epoch in range(1, EPOCHS+1):
    model.train()
    total_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 100 == 0:
            print(f"Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f}")

    # Evaluate on dev set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in dev_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    dev_acc = 100.0 * correct / total
    print(f"Epoch {epoch} | Dev Accuracy: {dev_acc:.2f}%")

    if dev_acc > best_dev_acc:
        best_dev_acc = dev_acc
        torch.save(model.state_dict(), best_model_path)
        print(f"✅ Best model saved (acc={dev_acc:.2f}%)")

print(f"\n🏆 Best dev accuracy: {best_dev_acc:.2f}%")

# ==================== PREDICT ON TEST ====================
model.load_state_dict(torch.load(best_model_path))
model.eval()

all_preds = []
with torch.no_grad():
    for data in test_loader:
        data = data[0].to(device)
        outputs = model(data)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())

predictions = np.array(all_preds)

# ==================== SAVE PREDICTIONS ====================
print("\n💾 Saving test predictions to HDFS...")
pred_rows = [Row(filename=fname, predicted_label=int(pred)) for fname, pred in zip(test_filenames, predictions)]
pred_df = spark.createDataFrame(pred_rows)
pred_df.write.mode("overwrite").parquet(PREDICTIONS_OUTPUT)
print(f"✅ Predictions saved to {PREDICTIONS_OUTPUT}")

# Optional CSV
pred_df.coalesce(1).write.mode("overwrite").option("header","true").csv(PREDICTIONS_OUTPUT + "_csv")

spark.stop()
print("👋 Done.")