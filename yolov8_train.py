from ultralytics import YOLO
import os
import pandas as pd
from pathlib import Path

# Step 1: Rebuild data.yaml with correct Kaggle paths
fixed_yaml = """
train: /kaggle/input/dataset/train/images
val: /kaggle/input/dataset/valid/images
test: /kaggle/input/dataset/test/images

nc: 5
names: ['bicycle', 'bus', 'car', 'motorbike', 'person']
"""

with open("fixed_data.yaml", "w") as f:
    f.write(fixed_yaml)

# Step 2: Load and train YOLOv8-nano model
model = YOLO("yolov8n.pt")  # Nano version of YOLOv8

model.train(
    data="fixed_data.yaml",
    epochs=20,
    imgsz=640,
    batch=16,
    name="traffic_detector"
)

# Step 3: Load best trained model
trained_model = YOLO("runs/detect/traffic_detector/weights/best.pt")

#Step 4: Define traffic level classifier
def classify_traffic(count):
    if count <= 7:
        return "Low"
    elif count <= 15:
        return "Moderate"
    else:
        return "High"

# Step 5: Perform inference on validation set
val_dir = "/kaggle/input/dataset/valid/images"
results = []

for img_path in Path(val_dir).rglob("*.jpg"):
    preds = trained_model(img_path)
    count = len(preds[0].boxes)
    level = classify_traffic(count)
    results.append({
        "image": img_path.name,
        "vehicles": count,
        "traffic_level": level
    })

# Step 6: Convert to DataFrame and show results
df = pd.DataFrame(results)
print(df.head(10))

# ✅ Step 7: Save results
df.to_csv("traffic_results.csv", index=False)
