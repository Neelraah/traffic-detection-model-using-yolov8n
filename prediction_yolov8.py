from ultralytics import YOLO
import os
import pandas as pd

# Load the trained YOLOv8 model
model = YOLO('yolov8n.pt')

# Define test directory
test_dir = 'test/images'

# Define device
device = 'cpu'  # Change to 'cuda' if using GPU

# Function to classify traffic as Light, Moderate, or Heavy
def classify_traffic(image_path):
    results = model.predict(image_path, device=device)
    # Get predictions
    boxes = results[0].boxes.data
    # Convert to DataFrame for easier handling
    df = pd.DataFrame(boxes.cpu().numpy(), columns=['x1', 'y1', 'x2', 'y2', 'confidence', 'class'])
    # Filter for vehicle classes (bus=1, car=2, motorbike=3)
    vehicle_classes = [1, 2, 3]
    vehicle_count = df[df['class'].isin(vehicle_classes)].shape[0]
    
    # Classify traffic level
    if vehicle_count >= 15:
        traffic = 'Heavy'
    elif vehicle_count >= 8:
        traffic = 'Moderate'
    else:
        traffic = 'Light'
    
    return vehicle_count, traffic

# Get list of test images
image_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith(('.jpg', '.png'))]

# Store results
results_list = []

# Loop through each test image and classify traffic
for image in image_files:
    count, traffic = classify_traffic(image)
    print(f'{image}: {count} vehicles, Traffic: {traffic}')
    results_list.append({'image': image, 'vehicle_count': count, 'traffic': traffic})

# Save results to CSV
results_df = pd.DataFrame(results_list)
results_df.to_csv('test_traffic_classification_results.csv', index=False)
print("Results saved to test_traffic_classification_results.csv")
