from ultralytics import YOLO

# Load YOLOv8 model (nano version - change to s, m, l, x for larger models)
model = YOLO('yolov8n.pt')

# Train the model
model.train(
    data='data.yaml',          # Path to your data.yaml file
    epochs=10,                 # Number of training epochs
    batch=8,                  # Batch size (lower if running out of memory)
    imgsz=640,                 # Image resolution (can adjust to 512 for faster training)
    save=True,                 # Save the best model
    save_period=5,             # Save every 5 epochs
    workers=4,                 # Number of CPU threads to use
    device='cpu'                   # GPU device (set to 'cpu' for CPU-only)
)

# Save the final model after training
model.export(format='torchscript', path='yolov8_traffic_model.pt')

print("Training complete! Model saved to 'yolov8_traffic_model.pt'")
