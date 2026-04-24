from ultralytics import YOLO

# Load model (choose size)
model = YOLO("yolo26n.pt")   # n/s/m/l/x

# Train
model.train(
    data="muroranit_t-hook_combined/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device=0  # GPU (or "cpu")
)