import os
import cv2
from ultralytics import YOLO

# ---------- setup ----------
output_dir = "test_predict"
base_name = "predict_result.jpg"

# create folder
os.makedirs(output_dir, exist_ok=True)

# filename
def get_unique_filename(folder, filename):
    name, ext = os.path.splitext(filename)
    counter = 0
    new_name = filename

    while os.path.exists(os.path.join(folder, new_name)):
        counter += 1
        new_name = f"{name}_{counter}{ext}"

    return os.path.join(folder, new_name)

# ---------- model ----------
model = YOLO("realsense_rgb.pt")

results = model("ir_raw_1776668485.png", conf=0.5)

# ---------- save ----------
for r in results:
    img = r.plot()  # draw bbox
    save_path = get_unique_filename(output_dir, base_name)
    cv2.imwrite(save_path, img)
    print(f"Saved to: {save_path}")