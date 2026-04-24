import os
import cv2
from ultralytics import YOLO

# ---------- config ----------
MODEL      = "realsense_merged.pt"
IR_IMAGE   = "ir_raw_1776668485.png"
RGB_IMAGE  = "color_raw_1776668432.png"
OUTPUT_DIR = "test_predict"
BASE_NAME  = f"predict_result_pair_{MODEL}.jpg"
CONF       = 0.5

os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_unique_filename(folder, filename):
    name, ext = os.path.splitext(filename)
    counter = 0
    new_name = filename
    while os.path.exists(os.path.join(folder, new_name)):
        counter += 1
        new_name = f"{name}_{counter}{ext}"
    return os.path.join(folder, new_name)

def add_label(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (img.shape[1], 36), (20, 20, 20), -1)
    cv2.putText(out, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 100), 2)
    return out

# ---------- predict ----------
model = YOLO(MODEL)

r_ir  = model(IR_IMAGE,  conf=CONF, verbose=False)[0]
r_rgb = model(RGB_IMAGE, conf=CONF, verbose=False)[0]

img_ir  = add_label(r_ir.plot(),  f"IR Image  | {len(r_ir.boxes)} det")
img_rgb = add_label(r_rgb.plot(), f"RGB Image | {len(r_rgb.boxes)} det")

# same height, side by side (2x1)
h = max(img_ir.shape[0], img_rgb.shape[0])
img_ir  = cv2.resize(img_ir,  (int(img_ir.shape[1]  * h / img_ir.shape[0]),  h))
img_rgb = cv2.resize(img_rgb, (int(img_rgb.shape[1] * h / img_rgb.shape[0]), h))

combined = cv2.hconcat([img_ir, img_rgb])

save_path = get_unique_filename(OUTPUT_DIR, BASE_NAME)
cv2.imwrite(save_path, combined)
print(f"Saved to: {save_path}")
