import os
import cv2
import math
import numpy as np
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO

# ---------- load model ----------
model_path = "realsense_merged.pt"
model = YOLO(model_path)

# ---------- config ----------
input_dir = "/home/mannaja/intern_in_MuroranIT/intern_program/pictures/realsense/train_model_pictures"
output_dir = f"predict_{model_path}"
max_images_per_grid = 16

os.makedirs(output_dir, exist_ok=True)

# ---------- search all images file (recursive) ----------
image_paths = list(Path(input_dir).rglob("*"))
image_paths = [p for p in image_paths if p.suffix.lower() in [".jpg", ".png", ".jpeg"]]
image_paths.sort()

# ---------- count per folder ----------
folder_stats = defaultdict(lambda: {"found": 0, "not_found": 0})

# ---------- function: create grid ----------
def create_grid(images):
    h, w = images[0].shape[:2]

    resized_images = []
    for img in images:
        img = cv2.resize(img, (w, h))
        img = img.astype("uint8")
        resized_images.append(img)

    grid_size = int(math.ceil(math.sqrt(len(resized_images))))

    while len(resized_images) < grid_size * grid_size:
        resized_images.append(255 * np.ones((h, w, 3), dtype="uint8"))

    rows = []
    for i in range(0, len(resized_images), grid_size):
        row_imgs = resized_images[i:i+grid_size]
        row_imgs = [cv2.resize(img, (w, h)) for img in row_imgs]
        row = cv2.hconcat(row_imgs)
        rows.append(row)

    max_width = max(r.shape[1] for r in rows)
    fixed_rows = []

    for r in rows:
        if r.shape[1] != max_width:
            r = cv2.resize(r, (max_width, r.shape[0]))
        fixed_rows.append(r)

    return cv2.vconcat(fixed_rows)

# ---------- process ----------
grid_index = 0

for i in range(0, len(image_paths), max_images_per_grid):
    batch_paths = image_paths[i:i + max_images_per_grid]

    result_images = []

    for img_path in batch_paths:
        results = model(str(img_path), conf=0.5)

        # 🔥 folder name
        folder_name = str(img_path.parent)

        for r in results:
            plotted = r.plot()

            # ---------- count detection ----------
            if len(r.boxes) > 0:
                folder_stats[folder_name]["found"] += 1
                label = "FOUND"
                color = (0, 255, 0)
            else:
                folder_stats[folder_name]["not_found"] += 1
                label = "NOT FOUND"
                color = (0, 0, 255)

            # ---------- input file name ----------
            filename = img_path.name
            cv2.putText(plotted, filename, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # ---------- input status ----------
            cv2.putText(plotted, label, (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            result_images.append(plotted)

    if len(result_images) == 0:
        continue

    grid_img = create_grid(result_images)

    save_path = os.path.join(output_dir, f"grid_{grid_index}.jpg")
    cv2.imwrite(save_path, grid_img)

    print(f"Saved: {save_path}")
    grid_index += 1

# ---------- summary ----------
print("\n===== SUMMARY PER FOLDER =====")
for folder, stats in folder_stats.items():
    found = stats["found"]
    not_found = stats["not_found"]
    total = found + not_found

    percent = (found / total * 100) if total > 0 else 0

    print(f"{folder}")
    print(f"   Found: {found}")
    print(f"   Not Found: {not_found}")
    print(f"   Total: {total}")
    print(f"   Detection Rate: {percent:.2f}%")