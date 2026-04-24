import os
import cv2
import math
import numpy as np
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO

# ---------- config ----------
MODEL_1     = "realsense_rgb.pt"   # RGB model
MODEL_2     = "realsense_ir.pt"    # IR  model
INPUT_DIR   = "/home/mannaja/intern_in_MuroranIT/intern_program/pictures/realsense/train_model_pictures"
OUTPUT_DIR  = "predict_ensemble"
CONF_THRESH = 0.5
IOU_THRESH  = 0.5          # NMS IoU threshold for merging boxes
MAX_PER_GRID = 16

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- load models ----------
model_rgb = YOLO(MODEL_1)
model_ir  = YOLO(MODEL_2)

# ---------- helpers ----------
def xywh_to_xyxy(box):
    x, y, w, h = box
    return [x - w/2, y - h/2, x + w/2, y + h/2]

def nms(boxes_xyxy, scores, iou_thresh):
    """Simple IoU-based NMS, returns kept indices."""
    if len(boxes_xyxy) == 0:
        return []
    boxes  = np.array(boxes_xyxy, dtype=float)
    scores = np.array(scores, dtype=float)
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    areas  = (x2 - x1) * (y2 - y1)
    order  = scores.argsort()[::-1]
    keep   = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w   = np.maximum(0, xx2 - xx1)
        h   = np.maximum(0, yy2 - yy1)
        inter = w * h
        iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou <= iou_thresh]
    return keep

def ensemble_and_draw(img_bgr, r1, r2, names):
    """Merge detections from 2 results and draw on image."""
    all_boxes  = []
    all_scores = []
    all_cls    = []

    for r in [r1, r2]:
        if r.boxes is None or len(r.boxes) == 0:
            continue
        for box in r.boxes:
            xyxy  = box.xyxy[0].cpu().numpy().tolist()
            score = float(box.conf[0].cpu())
            cls   = int(box.cls[0].cpu())
            if score >= CONF_THRESH:
                all_boxes.append(xyxy)
                all_scores.append(score)
                all_cls.append(cls)

    kept = nms(all_boxes, all_scores, IOU_THRESH)

    out = img_bgr.copy()
    for idx in kept:
        x1, y1, x2, y2 = [int(v) for v in all_boxes[idx]]
        score = all_scores[idx]
        cls   = all_cls[idx]
        label = f"{names.get(cls, cls)} {score:.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(out, label, (x1, max(y1 - 6, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 2)

    return out, len(kept)

def create_grid(images, cell_w=640, cell_h=480):
    resized = [cv2.resize(img, (cell_w, cell_h)).astype("uint8") for img in images]
    gs = int(math.ceil(math.sqrt(len(resized))))
    while len(resized) < gs * gs:
        resized.append(np.ones((cell_h, cell_w, 3), dtype="uint8") * 255)
    rows = []
    for i in range(0, len(resized), gs):
        rows.append(cv2.hconcat(resized[i:i+gs]))
    return cv2.vconcat(rows)

# ---------- collect images ----------
image_paths = sorted([
    p for p in Path(INPUT_DIR).rglob("*")
    if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
])

# class names (same for both models)
names = model_rgb.names   # dict {id: name}

folder_stats = defaultdict(lambda: {"found": 0, "not_found": 0})
grid_index   = 0

# ---------- process batches ----------
for i in range(0, len(image_paths), MAX_PER_GRID):
    batch = image_paths[i:i + MAX_PER_GRID]
    result_images = []

    for img_path in batch:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue

        r1 = model_rgb(img_bgr, conf=CONF_THRESH, verbose=False)[0]
        r2 = model_ir (img_bgr, conf=CONF_THRESH, verbose=False)[0]

        merged, n_det = ensemble_and_draw(img_bgr, r1, r2, names)

        folder = str(img_path.parent)
        if n_det > 0:
            folder_stats[folder]["found"]     += 1
            status, color = "FOUND",     (0, 255, 0)
        else:
            folder_stats[folder]["not_found"] += 1
            status, color = "NOT FOUND", (0, 0, 255)

        cv2.putText(merged, img_path.name, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(merged, status, (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        result_images.append(merged)

    if not result_images:
        continue

    grid = create_grid(result_images)
    save_path = os.path.join(OUTPUT_DIR, f"grid_{grid_index}.jpg")
    cv2.imwrite(save_path, grid)
    print(f"Saved: {save_path}")
    grid_index += 1

# ---------- summary ----------
print("\n===== SUMMARY PER FOLDER =====")
for folder, stats in folder_stats.items():
    found     = stats["found"]
    not_found = stats["not_found"]
    total     = found + not_found
    pct       = found / total * 100 if total else 0
    print(f"{folder}")
    print(f"   Found:          {found}")
    print(f"   Not Found:      {not_found}")
    print(f"   Total:          {total}")
    print(f"   Detection Rate: {pct:.2f}%")
