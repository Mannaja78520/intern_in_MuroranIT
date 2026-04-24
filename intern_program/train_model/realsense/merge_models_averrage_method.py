"""
Merge 2 YOLO models (same architecture, same labels) into 1 by averaging weights.
Method: Model Soup — merged_weight = (w1 * alpha) + (w2 * (1 - alpha))
"""
import torch
from ultralytics import YOLO

# ---------- config ----------
MODEL_1    = "realsense_rgb.pt"
MODEL_2    = "realsense_ir.pt"
ALPHA      = 0.7   # 0.5 = equal weight, 0.7 = trust model1 more
OUTPUT     = f"realsense_merged_{ALPHA}.pt"

# ---------- load ----------
model1 = YOLO(MODEL_1)
model2 = YOLO(MODEL_2)

state1 = model1.model.state_dict()
state2 = model2.model.state_dict()

# ---------- sanity check ----------
keys1, keys2 = set(state1.keys()), set(state2.keys())
if keys1 != keys2:
    raise ValueError("Models have different layers — cannot merge.")

print(f"Merging: {MODEL_1}  (alpha={ALPHA})")
print(f"     +   {MODEL_2}  (alpha={1-ALPHA})")

# ---------- merge ----------
merged_state = {}
for key in state1:
    t1, t2 = state1[key], state2[key]
    if t1.dtype.is_floating_point:
        merged_state[key] = ALPHA * t1 + (1 - ALPHA) * t2
    else:
        # integer tensors (e.g. num_batches_tracked) — keep from model1
        merged_state[key] = t1

# ---------- load merged weights into model1 and save ----------
model1.model.load_state_dict(merged_state)
model1.save(OUTPUT)
print(f"\nSaved merged model → {OUTPUT}")
print(f"Classes: {model1.names}")
