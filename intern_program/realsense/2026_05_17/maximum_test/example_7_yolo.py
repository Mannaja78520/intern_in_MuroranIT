"""Example 7 - YOLO object detection

One of 7 worked examples for the exported RealSense Cable pipeline.
Run:  python example_7_yolo.py [photo.jpg]
"""
import os
import sys

import cv2
import numpy as np

# Make the exported pipeline importable whether it sits next to this
# file or one folder up, then import it. The exported file is a
# self-contained library: every tuned setting lives in CONFIG.
_HERE = os.path.dirname(os.path.abspath(__file__))
for _d in (_HERE, os.path.dirname(_HERE)):
    if _d not in sys.path:
        sys.path.insert(0, _d)
from exported_pipeline import detect, run_yolo, find_objects, CONFIG


def demo_image():
    """A throw-away test image (red box on white) so the example runs
    even before you have a real photo."""
    img = np.full((240, 320, 3), 255, np.uint8)
    cv2.rectangle(img, (110, 80), (210, 160), (0, 0, 200), -1)
    return img


def main():
    """YOLO object detection.

    When the exported config had 'Use YOLO' ticked, detect() ALREADY
    runs YOLO - the result then has 'yolo_boxes' and 'yolo_mask'. You
    can also call run_yolo() directly, as shown here.
    Needs:  pip install ultralytics   + the model weights file."""
    p = sys.argv[1] if len(sys.argv) > 1 else None
    bgr = cv2.imread(p) if p and os.path.exists(p) else demo_image()
    if not CONFIG.get("export_yolo", {}).get("enabled"):
        print("example 7: this export had YOLO OFF - tick 'Use YOLO' "
              "in the analyzer and re-export to use it.")
        return
    try:
        boxes, ymask = run_yolo(bgr)
    except Exception as e:
        print("example 7: YOLO needs 'ultralytics' + weights -", e)
        return
    annotated = bgr.copy()
    for d in boxes:
        cv2.rectangle(annotated, (d["x"], d["y"]),
                      (d["x"] + d["w"], d["y"] + d["h"]), (0, 255, 0), 2)
        cv2.putText(annotated, "%s %.2f" % (d["name"], d["conf"]),
                    (d["x"], max(12, d["y"] - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imwrite("ex7_yolo.png", annotated)
    print("example 7: YOLO found %d object(s) -> ex7_yolo.png"
          % len(boxes))


if __name__ == "__main__":
    main()
