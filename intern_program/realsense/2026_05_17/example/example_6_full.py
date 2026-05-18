"""Example 6 - everything together

One of 7 worked examples for the exported RealSense Cable pipeline.
Run:  python example_6_full.py [photo.jpg]
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
    """EVERYTHING: detect, find objects, draw boxes, save."""
    p = sys.argv[1] if len(sys.argv) > 1 else None
    bgr = cv2.imread(p) if p and os.path.exists(p) else demo_image()
    masks = detect(bgr)
    # Min_area comes from the exported CONFIG (the value you tuned).
    min_area = CONFIG["params"].get("Min_area", 100)
    boxes = find_objects(masks["combined"], min_area)
    annotated = bgr.copy()
    for (x, y, w, h) in boxes:
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imwrite("ex6_annotated.png", annotated)
    cv2.imwrite("ex6_combined.png", masks["combined"])
    print("example 6: found %d object(s) -> ex6_annotated.png"
          % len(boxes))


if __name__ == "__main__":
    main()
