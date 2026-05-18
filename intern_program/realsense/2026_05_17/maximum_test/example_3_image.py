"""Example 3 - run on one image

One of 7 worked examples for the exported RealSense Cable pipeline.
Run:  python example_3_image.py [photo.jpg]
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
    """Feed in ONE IMAGE - the simplest use. Save every mask."""
    p = sys.argv[1] if len(sys.argv) > 1 else None
    if p and os.path.exists(p):
        bgr = cv2.imread(p)
    else:
        print("no image given - using a generated test image")
        bgr = demo_image()
    masks = detect(bgr)                    # dict {name: array}
    for name, value in masks.items():
        if not isinstance(value, np.ndarray):
            continue                       # skip non-image keys
        cv2.imwrite("ex3_%s.png" % name, value)
        print("example 3: wrote ex3_%s.png" % name)


if __name__ == "__main__":
    main()
