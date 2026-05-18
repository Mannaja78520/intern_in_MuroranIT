"""Example 4 - use only one mask

One of 7 worked examples for the exported RealSense Cable pipeline.
Run:  python example_4_one_pipeline.py [photo.jpg]
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
    """Use ONLY ONE pipeline / one mask.

    detect() returns rgb_mask, ir_mask, combined and every branch. If
    you only need one, just take that key and ignore the rest."""
    p = sys.argv[1] if len(sys.argv) > 1 else None
    bgr = cv2.imread(p) if p and os.path.exists(p) else demo_image()
    rgb_only = detect(bgr)["rgb_mask"]     # <-- the only thing we want
    cv2.imwrite("ex4_rgb_only.png", rgb_only)
    print("example 4: rgb_mask has %d detected pixels"
          % int((rgb_only > 0).sum()))


if __name__ == "__main__":
    main()
