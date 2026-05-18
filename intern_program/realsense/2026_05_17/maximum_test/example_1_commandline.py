"""Example 1 - command-line arguments

One of 7 worked examples for the exported RealSense Cable pipeline.
Run:  python example_1_commandline.py [photo.jpg] [ir.png]
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
    """COMMAND-LINE ARGUMENTS.

    The exported pipeline file is runnable on its own:
        python exported_pipeline.py photo.jpg
        python exported_pipeline.py photo.jpg ir.png
    argv[1] = RGB image, argv[2] = optional IR image. Here we read
    those arguments ourselves and run the pipeline."""
    img_path = sys.argv[1] if len(sys.argv) > 1 else None
    ir_path  = sys.argv[2] if len(sys.argv) > 2 else None
    if img_path and os.path.exists(img_path):
        bgr = cv2.imread(img_path)
    else:
        print("no image given - using a generated test image")
        bgr = demo_image()
    ir = cv2.imread(ir_path) if ir_path and os.path.exists(ir_path) else None
    masks = detect(bgr, ir)
    cv2.imwrite("ex1_combined.png", masks["combined"])
    print("example 1: wrote ex1_combined.png")


if __name__ == "__main__":
    main()
