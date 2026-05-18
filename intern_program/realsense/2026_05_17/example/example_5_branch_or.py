"""Example 5 - branch + OR with main RGB

One of 7 worked examples for the exported RealSense Cable pipeline.
Run:  python example_5_branch_or.py [photo.jpg]
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
    """Combine a BRANCH with the main RGB mask using OR.

    Two ways to do it:
    A) INSIDE the analyzer GUI: add a step to the main RGB pipeline,
       set Combine = OR, source = up_branch1. Then the OR is baked in
       and detect()['rgb_mask'] already contains it.
    B) OUTSIDE, on the returned masks - shown here."""
    p = sys.argv[1] if len(sys.argv) > 1 else None
    bgr = cv2.imread(p) if p and os.path.exists(p) else demo_image()
    masks  = detect(bgr)
    branch = masks.get("up_branch1")
    if branch is None:
        print("example 5: no branch 'up_branch1' in this export - "
              "add one in the analyzer and re-export.")
        return
    merged = cv2.bitwise_or(masks["rgb_mask"], branch)   # <-- the OR
    cv2.imwrite("ex5_merged.png", merged)
    print("example 5: wrote ex5_merged.png  (rgb_mask OR up_branch1)")


if __name__ == "__main__":
    main()
