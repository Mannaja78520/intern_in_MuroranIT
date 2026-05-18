"""Example 2 - process a video file

One of 7 worked examples for the exported RealSense Cable pipeline.
Run:  python example_2_video.py clip.mp4
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
    """Process a VIDEO file frame by frame and save a mask video."""
    if len(sys.argv) < 2 or not os.path.exists(sys.argv[1]):
        print("usage: python example_2_video.py clip.mp4")
        return
    cap = cv2.VideoCapture(sys.argv[1])
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter("ex2_mask.mp4",
                          cv2.VideoWriter_fourcc(*"mp4v"),
                          fps, (w, h), isColor=False)
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        masks = detect(frame)              # <-- run the pipeline
        out.write(masks["combined"])
        n += 1
    cap.release()
    out.release()
    print("example 2: processed %d frames -> ex2_mask.mp4" % n)


if __name__ == "__main__":
    main()
